#pragma once

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// COMPILER FLAG PARAMETERS

// #define VERBOSE

#define INIT__ORIG        // initialize w/ original code
// #define GRAD__PERMUTE     // shuffle nodes of B -- this improves results (more randomness), but does slow things down
// #define LAP__FULLSORT     // stable sort instead of introselect for topk
// #define LAP__HEAPSORT     // queue sort instead of introselect

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

// base
#include "algorithm"
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <random>
#include <cmath>

// external
#include "extern/lapjv.h"

// ours
#include "data.h"
#include "timer.h"
#include "matrix_fun.h"
#include "lap_fun.h"
#include "init_fun.h"

void mgmmf(
  int_t*           ind,     // output
  int_t            nc,      // number of channels
  int_t            nt,      // number of tmplt nodes
  int_t            nw,      // number of world nodes
  csr_t*           A,       // stacked adj of tmplt
  csr_t*           At,      // (stacked adj of tmplt).T
  csr_t*           B,       // stacked adj of world
  csr_t*           Bt,      // (stacked adj of world).T
  csr_t*           sim_sp,  // similarity matrix (sparse representation)
  arr2d_t<cost_t>* sim,     // similarity matrix (dense representation)
  csr_t*           P_ex      = nullptr,
  int_t*           w_p_ex    = nullptr,
  uint32_t         seed      = 123,
  
  arr2d_t<int_t>*  solution_counter = nullptr,
  cost_t           scale_eps        = 1.0,
  bool             scale_init       = false,
  bool             scale_sim        = false,
  bool             scale_grad       = false,
  
  int_t            max_iter  = 20,
  
  int_t            init_iter             = 20,  // Number of iterations for double-stochastic initialization
  bool             init_doublestochastic = true // Use double-stochastic initialization (vs just row normalization)
) {
    cost_t c, d0, u, d1, e, v;
    cost_t d, z0, z1, f1;
    cost_t alpha, falpha;

    arr2d_t<cost_t> _grad(sim->nrow, sim->ncol);
    arr2d_t<cost_t> _Z0(sim->nrow, sim->ncol);
    arr2d_t<cost_t> _Z1(sim->nrow, sim->ncol);
    arr2d_t<cost_t> _Z0p(sim->nrow, sim->ncol);
    arr2d_t<cost_t> _Z1p(sim->nrow, sim->ncol);
    
    arr2d_t<cost_t>* grad = &_grad;
    arr2d_t<cost_t>* Z0   = &_Z0;
    arr2d_t<cost_t>* Z1   = &_Z1;
    arr2d_t<cost_t>* Z0p  = &_Z0p;
    arr2d_t<cost_t>* Z1p  = &_Z1p;
    
    // --
    // Initialize random shuffle

    int_t* w_p = (int_t*)malloc(nw * sizeof(int_t));
#if defined(GRAD__PERMUTE) || defined(LAP__HEAPSORT)
    if(w_p_ex == nullptr) {
      for(int i = 0 ; i < nw ; i++)
        w_p[i] = i;
      
      srand(seed);
      random_shuffle(w_p, w_p + nw);
    } else {
      for(int i = 0 ; i < nw ; i++)
        w_p[i] = w_p_ex[i];
    }
#endif

    // --
    // Initialize P
    
    csr_t _P;
    csr_t* P = &_P;    
    if(P_ex != nullptr) {
      P->copy(P_ex);
    } else {
      if(init_doublestochastic) {
        do_init_doublestochastic(sim_sp, P, seed, init_iter);
      } else {
        do_init_rownorm(sim_sp, P, seed);
      }
    }
    
    // --
    // Run
    
    stacked_multiply(Z0, nc, At, P, B);
    stacked_multiply(Z1, nc, A, P, Bt);
    
    mu_timer_t tt;
    for(int_t it = 0; it < max_iter; it++) {
      
      // Compute grad
#ifdef GRAD__PERMUTE
      for(int_t r = 0; r < sim->nrow; r++) {
        for(int_t c = 0; c < sim->ncol; c++) {
          int_t wi = r * sim->ncol + c;
          int_t ri = r * sim->ncol + w_p[c];
          grad->data[wi] = Z0->data[ri] + Z1->data[ri] + sim->data[ri]; // bad read locality
          if(scale_grad) {
            grad->data[wi] *= pow(scale_eps, solution_counter->data[ri]); // scale gradient - untested
          }
        }
      }
#else
      #pragma GCC ivdep
      for(int_t i = 0; i < sim->nnz; i++) {
        grad->data[i] = Z0->data[i] + Z1->data[i] + sim->data[i];
        if(scale_grad) {
          grad->data[i] *= pow(scale_eps, solution_counter->data[i]); // scale gradient
        }
      }
#endif

      // Solve LAP
      rect_lap(grad->nrow, grad->ncol, grad->data, grad->data, ind, w_p);

#ifdef GRAD__PERMUTE
      for(int_t i = 0; i < nt; i++)
        ind[i] = w_p[ind[i]];
#endif

      stacked_multiply(Z0p, nc, At, ind, B);
      stacked_multiply(Z1p, nc, A, ind, Bt);
      
      compute_traces(c, d0, u, Z0, Z0p, sim, P);
      compute_traces(d1, e, v, Z0, Z0p, sim, ind, nt);

      d  = d0 + d1;
      z0 = c - d + e;
      z1 = d - 2 * e + u - v;
      f1 = c - e + u - v;
      
      if((z0 == 0) && (z1 == 0)) {
        alpha = 0;
        falpha = z0 * std::pow(alpha, 2) + z1 * alpha;
      } else if(z0 == 0) {
        alpha  = std::numeric_limits<cost_t>::max();
        falpha = std::numeric_limits<cost_t>::max();
      } else {
        alpha  = - z1 / (2 * z0);
        falpha = z0 * std::pow(alpha, 2) + z1 * alpha;
      }
      
      if((alpha < 1) && (alpha > 0) && (falpha > 0) && (falpha > f1)) {
        sparse_convex_combination(alpha, ind, P);
        dense_convex_combination(alpha, Z0p, Z0);
        dense_convex_combination(alpha, Z1p, Z1);
      } else if (f1 < 0) {
        free(P->indptr);
        free(P->indices);
        free(P->data);
        
        P->indptr  = (int_t*)malloc((nt + 1) * sizeof(int_t));
        P->indices = (int_t*)malloc(nt * sizeof(int_t));
        P->data    = (cost_t*)malloc(nt * sizeof(cost_t));
        
        P->nnz = nt;
        
        for(int i = 0; i < nt + 1; i++) P->indptr[i]  = i;
        for(int i = 0; i < nt    ; i++) P->indices[i] = ind[i];
        for(int i = 0; i < nt    ; i++) P->data[i]    = 1;
        
        std::swap(Z0, Z0p);
        std::swap(Z1, Z1p);
      } else {
        break;
      }
    }
    
    arr2d_t<cost_t> P_dense(P->nrow, P->ncol);
    for(int_t i = 0; i < P->nrow; i++) {
      for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
        int_t j = P->indices[offset];
        P_dense.data[i * P->ncol + j] = P->data[offset];
      }
    }
    rect_lap(P_dense.nrow, P_dense.ncol, P_dense.data, P_dense.data, ind, nullptr);
    
    // --
    // Free memory
    
    free(P_dense.data);
    
    free(P->indptr);
    free(P->indices);
    free(P->data);
    
    free(grad->data);
    free(Z0->data);
    free(Z1->data);
    free(Z0p->data);
    free(Z1p->data);

    for(int_t i = 0; i < nt; i++) {
      solution_counter->data[i * solution_counter->ncol + ind[i]]++;
      
      // Scaling sim in objective - should double check
      if(scale_sim) {
        sim->data[i * sim->ncol + ind[i]] *= scale_eps;
      }

      // Scaling sim in restart - should double check
      if(scale_init) {
        for(int_t offset = sim_sp->indptr[i] ; offset < sim_sp->indptr[i + 1]; offset++) {
          int_t idx = sim_sp->indices[offset];
          if(idx == ind[i]) {
            sim_sp->data[offset] *= scale_eps;
          }
        }
      }
    }

#ifdef GRAD__PERMUTE
    free(w_p);
#endif
}
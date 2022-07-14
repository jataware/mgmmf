#pragma once

void do_init_rownorm(csr_t* sim_sp, csr_t* P, uint32_t seed) {
  P->nrow    = sim_sp->nrow;
  P->ncol    = sim_sp->ncol;
  P->nnz     = sim_sp->nnz;
  P->indptr  = (int_t* )malloc((P->nrow + 1) * sizeof(int_t));
  P->indices = (int_t* )malloc(P->nnz        * sizeof(int_t));
  P->data    = (cost_t*)malloc(P->nnz        * sizeof(cost_t));
  
  memcpy(P->indptr,  sim_sp->indptr,  (P->nrow + 1) * sizeof(int_t));
  memcpy(P->indices, sim_sp->indices,  P->nnz * sizeof(int_t));
  
  std::mt19937 gen{seed};
  std::normal_distribution<cost_t> norm{0, 2};
  for(int_t i = 0; i < P->nnz; i++) 
    P->data[i] = exp(sim_sp->data[i] + norm(gen));

  cost_t* r_row = (cost_t*)malloc(P->nrow * sizeof(cost_t));
  
  // Row norm
  for(int_t i = 0; i < P->nrow; i++) r_row[i] = 0;
  
  for(int_t i = 0; i < P->nrow; i++) {
    for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
      r_row[i] += P->data[offset];
    }
  }
  
  for(int_t i = 0; i < P->nrow; i++) r_row[i] = r_row[i] == 0 ? 1 : 1 / r_row[i];
  
  for(int_t i = 0; i < P->nrow; i++) {
    for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
      P->data[offset] *= r_row[i];
    }
  }
  
  free(r_row);
}

#ifdef INIT__ORIG

void do_init_doublestochastic(csr_t* sim_sp, csr_t* P, uint32_t seed, int_t init_iter) {
  // Possibly doing this on a dense matrix is better?

  P->nrow    = sim_sp->nrow;
  P->ncol    = sim_sp->ncol;
  P->nnz     = sim_sp->nnz;
  P->indptr  = (int_t* )malloc((P->nrow + 1) * sizeof(int_t));
  P->indices = (int_t* )malloc(P->nnz        * sizeof(int_t));
  P->data    = (cost_t*)malloc(P->nnz        * sizeof(cost_t));
  
  memcpy(P->indptr,  sim_sp->indptr,  (P->nrow + 1) * sizeof(int_t));
  memcpy(P->indices, sim_sp->indices,  P->nnz * sizeof(int_t));
  
  std::mt19937 gen{seed};
  std::normal_distribution<cost_t> norm{0, 2};
  for(int_t i = 0; i < P->nnz; i++) 
    P->data[i] = exp(sim_sp->data[i] + norm(gen));

  cost_t* r_row = (cost_t*)malloc(P->nrow * sizeof(cost_t));
  cost_t* r_col = (cost_t*)malloc(P->ncol * sizeof(cost_t));
  
  for(int_t it = 0 ; it < 1 ; it++) {
    
    // Column norm
    for(int_t i = 0; i < P->ncol; i++) r_col[i] = 0;
    
    for(int_t i = 0; i < P->nrow; i++) {
      for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
        int_t j = P->indices[offset];
        r_col[j] += P->data[offset];
      }
    }
    
    for(int_t i = 0; i < P->ncol; i++) r_col[i] = r_col[i] == 0 ? 1 : 1 / r_col[i];
    
    for(int_t i = 0; i < P->nrow; i++) {
      for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
        int_t j = P->indices[offset];
        P->data[offset] *= r_col[j]; 
      }
    }
    
    // Row norm
    for(int_t i = 0; i < P->nrow; i++) r_row[i] = 0;
    
    for(int_t i = 0; i < P->nrow; i++) {
      for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
        r_row[i] += P->data[offset];
      }
    }
    
    for(int_t i = 0; i < P->nrow; i++) r_row[i] = r_row[i] == 0 ? 1 : 1 / r_row[i];
    
    for(int_t i = 0; i < P->nrow; i++) {
      for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
        P->data[offset] *= r_row[i];
      }
    }
  }
  
  free(r_row);
  free(r_col);
}

#else

void do_init_doublestochastic(csr_t* sim_sp, csr_t* P, uint32_t seed, int_t init_iter) {
  mu_timer_t t;
  
  P->nrow    = sim_sp->nrow;
  P->ncol    = sim_sp->ncol;
  P->nnz     = sim_sp->nnz;

  P->indptr  = (int_t* )malloc((P->nrow + 1) * sizeof(int_t));
  P->indices = (int_t* )malloc(P->nnz        * sizeof(int_t));
  P->data    = (cost_t*)malloc(P->nnz        * sizeof(cost_t));
  
  memcpy(P->indptr,  sim_sp->indptr,  (P->nrow + 1) * sizeof(int_t));
  memcpy(P->indices, sim_sp->indices,  P->nnz       * sizeof(int_t));
  
  std::mt19937 gen{seed};
  std::normal_distribution<cost_t> norm{0, 2};
  for(int_t i = 0; i < P->nnz; i++)  {
    P->data[i] = exp(sim_sp->data[i] + norm(gen));
  }


  cost_t* u = (cost_t*)malloc(P->nrow * sizeof(cost_t));
  cost_t* v = (cost_t*)malloc(P->ncol * sizeof(cost_t));
  
  for(int_t i = 0; i < P->nrow; i++) u[i] = 1.0 / P->nrow;
  for(int_t i = 0; i < P->ncol; i++) v[i] = 1.0 / P->ncol;
  
  std::cout << "P->ncol: " << P->ncol << std::endl;
  std::cout << "P->nrow: " << P->nrow << std::endl;
  
  for(int_t it = 0; it < init_iter; it++) {
    
    // v = 1 / (u @ M)
    
    for(int_t i = 0; i < P->ncol; i++) v[i] = 0;
    
    for(int_t i = 0; i < P->nrow; i++) {
      for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
        int_t j = P->indices[offset];
        v[j] += P->data[offset] * u[i];
      }
    }
    
    
    for(int_t i = 0; i < P->ncol; i++) v[i] = 1 / v[i];
    
    // u = 1 / (M @ v)
    
    for(int_t i = 0; i < P->nrow; i++) u[i] = 0;
    
    
    for(int_t i = 0; i < P->nrow; i++) {
      for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
        int_t j = P->indices[offset];
        u[i] += P->data[offset] * v[j];
      }
    }
    
    
    for(int_t i = 0; i < P->nrow; i++) u[i] = 1 / u[i];
  }
  
  
  for(int_t i = 0; i < P->nrow; i++) {
    for(int_t offset = P->indptr[i] ; offset < P->indptr[i + 1]; offset++) {
      int_t j = P->indices[offset];
      P->data[offset] *= v[j] * u[i];
    }
  } 
    
  free(u);
  free(v);
}
#endif
#pragma once

void stacked_multiply(
  arr2d_t<cost_t>* out,
  int_t n,
  csr_t* X,
  csr_t* P,
  csr_t* Y
) {
  for(int_t i = 0; i < X->nrow; i++) {
    
    for(int_t j = 0; j < out->ncol; j++) {
      out->data[i * out->ncol + j] = 0;
    }
    
    for(int_t jA = X->indptr[i]; jA < X->indptr[i + 1]; jA++) {
      int_t j      = X->indices[jA] % (X->ncol / n);
      int_t m      = X->indices[jA] / (X->ncol / n);
      cost_t j_val = X->data[jA];
      
      for(int_t kB = P->indptr[j]; kB < P->indptr[j + 1]; kB++) {
        int_t k      = P->indices[kB];
        cost_t k_val = P->data[kB];
        
        for(int_t hP = Y->indptr[m * Y->ncol + k]; hP < Y->indptr[m * Y->ncol + k + 1]; hP++) {
          int_t h      = Y->indices[hP];
          cost_t h_val = Y->data[hP];
          out->data[i * out->ncol + h] += j_val * k_val * h_val;
        }
      }
    }
  }
}

void stacked_multiply(
  arr2d_t<cost_t>* out,
  int_t n,
  csr_t* X,
  int_t* P,
  csr_t* Y
) {
  for(int_t i = 0; i < X->nrow; i++) {

    for(int_t j = 0; j < out->ncol; j++) {
      out->data[i * out->ncol + j] = 0;
    }

    for(int_t jA = X->indptr[i]; jA < X->indptr[i + 1]; jA++) {
      int_t j      = X->indices[jA] % (X->ncol / n);
      int_t m      = X->indices[jA] / (X->ncol / n);
      cost_t j_val = X->data[jA];
      int_t k      = P[j];
      for(int_t hP = Y->indptr[m * Y->ncol + k]; hP < Y->indptr[m * Y->ncol + k + 1]; hP++) {
        int_t h      = Y->indices[hP];
        cost_t h_val = Y->data[hP];
        out->data[i * out->ncol + h] += j_val * h_val;
      }
    }
  }
}

void compute_traces(
  cost_t& a,
  cost_t& b,
  cost_t& c,
  arr2d_t<cost_t> *x, 
  arr2d_t<cost_t> *y, 
  arr2d_t<cost_t> *z, 
  csr_t *A
) {
  a = 0;
  b = 0;
  c = 0;

  for(int_t i = 0; i < A->nrow; i++) {
    for(int_t offset = A->indptr[i] ; offset < A->indptr[i + 1]; offset++) {
      int_t j = A->indices[offset];
      a += x->data[i * x->ncol + j] * A->data[offset];
      b += y->data[i * y->ncol + j] * A->data[offset];
      c += z->data[i * z->ncol + j] * A->data[offset];
    }
  }
}

void compute_traces(
  cost_t& a,
  cost_t& b,
  cost_t& c,
  arr2d_t<cost_t> *x, 
  arr2d_t<cost_t> *y, 
  arr2d_t<cost_t> *z, 
  int_t* A,
  int_t n
) {
  a = 0;
  b = 0;
  c = 0;
  for(int_t i = 0; i < n; i++) {
    a += x->data[i * x->ncol + A[i]];
    b += y->data[i * y->ncol + A[i]];
    c += z->data[i * z->ncol + A[i]];
  }
}

void dense_convex_combination(
  cost_t alpha,
  arr2d_t<cost_t> *a,
  arr2d_t<cost_t> *b
) {
  // #pragma GCC ivdep
  for(int_t i = 0; i < a->nnz; i++) {
    b->data[i] = alpha * b->data[i] + (1 - alpha) * a->data[i];
  }
}

void sparse_convex_combination(
  cost_t alpha,
  int_t* a,
  csr_t* b
) {
  
  // cost_t alpha = 1 - _alpha; // !! Due to a bug, I think this is what what happening before
  
  int_t* X_pos    = (int_t*)malloc((b->nrow + 1) * sizeof(int_t));
  int_t* X_crd    = (int_t*)malloc((b->nrow + b->nnz) * sizeof(int_t));
  cost_t* X_vals  = (cost_t*)malloc((b->nrow + b->nnz) * sizeof(cost_t));
  
  X_pos[0] = 0;
  
  int_t jX = 0;

  for (int_t i = 0; i < b->nrow; i++) {
    int_t pX2_begin = jX;
    
    int_t jA      = i;
    int_t pA2_end = (i + 1);
    int_t jB      = b->indptr[i];
    int_t pB2_end = b->indptr[(i + 1)];

    while (jA < pA2_end && jB < pB2_end) {
      int_t jA0 = a[jA];
      int_t jB0 = b->indices[jB];
      int_t j   = min(jA0, jB0);
      
      if (jA0 == j && jB0 == j) {
        X_vals[jX] = (1 - alpha) + alpha * b->data[jB];
        X_crd[jX] = j;
        jX++;
      } else if (jA0 == j) {
        X_vals[jX] = 1 - alpha;
        X_crd[jX] = j;
        jX++;
      } else {
        X_vals[jX] = alpha * b->data[jB];
        X_crd[jX] = j;
        jX++;
      }
      jA += (int_t)(jA0 == j);
      jB += (int_t)(jB0 == j);
    }
    
    while (jA < pA2_end) {
      int_t j = a[jA];
      X_vals[jX] = 1 - alpha;
      X_crd[jX] = j;
      jX++;
      jA++;
    }
    
    // #pragma GCC ivdep
    while (jB < pB2_end) {
      int_t j = b->indices[jB];
      X_vals[jX] = alpha * b->data[jB];
      X_crd[jX] = j;
      jX++;
      jB++;
    }
    
    X_pos[i + 1] = jX - pX2_begin;
  }
  
  int_t acc = 0;
  for(int_t i = 1; i < b->nrow + 1; i++) {
    acc += X_pos[i];
    X_pos[i] = acc;
  }
  
  free(b->indptr);
  free(b->indices);
  free(b->data);
  
  b->indptr  = X_pos;
  b->indices = X_crd;
  b->data    = X_vals;
  b->nnz     = X_pos[b->nrow];
}
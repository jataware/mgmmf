#pragma once

using namespace std;

// --
// Data Structures

struct csr_t {
  int_t nrow;
  int_t ncol;
  int_t nnz;
  
  int_t* indptr;
  int_t* indices;
  cost_t* data;
  
  void read(FILE* ptr) {
    fread(&nrow, sizeof(int_t), 1, ptr);
    fread(&ncol, sizeof(int_t), 1, ptr);
    fread(&nnz, sizeof(int_t), 1, ptr);
    indptr  = (int_t* )malloc((nrow + 1) * sizeof(int_t));
    indices = (int_t* )malloc(nnz        * sizeof(int_t));
    data    = (cost_t*)malloc(nnz        * sizeof(cost_t));
    
    fread(indptr,  sizeof(int_t),  nrow + 1, ptr);
    fread(indices, sizeof(int_t),  nnz, ptr);
    fread(data,    sizeof(cost_t), nnz, ptr);

#ifdef VERBOSE
        printf("----------------------------\n");
        printf("nrow   = %d\n", nrow);
        printf("ncol   = %d\n", ncol);
        printf("nnz    = %d\n", nnz);
        printf("----------------------------\n");
#endif
  }
  
  void copy(csr_t* src) {
      nnz     = src->nnz;
      nrow    = src->nrow;
      ncol    = src->ncol;
      
      indptr  = (int_t*)malloc((nrow + 1) * sizeof(int_t));
      indices = (int_t*)malloc(nnz * sizeof(int_t));
      data    = (cost_t*)malloc(nnz * sizeof(cost_t));
      
      memcpy(indptr,  src->indptr,  (nrow + 1) * sizeof(int_t));
      memcpy(indices, src->indices, nnz        * sizeof(int_t));
      memcpy(data,    src->data,    nnz        * sizeof(cost_t));
  }
};

template <typename T>
struct arr2d_t {
  
  int_t nrow;
  int_t ncol;
  int_t nnz;
  T* data;
  
  arr2d_t() {}
  
  void read(FILE* ptr) {
    fread(&nrow, sizeof(int_t), 1, ptr);
    fread(&ncol, sizeof(int_t), 1, ptr);
    
    nnz  = nrow * ncol;
    data = (T*)malloc(nnz * sizeof(T));
    fread(data, sizeof(cost_t), nnz, ptr);

  #ifdef VERBOSE
          printf("----------------------------\n");
          printf("nrow   = %d\n", nrow);
          printf("ncol   = %d\n", ncol);
          printf("nnz    = %d\n", nnz);
          printf("----------------------------\n");
  #endif
  }
  
  arr2d_t(
    int_t _nrow,
    int_t _ncol
  ) : nrow(_nrow), ncol(_ncol), nnz(_nrow * _ncol) {
    data = (T*)malloc(nnz * sizeof(T));
    for(int_t i = 0; i < nnz; i++) data[i] = 0;
  }
};

template <typename T>
struct arr1d_t {
  int_t nnz;
  T* data;
  
  arr1d_t() {}
  
  void read(FILE* ptr) {
    fread(&nnz, sizeof(int_t), 1, ptr);
    
    data = (T*)malloc(nnz * sizeof(T));
    fread(data, sizeof(cost_t), nnz, ptr);

  #ifdef VERBOSE
          printf("----------------------------\n");
          printf("nnz   = %d\n", nnz);
          printf("----------------------------\n");
  #endif
  }
};


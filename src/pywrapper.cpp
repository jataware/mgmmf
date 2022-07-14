#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <iostream>
#include <omp.h>

#include "mgmmf.h"

// --
// Helpers

void py2csr(
  csr_t* out,
  int_t nrow, int_t ncol, int_t nnz,
  py::array_t<int_t>& indptr, py::array_t<int_t>& indices, py::array_t<cost_t>& data
) {
    out->nrow    = nrow;
    out->ncol    = ncol;
    out->nnz     = nnz;
    out->indptr  = static_cast<int_t*>(indptr.request().ptr);
    out->indices = static_cast<int_t*>(indices.request().ptr);
    out->data    = static_cast<cost_t*>(data.request().ptr);
}

template<typename T>
void py2arr2d(arr2d_t<T>* out, int_t nrow, int_t ncol, py::array_t<T>& dense) {
    out->nrow = nrow;
    out->ncol = ncol;
    out->nnz  = nrow * ncol;
    out->data = static_cast<cost_t*>(dense.request().ptr);
}

void _wrapped_mgmmf(
  py::array_t<int_t> ind_arr, 
  
  int_t nc, int_t nt, int_t nw,
  int_t A_nnz,
  int_t B_nnz,
  int_t sim_nnz,
  
  py::array_t<int_t> A_indptr, 
  py::array_t<int_t> A_indices,
  py::array_t<cost_t> A_data,
  
  py::array_t<int_t> At_indptr, 
  py::array_t<int_t> At_indices,
  py::array_t<cost_t> At_data,

  py::array_t<int_t> B_indptr, 
  py::array_t<int_t> B_indices,
  py::array_t<cost_t> B_data,
  
  py::array_t<int_t> Bt_indptr, 
  py::array_t<int_t> Bt_indices,
  py::array_t<cost_t> Bt_data,
  
  py::array_t<int_t> sim_indptr, 
  py::array_t<int_t> sim_indices,
  py::array_t<cost_t> sim_data,
  
  py::array_t<cost_t> sim_dense,
  
  std::optional<py::array_t<int_t>> P_ex_indptr,
  std::optional<py::array_t<int_t>> P_ex_indices,
  std::optional<py::array_t<cost_t>> P_ex_data,
  std::optional<py::array_t<int_t>> w_p_ex,
  
  int_t seed,

  cost_t scale_eps,
  bool scale_init,
  bool scale_sim,
  bool scale_grad,
  
  int_t max_iter,
  
  int_t init_iter,
  bool init_doublestochastic,
  
  int_t n_runs
) {
    if(scale_eps != 1 && (scale_init || scale_sim || scale_grad)) {
      std::cout << "diversity -> 1 thread" << std::endl;
      omp_set_dynamic(0);
      omp_set_num_threads(1);
    }
  
    csr_t _A, _At, _B, _Bt, _sim_sp, _P;
    
    csr_t* A      = &_A;
    csr_t* At     = &_At;
    csr_t* B      = &_B;
    csr_t* Bt     = &_Bt;
    csr_t* sim_sp = &_sim_sp;
    
    py2csr(A,  nt, nt * nc, A_nnz,  A_indptr,  A_indices, A_data);
    py2csr(At, nt, nt * nc, A_nnz, At_indptr, At_indices, At_data);
    
    py2csr(B,  nw * nc, nw, B_nnz,  B_indptr,  B_indices,  B_data);
    py2csr(Bt, nw * nc, nw, B_nnz, Bt_indptr, Bt_indices, Bt_data);
    
    py2csr(sim_sp, nt, nw, sim_nnz, sim_indptr, sim_indices, sim_data);

    arr2d_t<cost_t> _sim;
    arr2d_t<cost_t>* sim = &_sim;
    py2arr2d(sim, nt, nw, sim_dense);
    
    csr_t* P;
    if(P_ex_indptr.has_value()) {
      P = &_P;
      py2csr(P, nt, nw, sim_nnz, P_ex_indptr.value(), P_ex_indices.value(), P_ex_data.value());
    } else {
      P = nullptr;
    }
    
    int_t* w_p;
    if(w_p_ex.has_value()) {
      w_p = static_cast<int_t*>(w_p_ex.value().request().ptr);
    } else {
      w_p = nullptr;
    }
    
    int_t* ind = static_cast<int_t*>(ind_arr.request().ptr);

    arr2d_t<int_t> _solution_counter(nt, nw);    
    arr2d_t<int_t>* solution_counter = &_solution_counter;
    
    if(n_runs > 1) {
      // progress bar
      int log_interval = n_runs > 100 ? (int)((float)n_runs / 100) : 1;
      for(int_t run_id = 0; run_id < max(0, n_runs - 2 * log_interval) ; run_id++) {
        if(run_id % log_interval == 0) cerr << "-";
      }
      cerr << ">|" << endl;
      
      #pragma omp parallel for schedule(dynamic)
      for(int_t run_id = 0; run_id < n_runs ; run_id++) {
        mgmmf(
          ind + run_id * nt, 
          nc, nt, nw,
          A, At, B, Bt, sim_sp, sim, P, w_p,
          seed * run_id,
          solution_counter, scale_eps, scale_init, scale_sim, scale_grad, 
          max_iter, init_iter, init_doublestochastic
        );
        if(run_id % log_interval == 0) cerr << "|";
      }
      cerr << endl;
    } else {
        mgmmf(
          ind,
          nc, nt, nw,
          A, At, B, Bt, sim_sp, sim, P, w_p,
          seed,
          solution_counter, scale_eps, scale_init, scale_sim, scale_grad,
          max_iter, init_iter, init_doublestochastic
        );
    }
}

PYBIND11_MODULE(_mgmmf_cpp, m) {
    m.def("_mgmmf_cpp", &_wrapped_mgmmf, "M-GMMF Subgraph Matching", 
      py::arg("ind"),
      py::arg("nc"), py::arg("nt"), py::arg("nw"),
      py::arg("A_nnz"),
      py::arg("B_nnz"),
      py::arg("sim_nnz"),
      
      py::arg("A_indptr"),  py::arg("A_indices"),  py::arg("A_data"),
      py::arg("At_indptr"), py::arg("At_indices"), py::arg("At_data"),
      
      py::arg("B_indptr"),  py::arg("B_indices"),  py::arg("B_data"),
      py::arg("Bt_indptr"), py::arg("Bt_indices"), py::arg("Bt_data"),

      py::arg("sim_indptr"), py::arg("sim_indices"), py::arg("sim_data"),
      py::arg("sim_dense"),

      py::arg("P_ex_indptr")  = py::none(),
      py::arg("P_ex_indices") = py::none(),
      py::arg("P_ex_data")    = py::none(),
      py::arg("w_p_ex")       = py::none(),
      
      py::arg("seed") = 234,

      py::arg("scale_eps")  = 1.0,
      py::arg("scale_init") = false,
      py::arg("scale_sim")  = false,
      py::arg("scale_grad") = false,
      
      py::arg("max_iter")  = 20, 
      
      py::arg("init_iter")             = 20,
      py::arg("init_doublestochastic") = true,
      
      py::arg("n_runs")    = 1
    );
}
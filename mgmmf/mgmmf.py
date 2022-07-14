#!/usr/bin/env python

import numpy as np
import pandas as pd
from time import time
from scipy import sparse as sp

from mgmmf._mgmmf_cpp import _mgmmf_cpp

def _prep_adjs(X, dim):
  X   = {k:v             for k,(_,v) in enumerate(X.items())}
  Xt  = {k:v.T.tocsr()   for k,(_,v) in enumerate(X.items())}

  assert dim in [0, 1]
  if dim == 0:
    X  = sp.vstack(list(X.values())).tocsr()
    Xt = sp.vstack(list(Xt.values())).tocsr()
  elif dim == 1:
    X  = sp.hstack(list(X.values())).tocsr()
    Xt = sp.hstack(list(Xt.values())).tocsr()

  X.data  = X.data.astype(np.float64)
  Xt.data = Xt.data.astype(np.float64)
  
  return X, Xt


def run_mgmmf(
  t_adjs,
  w_adjs,
  nodesim,
  n_runs=1,
  seed=123,
  scale_eps=1,
  scale_init=False,
  scale_sim=False,
  scale_grad=False,
  init_doublestochastic=True,
):
  
  nt = t_adjs[0].shape[0]
  nw = w_adjs[0].shape[0]
  
  assert len(t_adjs) == len(w_adjs)
  nc = len(t_adjs)
  
  A, At = _prep_adjs(t_adjs, dim=1)
  B, Bt = _prep_adjs(w_adjs, dim=0)
  
  print(nt, nw, nc)
  
  assert nt      == A.shape[0]
  assert nt * nc == A.shape[1]
  assert nw * nc == B.shape[0]
  assert nw      == B.shape[1]
  assert nt      == nodesim.shape[0]
  assert nw      == nodesim.shape[1]

  nodesim    = np.ascontiguousarray(nodesim.astype(np.float64))
  nodesim_sp = sp.csr_matrix(nodesim)

  n_single_cand = ((nodesim != 0).sum(axis=0) == 1).sum()
  if n_single_cand > 0:
    print('!! run_mgmmf: Only one candidate for some world nodes.  Forcing `init_doublestochastic=False` ...')
    init_doublestochastic = False

  ind    = np.zeros(n_runs * nt, dtype=np.int32)
  
  t = time()
  _mgmmf_cpp(
    ind         = ind,
    
    nc          = nc,
    nt          = nt,
    nw          = nw,
    
    A_nnz       = A.nnz,
    B_nnz       = B.nnz,
    sim_nnz     = nodesim_sp.nnz,

    A_indptr    = A.indptr,
    A_indices   = A.indices,
    A_data      = A.data,

    At_indptr   = At.indptr,
    At_indices  = At.indices,
    At_data     = At.data,

    B_indptr    = B.indptr,
    B_indices   = B.indices,
    B_data      = B.data,

    Bt_indptr   = Bt.indptr,
    Bt_indices  = Bt.indices,
    Bt_data     = Bt.data,

    sim_indptr  = nodesim_sp.indptr,
    sim_indices = nodesim_sp.indices,
    sim_data    = nodesim_sp.data,

    sim_dense   = nodesim.ravel(),

    P_ex_indptr  = None,
    P_ex_indices = None,
    P_ex_data    = None,
    w_p_ex       = None,

    seed   = seed,
    
    scale_eps=scale_eps,
    scale_init=scale_init,
    scale_sim=scale_sim,
    scale_grad=scale_grad,
    
    init_doublestochastic=init_doublestochastic,

    n_runs = n_runs,
  )
  elapsed = time() - t
  
  return ind.reshape(n_runs, -1), elapsed
  
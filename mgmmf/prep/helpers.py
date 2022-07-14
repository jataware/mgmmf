#!/usr/bin/env python
"""
  mgmmf/prep/helpers.py
"""

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit
from numba.types import bool_
from scipy import sparse as sp

def coarse_bounds_filter(t, w, rtype):
  """ drop nodes in w that are outside acceptable range for _all_ template nodes """
  
  tmp = t[t.rdf_type == rtype]
  assert tmp.numericValue_min.notnull().all()
  assert tmp.numericValue_max.notnull().all()
  
  min_val = tmp.numericValue_min.min()
  max_val = tmp.numericValue_max.max()
  
  sel = (
    (w.rdf_type != rtype) | (
      (w.rdf_type == rtype) & (
        (w.numericValue.isnull()) | (    # Are nulls allowed here
          (w.numericValue >= min_val) &
          (w.numericValue <= max_val)
        )
      )
    )
  )
  
  print(f"coarse_bound_node_filter: Drop non-null {rtype} outside [{min_val}, {max_val}]. Keep null values.", file=sys.stderr)
  print(f"coarse_bound_node_filter: {sel.sum()} / {w.shape[0]} entries left", file=sys.stderr)
  
  return w[sel]


def missing_node_edge_filter(node, edge):
  """ drop edges that connect to nodes that don't exist """
  
  unodes = set(node.name)
  sel    = edge.node1.isin(unodes) & edge.node2.isin(unodes)
  
  print(f"missing_node_edge_filter: {sel.sum()} / {edge.shape[0]} edges left", file=sys.stderr)
  return edge[sel]


def node_attr_filter(t_node, w_node, attr, rdf_type=None):
  """ filter on node attributes (eg, linkTarget) """
  if rdf_type is not None:
    uattr = t_node[attr][t_node.rdf_type == rdf_type].dropna().unique()
  else:
    uattr = t_node[attr].dropna().unique()
  
  sel = w_node[attr].isin(uattr)
  sel |= w_node[attr].isnull()
  if rdf_type is not None:
    sel |= (w_node.rdf_type != rdf_type)
  
  print(f"node_attr_filter        : {sel.sum()} / {w_node.shape[0]} nodes left (kept NaNs)", file=sys.stderr)
  return w_node[sel]


def node_attr_filter_expand(t_node, w_node, attr, rdf_type, labs, neibs):
  """ filter on node attributes, given NN list (eg, linkTarget)"""
  uattr     = t_node[attr][t_node.rdf_type == rdf_type].dropna().unique()
  
  # expand uattr w/ nn list
  uattr_exp = neibs[np.in1d(labs, uattr)]
  uattr_exp = np.unique(uattr_exp)
  
  # check whether there are neighbors for all uattr
  # if not, it'll get treated as a hard constraint, for now
  tmp = np.in1d(uattr, uattr_exp)
  if not tmp.all():
    print('missing qlabs:', uattr[~tmp], file=sys.stderr)
    uattr_exp = np.unique(np.hstack([uattr, uattr_exp]))
  
  sel = w_node[attr].isin(uattr_exp)
  sel |= w_node[attr].isnull()
  sel |= (w_node.rdf_type != rdf_type)
  
  print(f"node_attr_filter_expand : {sel.sum()} / {w_node.shape[0]} nodes left (kept NaNs)", file=sys.stderr)
  return w_node[sel]


def channel_edge_filter(t_edge, w_edge, attr='channel'):
  """ drop edges that have channels that don't exist in the template """
  
  uattr = set(t_edge[attr])
  sel   = w_edge[attr].isin(uattr)
  
  print(f"channel_edge_filter     : {sel.sum()} / {w_edge.shape[0]} edges left", file=sys.stderr)
  return w_edge[sel]


def zero_degree_node_filter(node, edge):
  """ drop nodes w/ zero degree """
  unodes = set(edge.node1) | set(edge.node2)
  sel    = node.name.isin(unodes)
  
  print(f"zero_degree_node_filter : {sel.sum()} / {node.shape[0]} nodes left", file=sys.stderr)
  return node[sel]


def compute_degree(df_node, df_edge, mode):
  """ compute in/out degree by channel """
  if mode == 'in':
    d_in  = pd.crosstab(df_edge.node2, df_edge.channel)
    d_in  = d_in.reindex(df_node.name.values).fillna(0)
    
    return d_in
  elif mode == 'out':
    d_out = pd.crosstab(df_edge.node1, df_edge.channel)
    d_out = d_out.reindex(df_node.name.values).fillna(0)
    return d_out
  
  else:
    raise Exception(f'!! unsupported mode {mode}')


@njit(cache=True)
def degree_filter(t, w, slack=0):
  """ filter world nodes that have incompatible degrees w/ template """
  nt = t.shape[0]
  nw = w.shape[0]
  nc = t.shape[1]
  
  out = np.ones((nt, nw), dtype=bool_)
  
  for i in range(nt):
    for j in range(nw):
      for k in range(nc):
        if t[i, k] > w[j, k] + slack:
          out[i, j] = False
  
  return out


def edgelist2adjs(df_node, df_edge, importance_weight=False):
  """ convert pandas edgelist to scipy.sparse.csr matrices"""
  
  n_nodes     = df_node.shape[0]
  node_lookup = dict(zip(df_node.name.values, range(n_nodes)))
  
  adjs = {}
  for channel in tqdm(sorted(df_edge.channel.unique())):
    sub = df_edge[df_edge.channel == channel]
    
    r = sub.node1.apply(lambda x: node_lookup[x])
    c = sub.node2.apply(lambda x: node_lookup[x])
    
    if importance_weight:
      v = 4 / np.sqrt(sub.importance.values)
    else:
      v = np.ones(sub.shape[0])
    
    adj = sp.csr_matrix((v, (r, c)), shape=(n_nodes, n_nodes))
    adjs[channel] = adj
  
  return adjs
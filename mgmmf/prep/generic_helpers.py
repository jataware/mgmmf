#!/usr/bin/env python

"""
  generic_helpers.py
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from scipy import sparse as sp

# --
# "Similarity" functions

class BinarySimilarity:
  @staticmethod
  def exact(x, values):
    if isinstance(x, pd.Series):
      return x.isin(values).values.astype(np.float64)
    else:
      return x in values
  
  @staticmethod
  def range(x, min_value, max_value):
    if isinstance(x, pd.Series):
      return (
        x.isnull() | 
        (
          (x >= min_value) & 
          (x <= max_value)
        )
      ).astype(np.float64)
    else:
      return (x >= min_value) and (x <= max_value)
  
  @staticmethod
  def apply(x, sim):
    sim = deepcopy(sim)
    field_name = sim['field_name'].replace(':', '_')
    func       = sim['function']
    
    sim_type   = func.pop('type')
    sim_fn     = getattr(BinarySimilarity, sim_type)
    return sim_fn(x[field_name], **func)  


def generic_sim(t, w_df, mode=BinarySimilarity):
  tn  = len(t)
  wn  = w_df.shape[0]
  
  out = np.ones((tn, wn))
  for idx, tt in enumerate(t):
    for sim in tt['similarities']:
      s         = BinarySimilarity.apply(w_df, sim)    
      out[idx] *= s # boolean AND on similarities - could do something different
  
  return out

# --
# Adjacency matrix functions

def __make_lookup(x):
  return dict(zip(x, range(len(x))))


def make_multiplex(tmplt, w_node, w_edge, edgesim):
  assert 'node1' in w_edge.columns
  assert 'node2' in w_edge.columns
  
  tn_node = len(tmplt['nodedef'])
  wn_node = w_node.shape[0]
  
  t_node = np.array([xx['template_id'] for xx in tmplt['nodedef']])
  t_src  = np.array([xx['node1'] for xx in tmplt['edgedef']])
  t_dst  = np.array([xx['node2'] for xx in tmplt['edgedef']])
  
  tnode_lookup = __make_lookup(t_node)
  wnode_lookup = __make_lookup(w_node.name.values)
  
  t_adjs = {}
  w_adjs = {}
  
  # compute channels as "equivalency classes" of edge similarities
  edge2channel = [hash(tuple(e)) for e in edgesim]
  
  # template adjacency
  for c_id, c in enumerate(np.unique(edge2channel)):
    tsel  = edge2channel == c
    
    tv    = np.ones(tsel.sum()) # TODO: could use importance here
    tr    = [tnode_lookup[xx] for xx in t_src[tsel]]
    tc    = [tnode_lookup[xx] for xx in t_dst[tsel]]
    
    t_adj = sp.csr_matrix((tv, (tr, tc)), shape=(tn_node, tn_node))
    t_adjs[c_id] = t_adj
  
  # world adjacency
  for c_id, c in enumerate(np.unique(edge2channel)):
    tsel  = edge2channel == c
    i     = np.where(tsel)[0][0] # row of edgesim w/ hash == c 
    e     = edgesim[i]
    esel  = e > 0
    
    wv    = e[esel]
    wr    = w_edge[esel].node1.apply(lambda x: wnode_lookup[x])
    wc    = w_edge[esel].node2.apply(lambda x: wnode_lookup[x])
    
    w_adj = sp.csr_matrix((wv, (wr, wc)), shape=(wn_node, wn_node))
    w_adjs[c_id] = w_adj
  
  return t_adjs, w_adjs
#!/usr/bin/env python

"""
  mgmmf/prep/generic_exact.py
  
  - Preprocessing for exact matching.  Built off of `prep.generic`, but adds:
    - iterated degree filtering
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from joblib import dump

from mgmmf.prep.helpers import missing_node_edge_filter, degree_filter, zero_degree_node_filter
from mgmmf.prep.generic_helpers import generic_sim, make_multiplex

# --
# Helpers

def check_feasible(nodesim, edgesim):
  if (nodesim.max(axis=-1) == 0).any():
    print('!! Found a node w/ no possible exact matches.  Exiting...')
    os._exit(0)

  if (edgesim.max(axis=-1) == 0).any():
    print('!! Found a node w/ no possible exact matches.  Exiting...')
    os._exit(0)

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world',  type=str, required=True)
    parser.add_argument('--tmplt',  type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    
    args.outdir = os.path.join(
      args.outdir,
      os.path.basename(args.world),
      os.path.basename(args.tmplt),
    )

    assert '.gdf' not in args.world
    assert '.json' not in args.tmplt    
    args.tmplt = args.tmplt + '.gen.json'
    
    return args

args = parse_args()

# --
# IO

print('prep.generic: loading...', file=sys.stderr)

tmplt  = json.load(open(args.tmplt))
w_node = pd.read_feather(args.world + '.nodes.feather')
w_edge = pd.read_feather(args.world + '.edges.feather')

# --
# Clean messy data

print('prep.generic: filtering...', file=sys.stderr)
w_edge = missing_node_edge_filter(w_node, w_edge)

print('prep.generic: compute node/edge similarities...', file=sys.stderr)
nodesim = generic_sim(tmplt['nodedef'], w_node)
edgesim = generic_sim(tmplt['edgedef'], w_edge)
check_feasible(nodesim, edgesim)

# --
# Form multiplex graph

t_adjs, w_adjs = make_multiplex(tmplt, w_node, w_edge, edgesim)

# --
# EXACT MATCHING CUSTOMIZATION: Iterated degree similarity filter

n_node_last = w_node.shape[0]

while True:
  t_deg = np.column_stack([
    np.column_stack([(a != 0).sum(axis=0).A.squeeze() for a in t_adjs.values()]),
    np.column_stack([(a != 0).sum(axis=1).A.squeeze() for a in t_adjs.values()]),
  ])
  
  w_deg = np.column_stack([
    np.column_stack([(a != 0).sum(axis=0).A.squeeze() for a in w_adjs.values()]),
    np.column_stack([(a != 0).sum(axis=1).A.squeeze() for a in w_adjs.values()]),
  ])
  
  nodesim *= degree_filter(t_deg, w_deg).astype(np.float32)
  
  sel     = nodesim.any(axis=0)
  w_node  = w_node[sel]
  w_edge  = missing_node_edge_filter(w_node, w_edge)
  w_node  = zero_degree_node_filter(w_node, w_edge)
  if w_node.shape[0] == n_node_last:
    break
    
  n_node_last = w_node.shape[0]

  nodesim = generic_sim(tmplt['nodedef'], w_node)
  edgesim = generic_sim(tmplt['edgedef'], w_edge)
  
  t_adjs, w_adjs = make_multiplex(tmplt, w_node, w_edge, edgesim)

# --
# Save

meta = {
  "t_node" : [xx['template_id'] for xx in tmplt['nodedef']],
}

w_node = w_node.reset_index(drop=True)
w_edge = w_edge.reset_index(drop=True)

os.makedirs(args.outdir, exist_ok=True)
print(f'mgmmf.prep.generic_exact: writing to {args.outdir}', file=sys.stderr)

_ = dump(t_adjs, os.path.join(args.outdir, 't_adjs.pkl'))
_ = dump(w_adjs, os.path.join(args.outdir, 'w_adjs.pkl'))
_ = dump(meta, os.path.join(args.outdir, 'meta.pkl'))

_ = w_node.to_feather(os.path.join(args.outdir, 'w_node.feather'))
_ = w_edge.to_feather(os.path.join(args.outdir, 'w_edge.feather'))

np.save(os.path.join(args.outdir, 'nodesim.npy'), nodesim)
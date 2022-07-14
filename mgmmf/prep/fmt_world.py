#!/usr/bin/env python

"""
  mgmmf/prep/fmt_world.py
  
  Convert .gdf format to two feather files
"""

import sys
import argparse
import numpy as np
from mgmmf.io_helpers import load_gdf

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath',        type=str)
    parser.add_argument('--remap_strings', action="store_true")
    
    args = parser.parse_args()
    
    if args.inpath[-4:] != '.gdf':
      args.inpath += '.gdf'
    
    return args


args = parse_args()
print(f'fmt_world.py: formatting {args.inpath}', file=sys.stderr)

df_node, df_edge = load_gdf(args.inpath)

if 'id' in df_edge.columns:
  del df_edge['id']

if not args.remap_strings:
  df_node.to_feather(args.inpath.replace('.gdf', '.nodes.feather'))
  df_edge.to_feather(args.inpath.replace('.gdf', '.edges.feather'))
  
else:
  unode  = df_node.name.unique()
  lookup = dict(zip(unode, np.arange(len(unode))))

  df_node['oname'] = df_node.name
  df_node['name']  = df_node.name.apply(lambda x: lookup[x])
  df_edge['node1'] = df_edge.node1.apply(lambda x: lookup[x])
  df_edge['node2'] = df_edge.node2.apply(lambda x: lookup[x])
  
  assert (df_node.name == np.arange(df_node.shape[0])).all()
  
  df_node.to_feather(args.inpath.replace('.gdf', '.nodes.feather'))
  df_edge.to_feather(args.inpath.replace('.gdf', '.edges.feather'))

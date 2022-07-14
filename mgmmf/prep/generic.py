#!/usr/bin/env python

"""
  mgmmf/prep/generic.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from joblib import dump

from mgmmf.prep.helpers import missing_node_edge_filter
from mgmmf.prep.generic_helpers import generic_sim, make_multiplex

# --
# CLI

def prep_generic(w_node, w_edge, tmplt):
    # --
    # Clean messy data

    print('prep.generic: filtering...', file=sys.stderr)
    w_edge = missing_node_edge_filter(w_node, w_edge)

    print('prep.generic: compute node/edge similarities...', file=sys.stderr)
    nodesim = generic_sim(tmplt['nodedef'], w_node)
    edgesim = generic_sim(tmplt['edgedef'], w_edge)

    # --
    # !! OPTIONAL: 
    # Apply hard pruning, iterative filtering, degree filters, etc for performance here

    # --
    # Form multiplex graph

    t_adjs, w_adjs = make_multiplex(tmplt, w_node, w_edge, edgesim)

    # --
    # !! OPTIONAL: 
    # Apply hard pruning, iterative filtering, degree filters, etc for performance here

    # --
    # Save

    meta = {
        "t_node" : [xx['template_id'] for xx in tmplt['nodedef']],
    }

    w_node = w_node.reset_index(drop=True)
    w_edge = w_edge.reset_index(drop=True)
    
    return t_adjs, w_adjs, nodesim, meta


if __name__ == "__main__":
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

    t_adjs, w_adjs, nodesim, meta = prep_generic(w_node, w_edge, tmplt)

    os.makedirs(args.outdir, exist_ok=True)
    print(f'mgmmf.prep.generic: writing to {args.outdir}', file=sys.stderr)

    _ = dump(t_adjs, os.path.join(args.outdir, 't_adjs.pkl'))
    _ = dump(w_adjs, os.path.join(args.outdir, 'w_adjs.pkl'))
    _ = dump(meta, os.path.join(args.outdir, 'meta.pkl'))

    _ = w_node.to_feather(os.path.join(args.outdir, 'w_node.feather'))
    _ = w_edge.to_feather(os.path.join(args.outdir, 'w_edge.feather'))

    np.save(os.path.join(args.outdir, 'nodesim.npy'), nodesim)
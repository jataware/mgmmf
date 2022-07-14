#!/usr/bin/env python

"""
    example.py
    
    Simple example showing
        - Template formatting
        - Running M-GMMF on a small synthetic graph
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import isomorphism

from mgmmf import run_mgmmf
from mgmmf.prep.generic import prep_generic

np.random.seed(123)

# --
# Generate world graph

w_G = nx.erdos_renyi_graph(1000, p=0.1, seed=234, directed=True)

for node in w_G.nodes:
    w_G.nodes[node]['name']      = node
    w_G.nodes[node]['node_type'] = np.random.choice(('a', 'b', 'c'))

for edge in w_G.edges:
    w_G.edges[edge]['node1']     = edge[0]
    w_G.edges[edge]['node2']     = edge[1]
    w_G.edges[edge]['edge_type'] = np.random.choice(('x', 'y', 'z'))

w_node = pd.DataFrame([w_G.nodes[node] for node in w_G.nodes])
w_edge = pd.DataFrame([w_G.edges[edge] for edge in w_G.edges])

# --
# Define template graph

tmplt = {
    "nodedef" : [
        {
            "template_id"  : "TemplateNode0",
            "importance"   : 1,
            "similarities" : [
                {
                    "field_name" : "node_type",
                    "function"   : {
                        "type"   : "exact",
                        "values" : [
                            "a"
                        ]
                    }
                }
            ]
        },
        {
            "template_id"  : "TemplateNode1",
            "importance"   : 1,
            "similarities" : [
                {
                    "field_name" : "node_type",
                    "function"   : {
                        "type"   : "exact",
                        "values" : [
                            "b"
                        ]
                    }
                }
            ]
        },
        {
            "template_id"  : "TemplateNode2",
            "importance"   : 1,
            "similarities" : [
                {
                    "field_name" : "node_type",
                    "function"   : {
                        "type"   : "exact",
                        "values" : [
                            "c"
                        ]
                    }
                }
            ]
        }
    ],
    "edgedef" : [
        {
            "template_id"  : "TemplateEdge1",
            "importance"   : 1,
            "node1"        : "TemplateNode0",
            "node2"        : "TemplateNode1",
            "similarities" : [
                {
                    "field_name" : "edge_type",
                    "function"   : {
                        "type"   : "exact",
                        "values" : ["x"]
                    }
                }
            ]
        },
        {
            "template_id"  : "TemplateEdge2",
            "importance"   : 1,
            "node1"        : "TemplateNode1",
            "node2"        : "TemplateNode2",
            "similarities" : [
                {
                    "field_name" : "edge_type",
                    "function"   : {
                        "type"   : "exact",
                        "values" : ["y"]
                    }
                }
            ]
        },
        {
            "template_id"  : "TemplateEdge3",
            "importance"   : 1,
            "node1"        : "TemplateNode2",
            "node2"        : "TemplateNode0",
            "similarities" : [
                {
                    "field_name" : "edge_type",
                    "function"   : {
                        "type"   : "exact",
                        "values" : ["z"]
                    }
                }
            ]
        }
    ]
}

# --
# Pre-process

t_adjs, w_adjs, nodesim, meta = prep_generic(w_node, w_edge, tmplt)

print('nodesim.shape: ', nodesim.shape)

# --
# Run match

cands, elapsed = run_mgmmf(t_adjs, w_adjs, nodesim, n_runs=1000, seed=567)

# --
# Verify

cands = w_node.name.values[cands]
cands = pd.DataFrame(cands).drop_duplicates().values

T = nx.DiGraph()
for node in tmplt['nodedef']:
    T.add_node(
        node['template_id'], 
        node_type=node['similarities'][0]['function']['values'][0]
    )

for edge in tmplt['edgedef']:
    T.add_edge(edge['node1'], edge['node2'], edge_type=edge['similarities'][0]['function']['values'][0])

def node_match(a, b):
    return a['node_type'] == b['node_type']

def edge_match(a, b):
    return a['edge_type'] == b['edge_type']

counter = 0
for cand in cands:
    sub   = w_G.subgraph(cand)
    GM    = isomorphism.GraphMatcher(sub, T, node_match=node_match, edge_match=edge_match)
    n_iso = len(list(GM.subgraph_isomorphisms_iter()))
    
    counter += (n_iso > 0)

print(f'number of unique candidates = {cands.shape}')
print(f'number of correct matches   = {counter}')
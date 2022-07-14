import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from haversine import haversine
from scipy.spatial.distance import cdist

def _geo_constraint(c, t_node, w_node):
  # Candidates for node1
  node1  = t_node[t_node.name == c['node1']].iloc[0].to_dict()
  cand1  = w_node[w_node.rdf_type == node1['rdf_type']]
  cand1  = cand1[cand1.latitude.notnull()]
  if node1['linkTarget'] is not None:
    cand1 = cand1[cand1.linkTarget == node1['linkTarget']] # !! hard linkTarget constraint
  
  # Candidates for node2
  node2  = t_node[t_node.name == c['node2']].iloc[0].to_dict()
  cand2  = w_node[w_node.rdf_type == node2['rdf_type']]
  cand2  = cand2[cand2.latitude.notnull()]
  if node2['linkTarget'] is not None:
    cand2 = cand2[cand2.linkTarget == node2['linkTarget']] # !! hard linkTarget constraint
    
  # Compute distance
  vals1 = cand1[['latitude', 'longitude']].values.astype(np.float64)
  vals2 = cand2[['latitude', 'longitude']].values.astype(np.float64)
  
  dist = cdist(vals1, vals2, metric=lambda a,b: haversine(a, b, unit='m'))
  dist = (dist >= c['minValue']) & (dist <= c['maxValue'])
  i, j = np.where(dist)
  
  # Return new edges
  return pd.DataFrame({
    "node1"        : cand1.iloc[i].name.values,
    "node2"        : cand2.iloc[j].name.values,
    "rdf_type"     : c['rdf_type'],
    "argument"     : c['argument'],
    "channel"      : c['channel'],
  })

def add_geo_constraints(t_node, t_edge, w_node, w_edge, t_cnst):
  t_cnst = t_cnst[t_cnst.constraint == 'geoDistance'].copy()
  
  new_edges = []
  for _, c in tqdm(t_cnst.iterrows(), total=t_cnst.shape[0]):
    c       = c.to_dict()
    c_edges = _geo_constraint(c, t_node, w_node)
    new_edges.append(c_edges)
  
  if len(new_edges) > 0:
    new_edges = pd.concat(new_edges, ignore_index=True)
    w_edge    = pd.concat([w_edge, new_edges], ignore_index=True)
    
    print(f'add_geo_constraints: adding {new_edges.shape[0]} new edges', file=sys.stderr)
    
    t_edge    = pd.concat([t_edge, t_cnst], ignore_index=True)
    
    # Drop channels from template if there are no valid edges in the world graph.
    t_edge = t_edge[t_edge.channel.isin(w_edge.channel)]
    t_edge = t_edge.reset_index(drop=True)
  
  return t_edge, w_edge

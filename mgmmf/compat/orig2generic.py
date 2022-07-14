#!/usr/bin/env python

"""
  mgmmf/compat/orig2generic.py
"""

import sys
import json
import argparse

# --
# Helpers

def parse_node(x):
  out = {
    "template_id"  : x["template_id"],
    "importance"   : x["importance"],
  }
  similarities = []
  for k,v in x.items():
    if k in out.keys(): continue
    sim = {
      "field_name" : k,
      "importance" : None,
    }
    
    if isinstance(v, dict):
      # "range"
      assert "minValue" in v
      assert "maxValue" in v
      
      sim['function'] = {
        "type" : "range",
        "min_value" : v['minValue'],
        "max_value" : v['maxValue'],
      }
    
    else:
      # "exact"
      if not isinstance(v, list):
        v = [v]
      
      sim['function'] = {
        "type"   : "exact",
        "values" : v,
      }
    
    similarities.append(sim)
  
  out['similarities'] = similarities
  return out


def parse_edge(x):
  assert len(x['node1']) == 1
  assert len(x['node2']) == 1
  out = {
    "template_id"  : x["template_id"],
    "importance"   : x["importance"],
    "node1"        : x["node1"][0],
    "node2"        : x["node2"][0],
  }
  
  similarities = []
  for k,v in x.items():
    if k in out.keys(): continue
    sim = {
      "field_name" : k,
      "importance" : None,
    }
    
    if isinstance(v, list):
      sim['function'] = {
        "type"   : "exact",
        "values" : v,
      }
    
    elif isinstance(v, dict):
      assert "minValue" in v
      assert "maxValue" in v
      
      sim['function'] = {
        "type" : "range",
        "min_value" : v['minValue'],
        "max_value" : v['maxValue'],
      }
    
    similarities.append(sim)
  
  out['similarities'] = similarities
  return out

def parse_tmplt_generic(tmplt):
  out            = {}
  out['name']    = tmplt.get('name', None)
  out['summary'] = tmplt.get('summary', None)
  out['nodedef'] = [parse_node(node) for node in tmplt['nodedef']]
  out['edgedef'] = [parse_edge(edge) for edge in tmplt['edgedef']]
  return out

# --
# Run

if __name__ == "__main__":
  def parse_args():
      parser = argparse.ArgumentParser()
      parser.add_argument('--tmplt',  type=str, required=True)
      args = parser.parse_args()
      
      # Clean paths
      assert '.json' not in args.tmplt    
      args.outpath = args.tmplt + '.gen.json'
      args.tmplt   = args.tmplt + '.json'
      
      return args
  
  args  = parse_args()
  tmplt = json.load(open(args.tmplt))
  tmplt = parse_tmplt_generic(tmplt)
  
  print(f'mgmmf.compat.orig2generic: {args.tmplt} -> {args.outpath}', file=sys.stderr)
  with open(args.outpath, 'w') as f:
    f.write(json.dumps(tmplt, indent=2))
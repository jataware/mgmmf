#!/usr/bin/env python

"""
  mgmmf/prep/fmt_tmplt.py
  
  Normalize schema for our purposes
"""

import os
import sys
import json
import argparse

# --
# Helpers

def unlist(x):
  if isinstance(x, list):
    assert len(x) == 1, x
    return x[0]
  else:
    return x

def fmt_tmplt(x):
  assert 'name'    in x
  assert 'nodedef' in x
  assert 'edgedef' in x

  if 'summary' not in x:         x['summary']         = None
  if 'geoConstraints' not in x:  x['geoConstraints']  = []
  if 'timeConstraints' not in x: x['timeConstraints'] = []

  for xx in x['nodedef']:
    assert 'template_id' in xx
    assert 'rdf:type'    in xx, f'error at {xx}'
    assert 'importance'  in xx
    
    if 'longitude'    not in xx: xx['longitude']    = None
    if 'latitude'     not in xx: xx['latitude']     = None
    if 'textValue'    not in xx: xx['textValue']    = None
    if 'linkTarget'   not in xx: xx['linkTarget']   = None
    
    xx['rdf:type'] = unlist(xx['rdf:type'])
    
    if 'numericValue' in xx:
      assert 'minValue' in xx['numericValue']
      assert 'maxValue' in xx['numericValue']
      xx['numericValue_min'] = xx['numericValue']['minValue']
      xx['numericValue_max'] = xx['numericValue']['maxValue']
      del xx['numericValue']
      
    else:
      xx['numericValue_min'] = None
      xx['numericValue_max'] = None

  for xx in x['edgedef']:
    assert 'template_id' in xx
    assert 'node1'       in xx
    assert 'node2'       in xx
    assert 'rdf:type'    in xx
    assert 'importance'  in xx
        
    if 'argument' not in xx: xx['argument'] = '_'
    
    xx['node1']    = unlist(xx['node1'])
    xx['node2']    = unlist(xx['node2'])
    xx['rdf:type'] = unlist(xx['rdf:type'])
    xx['argument'] = unlist(xx['argument'])
    
    if 'numericValue' in xx:
      assert 'minValue' in xx['numericValue']
      assert 'maxValue' in xx['numericValue']
      xx['numericValue_min'] = xx['numericValue']['minValue']
      xx['numericValue_max'] = xx['numericValue']['maxValue']
      del xx['numericValue']
      
    else:
      xx['numericValue_min'] = None
      xx['numericValue_max'] = None

  for xx in x['timeConstraints']:
    assert 'template_id' in xx
    assert 'node1'       in xx
    assert 'node2'       in xx
    assert 'importance'  in xx
    
    assert 'units' in xx
    assert 'minValue' in xx
    
    if 'maxValue'   not in xx: xx['maxValue']   = None
    if 'constraint' not in xx: xx['constraint'] = 'startTime' # !! always correct?

  for xx in x['geoConstraints']:
    assert 'template_id' in xx
    assert 'node1'       in xx
    assert 'node2'       in xx
    assert 'importance'  in xx
    
    assert 'units' in xx
    assert 'minValue' in xx
    
    if 'maxValue'   not in xx: xx['maxValue']   = None
    if 'constraint' not in xx: xx['constraint'] = 'startTime' # !! always correct?
  
  return x

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    args = parser.parse_args()
    
    assert '.json' not in args.inpath
    
    args.outpath = args.inpath + '.prep.json'
    args.inpath  = args.inpath + '.json'
    
    return args

if __name__ == "__main__":
  
  args = parse_args()
  print(f'{args.inpath} -> {args.outpath}', file=sys.stderr)
  
  #--
  # Run
  
  x = json.load(open(args.inpath))
  x = fmt_tmplt(x)

  # --
  # Save
  
  json.dump(x, open(args.outpath, 'w'), indent=2)


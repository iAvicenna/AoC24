#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:37:32 2024

@author: avicenna
"""

import pickle
import networkx as nx
import numpy as np
from pathlib import Path
from functools import lru_cache
from itertools import combinations
from collections import defaultdict


cwd = Path(__file__).parent.resolve()
nbits = 16

def parse_input(file_path):
  with file_path.open("r") as fp:
    data = fp.read().split("\n")

  inputs = {x.split(': ')[0]:x.split(': ')[-1] for x in data[:data.index("")]}
  outputs = {x.split(" -> ")[-1]:(x.split(' ')[1], x.split(' ')[0], x.split(' ')[2])
             for x in data[data.index("")+1:-1]}


  return inputs, outputs

def construct_graph(inputs, outputs):

  graph = nx.DiGraph()

  for counter,var_name in enumerate(outputs):

    op = f"{outputs[var_name][0]}-{counter}"
    in1 = outputs[var_name][1]
    in2 = outputs[var_name][2]

    if in1 in inputs:
      val1 = int(inputs[in1])
    else:
      val1 = None
    if in2 in inputs:
      val2 = int(inputs[in2])
    else:
      val2 = None

    graph.add_node(in1, val=val1, init=val1, op=None)
    graph.add_node(in2, val=val2, init=val2, op=None)

    graph.add_node(op, val=None, init=None, op=_op(outputs[var_name][0]))
    graph.add_node(var_name, val=None, init=None, op=None)

    graph.add_edge(in1, op)
    graph.add_edge(in2, op)
    graph.add_edge(op, var_name)


  return graph

def _op(name):

  if name == "OR":
    return lru_cache(20000)(np.bitwise_or)
  elif name == "XOR":
    return lru_cache(20000)(np.bitwise_xor)
  elif name == "AND":
    return lru_cache(20000)(np.bitwise_and)
  else:
    raise ValueError(f"Unknown gate {name}")

def _eval(graph, node_name):

  pre = list(graph.predecessors(node_name))
  node = graph.nodes[node_name]

  if node["val"] is not None:
    return node["val"]
  elif node["op"] is not None:
    node["val"] = node["op"](_eval(graph, pre[0]), _eval(graph, pre[1]))
    return node["val"]
  else:
    node["val"] = _eval(graph, pre[0])
    return node["val"]

def get_out(graph, outpins):

  outvals = [str(graph.nodes[node]["val"]) for node in outpins]

  return ''.join(outvals)

def eval_graph(graph, outpins):
  '''
  returns in the same order as outpins
  '''

  outvals = []
  for out_var in outpins:
    outvals.append(int(_eval(graph, out_var)))

  return outvals

def set_init(graph):

  for node in graph.nodes():
    graph.nodes[node]["init"] = graph.nodes[node]["val"]

def reset(graph, gate_names):

  out = [node for gate_name in gate_names for node in
          list(nx.neighbors(graph, gate_name))]

  desc = list(set(out + list(nx.descendants(graph, out[0]))))

  for node in gate_names + desc:
    graph.nodes[node]["val"] = graph.nodes[node]["init"]


def swap_gates(graph, gate_name1, gate_name2):

  out1 = list(nx.neighbors(graph, gate_name1))
  out2 = list(nx.neighbors(graph, gate_name2))

  desc1 = list(nx.descendants(graph, gate_name1))
  desc2 = list(nx.descendants(graph, gate_name2))

  graph.remove_edge(gate_name1, out1[0])
  graph.remove_edge(gate_name2, out2[0])
  graph.add_edge(gate_name2, out1[0])
  graph.add_edge(gate_name1, out2[0])

  for node in [gate_name1, gate_name2] + desc1 + desc2:
    graph.nodes[node]["val"] = None

  return graph


def find_valid_pairs(graphs, outpins, targets, init_outs, comb_gates=None,
                     verbose=False):


  if comb_gates is None:
    gates = [node for node in graphs[0].nodes if
             graphs[0].nodes[node]["op"] is not None]
    comb_gates = list(combinations(gates, 2))

  valid_pairs = []
  dists = []


  for indg, (gate1, gate2) in enumerate(comb_gates):

    dif_out = 0
    dif_init = 0

    if indg%100==0 and verbose:
      print(indg)

    for graph,target,init_out in zip(graphs, targets, init_outs):


      dif_init += dist(target, init_out)
      swap_gates(graph, gate1, gate2)

      try:
        out_val = eval_graph(graph, outpins)
        swap_gates(graph, gate1, gate2)
        reset(graph, [gate1, gate2])
      except RecursionError: # cyclic graphs, faster than find_cycle
        swap_gates(graph, gate1, gate2)
        reset(graph, [gate1, gate2])
        dif_out = np.inf
        break

      dif_out += dist(target, out_val)

    if dif_out < dif_init:
      valid_pairs.append([gate1, gate2])
      dists.append(dif_init-dif_out)

  return valid_pairs, dists

def dist(pins1, pins2):

  return\
  np.sum(np.abs(np.array(list(pins1)).astype(int)-
                           np.array(list(pins2)).astype(int)))


def set_input(graph, x, y):

  '''assumes values are ordered from 0 to N'''

  xnodes = [node for node in graph.nodes if node[0]=='x']
  ynodes = [node for node in graph.nodes if node[0]=='y']
  xnodes = sorted(xnodes, key=lambda x: int(x[1:]))
  ynodes = sorted(ynodes, key=lambda y: int(y[1:]))

  assert len(xnodes) == len(x)
  assert len(ynodes) == len(y)

  for indx,xnode in enumerate(xnodes):
    graph.nodes[xnode]["val"] = x[indx]

  for indy,ynode in enumerate(ynodes):
    graph.nodes[ynode]["val"] = y[indy]

def solve_problem1(file_name):

  inputs,outputs = parse_input(Path(cwd, file_name))

  graph = construct_graph(inputs, outputs)

  outpins = [var_name for var_name in outputs if
             all(var_name not in val[1:3] for val in outputs.values())]
  outpins = sorted(outpins, key = lambda x: int(x[1:]))

  return int(''.join([str(x) for x in eval_graph(graph, outpins[::-1])]),2)

def sum_bits(xval, yval, nbits):

  '''assumes values are ordered from N to 0'''

  xval = ''.join([str(x) for x in xval])
  yval = ''.join([str(y) for y in yval])

  val = [int(x) for x in list(np.array(list(bin(int(xval,2) + int(yval,2))[2:])).astype(int))]

  if nbits is not None:
    val = [0]*(nbits-len(val)) + val

  return val

def solve_problem2(file_name, seed=0, N=10, find_valid=False,
                   xinputs=None, yinputs=None):

  rng = np.random.default_rng(seed)
  answer = []

  inputs,outputs = parse_input(Path(cwd, file_name))
  graphs = [construct_graph(inputs, outputs) for _ in range(N)]


  outpins = [var_name for var_name in outputs if
              all(var_name not in val[1:3] for val in outputs.values())]
  outpins = sorted(outpins, key = lambda x: int(x[1:]))[::-1]

  N = len([key for key in inputs if key[0]=='x'])


  if xinputs is None and yinputs is None:
    xinputs = [np.zeros(N, int), np.ones(N, int), np.ones(N, int)]
    yinputs = [np.zeros(N, int), np.ones(N, int), np.zeros(N, int)]
    xinputs += [rng.integers(0, 2, N, int) for _ in range(N)] #ordered from 0 to N
    yinputs += [rng.integers(0, 2, N, int) for _ in range(N)] #ordered from 0 to N
    targets = [sum_bits(x[::-1].astype(int), y[::-1].astype(int), N+1) for x,y in zip(xinputs, yinputs)] # ordered from N to 0


  [set_input(graph, list(x), list(y)) for x,y,graph in zip(xinputs, yinputs, graphs)]
  init_outs = [eval_graph(graph, outpins) for graph in graphs]
  [set_init(graph) for graph in graphs]

  if find_valid:
    valid_gate_pairs, dists = find_valid_pairs(graphs, outpins, targets, init_outs,
                                               verbose=True)

    with open("vgp", "wb") as fp:
      pickle.dump([valid_gate_pairs,dists], fp)

  else:
    with open("vgp", "rb") as fp:
      valid_gate_pairs,dists = pickle.load(fp)


  I = np.argmax(dists)
  gp = valid_gate_pairs[I]
  valid_gate_pairs.remove(gp)
  answer.append(gp)
  [swap_gates(graph, *gp) for graph in graphs]
  outs = np.array([eval_graph(graph, outpins) for graph in graphs])
  [set_init(graph) for graph in graphs]

  distances = np.array([dist(out, target) for out,target in zip(outs, targets)])
  counter = 0


  while sum(distances)!=0:

    print(f"cycle: {counter}, dist: {sum(distances)}")

    valid_gate_pairs, dists = find_valid_pairs(graphs, outpins, targets, outs,
                                               comb_gates=valid_gate_pairs)

    if len(dists)==0:
      break

    I = np.argmax(dists)
    gp = valid_gate_pairs[I]
    valid_gate_pairs.remove(gp)

    [swap_gates(graph, *gp) for graph in graphs]
    outs = np.array([eval_graph(graph, outpins) for graph in graphs])

    [set_init(graph) for graph in graphs]

    distances = np.array([dist(out, target) for out,target in zip(outs, targets)])
    answer.append(gp)
    counter += 1

  wires = ','.join(sorted([list(nx.neighbors(graphs[0], node))[0] for gp in answer for node in gp]))
  print(f"final distance: {sum(distances)}")

  return answer, wires, sum(distances)

if __name__ == "__main__":

  result = solve_problem1("test_input24")
  print(f"test 24-1 result: {result}")
  assert result == 2024

  result = solve_problem1("input24")
  print(f"test 24-1 result: {result}")
  assert result == 64755511006320

  comb_found = False
  seed = 0
  np.random.seed(0)
  counter = 0
  N = 40

  # an initial search for reducing search space though it is a bit slow
  # so I dont repeat it. For this I have set N=40
  #solve_problem2("input24", seed=seed, N=40, find_valid=True)

  while not comb_found:

    seed = np.random.randint(0, 2*16)

    answer, wires, final_dist = solve_problem2("input24", seed=seed, N=N)

    if final_dist==0 and len(answer)==4:
      comb_found = True
      print(f"problem 24-2 result: {wires}")

    counter += 1

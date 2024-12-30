#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:37:32 2024

@author: avicenna
"""

import networkx as nx
import numpy as np
from pathlib import Path

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
      in1 = f"{in1}:{inputs[in1]}"
    if in2 in inputs:
      in2 = f"{in2}:{inputs[in2]}"


    graph.add_edge(in1, op)
    graph.add_edge(in2, op)
    graph.add_edge(op, var_name)

  return graph

def _op(name):

  if name == "OR":
    return np.bitwise_or
  elif name == "XOR":
    return np.bitwise_xor
  elif name == "AND":
    return np.bitwise_and
  else:
    raise ValueError(f"Unknown gate {name}")

def _eval(graph, node):

  pre = list(graph.predecessors(node))

  if ':' in node:
    return int(node.split(':')[-1])
  elif (op_name:=str(node.split('-')[0])) in ["OR", "AND", "XOR"]:
    return _op(op_name)(_eval(graph, pre[0]), _eval(graph, pre[1]))
  else:

    assert len(pre)==1
    return _eval(graph, pre[0])


def eval_graph(graph, outputs):
  outpins = [var_name for var_name in outputs if
              all(var_name not in val[1:3] for val in outputs.values())]
  outpins = sorted(outpins, key = lambda x: int(x[1:]))
  outvals = []
  for out_var in outpins:
    outvals.append(str(_eval(graph, out_var)))

  return ''.join(outvals[::-1])

def solve_problem1(file_name):

  inputs,outputs = parse_input(Path(cwd, file_name))

  graph = construct_graph(inputs, outputs)

  return int(eval_graph(graph, outputs),2)


def solve_problem2(file_name):

  inputs,outputs = parse_input(Path(cwd, file_name))

  outpins = [var_name for var_name in outputs if
              all(var_name not in val[1:3] for val in outputs.values())]
  outpins = sorted(outpins, key = lambda x: int(x[1:]))

  xval = ''.join([inputs[key] for key in inputs if key[0]=='x'])
  yval = ''.join([inputs[key] for key in inputs if key[0]=='y'])

  graph = construct_graph(inputs, outputs)
  zval = eval_graph(graph, outputs)
  sum_xt = bin(int(xval,2) + int(yval,2))[2:]

  I = [outpins[ind] for ind,(val1,val2) in enumerate(zip(sum_xt, zval)) if
       val1!=val2]

  gates = set([node for pin in I for node in nx.ancestors(graph, pin)
               if str(node.split('-')[0]) in ["OR", "AND", "XOR"]])
  breakpoint()
  nx.draw_networkx(graph)


if __name__ == "__main__":

  # result = solve_problem1("test_input24")
  # print(f"test 24-1 result: {result}")

  result = solve_problem2("test_input24")
  print(f"problem 24-1 result: {result}")

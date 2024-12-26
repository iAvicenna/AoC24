#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 16:21:26 2024

@author: avicenna
"""

import networkx as nx
from itertools import combinations
from pathlib import Path

cwd = Path(__file__).parent.resolve()

def parse_input(file_path):
  with file_path.open("r") as fp:
    data = fp.readlines()

  edges = [tuple(x.strip().split('-')) for x in data]
  graph = nx.Graph()
  graph.add_edges_from(edges)

  return graph

def solve_problem1(file_name):

  lan_graph = parse_input(Path(cwd, file_name))

  cycles = [','.join(x) for x in nx.simple_cycles(lan_graph, 3)]

  return len([x for x in cycles if x[0]=='t' or ',t' in x])

def solve_problem2(file_name):

  lan_graph = parse_input(Path(cwd, file_name))

  for nnodes in range(len(lan_graph),-1,-1):

    for node in lan_graph.nodes:
      all_nodes = [node] + list(lan_graph.neighbors(node))

      if len(all_nodes)<nnodes:
        continue

      for nodes in combinations(all_nodes, nnodes):
        edges = set([','.join(sorted(e)) for n in nodes
                     for e in lan_graph.edges(n) if e[0] in nodes and
                     e[1] in nodes])

        if len(edges) == sum(range(0, nnodes)):
          return ','.join(sorted(nodes))


if __name__ == "__main__":

  result = solve_problem1("test_input23")
  print(f"test 23-1 result: {result}")
  assert result==7

  result = solve_problem1("input23")
  print(f"problem 23-1 result: {result}")
  assert result==1423

  result = solve_problem2("input23")
  print(f"problem 23-2 result: {result}")
  assert result=="gt,ha,ir,jn,jq,kb,lr,lt,nl,oj,pp,qh,vy"

  result = solve_problem2("test_input23")
  print(f"test 23-2 result: {result}")
  assert result=="co,de,ka,ta"

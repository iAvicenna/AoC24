#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:05:24 2024

@author: avicenna
"""

import numpy as np
import networkx as nx
from pathlib import Path

cwd = Path(__file__).parent.resolve()

def parse_input(file_path):
  with file_path.open("r") as fp:
    data = fp.readlines()

  return [list(map(int,x.split(','))) for x in data]

def construct_graph(grid):

  graph = nx.Graph()

  for i0 in range(grid.shape[0]):
    for i1 in range(grid.shape[1]):
      if not grid[i0,i1]:
        graph.add_node((i0,i1))

        if i0>0 and not grid[i0-1,i1]:
          graph.add_edge((i0-1,i1), (i0,i1))
        if i1>0 and not grid[i0,i1-1]:
          graph.add_edge((i0,i1-1), (i0,i1))

  return graph

def solve_problem1(file_name, gs, max_bytes):

  incoming_positions = parse_input(Path(cwd, file_name))[:max_bytes]

  grid = np.zeros((gs,gs))
  grid[tuple(np.array(incoming_positions).T)[::-1]] = 1

  graph = construct_graph(grid)

  return nx.shortest_path_length(graph, (0,0), (gs-1,gs-1))

def nodes_connected(G, u, v):
  return u in G.neighbors(v)

def solve_problem2(file_name, gs):

  incoming_positions = [tuple(x)[::-1] for x in parse_input(Path(cwd, file_name))]

  connected = [0]
  not_connected = [len(incoming_positions)]

  grid = np.zeros((gs,gs))
  graph = construct_graph(grid)
  edges = {node:[tuple(x) for x in graph.edges(node)] for node in incoming_positions}

  while True:

    if not_connected[-1] - connected[-1] == 1:
      return ','.join([str(x) for x in incoming_positions[not_connected[-1]][::-1]])

    max_bytes = int((not_connected[-1] - connected[-1])/2) + connected[-1]

    nodes = [x for x in incoming_positions[:max_bytes+1]]
    graph.remove_nodes_from(nodes)

    if nx.has_path(graph, (0,0), (gs-1, gs-1)):
      connected.append(max_bytes)
    else:
      not_connected.append(max_bytes)

    graph.add_nodes_from(nodes)
    graph.add_edges_from([y for node in nodes for y in edges[node]])


if __name__ == "__main__":

  result = solve_problem1("test_input18", 7, 12)
  print(f"test 18-1 result: {result}")
  assert result==22

  result = solve_problem1("input18", 71, 1024)
  print(f"problem 18-1 result: {result}")

  result = solve_problem2("test_input18", 7)
  print(f"test 18-2 result: {result}")
  assert result == "6,1"

  result = solve_problem2("input18", 71)
  print(f"problem 18-2 result: {result}")
  assert result == "44,64"

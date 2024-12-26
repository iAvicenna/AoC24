#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 08:46:53 2024

@author: avicenna
"""

import numpy as np
import networkx as nx
from collections import Counter
from pathlib import Path


cwd = Path(__file__).parent.resolve()
_Ad = ['S','E','.']

def parse_input(file_path):
  with file_path.open("r") as fp:
    racetrack = np.array([list(x.strip()) for x in fp.readlines()], dtype=object)

  return racetrack


def connect(i0, i1, racetrack, graph):

  for d0,d1 in [[-1,0],[1,0],[0,-1],[0,1]]:

    j0 = i0+d0
    j1 = i1+d1

    if j0>=0 and j1>=0 and j0<racetrack.shape[0] and j1<racetrack.shape[1] and\
      racetrack[j0, j1] in _Ad and ((j0,j1),(i0,i1)) not in graph.edges:
        graph.add_edge((j0,j1), (i0, i1))


def construct_graph(racetrack):

  graph = nx.Graph()

  for i0 in range(racetrack.shape[0]):
    for i1 in range(racetrack.shape[1]):

      if racetrack[i0,i1] not in _Ad:
        continue

      graph.add_node((i0,i1))
      connect(i0, i1, racetrack, graph)

  return graph


def admissible_obstacles(racetrack):

  obs_idx = np.argwhere(racetrack == '#')
  obs_idx = obs_idx[(obs_idx[:,0]>0) & (obs_idx[:,1]>0) &
                    (obs_idx[:,0]<racetrack.shape[0]-1) &
                    (obs_idx[:,1]<racetrack.shape[1]-1)]

  adm_obs_idx = []

  for i0,i1 in obs_idx:

    if np.count_nonzero(np.isin(racetrack[tuple([[i0-1, i0+1], [i1, i1]])],_Ad))==2\
      or np.count_nonzero(np.isin(racetrack[tuple([[i0, i0], [i1-1, i1+1]])],_Ad))==2:

        adm_obs_idx.append((i0,i1))

  return adm_obs_idx


def solve_problem1(file_name):

  racetrack = parse_input(Path(cwd, file_name))
  s = tuple(np.argwhere(racetrack=='S')[0])
  e = tuple(np.argwhere(racetrack=='E')[0])
  graph = construct_graph(racetrack)
  path = list(nx.shortest_path(graph, s, e))

  idx = admissible_obstacles(racetrack)

  dif_lens = []
  counter = 0

  for i0,i1 in idx:

    graph.add_node((i0,i1))
    connect(i0, i1, racetrack, graph)

    dif_lens.append(len(path) - nx.shortest_path_length(graph, s, e))

    graph.remove_node((i0,i1))

    counter += 1

  dif_lens = Counter(dif_lens)

  return sum([dif_lens[x] for x in dif_lens if x>=100])


if __name__ == "__main__":

  result = solve_problem1("input20")
  print(f"problem 20 result: {result}")
  assert result==1197

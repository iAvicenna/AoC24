#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:23:59 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path

cwd = Path(__file__).parent

cross = np.array([[-1,0],[1,0],[0,-1],[0,1]])

class Node():
  def __init__(self, coord, parent):
    self.coord = coord
    self.parent = parent

  def __repr__(self):
    return f"{self.coord}"

def parse_input(file_path):

  with file_path.open("r") as fp:
    data = list(map(list, fp.read().splitlines()))

  return np.array(data, dtype=int)

def find_neighbours(node_pos, grid):

  I = list(filter(lambda x: all([c>=0 and o-c>0 for c,o in zip(x,grid.shape)]),
                  list(cross + node_pos)))

  candidates = grid[tuple(np.array(I).T)]
  J = np.argwhere(candidates-grid[tuple(node_pos)]==1).flatten()

  return list(np.array(I).T[:, J].T)

def construct_tree_paths(grid):

  roots = list(np.argwhere(grid==0))
  trees = []

  for root in roots:

    levels = [[Node(root, None)]]
    while len(levels[-1])>0 or len(levels)==1:
      levels.append([Node(node, root) for root in levels[-1] for node in
                     find_neighbours(root.coord, grid)])
    trees.append(levels)

  return trees

def trace_back(tree_paths, grid):

  paths = []

  for levels in tree_paths:
    for node in levels[-2]:

      path = ""
      while node is not None:
        coord = ",".join(node.coord.astype(str))
        path += f"{coord} "
        node = node.parent
      paths.append(path)

  return paths

def solve_problem(file_name):

  grid = parse_input(Path(cwd, file_name))
  tree_paths = construct_tree_paths(grid)
  trails = trace_back(tree_paths, grid)
  ntrails = len(set(trails))
  nreached = sum([len(set([tuple(x.coord) for x in levels[-2]]))
                  for levels in tree_paths])

  return nreached, ntrails

if __name__ == "__main__":

  nreached, ntrails = solve_problem("test_input10")
  print(f"test 10-1: {nreached}")
  print(f"test 10-2: {ntrails}")
  assert nreached==36
  assert ntrails==81

  nreached, ntrails = solve_problem("input10")
  print(f"problem 10-1: {nreached}")
  print(f"problem 10-2: {ntrails}")
  assert  nreached==746
  assert ntrails==1541

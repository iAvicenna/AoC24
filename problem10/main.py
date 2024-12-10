#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:23:59 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path

cwd = Path(__file__).parent

def parse_input(file_path):

  with file_path.open("r") as fp:
    data = list(map(list, fp.read().splitlines()))

  return np.array(data, dtype=int)

class Node():

  def __init__(self, coord, parent):
    self.coord = coord
    self.parent = parent

  def __repr__(self):
    return f"{self.coord}"

def find_neighbours(node_pos, grid):

  Ix = list(range(node_pos[0]-1, node_pos[0]+2)) + [node_pos[0] for _ in range(3)]
  Iy = [node_pos[1] for _ in range(3)] + list(range(node_pos[1]-1,node_pos[1]+2))

  I = list(filter(lambda x: x[0]>=0 and x[0]<grid.shape[0] and
                  x[1]>=0 and x[1]<grid.shape[1], zip(Ix, Iy)))
  I = tuple([[x[0] for x in I], [x[1] for x in I]])

  if len(I[0])==0:
    return []

  candidates = grid[I]
  J = np.argwhere(candidates-grid[tuple(node_pos)]==1).flatten()

  return list(np.array(I)[:, J].T)

def construct_trees(grid):

  roots = list(np.argwhere(grid==0))
  trees = []

  for root in roots:

    levels = [[Node(root, None)]]
    while len(levels[-1])>0 or len(levels)==1:
      levels.append([Node(node, root) for root in levels[-1] for node in
                     find_neighbours(root.coord, grid)])
    trees.append(levels)

  return trees

def reconstruct_trails(trees, grid):

  paths = []

  for levels in trees:
    for node in levels[-2]:

      path = ""

      while node.parent is not None:
        coord = ",".join(node.coord.astype(str))
        path += f"{coord} "
        node = node.parent

      coord = ",".join(node.coord.astype(str))
      path += f"{coord} "

      paths.append(path)

  return paths

def solve_problem(file_name, return_trails):

  grid = parse_input(Path(cwd, file_name))

  trees = construct_trees(grid)

  trails = reconstruct_trails(trees, grid)
  ntrails = len(set(trails))

  ntargets = sum([len(set([tuple(x.coord) for x in levels[-2]])) for levels in trees])

  return ntargets, ntrails

if __name__ == "__main__":

  ntargets, ntrails = solve_problem("test_input10", False)
  print(f"test 10-1: {ntargets}")
  print(f"test 10-2: {ntrails}")

  ntargets, ntrails = solve_problem("input10", False)
  print(f"problem 10-1: {ntargets}")
  print(f"problem 10-2: {ntrails}")

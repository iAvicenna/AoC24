#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:26:40 2024

@author: avicenna
"""

import numpy as np
from functools import partial
from pathlib import Path
cwd = Path(__file__).parent

def parse_input(file_path, correction):

  with file_path.open("r") as fp:
    instructions = fp.readlines()

  machine_instructions = []
  for ind in range(0,len(instructions)+1,4):

    mins = instructions[ind:ind+3]
    machine_instructions.append([])
    for i,s in zip(range(3),['+','+','=']):
      machine_instructions[-1].append([int(mins[i].split(',')[0].split(s)[-1]),
                                   int(mins[i].split(',')[1].split(s)[-1])])

    for i in range(2):
      machine_instructions[-1][-1][i] += correction

  return machine_instructions


def solve(threshold, maxn, vectors):

  c = np.array([3, 1])

  M = np.concat([np.array(vectors[0])[:,None],
                 np.array(vectors[1])[:,None]],axis=1).astype(int)

  if np.linalg.det(M)==0:
    return np.nan

  Minv = np.linalg.inv(M)
  nmoves = Minv @ np.array(vectors[2])

  if np.any(np.abs(nmoves - np.round(nmoves))>threshold) or\
    np.any(nmoves>maxn) or np.any(nmoves<0):
      return np.nan

  return np.sum(c * (Minv @ np.array(vectors[2])))


def solve_problem(file_name, correction=0, maxn=100, threshold=1e-5):

  machine_instructions = parse_input(Path(cwd, file_name), correction)

  _solve = partial(solve, threshold, maxn)

  tokens = list(map(_solve, machine_instructions))

  return int(np.nansum(list(tokens)))

if __name__ == "__main__":

  cost = solve_problem("test_input13")
  print(f"test13-1 result: {cost}")
  assert cost==480

  cost = solve_problem("input13")
  print(f"problem13-1 result: {cost}")
  assert cost==39748

  cost = solve_problem("test_input13", 10000000000000, np.inf, 1e-4)
  print(f"test13-2 result: {cost}")
  assert cost==875318608908

  cost = solve_problem("input13", 10000000000000, np.inf, 1e-4)
  print(f"test13-2 result: {cost}")
  assert cost==74478585072604

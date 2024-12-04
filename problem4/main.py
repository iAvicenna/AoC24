#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:37:59 2024

@author: avicenna
"""

import numpy as np
import itertools as it
from pathlib import Path
cwd = Path(__file__).parent


def parse_instructions(path):

  with path.open("r") as fp:
    data = fp.read().splitlines()

  return np.array(list(map(list, data)), dtype=object)


def get_windows(irow, icol, n0, n1):

  dirs = it.product([-1, 0, 1], [-1, 0, 1])
  I0 = []
  I1 = []

  for d0,d1 in dirs:

    if d0==0 and d1==0:
      continue

    J0 = np.array([irow + d0*i for i in range(4) if irow + d0*i>=0 and irow + d0*i<n0])
    J1 = np.array([icol + d1*i for i in range(4) if icol + d1*i>=0 and icol + d1*i<n1])

    if J0.size==4 and J1.size==4:
      I0 += list(J0)
      I1 += list(J1)

  return tuple([I0, I1])


def solve_problem1(file_name):

  path = Path(cwd, file_name)

  grid = parse_instructions(path)
  n0,n1 = grid.shape
  target = "XMAS"

  counter = 0

  for irow in range(n0):
    for icol in range(n1):

      r,c = get_windows(irow, icol, n0, n1)
      convs = grid[tuple([r,c])]

      counter += len([ind0 for ind0 in range(0, len(r), 4)
                      if "".join(convs[ind0:ind0+4])==target])


  return counter


def solve_problem2(file_name):

  path = Path(cwd, file_name)

  grid = parse_instructions(path)
  n0,n1 = grid.shape
  target = {'MAS','SAM'}

  w1 = tuple([[0, 1, 2], [0, 1, 2]])
  w2 = tuple([[0, 1, 2], [2, 1, 0]])

  counter = 0

  for irow in range(n0):
    for icol in range(n1):

      I0 = np.array(range(irow, irow+3))
      I1 = np.array(range(icol, icol+3))

      if np.all(I0>=0) & np.all(I0<n0) & np.all(I1>=0) & np.all(I1<n1):
        conv = grid[irow:irow+3, icol:icol+3]

        if set([''.join(conv[w1]),''.join(conv[w2])]).issubset(target):
          counter += 1

  return counter



if __name__ == "__main__":

  result = solve_problem1("test_input4")
  print(f"test 4-1: {result}")
  assert result==18

  result = solve_problem1("input4")
  print(f"problem 4-1: {result}")
  assert result==2583

  result = solve_problem2("test_input4")
  print(f"test 4-2: {result}")
  assert result==9

  result = solve_problem2("input4")
  print(f"problem 4-2: {result}")
  assert result==1978

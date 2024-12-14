#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:35:10 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path
from collections import Counter
cwd = Path(__file__).parent


def parse_input(file_path):
  with file_path.open("r") as fp:
    robot_info = fp.readlines()

  _split = lambda x,p: [int(x.split(' ')[p].split(',')[0].split('=')[-1]),
                        int(x.split(' ')[p].split(',')[-1])]

  robot_pos = np.array([_split(x, 0) for x in robot_info])
  robot_vel = np.array([_split(x, 1) for x in robot_info])

  return robot_pos, robot_vel


def solve_problem1(file_name, nrows, ncols, nmoves):

  robot_pos, robot_vel = parse_input(Path(cwd, file_name))

  final_pos = robot_pos + nmoves*robot_vel
  final_pos = [(x[0]%ncols, x[1]%nrows) for x in list(final_pos)]

  pos_counts = Counter(final_pos)
  coords = np.array(list(pos_counts.keys()))[:,::-1] #x is cols, y is rows
  coords = tuple(coords.T)

  grid = np.zeros((nrows, ncols), dtype=int)
  grid[coords] += list(pos_counts.values())

  counts = [np.sum(grid[:nrows>>1, :ncols>>1]),
            np.sum(grid[:nrows>>1, -(ncols>>1):]),
            np.sum(grid[-(nrows>>1):, :ncols>>1]),
            np.sum(grid[-(nrows>>1):, -(ncols>>1):])]

  return int(np.prod(counts))


def solve_problem2(file_name, nrows, ncols):

  robot_pos, robot_vel = parse_input(Path(cwd, file_name))

  grid = np.zeros((nrows, ncols), dtype=object)

  # update all positions in a vectorised manner
  final_positions = robot_pos[None, :, :] + np.arange(1,10000)[:,None,None]*robot_vel[None,:,:]
  final_positions[:,:,0] = final_positions[:,:,0]%ncols
  final_positions[:,:,1] = final_positions[:,:,1]%nrows

  for s in range(final_positions.shape[0]):
    grid[:,:] = 0

    final_pos = map(tuple, tuple(final_positions[s,:,:]))

    pos_counts = Counter(final_pos)
    coords = np.array(list(pos_counts.keys()))[:,::-1] #x is cols, y is rows
    coords = tuple(coords.T)

    grid[coords] += list(pos_counts.values())

    xmarg = np.sum(grid, axis=0)
    tops = set(np.argsort(xmarg)[::-1][:10])
    p_near_center = len(tops.intersection(set(range((ncols>>1)-5, (ncols>>1) + 6))))/10

    ymarg = np.sum(grid, axis=1)
    ysym = 1 - abs(ymarg[:nrows>>1].sum() - ymarg[nrows>>1:].sum())/ymarg.sum()

    if p_near_center>0.5 and ysym<0.8:
      return s+1


if __name__ == "__main__":

  result = solve_problem1("test_input14", 7, 11, 100)
  print(f"test14 result: {result}")
  assert result == 12

  result = solve_problem1("input14", 103, 101, 100)
  print(f"problem14-1 result: {result}")
  assert result == 216772608

  result = solve_problem2("input14", 103, 101)
  print(f"problem14-2 result: {result}")
  assert result == 6888

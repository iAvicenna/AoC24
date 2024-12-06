#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:55:57 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path
cwd = Path(__file__).parent

rotate_direction = {'>':'v',
                    'v':'<',
                    '<':'^',
                    '^':'>'}

direction_change = {
  '>':tuple([0,1]),
  'v':tuple([1,0]),
  '<':tuple([0,-1]),
  '^':tuple([-1,0])
  }


class Guard():

  def __init__(self, name, initial_position, initial_direction):
    self.name = name
    self.directions = [initial_direction]
    self.path = [tuple(initial_position)]
    self.state = 0 #0: moving, 1: out, 2: loop


  def set_marked_map(self, shape): # for problem 2
    self.marked_map = np.empty(shape,dtype=object)
    self.marked_map = np.array([[set() for _ in range(shape[0])]
                                for _ in range(shape[1])])

  def rotate(self):
    self.directions.append(rotate_direction[self.directions[-1]])

  def move(self, lab_map, check_loop=False):

    next_position = tuple([self.path[-1][i] +
                           direction_change[self.directions[-1]][i]
                           for i in range(2)])

    self.check_state(1, next_position, lab_map.shape)

    if self.state ==1:
      return True

    if lab_map[tuple(next_position)] == '#':
      self.rotate()
      return self.move(lab_map)

    self.path.append(next_position)

    if len(self.directions) == len(self.path)-1:
      self.directions.append(self.directions[-1])

    if check_loop:
      self.check_state(2)
      self.marked_map[next_position].add(self.directions[-1])

    if self.state==2:
      return True

    return False

  def check_state(self, state_type, next_position=None, boundaries=None):

    if state_type==1 and any(x<0 or x>=boundaries[indx] for indx,x
                             in enumerate(next_position)):
      self.state = 1

    if state_type==2 and len(self.marked_map[self.path[-1]])>0:
      if self.directions[-1] in self.marked_map[self.path[-1]]:
        self.state = 2

  def reset(self, initial_pos, initial_dir):

    for pos in self.path:
      self.marked_map[pos] = set()

    if initial_dir is None:
      self.directions = self.directions[0:1]
    else:
      self.directions = [initial_dir]

    if initial_pos is None:
      self.path = self.path[0:1]
    else:
      self.path = [initial_pos]

    self.state = 0

def parse_input(path):

  with path.open("r") as fp:
    data = map(list,fp.read().splitlines())

  lab_map = np.array(list(data), dtype=object)

  guard_position = tuple(np.argwhere(np.isin(lab_map, ['<','>','^','v']))[0,:])
  guard_direction = lab_map[guard_position]
  lab_map[guard_position] = '.'

  guard = Guard("Terry", guard_position, guard_direction)

  return guard, lab_map

def solve_problem1(file_name):

  guard, lab_map = parse_input(Path(cwd, file_name))
  terminated = False

  while not terminated:
    terminated = guard.move(lab_map)

  return len(set(guard.path))


def solve_problem2(file_name, verbose=False):

  guard, lab_map = parse_input(Path(cwd, file_name))
  guard.set_marked_map(lab_map.shape)
  terminated = False

  while not terminated:
    terminated = guard.move(lab_map)

  nobstacles = 0
  path = guard.path.copy()
  directions = guard.directions.copy()

  # subset to unique positions
  I = sorted(list(set([path.index(pos) for pos in set(path)])))
  path = [path[i] for i in I]
  directions = [directions[i] for i in I]

  # place obstacles on the path
  for indp,pos in enumerate(path[1:], start=1):

    if indp%100==0 and verbose:
      print(f"%{np.round(100*indp/len(path),2)}".ljust(10), end="\r")

    copy_map = lab_map.copy()
    copy_map[pos] = '#'

    # we do not need to re-traverse the previous steps
    guard.reset(path[indp-1], directions[indp-1])

    terminated = False

    while not terminated:
      terminated = guard.move(copy_map, True)

    if guard.state==2:
      nobstacles += 1

  return nobstacles


if __name__ == "__main__":

  result = solve_problem1("test_input6")
  print(f"test 6-1: {result}")
  assert result==41

  result = solve_problem1("input6")
  print(f"problem 6-1: {result}")
  assert result==5531

  result = solve_problem2("test_input6", False)
  print(f"test 6-2: {result}")
  assert result==6

  result = solve_problem2("input6", True)
  print(f"problem 6-1: {result}")
  assert result==2165

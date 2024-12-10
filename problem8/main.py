#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:55:57 2024

@author: avicenna
"""

import numpy as np
from itertools import permutations
from pathlib import Path
cwd = Path(__file__).parent

def parse_input(path):
  with path.open("r") as fp:
    lines = fp.read().splitlines()

  return np.array([list(x) for x in lines], dtype=object)

def find_antinodes(antenna_type, city_map, max_ans=1):

  I = np.argwhere(city_map == antenna_type)

  valid_antinodes = []

  for coord1, coord2 in permutations(I,2):

    if max_ans != 1:
      valid_antinodes += [tuple(coord1), tuple(coord2)]

    arrow = coord2 - coord1

    an1 = coord1 - arrow
    counter = 0
    while np.all((an1>=0) & (an1<city_map.shape)) and counter<max_ans:
      valid_antinodes.append(tuple(an1))
      an1 = an1 - arrow
      counter += 1

    an2 = coord2 + arrow
    counter = 0
    while np.all((an2>=0) & (an2<city_map.shape)) and counter<max_ans:
      valid_antinodes.append(tuple(an2))
      an2 = an2 + arrow
      counter +=1

  return valid_antinodes

def solve_problem(file_name, max_antinodes):
  city_map = parse_input(Path(cwd, file_name))

  antenna_types = set(city_map.flatten()).difference(['.'])
  antinodes = []

  for antenna_type in antenna_types:
    antinodes += find_antinodes(antenna_type, city_map, max_antinodes)

  return len(set(antinodes))

if __name__ == "__main__":

  result = solve_problem("test_input8", 1)
  print(f"test 8-1: {result}")
  assert result==14

  result = solve_problem("input8", 1)
  print(f"problem 8-1: {result}")
  assert result==426

  result = solve_problem("test_input8", np.inf)
  print(f"test 8-2: {result}")
  assert result==34

  result = solve_problem("input8", np.inf)
  print(f"problem 8-2: {result}")
  assert result==1359

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:15:54 2024

@author: avicenna
"""

from pathlib import Path
cwd = Path(__file__).parent


def parse_list(path):

  list1, list2 = [], []

  with path.open("r") as fp:
    for line in fp:
      val1, val2 = line.strip("\n").split()
      list1.append(int(val1))
      list2.append(int(val2))

  return list1,list2


def solve_problem1():

  list1, list2 = parse_list(Path(cwd, "input1"))

  return sum([abs(x-y) for x,y in zip(sorted(list1), sorted(list2))])


def solve_problem2():

  list1, list2 = parse_list(Path(cwd, "input1"))

  num_to_count = {num:list2.count(num) for num in set(list1)}

  return sum([num*num_to_count[num] for num in list1])



if __name__ == "__main__":

  sol1 = solve_problem1()
  print(f"solution 1-1: {sol1}")
  assert sol1 == 1530215

  sol2 = solve_problem2()
  print(f"solution 1-2: {sol2}")
  assert sol2 == 26800609

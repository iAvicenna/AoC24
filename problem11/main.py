#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 08:21:10 2024

@author: avicenna
"""
from pathlib import Path
from collections import defaultdict
cwd = Path(__file__).parent

def parse_input(file_path):
  with file_path.open("r") as fp:
    numbers = list(map(int, fp.read().splitlines()[0].split(' ')))

  return numbers

def calculate_next(val):

  if val == 0:
    return [1]
  if (l:=len(str(val)))%2==0:
    return [int(str(val)[:int(l/2)]), int(str(val)[int(l/2):])]
  else:
    return [2024*val]

def solve_problem(file_name, nblinks):

  numbers = parse_input(Path(cwd, file_name))
  nvals = 0

  for indt, node in enumerate(numbers):

    last_nodes = {node:1}
    counter = 0

    while counter<nblinks:
      new_nodes = defaultdict(int)

      for val,count in last_nodes.items():
        val_next_nodes = calculate_next(val)

        for node in val_next_nodes:
          new_nodes[node] += count

      last_nodes = new_nodes
      counter += 1
    nvals += sum(last_nodes.values())

  return nvals

if __name__ == "__main__":

  result = solve_problem("test_input11", 6)
  print(f"test 11-1: {result}")
  assert result==22

  result = solve_problem("input11", 25)
  print(f"problem 11-1: {result}")
  assert result==199986

  result = solve_problem("input11", 75)
  print(f"problem 11-2: {result}")
  assert result==236804088748754

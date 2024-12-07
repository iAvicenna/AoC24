#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:55:57 2024

@author: avicenna
"""

from pathlib import Path
cwd = Path(__file__).parent

def parse_input(path):

  with path.open("r") as fp:
    lines = fp.read().splitlines()

  roots = [int(line.split(':')[0]) for line in lines]
  node_lists = [[int(x)  for x in line.split(':')[1][1:].split(' ')] for line in lines]

  return roots, node_lists

def construct_tree(root, nodes, include_concat):

  levels = [[] for _ in range(len(nodes)+1)]
  levels[0] = [(str(root), "")]
  # level nodes are tuples of the form (val, operation) where both are str
  # val can be numerical or empty string
  # operation can be *, +, || or empty string

  for indl, level in enumerate(levels[1:], start=1):

    node = nodes[indl-1]

    for elem in levels[indl-1]:

      if elem[0]=='':
        continue

      if elem[0][-len(str(node)):] == str(node) and include_concat:
        levels[indl].append((elem[0][:-len(str(node))], "||"))
      if (a:=int(elem[0]))%(b:=int(node))==0:
        levels[indl].append((str(int(a/b)), '*'))
      if (a:=int(elem[0])) - (b:=int(node))>0:
        levels[indl].append((str(a - b), "+{node}"))

  return levels[-1]

def solve_problem(file_name, include_concat):

  roots, node_lists = parse_input(Path(cwd, file_name))
  valid_roots = []

  for root, nodes in zip(roots, node_lists):

    top = construct_tree(root, nodes[::-1], include_concat)

    if any((x[0]=='1' and x[1]=='*') or (x[0]=='0' and x[1]=='+') or
           (x[0]=='' and x[1]=='||')
           for x in top):

      valid_roots.append(root)

  return sum(valid_roots)

if __name__ == "__main__":

  result = solve_problem("test_input7", False)
  print(f"test 7-1: {result}")
  assert result==3749

  result = solve_problem("input7", False)
  print(f"problem 7-1: {result}")

  result = solve_problem("test_input7", True)
  print(f"test 7-1: {result}")
  assert result==11387

  result = solve_problem("input7", True)
  print(f"problem 7-1: {result}")
  assert result==110365987435001

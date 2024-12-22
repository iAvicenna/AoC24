#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:35:10 2024

@author: avicenna
"""
import numpy as np
import networkx as nx
from pathlib import Path
cwd = Path(__file__).parent

def parse_input(file_path, extend=False):
  process = lambda x: x.strip("\n")

  with file_path.open("r") as fp:
    maze = list(map(process, fp.readlines()))

  maze = np.array(list(map(list, maze)))

  return maze

def get_admissible_dirs(i0,i1, maze):

  dirs = []

  # start and end don't have directions I need to keep track of
  # for end it does not matter for start it is always >
  if maze[i0,i1] in ['S','E']:
    return [maze[i0,i1]]

  if i0>0 and maze[i0-1,i1] != '#':
    dirs.append('^')
  if i0<maze.shape[0]-1 and maze[i0+1,i1] != '#':
    dirs.append('v')
  if i1>0 and maze[i0, i1-1] != '#':
    dirs.append('<')
  if i1<maze.shape[1]-1 and maze[i0, i1+1] != '#':
    dirs.append('>')

  return dirs

def l(dir1, dir2):

  if dir1 == dir2 or dir1 == 'E' or dir2 == 'E' or\
    set([dir1,dir2]).issubset((['>','<','S'])):
    return 1
  else:
    return 1001

def construct_graph(maze):

  graph = nx.DiGraph()

  admissible_dirs = {(i0,i1):set(get_admissible_dirs(i0, i1, maze))
                     for i0 in range(maze.shape[0]) for i1 in range(maze.shape[1])}

  for i0 in range(maze.shape[0]):
    for i1 in range(maze.shape[1]):

      if maze[i0,i1] in ['E','S','.']:
        for d in admissible_dirs[(i0,i1)]:
          graph.add_node((i0,i1,d))

        if i0>0 and maze[i0-1, i1] in ['E','S','.']:

          for d in admissible_dirs[(i0,i1)].difference(['^']):
            graph.add_edge((i0-1, i1, 'v'), (i0,i1,d), length=l('v', d))

          for d0 in admissible_dirs[(i0-1, i1)].difference('v'):
            for d1 in admissible_dirs[(i0,i1)].intersection(['^','S','E']):
              graph.add_edge((i0, i1, d1), (i0-1, i1, d0), length=l(d0, d1))

        if i1>0 and maze[i0, i1-1] in ['E','S','.']:
          for d in admissible_dirs[(i0,i1)].difference(['<']):
            graph.add_edge((i0, i1-1, '>'), (i0,i1,d), length=l('>', d))

          for d0 in admissible_dirs[(i0, i1-1)].difference('>'):
            for d1 in admissible_dirs[(i0,i1)].intersection(['<','S','E']):
              graph.add_edge((i0, i1, d1), (i0, i1-1, d0), length=l(d0, d1))

  return graph

def solve_problem1(file_name, all_shortest=False):

  maze = parse_input(Path(cwd, file_name))
  maze_graph = construct_graph(maze)

  s = tuple(np.argwhere(maze=='S')[0])
  e = tuple(np.argwhere(maze=='E')[0])

  s = (s[0], s[1],'S')
  e = (e[0], e[1], 'E')

  if not all_shortest:
    return nx.shortest_path_length(maze_graph, s, e, weight="length")

  all_path_nodes = set()

  for indp,path in enumerate(nx.all_shortest_paths(maze_graph, s, e, weight="length")):
    all_path_nodes = all_path_nodes.union(set([(x[0], x[1]) for x in set(path)]))

  return len(all_path_nodes)


if __name__ == "__main__":

  result = solve_problem1("test_input16-1")
  print(f"test 16-1: result={result}")
  assert result==7036

  result = solve_problem1("input16")
  print(f"problem 16-1: result={result}")
  assert result==99488

  result = solve_problem1("test_input16-1", True)
  print(f"test 16-2: result={result}")
  assert result==45

  result = solve_problem1("input16", True)
  print(f"problem 16-2: result={result}")
  assert result==516

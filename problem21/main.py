#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 23:05:20 2024

@author: avicenna
"""

import networkx as nx
import numpy as np
import itertools as it
from pathlib import Path

cwd = Path(__file__).parent.resolve()

def ind_to_dir_keypad(val):

  dirs = ['o','<','>','^','v']

  if val in [7,8,9]:
    dirs.remove('^')
  if val in [7,4,1]:
    dirs.remove('<')
  if val in [9,6,3,'A']:
    dirs.remove('>')
  if val in [1,0,'A']:
    dirs.remove('v')

  return dirs

def ind_to_dir_ctrlpad(val):

  dirs = ['o','<','>','^','v']

  if val in ['^','A']:
    dirs.remove('^')
  if val in ['A','>']:
    dirs.remove('>')
  if val in ['<']:
    dirs.remove('<')
  if val in ['<','v','>']:
    dirs.remove('v')

  return dirs



def parse_input(file_path):
  with file_path.open("r") as fp:
    codes = fp.read().split('\n')[:-1]

  return codes


def add_edges(i0, i1, graph, pad):

  val1 = pad[i0, i1]

  if val1 is None:
    return


  if i0>0 and (val2 := pad[i0-1, i1]) is not None:

    graph.add_edges_from([(f'v{val2}', f'{y}{val1}') for y
                          in ind_to_dir_keypad(val1)])
    graph.add_edges_from([(f'^{val1}',f'{y}{val2}') for y
                          in ind_to_dir_keypad(val2)])


  if i1>0 and (val2 := pad[i0, i1-1]) is not None:

    graph.add_edges_from([(f'>{val2}', f'{y}{val1}') for y
                          in ind_to_dir_keypad(val1)])
    graph.add_edges_from([(f'<{val1}',f'{y}{val2}') for y
                          in ind_to_dir_keypad(val2)])



def keypad_graph():

  graph = nx.DiGraph()

  keypad = np.array([np.array([7,8,9]),
                     np.array([4,5,6]),
                     np.array([1,2,3]),
                     np.array([None,0,'A'], dtype=object)], dtype=object)

  graph.add_nodes_from([f'{y}{x}' for x in keypad.flatten()
                        for y in ind_to_dir_keypad(x) if x is not None])


  for i0 in range(4):
    for i1 in range(3):
      val1 = keypad[i0, i1]

      if val1 is None:
        continue

      graph.add_edges_from([(f'{y}{val1}',f'o{val1}') for y
                            in ind_to_dir_keypad(val1) if y!='o'])
      graph.add_edges_from([(f'o{val1}',f'{y}{val1}') for y
                            in ind_to_dir_keypad(val1) if y!='o'])

      add_edges(i0, i1, graph, keypad)


  return graph



def ctrlpad_graph():

  graph = nx.DiGraph()

  ctrlpad = np.array([np.array([None,'^','o']),
                     np.array(['<','v','>'])], dtype=object)

  graph.add_nodes_from([f'{y}{x}' for x in ctrlpad.flatten()
                        for y in ind_to_dir_ctrlpad(x) if x is not None])


  for i0 in range(2):
    for i1 in range(3):
      val1 = ctrlpad[i0, i1]

      if val1 is None:
        continue

      graph.add_edges_from([(f'{y}{val1}',f'o{val1}') for y
                            in ind_to_dir_keypad(val1) if y!='o'])
      graph.add_edges_from([(f'o{val1}',f'{y}{val1}') for y
                            in ind_to_dir_keypad(val1) if y!='o'])

      add_edges(i0, i1, graph, ctrlpad)

  return graph


def lift_to_shortest_path(path, shortest_dict):

  lifted_path = ""

  for let1,let2 in zip(path[:-1], path[1:]):
    if let1!=let2:
      lifted_path += shortest_dict[(f"{let1}{let2}",1)][1:]
    else:
      lifted_path += 'o'

  return lifted_path

def lift_path_ntimes(path, shortest_dict, n, add_o=True):

  counter = 0

  if (path,n) in shortest_dict:
    return shortest_dict[(path,n)]

  while counter<n:
    path = 'o'*int(add_o) + lift_to_shortest_path(path, shortest_dict)
    counter += 1

  return path[1:]

def lift_path_ntimes_using_pair_lifts(path, shortest_dict, n, return_sum=False):

  lifted_path = ""
  if return_sum:
    sum_val = 0

  for let1,let2 in zip(path[:-1], path[1:]):
    if let1!=let2:

      if (f"{let1}{let2}",n) not in shortest_dict:
        lifted_path1 = lift_path_ntimes(f"{let1}{let2}", shortest_dict, n, True)
        shortest_dict[(f"{let1}{let2}",n)] = lifted_path1

      if return_sum:
        sum_val += len(shortest_dict[(f"{let1}{let2}",n)])
      else:
        lifted_path += shortest_dict[(f"{let1}{let2}",n)]
      #for each lifted path I take [1:] because it involves pressing both from the
      #previous path and the current path. ex for o<:
      # ['oo', 'vo', '<>', '<v', 'o<'] ['o<', '><', '>v', '^>', 'oo']
      # and for the beginning of the path we add 'o' which is an unpressed button
      # and is just a starting position so also first element which is 'oo' is
      # not required
    else:
      if return_sum:
        sum_val += 1
      else:
        lifted_path += 'o'


  if return_sum:
    return sum_val
  else:
    return lifted_path

def solve_problem1(file_name, n1, n2):

  codes = parse_input(Path(cwd, file_name))

  graph1 = keypad_graph()
  code_to_paths1 = {}

  shortest_dict = {}
  extended_codes = ['A'+x for x in codes]
  for let1 in set(''.join(codes)):
    for let2 in set(''.join(codes)):
      if not any(f"{let1}{let2}" in c for c in extended_codes):
        continue
      shortest_dict[(let1, let2)] = list(nx.all_shortest_paths(graph1, f'o{let1}', f'o{let2}'))

  # not very proud but a case by case analysis to reduce the search
  # space resulted in these
  if ('3','7') in shortest_dict:
    shortest_dict[('3','7')] = shortest_dict[('3','7')][0:1]
  if ('A', '7') in shortest_dict:
    shortest_dict[('A','7')] = shortest_dict[('A','7')][8:9]
  if ('A', '8') in shortest_dict:
    shortest_dict[('A','8')] = shortest_dict[('A','8')][0:1]
  if ('A', '5') in shortest_dict:
    shortest_dict[('A','5')] = shortest_dict[('A','5')][0:1]
  if ('A', '2') in shortest_dict:
    shortest_dict[('A','2')] = shortest_dict[('A','2')][0:1]
  if ('A', '1') in shortest_dict:
    shortest_dict[('A','1')] = shortest_dict[('A','1')][1:2]
  if ('A', '4') in shortest_dict:
    shortest_dict[('A','4')] = shortest_dict[('A','4')][4:5]
  if ('8', '6') in shortest_dict:
    shortest_dict[('8','6')] = shortest_dict[('8','6')][1:2]
  if ('2', '7') in shortest_dict:
    shortest_dict[('2','7')] = shortest_dict[('2','7')][0:1]
  if ('1', '6') in shortest_dict:
    shortest_dict[('1','6')] = shortest_dict[('1','6')][2:3]
  if ('4', '0') in shortest_dict:
    shortest_dict[('4','0')] = shortest_dict[('4','0')][0:1]
  if ('2', '9') in shortest_dict:
    shortest_dict[('2','9')] = shortest_dict[('2','9')][0:1]


  for code in codes:

    paths = []
    code = 'A' + code

    for let1,let2 in zip(code[:-1], code[1:]):
      paths.append([x[1:] for x in shortest_dict[(let1,let2)]])

    code_to_paths1[code[1:]] = [''.join([z[0] for x in y for z in x]) for y in list(it.product(*paths))]

  graph2 = ctrlpad_graph()
  shortest_dict = {}
  for let1 in ['<','^','>','v','o']:
    for let2 in ['<','^','>','v','o']:
      if let1==let2:
        continue
      shortest_dict[(let1, let2)] = list(nx.all_shortest_paths(graph2, f'o{let1}', f'o{let2}'))

  shortest_dict[('<', 'o')] = shortest_dict[('<', 'o')][0:1]
  shortest_dict[('^', '>')] = shortest_dict[('^', '>')][1:2]
  shortest_dict[('>', '^')] = shortest_dict[('>', '^')][0:1]
  shortest_dict[('o', '<')] = shortest_dict[('o', '<')][1:2]
  shortest_dict[('o', 'v')] = shortest_dict[('o', 'v')][0:1]
  shortest_dict[('v', 'o')] = shortest_dict[('v', 'o')][1:2]


  shortest_dict = {(''.join(key),1):''.join([x[0] for x in shortest_dict[key][0]])
                                            for key in shortest_dict}
  code_to_paths1 = {code:code_to_paths1[code][0] for code in code_to_paths1}

  for key,_ in list(shortest_dict.keys()).copy():

    if n1>0:
      lifted_path1 = lift_path_ntimes(key, shortest_dict, n1)
      shortest_dict[(key,n1)] = lifted_path1

    lifted_path1 = lift_path_ntimes(key, shortest_dict, n2)
    shortest_dict[(key,n2)] = lifted_path1


  code_to_length = {}
  for code in codes:
    if n1>0:
      code_to_paths1[code] = lift_path_ntimes_using_pair_lifts('o'+code_to_paths1[code],
                                                               shortest_dict, n1)

    code_to_length[code] = lift_path_ntimes_using_pair_lifts('o'+code_to_paths1[code],
                                                             shortest_dict, n2,
                                                             return_sum=True)

  min_lens = [code_to_length[code] for code in codes]
  numeric = [int(''.join([x for x in code if x.isnumeric()])) for code in codes]

  return sum([a*b for a,b in zip(min_lens, numeric)])


if __name__ == "__main__":

  result = solve_problem1("input21", 0, 2)
  print(f"problem 21-1 result: {result}")
  assert result==184716

  result = solve_problem1("input21", 10, 15)
  print(f"problem 21-1 result: {result}")
  assert result==229403562787554

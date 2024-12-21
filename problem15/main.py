#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:35:10 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path
from itertools import takewhile
from networkx import Graph
cwd = Path(__file__).parent


dir_to_nrots = {'>':0, 'v':1, '<':2, '^':3}
rotate_dir = {'>':'^', '^':'<', '<':'v', 'v':'>'}
rotate_box = {'[':'⎵', ']':'⎴', '⎵':']', '⎴':'['}
rotate_box_inv = {y:x for x,y in rotate_box.items()}
connect = {'[':[0,1], ']':[0,-1], '⎵':[1,0], '⎴':[-1,0]}


def parse_input(file_path, extend=False):


  process = lambda x: x.strip("\n")

  with file_path.open("r") as fp:
    robot_info = list(map(process, fp.readlines()))

  warehouse_map = np.array(list(map(list, robot_info[:robot_info.index("")])))
  instructions = list(''.join(robot_info[robot_info.index("")+1:]))

  if extend:
    ext_map = np.tile(warehouse_map, (1, 2))
    cols1 = warehouse_map.copy()
    cols2 = warehouse_map.copy()
    cols1[cols1=='O'] = '['
    cols2[cols2=='O'] = ']'
    cols2[cols2=='@'] = '.'

    ext_map[:,range(0, ext_map.shape[1],2)] = cols1
    ext_map[:,range(1, ext_map.shape[1],2)] = cols2

    warehouse_map = ext_map

  return warehouse_map, instructions


def rotate_direction(direction, nrots):

  for _ in range(nrots%4):
    direction = rotate_dir[direction]

  return direction


def rotate_map(rpos, wmap, direction):
  '''
  rotate map and robot position until direction points to '>'
  '''

  rotated_map = wmap[0].copy()
  nrots = wmap[1]
  rdir = direction

  for _ in range(dir_to_nrots[direction]):
    rotated_map = np.rot90(rotated_map)
    rpos = np.array([wmap[0].shape[0]-1-rpos[1], rpos[0]])
    nrots += 1
    rdir = rotate_dir[rdir]

    for box_edge in set(rotate_box).intersection(set(rotated_map.flatten())):
      rotated_map[rotated_map==box_edge] = rotate_box[box_edge]

  return rpos, (rotated_map, nrots%4), rdir

def invert_rotations(wmap):

  rmap = wmap[0]

  for _ in range(wmap[1]):
    rmap = np.rot90(rmap, axes=[1, 0])

    for box_edge in set(rotate_box).intersection(set(rmap.flatten())):
      rmap[rmap==box_edge] = rotate_box_inv[box_edge]

  return rmap

def push(lane, nmove):

  pos_objs = np.argwhere(np.isin(lane, ['O','@'])).flatten()
  syms = lane[pos_objs]
  if pos_objs.size==0 or lane.size==1:
    return lane

  nmoves = np.array([nmove - np.count_nonzero(np.isin(lane[:pos], ['.'])) for pos
                     in pos_objs])
  pos_objs = pos_objs[nmoves>0]
  syms = syms[nmoves>0]
  nmoves = nmoves[nmoves>0]

  for pos, sym, nmove in zip(pos_objs[::-1], syms[::-1], nmoves[::-1]):

    lane[pos] = '.'
    lane[pos + nmove] = sym

  return lane


def move(wmap, rpos, ndirections):

  direction = rotate_direction(ndirections[0], wmap[1])
  nmove = len(ndirections)

  rpos, rwmap, rdir = rotate_map(rpos, wmap, direction)

  I = np.argwhere(rwmap[0][rpos[0], rpos[1]:]=='#').flatten()

  if I.size==0:
    end = None
  else:
    end = rpos[1] + I[0]

  lane = rwmap[0][rpos[0], rpos[1]:end]
  max_nmove = min(nmove, lane.size - np.count_nonzero(lane=='O') - 1)
  push(lane, max_nmove)
  rpos[1] += max_nmove

  return rwmap, rpos


def printwmap(wmap):

  for i in range(wmap.shape[0]):
    print(''.join(wmap[i,:]))


def construct_graph(wmap):

  I = tuple(np.argwhere(wmap=='@')[0])
  box_edges = set(wmap.flatten()).intersection(set(rotate_box))
  assert box_edges in [set(['[',']']), set(['⎴','⎵'])]

  col_nodes = [[tuple(I)]]
  ncol = I[1]

  blocked = False


  while ncol<wmap.shape[1]-1:
    next_nodes = []

    for node in col_nodes[-1]:

      right_node = wmap[node[0], node[1]+1]

      if right_node == '⎴':
        next_nodes.append((node[0], node[1]+1))
        next_nodes.append((node[0]+1, node[1]+1))
      elif right_node == '⎵':
        next_nodes.append((node[0], node[1]+1))
        next_nodes.append((node[0]-1, node[1]+1))
      elif right_node in ['[',']','#']:
        next_nodes.append((node[0], node[1]+1))
        if right_node == '#':
          blocked = True

    if blocked:
      col_nodes.append(next_nodes)
      break
    if len(next_nodes)==0:
      break

    col_nodes.append(next_nodes)
    ncol+=1

  return np.array([x for nodes in col_nodes for x in nodes]), blocked


def move_graph(wmap, rpos, ndirections):

  direction = rotate_direction(ndirections[0], wmap[1])
  nmove = len(ndirections)

  rpos, rwmap, rdir = rotate_map(rpos, wmap, direction)


  for i in range(nmove):

    nodes, blocked = construct_graph(rwmap[0])

    if blocked:
      break

    next_nodes = np.array([(x[0], x[1]+1) for x in nodes])
    syms = rwmap[0][tuple(nodes.T)].copy()

    rwmap[0][tuple(nodes.T)] = '.'
    rwmap[0][tuple(next_nodes.T)] = syms


  return rwmap, rpos


def solve_problem1(file_name):

  wmap, inst = parse_input(Path(cwd, file_name))
  wmap = (wmap, 0) #map and number of rotations
  rpos = np.argwhere(wmap[0]=='@')[0,:]

  while len(inst)>0:

    ndirections = list(takewhile(lambda x: x == inst[0], inst))
    inst = inst[len(ndirections):]

    wmap, rpos = move(wmap, rpos, ndirections)


  rmap = invert_rotations(wmap)

  box_coords = np.argwhere(rmap=='O') #from left, from top
  gps_coords = box_coords*np.array([100, 1])

  return np.sum(gps_coords)

def solve_problem2(file_name):

  wmap, inst = parse_input(Path(cwd, file_name), True)
  wmap = (wmap, 0) #map and number of rotations
  rpos = np.argwhere(wmap[0]=='@')[0,:]
  while len(inst)>0:

    ndirections = list(takewhile(lambda x: x == inst[0], inst))
    inst = inst[len(ndirections):]

    wmap, rpos = move_graph(wmap, rpos, ndirections)

  rmap = invert_rotations(wmap)
  box_coords = np.argwhere(rmap=='[') #from left, from top
  gps_coords = box_coords*np.array([100, 1])

  return np.sum(gps_coords)


if __name__ == "__main__":

  result = solve_problem1("test_input15-1")
  print(f"test15-1 result: {result}")
  assert result==2028

  result = solve_problem1("test_input15-2")
  print(f"test15-2 result: {result}")
  assert result==10092

  result = solve_problem1("input15")
  print(f"problem 15-1 result: {result}")

  result = solve_problem2("test_input15-2")
  print(f"test15-3 result: {result}")
  assert result==9021

  result = solve_problem2("input15")
  print(f"problem 15-2 result: {result}")
  assert result==1381446

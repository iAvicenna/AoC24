#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:55:57 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path
cwd = Path(__file__).parent

def checksum(diskmap):
  I = np.argwhere(diskmap != '.').flatten()
  return sum(list(map(int, diskmap[I]))*I)

def parse_input(file_path):

  with file_path.open("r") as fp:
    data = fp.readlines()

  assert len(data)==1

  condensed_diskmap = np.array(list(data[0].strip("\n")), dtype=int)

  diskmap = [str(inde//2) if inde%2==0 else '.' for inde,entry in
             enumerate(condensed_diskmap) for _ in range(entry)]

  return np.array(diskmap, dtype=object)

def find_crossing(ind0, diskmap):

  left_val = np.count_nonzero(diskmap[:ind0+1] == '.')
  right_val = np.count_nonzero(diskmap[ind0:] !='.')

  I = np.argwhere(diskmap[ind0+1:]=='.').flatten()

  if I.size == 0:
    buffer = 0
  else:
    buffer = I[0]

  return right_val - left_val - buffer

def fill_empty_leftof(diskmap, ind):

  I = np.argwhere(diskmap[:ind+1]=='.').flatten()
  J = ind+1 + np.argwhere(diskmap[ind+1:]!='.').flatten()

  diskmap[I] = diskmap[J[:-I.size-1:-1]]
  diskmap[J[:-I.size-1:-1]] = '.'

  return diskmap

def get_data_chunks(diskmap):

  data_chunks = []

  for indx, x in enumerate(diskmap):

    if len(data_chunks)==0 and x!='.':
      data_chunks.append([x,indx])
    if x != data_chunks[-1][0] and len(data_chunks[-1])==2:
      data_chunks[-1] += [indx]
    if x != '.' and len(data_chunks[-1])==3:
      data_chunks.append([x,indx])
    if indx == len(diskmap)-1 and len(data_chunks[-1])==2:
      data_chunks[-1] += [indx+1]

  return data_chunks

def get_space_chunks(diskmap):

  space_chunks = []

  for indx, x in enumerate(diskmap):

    if len(space_chunks)==0 and x=='.':
      space_chunks.append([indx])
    if x != '.' and len(space_chunks)>0 and len(space_chunks[-1])==1:
      space_chunks[-1] += [indx]
    if x == '.' and len(space_chunks)>0 and len(space_chunks[-1])==2:
      space_chunks.append([indx])
    if indx == len(diskmap)-1 and len(space_chunks)>0 and len(space_chunks[-1])==2:
      space_chunks[-1] += [indx+1]

  return space_chunks

def fill_empty_leftof_contiguous(diskmap):

  data_chunks = get_data_chunks(diskmap)
  space_chunks = get_space_chunks(diskmap)

  space_sizes = [x[1] - x[0] for x in space_chunks]
  space_indices =   [[x[0], x[1]] for x in space_chunks]

  for data in data_chunks[::-1]:
    data_size = data[2] - data[1]
    data_arr = np.array([data[0] for _ in range(data_size)])

    I = [ind for ind,(space_size, space_address) in
         enumerate(zip(space_sizes, space_indices)) if
         space_size>=data_size and space_address[1]<=data[1]]

    if len(I)==0:
      continue

    i0,i1 = space_indices[I[0]]

    diskmap[i0:i0+data_size] = data_arr
    diskmap[data[1]:data[2]] = '.'

    if i1-i1==data_size:
      _ = space_indices.pop(I[0])
      _ = space_sizes.pop(I[0])
    else:
      space_indices[I[0]] = (i0+data_size,i1)
      space_sizes[I[0]] = i1 - i0 - data_size

  return diskmap

def solve_problem1(file_name):

  diskmap = parse_input(Path(cwd, file_name))

  I = np.argwhere(diskmap=='.').flatten()
  chunksize = int(np.ceil(I.size/500))

  guess_crossing = np.array([find_crossing(ind0, diskmap) for ind0
                             in I[::chunksize]])
  ind0 = np.argwhere(guess_crossing<=0).flatten()[0]-1
  I = I[ind0*chunksize:(ind0+1)*chunksize+1]

  crossing = np.array([find_crossing(ind0, diskmap) for ind0 in I])
  ind0 = I[np.argwhere(crossing<=0).flatten()[0]]

  diskmap = fill_empty_leftof(diskmap, ind0)

  return checksum(diskmap)

def solve_problem2(file_name):

  diskmap = parse_input(Path(cwd, file_name))
  diskmap = fill_empty_leftof_contiguous(diskmap)

  return checksum(diskmap)

if __name__ == "__main__":

  result = solve_problem1("test_input9")
  print(f"test 9-1: {result}")
  assert result==1928

  result = solve_problem1("input9")
  print(f"problem 9-1: {result}")
  assert result==6382875730645

  result = solve_problem2("test_input9")
  print(f"test 9-2: {result}")
  assert result==2858

  result = solve_problem2("input9")
  print(f"problem 9-2: {result}")
  assert result==6420913943576

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:37:59 2024

@author: avicenna
"""

from pathlib import Path
cwd = Path(__file__).parent

def parse_instructions(path):

  with path.open("r") as fp:
    corrupted_instructions = ''.join(fp.readlines())

  return corrupted_instructions

def clean_instructions(corrupted_instructions):

  parts = [tuple(x.split(')')[0].split(',')) for x in corrupted_instructions.split("mul(")
           if ')' in x and x.index(')')>2 and ',' in x and x.index(',')>0 and
           all(e.isnumeric() or e==',' for e in x[:x.index(')')])]

  return sum([int(num0)*int(num1) for num0,num1 in parts])

def solve_problem1(file_name):

  path = Path(cwd, file_name)

  corrupted_instructions = parse_instructions(path)

  return clean_instructions(corrupted_instructions)

def solve_problem2(file_name):

  path = Path(cwd, file_name)

  corrupted_instructions = parse_instructions(path)

  parts = corrupted_instructions.split("do()")

  return sum(map(clean_instructions, [x.split("don't()")[0] for x in parts]))


if __name__ == "__main__":
  # REGEXP FREE

  result = solve_problem1("test_input3-1")
  print(f"test 3-1: {result}")
  assert result==161

  result = solve_problem1("input3")
  print(f"problem 3-1: {result}")
  assert result == 188192787

  result = solve_problem2("test_input3-2")
  print(f"test 3-2: {result}")
  assert result==48

  result = solve_problem2("input3")
  print(f"problem 3-2: {result}")
  assert result==113965544

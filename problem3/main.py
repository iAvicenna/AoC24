#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:37:59 2024

@author: avicenna
"""

import re

from pathlib import Path
cwd = Path(__file__).parent


def parse_instructions(path):

  with path.open("r") as fp:
    corrupted_instructions = ''.join(fp.readlines())

  return corrupted_instructions

def clean_instructions(corrupted_instructions):

  pattern = "(mul)\((\d{1,3}),(\d{1,3})\)"
  instructions = list(re.findall(pattern, corrupted_instructions))

  if len(instructions)==0:
    return 0

  return sum([int(num1)*int(num2) for _,num1,num2 in instructions])

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

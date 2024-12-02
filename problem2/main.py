#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:37:59 2024

@author: avicenna
"""

from numpy import sign
from pathlib import Path
cwd = Path(__file__).parent


def parse_file(path):

  with path as fp:

    reports = []

    process = lambda x: list(map(int, x.strip("\n").split()))

    reports = list(map(process,fp))

  return reports


def condition(report):

  if len(report)==1:
    return True

  sgn = sign(report[0]-report[1])
  # if report[0] == report[1] sgn is 0 which will anyway violate abs(x-y)>1
  # so dont need to deal with that

  return all(1 <= abs(x-y) <= 3 and sign(x-y)==sgn for x,y in
             zip(report[:-1], report[1:]))


def loo_condition(report):

  return any([condition(report[:i]+report[i+1:]) for i in range(len(report))])


def solve_problem1(file):

  reports = parse_file(Path(cwd, file).open("r"))

  return sum(map(condition, reports))


def solve_problem2(file):

  reports = parse_file(Path(cwd, file).open("r"))

  return sum(map(loo_condition, reports))



if __name__ == "__main__":
  n_safe = solve_problem1("test_input2")
  print(f"test 2-1: {n_safe}")
  assert n_safe==2

  n_safe = solve_problem1("input2")
  print(f"problem 2-1: {n_safe}")
  assert n_safe==282

  n_safe = solve_problem2("test_input2")
  print(f"test 2-2: {n_safe}")
  assert n_safe==4

  n_safe = solve_problem2("input2")
  print(f"problem 2-2: {n_safe}")
  assert n_safe==349

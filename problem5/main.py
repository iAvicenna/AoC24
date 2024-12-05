#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:28:46 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path
from functools import partial, cmp_to_key
cwd = Path(__file__).parent


def parse_protocol(path):

  with path.open("r") as fp:
    data = fp.read().splitlines()

  rules = data[:data.index('')]
  page_to_rule = {r.split('|')[0]:[] for r in rules}
  [page_to_rule[r.split('|')[0]].append(r.split('|')[1]) for r in rules]

  updates = list(map(lambda x: x.split(','), data[data.index('')+1:]))

  return page_to_rule, updates


def compare_pages(pr, page1, page2):

  if page1 not in pr or page2 not in pr[page1]:
    return 0
  return -1


def sort_pages(pages, page_to_rule):

  cmp = partial(compare_pages, page_to_rule)
  return sorted(pages, key = cmp_to_key(cmp))


def check_and_get_updates(updates, page_to_rule):

  to_print = []

  for pages in updates:
    if pages == sort_pages(pages, page_to_rule):
      to_print.append(pages[int(np.floor(len(pages)/2))])

  return sum(map(int,to_print))


def correct_and_get_updates(updates, page_to_rule):

  to_print = []

  for pages in updates:

    sorted_pages = sort_pages(pages, page_to_rule)

    if pages != sorted_pages:
      to_print.append(sorted_pages[int(np.floor(len(sorted_pages)/2))])

  return sum(map(int,to_print))


def solve_problem1(file_name):

  page_to_rule, updates = parse_protocol(Path(cwd, file_name))

  return check_and_get_updates(updates, page_to_rule)


def solve_problem2(file_name):

  page_to_rule, updates = parse_protocol(Path(cwd, file_name))

  return correct_and_get_updates(updates, page_to_rule)


if __name__ == "__main__":

  result = solve_problem1("test_input5")
  print(f"test 5-1: {result}")
  assert result==143

  result = solve_problem1("input5")
  print(f"problem 5-1: {result}")
  assert result==5452

  result = solve_problem2("test_input5")
  print(f"test 5-2: {result}")
  assert result==123

  result = solve_problem2("input5")
  print(f"problem 5-2: {result}")
  assert result==4598

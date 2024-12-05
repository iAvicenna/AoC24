#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:28:46 2024

@author: avicenna
"""

from pathlib import Path
from functools import cmp_to_key
cwd = Path(__file__).parent


def parse_protocol(path):

  with path.open("r") as fp:
    data = fp.read().splitlines()

  rules = data[:data.index('')]
  page_to_rule = {r.split('|')[0]:[] for r in rules}
  [page_to_rule[r.split('|')[0]].append(r.split('|')[1]) for r in rules]

  updates = list(map(lambda x: x.split(','), data[data.index('')+1:]))

  return page_to_rule, updates


def sort_pages(pages, page_to_rule):

  compare_pages = lambda page1, page2:\
    0 if page1 not in page_to_rule or page2 not in page_to_rule[page1] else -1

  return sorted(pages, key = cmp_to_key(compare_pages))


def solve_problem(file_name, fix):

  page_to_rule, updates = parse_protocol(Path(cwd, file_name))

  to_print = [temp_p[int(len(pages)//2)] for pages in updates
              if (not fix and (temp_p:=pages) == sort_pages(pages, page_to_rule))
              or (fix and (temp_p:=sort_pages(pages, page_to_rule)) != pages)]

  return sum(map(int,to_print))


if __name__ == "__main__":

  result = solve_problem("test_input5", False)
  print(f"test 5-1: {result}")
  assert result==143

  result = solve_problem("input5", False)
  print(f"problem 5-1: {result}")
  assert result==5452

  result = solve_problem("test_input5", True)
  print(f"test 5-2: {result}")
  assert result==123

  result = solve_problem("input5", True)
  print(f"problem 5-2: {result}")
  assert result==4598

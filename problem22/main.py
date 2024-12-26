import numpy as np
from pathlib import Path
from multiprocessing import Pool
from functools import partial

cwd = Path(__file__).parent.resolve()

def parse_input(file_path):
  with file_path.open("r") as fp:
    initial_numbers = [int(x.strip()) for x in fp.readlines()]

  return initial_numbers

def generate_prs(ngens, number0):

  number1 = np.bitwise_xor(64*number0, number0)%16777216

  number2 = np.bitwise_xor(int(np.floor(number1/32)), number1)%16777216

  next_num = np.bitwise_xor(2048*number2, number2)%16777216

  if ngens==1:
    return [next_num]
  else:
    return [next_num] +  generate_prs(ngens-1, next_num)

def solve_problem(file_name, ngens):

  initial_numbers = parse_input(Path(cwd, file_name))
  _parfun = partial(generate_prs, ngens)

  with Pool(6) as p:
    numbers = list(p.imap(_parfun, initial_numbers))

  final_numbers = [num[-1] for num in numbers]

  numbers = [[num]+nums for num,nums in zip(initial_numbers,numbers)]
  difs = [''.join([chr((x%10-y%10)+85) for x,y in zip(nums[1:],nums[:-1])])
          for nums in numbers]

  patterns = set([d[i0:i0+4] for d in difs for i0 in range(0, len(d), 4)])

  profits = {}
  for pattern in patterns:
    I = [d.index(pattern) if pattern in d else None for d in difs]
    profits[tuple(ord(x)-85 for x in pattern)] =\
      sum([numbers[indi][i+4]%10 if i is not None else 0
           for indi,i in enumerate(I)])

  return sum(final_numbers), max(profits.values())


if __name__ == "__main__":

  result1, result2 = solve_problem("test_input22", 2000)
  print(f"test 22-1 result: {result1}")
  assert result1==37990510

  print(f"test 22-2 result: {result2}")
  assert result2==23

  result1, result2 = solve_problem("input22", 2000)
  print(f"problem 22-1 result: {result1}")
  assert result1==15335183969

  print(f"tproblem 22-2 result: {result2}")
  assert result2==1696

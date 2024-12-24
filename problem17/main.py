#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:56:10 2024

@author: avicenna
"""

from pathlib import Path

cwd = Path(__file__).parent.resolve()
_o_to_i = {0:0, 6:1, 7:2}

def parse_input(file_path):
  with file_path.open("r") as fp:
    data = fp.read().splitlines()

  registers = [int(x.split(': ')[-1]) for x in data[:data.index("")]]
  program = data[data.index("")+1:][0].split(': ')[-1].split(',')

  return registers, program

def _op(x, y, op_code):
  if op_code in [0,6,7]:
    return x//2**y
  elif op_code == 1:
    return x^y
  elif op_code==2:
    return y%x
  elif op_code==4:
    return x^y
  elif op_code==5:
    return y%x

def resolve(op_code, operand, registers, ptr):

  operand = int(operand)
  combo_operand = operand if operand <=3 else registers[operand-4] if operand <= 6 else None
  output_val = None

  if op_code in [0,6,7]:
    registers[_o_to_i[op_code]] =  _op(registers[0], combo_operand, op_code) #registers[0]//2**combo_operand

  elif op_code==1:
    registers[1] = _op(registers[1], int(operand), op_code) #registers[1]^int(operand)

  elif op_code==2:
    registers[1] = _op(8, combo_operand, op_code) #combo_operand%8

  elif op_code==3:
    if registers[0] != 0:
      ptr = int(operand)
    else:
      ptr += 2

  elif op_code==4:
    registers[1] = _op(registers[1], registers[2], op_code)#registers[1]^registers[2]

  elif op_code==5:
    output_val = _op(8, combo_operand, op_code)#combo_operand%8

  else:
    raise ValueError(f"Unknown opcode {op_code} of type {type(op_code)}")

  return ptr, output_val

def solve_problem1(file_name):

  registers, program = parse_input(Path(cwd, file_name))

  eop = False
  ptr = 0
  outputs = []

  while not eop:
    next_ptr, output_val =\
      resolve(int(program[ptr]), program[ptr+1], registers, ptr)

    if int(program[ptr]) != 3:
      ptr += 2
    else:
      ptr = next_ptr

    if ptr > len(program)-1:
      eop = True

    if output_val is not None:
      outputs.append(output_val)

  return ''.join(map(str, outputs)), registers


def solve_problem2(file_name):

  _, program = parse_input(Path(cwd, file_name))

  levels = [0]
  i1 = 1

  # the ends of the outputs repeat every 2**3 times (apart from getting longer)
  # start with regA=0, go from 0 to 8 and record every regA + i for which
  # end is 0 (end of the program). Take each of these, multiply by 2**3.
  # for each do the same and record values for which now the one before the
  # last of the output is 3 (same as the program) etc etc. this search is gonna
  # cover every possible interval for regA but eliminate those for which the
  # end of the output does not match the end of the program. note that this
  # is program specific and wont work for other programs

  while True:

    next_levels = []

    for regA in levels:

      for i0 in range(0, 8):

        registers = [regA+i0, 0 ,0]

        eop = False
        outputs = []
        ptr = 0

        while not eop:
          next_ptr, output_val =\
            resolve(int(program[ptr]), program[ptr+1], registers, ptr)

          if int(program[ptr]) != 3:
            ptr += 2
          else:
            ptr = next_ptr

          if ptr > len(program)-1:
            eop = True

          if output_val is not None:
            outputs.append(output_val)

        if ''.join(map(str, outputs)) == ''.join(program):
          return regA+i0

        if str(outputs[-i1]) == program[-i1]:
          next_levels.append((regA+i0)*2**3)

    i1 += 1
    levels = next_levels.copy()


if __name__ == "__main__":


  result, registers = solve_problem1("test_input17-1")
  print(f"test 17-1 result: {result}")
  assert result=="42567777310"
  assert registers[0] == 0

  result, registers = solve_problem1("test_input17-2")
  print(f"test 17-2 result: {result}")
  assert result=="012"

  result, registers = solve_problem1("test_input17-3")
  print(f"test 17-3 result: {result}")
  assert result=="4635635210"

  result,_ = solve_problem1("input17")
  print(f"problem 17-1 result: {result}")
  result=="167430506"

  result = solve_problem2("input17")
  print(f"problem 17-2 result: {result}")

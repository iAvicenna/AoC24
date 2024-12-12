#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:35:10 2024

@author: avicenna
"""

import numpy as np
from pathlib import Path
from shapely import box, union, MultiPolygon, Polygon, MultiLineString
cwd = Path(__file__).parent

def parse_input(file_path):
  with file_path.open("r") as fp:
    garden = list(map(list, fp.read().splitlines()))

  return np.array(garden)

def get_polygon(plant, garden):
  coords = list(map(tuple, list(np.argwhere(garden==plant))))
  for indc,coord in enumerate(coords):

    box_next = box(xmin=coord[0], ymin=coord[1], xmax=coord[0]+1,
                   ymax=coord[1]+1)

    if indc==0:
      polygon = box_next
    else:
      polygon = union(polygon, box_next)

  if isinstance(polygon, Polygon):
    polygon = MultiPolygon([polygon])

  return polygon

def are_collinear(coords, tol=None):
    coords = np.array(coords, dtype=float)
    coords -= coords[0]
    return np.linalg.matrix_rank(coords, tol=tol)==1

def simplify_boundary(boundary):

  # if the object has internal and external boundaries then split them
  # and recurse
  if isinstance(boundary, MultiLineString):
    coordinates = []
    for b in boundary.geoms:
      coordinates.append(simplify_boundary(b))
    return list(np.concat(coordinates, axis=0))

  simple_boundary = boundary.simplify(0)
  coords = [np.array(x) for x in list(simple_boundary.coords)[:-1]]
  resolved = False

  while not resolved:

    end_side=\
    np.concat([x[:,None] for x in [coords[-1], coords[0], coords[1]]], axis=1)

    if  are_collinear(end_side.T):
      coords = coords[1:]
    else:
      resolved = True

  return coords

def solve_problem(file_name):

  garden = parse_input(Path(cwd, file_name))
  unique_plants = set(garden.flatten())
  total_price = 0
  discounted_total_price = 0

  for plant in unique_plants:

    polygon = get_polygon(plant, garden)

    for geom in polygon.geoms:
      coordinates = simplify_boundary(geom.boundary)
      total_price += geom.area*geom.length
      discounted_total_price += geom.area*len(coordinates)

  return int(total_price), int(discounted_total_price)

if __name__ == "__main__":


  price, discounted_price = solve_problem("test_input12")
  assert price==1930
  assert discounted_price==1206
  print(f"test 12-1: {price}")
  print(f"test 12-2: {discounted_price}")

  price, discounted_price = solve_problem("input12")
  assert price==1424472
  assert discounted_price==870202
  print(f"problem 12-1: {price}")
  print(f"problem 12-2: {discounted_price}")


  #18915832 too high

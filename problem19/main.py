
import networkx as nx
from pathlib import Path

cwd = Path(__file__).parent.resolve()


def parse_input(file_path):
  with file_path.open("r") as fp:
    data = fp.readlines()

  available_towels = [x.strip() for x in data[0].split(',')]
  patterns = [x.strip() for x in data[2:]]

  return available_towels, patterns


def construct_graph(available_towels, pattern):

  graph = nx.DiGraph()
  graph.add_node(0, name=pattern)
  levels = [[0]]

  while len(levels[-1])!=0:
    next_levels = []

    for elem in levels[-1]:

      elem_name = graph.nodes[elem]["name"]

      for indt,towel in enumerate(available_towels):
        if elem_name[-len(towel):] == towel:

          node_name = elem_name[:-len(towel)]

          if node_name not in [node['name'] for node in graph.nodes.values()]:
            graph.add_node(len(graph), name=node_name)
            node_number = len(graph)-1
            next_levels.append(node_number)
          else:
            I = [ind for ind,node in enumerate(graph.nodes.values()) if
                 node['name']==node_name]
            node_number = I[0]

          graph.add_edge(elem, node_number)

    levels.append(next_levels)

  return graph

def backtrace(node, graph, memo):

  count = 0

  if node in memo:
    return memo[node]
  if graph.nodes.get(node)["name"] == "":
    return 1

  for nb_node in graph.neighbors(node):
    count += backtrace(nb_node, graph, memo)

  memo[node] = count
  return count


def solve_problem(file_name, count_paths=False):

  available_towels, patterns = parse_input(Path(cwd, file_name))
  npaths = 0

  for indp,pattern in enumerate(patterns):

    towels_to_use = [t for t in available_towels if t in pattern]
    graph = construct_graph(towels_to_use, pattern)
    memo = {}

    if count_paths:
      npaths += backtrace(0, graph, memo)
    else:
      npaths += backtrace(0, graph, memo)>0


  return npaths

if __name__ == "__main__":

  result = solve_problem("test_input19", False)
  print(f"test 19-1 result: {result}")
  assert result==6

  result = solve_problem("input19", False)
  print(f"problem 19-1 result: {result}")
  assert result==206

  result = solve_problem("test_input19", True)
  print(f"test 19-2 result: {result}")
  assert result==16

  result = solve_problem("input19", True)
  print(f"problem 19-2 result: {result}")
  assert result==622121814629343

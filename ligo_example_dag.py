from pathlib import Path
import importlib
import numpy as np
import matplotlib.pyplot as plt


# ---- Graph and scheduling config ----
# Resource is the fraction of a processor needed by a task (0 < resource <= 1).
# NODE_DEFS = {
# 	"A": {"resource": 1 / 3, "time": 1},
# 	"B": {"resource": 1 / 3, "time": 2},
# 	"C": {"resource": 1 / 3, "time": 1},
# 	"D": {"resource": 1.0, "time": 3},
# 	"E": {"resource": 1, "time": 1},
# 	"F": {"resource": 1, "time": 1},
# 	"H": {"resource": 1 / 3, "time": 1},
# 	"I": {"resource": 1 / 3, "time": 1},
# 	"J": {"resource": 1 / 3, "time": 2},
# 	"K": {"resource": 1.0, "time": 3},
# 	"L": {"resource": 1, "time": 1},
# 	"M": {"resource": 1, "time": 1},
# 	"N": {"resource": 1 / 3, "time": 2},
# 	"O": {"resource": 1 / 3, "time": 1},
# 	"P": {"resource": 1 / 3, "time": 1},
# 	"Q": {"resource": 1.0, "time": 3},
# 	"R": {"resource": 1, "time": 1},
# 	"S": {"resource": 1, "time": 1},
# }

NODE_DEFS = {
	"A": {},
	"B": {},
	"C": {},
	"D": {},
	"E": {},
	"F": {},
	"G": {},
	"H": {},
	"I": {},
}

EDGES = [
	["A", "D"],
	["B", "D"],
	["C", "D"],
	["D", "E"],
	["E", "F"],
	["E", "G"],
	["E", "I"],
	["E", "H"],
]

# scale to 4 concurrent graphs
EDGES = [[f"{x}{i}", f"{y}{i}"] for i in range(1, 4) for x, y in EDGES]
NODE_DEFS = {f"{node}{i}": {} for i in range(1, 4) for node in NODE_DEFS.keys()}

def compute_stage_levels(node_defs, edges):
	"""Assign each node a stage level from DAG structure (order-independent)."""
	nodes = list(node_defs.keys())
	parents = {node: [] for node in nodes}
	children = {node: [] for node in nodes}
	indegree = {node: 0 for node in nodes}

	for src, dst in edges:
		if src not in node_defs or dst not in node_defs:
			raise ValueError(f"Edge ({src}, {dst}) refers to an unknown node.")
		children[src].append(dst)
		parents[dst].append(src)
		indegree[dst] += 1

	queue = [node for node in nodes if indegree[node] == 0]
	levels = {node: 0 for node in queue}
	visited = 0

	while queue:
		node = queue.pop(0)
		visited += 1
		for child in children[node]:
			levels[child] = max(levels.get(child, 0), levels[node] + 1)
			indegree[child] -= 1
			if indegree[child] == 0:
				queue.append(child)

	if visited != len(nodes):
		raise ValueError("Input graph is not a DAG (cycle detected).")

	for node in nodes:
		levels.setdefault(node, 0)
	return levels


def _split_output_path(output_path: str) -> tuple[str, str]:
	"""Convert a path like dag_connected.png into (dag_connected, png)."""
	path = Path(output_path)
	if path.suffix:
		return str(path.with_suffix("")), path.suffix.lstrip(".")
	return str(path), "png"


def draw_connected_dag(output_path: str = "ligo_example_dag.png") -> None:
	"""Render the DAG with Graphviz and color nodes by assigned processor."""
	try:
		graphviz = importlib.import_module("graphviz")
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"Missing dependency: graphviz. Install it with: pip install graphviz"
		) from exc

	Digraph = graphviz.Digraph

	nodes = list(NODE_DEFS.keys())
	# assignment, start_time, finish_time, run_order = schedule_nodes(
	# 	NODE_DEFS,
	# 	EDGES,
	# 	PRIORITY_ORDERING,
	# 	num_processors,
	# 	look_ahead
	# )
	levels = compute_stage_levels(NODE_DEFS, EDGES)

	dot = Digraph("task_mapping", format="png")
	dot.attr(rankdir="TD", splines="spline", nodesep="0.45", ranksep="0.6")
	dot.attr("node", shape="circle", style="filled", fontcolor="white", fontsize="14", penwidth="1.5")
	dot.attr("edge", color="#2f2f2f", arrowsize="0.8", penwidth="1.0")

	for node in nodes:
		dot.node(node, label='', fillcolor='blue')

	for src, dst in EDGES:
		dot.edge(src, dst)

	# Force nodes in the same stage to share rank, making LR layout clearly horizontal.
	max_level = max(levels.values()) if levels else 0
	for level in range(max_level + 1):
		same_level_nodes = [node for node in nodes if levels[node] == level]
		if not same_level_nodes:
			continue
		with dot.subgraph() as sub:
			sub.attr(rank="same")
			for node in same_level_nodes:
				sub.node(node)

	base_name, output_format = _split_output_path(output_path)
	dot.format = output_format
	try:
		rendered_path = dot.render(filename=base_name, cleanup=True)
	except Exception as exc:
		raise RuntimeError(
			"Graphviz rendering failed. Install Graphviz system binaries (dot), "
			"then retry."
		) from exc

if __name__ == "__main__":
	draw_connected_dag()

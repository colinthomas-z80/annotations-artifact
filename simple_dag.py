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
	"A": {"resource": 1, "time": 1},
	"B": {"resource": 1, "time": 1},
	"N": {"resource": 1, "time": 1},
	"C": {"resource": 1, "time": 1},
	"Q": {"resource": 1, "time": 1},
	"R": {"resource": 1, "time": 1},
	"K": {"resource": 1, "time": 1},

	"D": {"resource": 1, "time": 1},
	"E": {"resource": 1, "time": 1},
	"O": {"resource": 1, "time": 1},
	"F": {"resource": 1, "time": 1},
	"S": {"resource": 1, "time": 1},
	"T": {"resource": 1, "time": 1},
	"L": {"resource": 1, "time": 1},

	"H": {"resource": 1, "time": 1},
	"I": {"resource": 1, "time": 1},
	"P": {"resource": 1, "time": 1},
	"W": {"resource": 1, "time": 1},
	"X": {"resource": 1, "time": 1},	
	"J": {"resource": 1, "time": 1},
	"M": {"resource": 1, "time": 1},
}

EDGES = [
	("A", "B"),
	("A", "N"),
	("B", "C"),
	("B", "Q"),
	("N", "C"),
	("N", "Q"),
	("C", "R"),
	("Q", "R"),
	("R", "K"),

	("D", "E"),
	("D", "O"),
	("E", "F"),
	("O", "F"),
	("E", "S"),
	("O", "S"),
	("F", "L"),
	("S", "L"),
	("L", "T"),
	
	("H", "I"),
	("H", "P"),
	("I", "W"),
	("P", "W"),
	("I", "M"),
	("P", "M"),
	("W", "X"),
	("M", "X"),
	("X", "J"),
]


# EDGES = [
# 	("A", "D"),
# 	("B", "D"),
# 	("C", "D"),
# 	("D", "E"),
# 	("D", "F"),
# 	("H", "K"),
# 	("I", "K"),
# 	("J", "K"),
# 	("K", "L"),
# 	("K", "M"),
# 	("N", "Q"),
#     ("O", "Q"),
# 	("P", "Q"),
#     ("Q", "R"),
# 	("Q", "S"),
# ]

# Total ordering used for list scheduling.
#TOTAL_ORDER = ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"]

#PRIORITY_ORDERING = ["R", "M", "F", "S", "E", "L", "D", "Q", "K", "B", "I", "J", "C", "H", "A", "P", "O", "N"]
#PRIORITY_ORDERING = ["F", "E", "D", "C", "B", "A", "M", "L", "K", "J", "I", "H", "S", "R", "Q", "P", "O", "N"]

PRIORITY_ORDERING = ["A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "W", "X"]

# Change this to control how many processors are available.
NUM_PROCESSORS = 1

# ---- Styling config ----
PROCESSOR_COLORS = [
	"#4C78A8",
	"#F58518",
	"#54A24B",
	"#E45756",
	"#72B7B2",
	"#EECA3B",
	"#B279A2",
	"#FF9DA6",
]
GRAPH_RANKDIR = "LR"
TASK_DURATION = 1


def schedule_nodes(node_defs, edges, priority_order, num_processors, look_ahead):
	"""Priority-list scheduler: each step runs the highest-priority ready task."""
	if num_processors < 1:
		raise ValueError("NUM_PROCESSORS must be at least 1.")

	nodes = list(node_defs.keys())
	if set(priority_order) != set(nodes) or len(priority_order) != len(nodes):
		raise ValueError("PRIORITY_ORDERING must contain each node exactly once.")

	parents = {node: [] for node in nodes}
	for src, dst in edges:
		if src not in node_defs or dst not in node_defs:
			raise ValueError(f"Edge ({src}, {dst}) refers to an unknown node.")
		parents[dst].append(src)

	# Per-processor utilization timeline: reservations[p][t] = used capacity at slot t.
	reservations = [{} for _ in range(num_processors)]
	assignment = {}
	start_time = {}
	finish_time = {}
	run_order = {}
	next_run_index = 1
	unscheduled = set(nodes)
	current_time = 0

	# Validate resource values once.
	for node in nodes:
		resource = float(node_defs[node]["resource"])
		if resource <= 0 or resource > 1:
			raise ValueError(
				f"Node {node} must have resource in the interval (0, 1]."
			)

	def task_ready(node, time_slot):
		return all(parent in finish_time and finish_time[parent] <= time_slot for parent in parents[node])

	def fits_on_processor(proc_idx, resource, start_slot, node):
		for slot in range(start_slot, start_slot + node["time"]):
			used = reservations[proc_idx].get(slot, 0.0)
			if used + resource > 1.0 + 1e-9:
				return False
		return True

	def reserve_on_processor(proc_idx, resource, start_slot, node):
		for slot in range(start_slot, start_slot + node["time"]):
			reservations[proc_idx][slot] = reservations[proc_idx].get(slot, 0.0) + resource

	while unscheduled:
		scheduled_this_tick = False

		# Keep selecting the highest-priority ready task until nothing else fits now.
		while True:
			selected_node = None
			selected_proc = None
			num_unready = 0

			for idx, node in enumerate(priority_order):
				if num_unready >= len(priority_order)/look_ahead:
					break
				if node not in unscheduled:
					continue
				if not task_ready(node, current_time):
					num_unready += 1
					continue

				resource = float(node_defs[node]["resource"])
				for proc_idx in range(num_processors):
					if fits_on_processor(proc_idx, resource, current_time, node_defs[node]):
						selected_node = node
						selected_proc = proc_idx
						break

				if selected_node is not None:
					break

			if selected_node is None or selected_proc is None:
				break

			resource = float(node_defs[selected_node]["resource"])
			reserve_on_processor(selected_proc, resource, current_time, node_defs[selected_node])
			assignment[selected_node] = selected_proc
			start_time[selected_node] = current_time
			finish_time[selected_node] = current_time + node_defs[selected_node]["time"]
			run_order[selected_node] = next_run_index
			next_run_index += 1
			unscheduled.remove(selected_node)
			scheduled_this_tick = True

		if unscheduled and not scheduled_this_tick:
			# Detect deadlock from cycles or impossible constraints.
			any_future_ready = any(
				all(parent in finish_time for parent in parents[node]) for node in unscheduled
			)
			if not any_future_ready:
				raise ValueError(
					"No schedulable tasks remain. Check for DAG cycles or invalid dependencies."
				)

		current_time += 1

	return assignment, start_time, finish_time, run_order


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


def draw_connected_dag(output_path: str = "dag_connected.png", look_ahead=len(PRIORITY_ORDERING), num_processors=NUM_PROCESSORS) -> None:
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
	dot.attr(rankdir="LR", splines="spline", nodesep="0.45", ranksep="0.6")
	dot.attr("node", shape="circle", style="filled", fontcolor="white", fontsize="14", penwidth="1.5")
	dot.attr("edge", color="#2f2f2f", arrowsize="0.8", penwidth="1.0")

	for node in nodes:
		# proc_idx = assignment[node]
		# fillcolor = PROCESSOR_COLORS[proc_idx % len(PROCESSOR_COLORS)]
		#label = str(run_order[node])
		dot.node(node, label='', fillcolor='green')

	for src, dst in EDGES:
		dot.edge(src, dst)

	# Force nodes in the same stage to share rank, making LR layout clearly horizontal.
	max_level = max(levels.values()) if levels else 0
	for level in range(max_level + 1):
		same_level_nodes = [node for node in PRIORITY_ORDERING if levels[node] == level]
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
	#print(f"Rendered DAG to: {rendered_path}")

	# scheduled_order = [
	# 	node for node, _ in sorted(run_order.items(), key=lambda item: item[1])
	# ]
	return None, None
	period_tasks = ["E", "L", "R"]

	expected_interval = len(PRIORITY_ORDERING) // len(period_tasks)
	# t/3 = 0, 0, 1, 1, 1, 2, 2, 2, ...
	
	expected = np.array([i // expected_interval for i in range(len(PRIORITY_ORDERING))])
	observed = []
	for t in PRIORITY_ORDERING:
		count = sum(1 for task in period_tasks if task in scheduled_order[:scheduled_order.index(t)+1])
		observed.append(count)

	observed = np.array(observed)
	
	# Calculate cumulative error (difference between expected and observed)
	cumulative_error = np.sum(observed - expected)
	

	total_runtime = max(finish_time.values(), default=0)
	#print("Scheduled task order:")
	#print("  " + ", ".join(scheduled_order))
	print(f"runtime: {total_runtime}, look_ahead: {look_ahead}, cumulative_error: {cumulative_error}")
	return total_runtime, cumulative_error


	


if __name__ == "__main__":
	plt.figure(figsize=(8, 4))

	for proc in [1]:
		runs = []
		errors = []
		for look_ahead in [1, 2, 3, 4, 5]:
			runtime, cumulative_error = draw_connected_dag(look_ahead=look_ahead, num_processors=proc)
			runs.append(runtime)
			errors.append(cumulative_error)
		plt.plot(errors, runs, marker='o', label=f"{proc} processors")

	plt.xlabel('Net Error')
	plt.ylabel('Time (t)')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig('net_error.png', dpi=300, bbox_inches='tight')
	#plt.show()

import matplotlib.pyplot as plt
import numpy as np

# breadth_first_priority_ordering = ["s", "t1", "t4", "t7", "t2", "t5", "t8", "t3", "t6", "t9"]
depth_first_priority_ordering = ["B", "I", "J", "C", "H", "A", "P", "O", "N", "D", "Q", "F", "R", "S", "E", "K", "M", "L"]
# region_ordering = ["s", "t7", "t8", "t9", "t4", "t5", "t6", "t1", "t2", "t3"]

region_ordering = ["C", "B", "A", "D", "F", "E", "J", "I", "H", "K", "M", "L", "P", "O", "N", "Q", "S", "R"]
#["C", "B", "A", "D", "J", "I", "H", "K", "F", "E", "M", "L", "P", "O", "N", "Q", "S", "R"]

period_tasks = ["F", "M", "S"]

#period_tasks = ["t3", "t6", "t9"]

fig, ax = plt.subplots(1, figsize=(10, 6))  # There are 2 orderings

# Expected: period tasks expected every order/num period tasks
expected_interval = len(region_ordering) // len(period_tasks)
# t/3 = 0, 0, 1, 1, 1, 2, 2, 2, ...
expected = np.array([i // expected_interval for i in range(len(region_ordering))])

for idx, l in enumerate([depth_first_priority_ordering, region_ordering]):
    # Generate time points from 0 to 9
    time = np.linspace(0, 17, 18)

    # Observed: count actual period tasks in the ordering up to each time point
    observed = []
    for t in l:
        count = sum(1 for task in period_tasks if task in l[:l.index(t)+1])
        observed.append(count)

    observed = np.array(observed)

    # Calculate cumulative error (difference between expected and observed)
    cumulative_error = observed - expected

    # Plot
    # ax[idx].plot(time, expected, label='Expected', marker='o', markersize=3)
    # ax[idx].plot(time, observed, label='Observed', marker='s', markersize=3)
    ax.plot(time, cumulative_error, label=["Depth-First", "Region"][idx], marker=['o', 'x'][idx])
    ax.set_xlabel('Schedule Index')
    ax.set_ylabel('Completion Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 18)
    ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig('cumulative_error.png', dpi=300, bbox_inches='tight')
plt.show()
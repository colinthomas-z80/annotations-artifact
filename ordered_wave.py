import matplotlib.pyplot as plt
import numpy as np


depth_first_priority_ordering = ["t7", "t8", "t9", "t4", "t5", "t6", "t1", "t2", "t3"]
#["t3", "t6", "t9", "t2", "t5", "t8", "t1", "t4", "t7"]

breadth_first_priority_ordering = ["t1", "t4", "t7", "t2", "t5", "t8", "t3", "t6", "t9"]

region_ordering = ["t7", "t8", "t9", "t4", "t5", "t6", "t1", "t2", "t3"]

period_tasks = ["start", "t3", "t6", "t9", "end"]

fig, ax = plt.subplots(3, figsize=(10, 6))  # There are 3 orderings

titles = ["Depth-First", "Breadth-First", "Region-Ordered"]

for idx, ordering in enumerate([depth_first_priority_ordering, breadth_first_priority_ordering, region_ordering]):

    ordering = ["start"] + ordering # + ["end"]
    last_period_end = 0
    for i, t in enumerate(ordering):
        if t in period_tasks:
            t_start = last_period_end 
            t_end = i
            period = t_end - t_start
            if period <= 0:
                continue
            x = np.linspace(t_start, t_end, 1000)
            y = np.zeros_like(x)
            mask = (x >= t_start) & (x <= t_end)
            x_interval = x[mask]
            phase = 2 * np.pi * (x_interval - t_start) / period
            y[mask] = np.sin(phase)

            ax[idx].yaxis.set_visible(False)
            if idx != 2:
                ax[idx].xaxis.set_visible(False)

            ax[idx].set_xticklabels([f't+{j}' for j in range(-1, len(ordering))])
            ax[idx].set_title(titles[idx])

            ax[idx].plot(x, y, linewidth=2)
            ax[idx].scatter([t_start, t_end], [0, 0], color='red', s=30, zorder=5)
            last_period_end = i

plt.savefig('ordered_waves.png', dpi=300)
plt.show()


       

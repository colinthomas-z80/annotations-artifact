import matplotlib.pyplot as plt
import numpy as np

# Abstract time t
t = np.linspace(0, 3, 4)

# y = x (where x is t, and y is inputs)
y = np.ones_like(t)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(t, y, linewidth=2)
plt.xlabel('Time (t)', fontsize=24)
plt.xticks([0, 1, 2, 3], fontsize=20)
plt.yticks([0, 1, 2, 3], fontsize=20)
plt.ylabel('Resource Demand', fontsize=24)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('constant_resource.png', dpi=300, bbox_inches='tight')

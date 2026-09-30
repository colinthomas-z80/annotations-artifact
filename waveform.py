import matplotlib.pyplot as plt
import numpy as np

# Create time array for 3 periods
t = np.linspace(0, 6 * np.pi, 1000)

# Create sine wave
y = np.sin(t)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, y, 'b-', linewidth=2)

# Mark the start of each waveform period
period_starts = np.arange(0, t[-1] + 1e-9, 2 * np.pi)
ax.scatter(period_starts, np.sin(period_starts), color='red', s=30, zorder=5)

# Use abstract time labels instead of numeric values
ax.set_xticks(period_starts)
ax.set_xticklabels(['', '', '', ''])
#ax.set_xticklabels(['t', 't+1', 't+2', 't+3'])

# Remove the y-axis
#ax.spines['left'].set_visible(False)
ax.yaxis.set_visible(False)

#ax.set_xlabel('Time')
#ax.set_title('Subgraph Completion Interval')
ax.grid(True, alpha=0.3)
plt.savefig('waveform.png', dpi=300, bbox_inches='tight')
plt.show()

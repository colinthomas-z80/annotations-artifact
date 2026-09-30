import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 4))

# Define colors for a balanced gradient with softer transitions.
# Orange and green each keep a similar share of the bar, while blue remains
# the central band but blends in gradually instead of switching abruptly.
colors = [
    (0.00, 'orange'),
    (0.22, 'orange'),
    (0.30, '#f4a460'),
    (0.38, 'blue'),
    (0.50, 'blue'),
    (0.62, '#5aa9e6'),
    (0.78, 'green'),
    (1.00, 'green'),
]
n_bins = 100
cmap = LinearSegmentedColormap.from_list('gradient', colors, N=n_bins)

# Create horizontal rectangle with gradient as a single image to avoid seams.
rect_height = 1
rect_width = 10
y_position = 0

gradient = np.linspace(0, 1, n_bins)
image = np.tile(gradient, (10, 1))
ax.imshow(
    image,
    cmap=cmap,
    aspect='auto',
    extent=[0, rect_width, y_position, y_position + rect_height],
    origin='lower',
    interpolation='bilinear',
)

# Set axis limits and remove axes
ax.set_xlim(0, rect_width)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/scuzee/Documents/annotations_paperfigs/gradient_rectangle.png', dpi=150, bbox_inches='tight')
#plt.show()

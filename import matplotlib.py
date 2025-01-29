import matplotlib.pyplot as plt
import numpy as np

# Define points in global coordinates for plotting
points_global = {
    "O": np.array([0, 0]),              # Frame pivot
    "A": np.array([-0.44, 0]),         # Rear wheel axle
    "B": np.array([-0.36091, 0.19454]),# BC damper left endpoint
    "C": np.array([0.1, 0.21]),        # BC damper right endpoint
    "D": np.array([0.57, 0]),          # DE damper bottom
    "E": np.array([0.51, 0.36]),       # DE damper top
    "F": np.array([0, 0.35]),          # Midpoint on frame
    "P": np.array([0, 0.90]),          # Rider load
    "Top_Vertical_Point": np.array([0.1, 0.275]), # Top of vertical bar
    "Bottom_Vertical_Point": np.array([0.1, 0.21])# Bottom of vertical bar
}

# Define links for visualization
links = [
    ("O", "A"),  # Rear triangle base
    ("O", "B"),  # Rear triangle left side
    ("A", "B"),  # Rear triangle right side
    ("B", "C"),  # BC damper
    ("D", "E"),  # DE damper
    ("O", "E"),  # Frame connection
    ("F", "E"),  # Frame diagonal
    ("F", "Top_Vertical_Point"), # Vertical link top
    ("Top_Vertical_Point", "Bottom_Vertical_Point"), # Vertical bar
]

# Plot the geometry
fig, ax = plt.subplots(figsize=(8, 6))
for link in links:
    start, end = points_global[link[0]], points_global[link[1]]
    ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', lw=2)

# Add points and labels
for point_name, coord in points_global.items():
    ax.plot(coord[0], coord[1], 'ro')  # Plot points
    ax.text(coord[0], coord[1], f' {point_name}', fontsize=10)

# Set plot limits and labels
ax.set_xlim(-0.6, 0.7)
ax.set_ylim(-0.1, 1.0)
ax.set_xlabel("X Coordinate (m)")
ax.set_ylabel("Y Coordinate (m)")
ax.set_title("Bicycle Geometry Visualization")
ax.grid(True)
plt.show()

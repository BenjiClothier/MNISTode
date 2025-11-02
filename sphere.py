import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# More irregular wavy sphere
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u, v)

# Radius with multiple frequency waves for irregularity
R = 1.5 + 0.25*np.sin(4*U)*np.cos(3*V) + 0.15*np.cos(6*U) + 0.1*np.sin(2*V)*np.cos(4*U)

# Spherical coordinates
x = R * np.sin(U) * np.cos(V)
y = R * np.sin(U) * np.sin(V)
z = R * np.cos(U)

# Plot surface with grid
ax.plot_surface(x, y, z, cmap='cool', alpha=0.6, edgecolor='gray', 
                linewidth=0.3, antialiased=True)

# Calculate a point on the manifold to end trajectory
# Choose specific u, v coordinates for endpoint
u_end = np.pi * 0.4  # latitude
v_end = np.pi * 1.2  # longitude
R_end = 1.5 + 0.25*np.sin(4*u_end)*np.cos(3*v_end) + 0.15*np.cos(6*u_end) + 0.1*np.sin(2*v_end)*np.cos(4*u_end)
x_end = R_end * np.sin(u_end) * np.cos(v_end)
y_end = R_end * np.sin(u_end) * np.sin(v_end)
z_end = R_end * np.cos(u_end)

# Trajectory from start to exact point on manifold
t = np.linspace(0, 1, 200)  # More points for finer line
x_start, y_start, z_start = -2.5, -2.8, 3.5

traj_x = x_start + (x_end - x_start) * t
traj_y = y_start + (y_end - y_start) * t  
traj_z = z_start + (z_end - z_start) * (t**1.5)  # Curved descent

# Plot trajectory with finer line
ax.plot(traj_x, traj_y, traj_z, 'r-', linewidth=1.5, alpha=0.9)

# Arrow at the end
arrow_start = 0.95
ax.quiver(traj_x[int(arrow_start*len(t))], traj_y[int(arrow_start*len(t))], traj_z[int(arrow_start*len(t))], 
          traj_x[-1]-traj_x[int(arrow_start*len(t))], 
          traj_y[-1]-traj_y[int(arrow_start*len(t))], 
          traj_z[-1]-traj_z[int(arrow_start*len(t))],
          color='r', arrow_length_ratio=0.5, linewidth=2, alpha=0.9)

# Start and end points
ax.scatter([x_start], [y_start], [z_start], color='blue', s=100, zorder=10)
ax.scatter([x_end], [y_end], [z_end], color='red', s=100, zorder=10)

# Remove all axes completely
ax.set_axis_off()

# Set viewing angle and limits
ax.view_init(elev=25, azim=130)
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-0.5, 3])

# Labels
ax.text(x_start-0.3, y_start, z_start+0.4, 'Noise', fontsize=12, color='blue', weight='bold')
ax.text(x_end+0.3, y_end, z_end-0.3, 'Data', fontsize=12, color='red', weight='bold')
ax.text(2, 2, 2, 'Data Manifold', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('wavy_sphere_manifold.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
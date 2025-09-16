import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches

import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
import plotting_fun as pf
from mpl_toolkits.mplot3d import Axes3D

from blocks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = MLPVectorField(dim=2, hiddens=[64,64,64,64]).to(device)
model.load_state_dict(torch.load('trained/model2025-09-10_12-27-45.pth')) 

# Initialise Path
path = LinearConditionalProbabilityPath(
    p_simple = Gaussian.isotropic(dim=2, std=1.0).to(device),
    p_data = UniformSampler(device)
)

ode = MoonODE(model)
simulator = HuenSimulator(ode)

num_samples = 50000
num_marginals = 5

scale = 6.0

# Your existing setup code
ts = torch.linspace(0, 1, 100).to(device)
record_every_idxs = pf.record_every(len(ts), len(ts) // (num_marginals - 1))
x0 = path.p_simple.sample(num_samples)
xts = simulator.simulate_with_trajectory(x0, ts.view(1, -1, 1).expand(num_samples, -1, 1))
xts = xts[:, record_every_idxs, :]

# Convert data to numpy for easier manipulation
xts_np = xts.cpu().numpy()  # Shape: (num_samples, num_time_points, 2)
ts_recorded = ts[record_every_idxs].cpu().numpy()

# Create animated version
fig_anim = plt.figure(figsize=(14, 10))
ax_anim = fig_anim.add_subplot(111, projection='3d')

# Limit samples for animation performance
num_samples_to_animate = min(num_samples, 100)
sample_indices = np.random.choice(num_samples, num_samples_to_animate, replace=False)

# Prepare colors for different samples
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Animation function
def animate_trajectories(frame):
    ax_anim.clear()
    
    # Current time index
    current_time_idx = frame
    current_time = ts_recorded[current_time_idx]
    
    # Plot trajectory lines up to current time
    for i, sample_idx in enumerate(sample_indices):
        sample_trajectory = xts_np[sample_idx, :current_time_idx+1, :]
        
        if current_time_idx > 0:  # Only plot if we have at least 2 points
            color = colors[i % len(colors)]
            ax_anim.plot(ts_recorded[:current_time_idx+1], 
                        sample_trajectory[:, 0], 
                        sample_trajectory[:, 1], 
                        alpha=0.6, linewidth=1.0, color=color)
    
    # Plot current points
    current_points = xts_np[sample_indices, current_time_idx, :]
    scatter = ax_anim.scatter(np.full(len(sample_indices), current_time), 
                             current_points[:, 0], 
                             current_points[:, 1],
                             c=current_time, cmap='viridis', 
                             s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Set labels and limits
    ax_anim.set_xlabel('Time', fontsize=12)
    ax_anim.set_ylabel('Y coordinate', fontsize=12)
    ax_anim.set_zlabel('Z coordinate', fontsize=12)
    ax_anim.set_xlim(0, 1)
    ax_anim.set_ylim(-scale, scale)
    ax_anim.set_zlim(-scale, scale)
    ax_anim.set_title(f'Trajectory Evolution: t = {current_time:.3f}\nSample Count: {num_samples_to_animate}', 
                     fontsize=14)
    
    # Add time progress bar
    progress = current_time_idx / (len(ts_recorded) - 1)
    ax_anim.text2D(0.02, 0.95, f'Progress: {progress*100:.1f}%', 
                   transform=ax_anim.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Create animation
anim = FuncAnimation(fig_anim, animate_trajectories, frames=len(ts_recorded), 
                    interval=200, repeat=True, blit=False)

# Display the animation
plt.tight_layout()
plt.show()

# Save animation as GIF (optional - uncomment to save)
# anim.save('trajectory_evolution.gif', writer=PillowWriter(fps=5))

print(f"Animation created with {num_samples_to_animate} samples over {len(ts_recorded)} time steps")

# Alternative: Side-by-side animation showing both current state and full trajectory
fig_side = plt.figure(figsize=(20, 8))

# Current state subplot
ax_current = fig_side.add_subplot(121, projection='3d')
# Full trajectory subplot  
ax_full = fig_side.add_subplot(122, projection='3d')

def animate_side_by_side(frame):
    # Clear both axes
    ax_current.clear()
    ax_full.clear()
    
    current_time_idx = frame
    current_time = ts_recorded[current_time_idx]
    
    # LEFT PLOT: Current marginal distribution
    current_points = xts_np[sample_indices, current_time_idx, :]
    ax_current.scatter(np.full(len(sample_indices), current_time), 
                      current_points[:, 0], 
                      current_points[:, 1],
                      c='red', s=30, alpha=0.7)
    
    ax_current.set_xlabel('Time', fontsize=10)
    ax_current.set_ylabel('Y coordinate', fontsize=10)
    ax_current.set_zlabel('Z coordinate', fontsize=10)
    ax_current.set_xlim(0, 1)
    ax_current.set_ylim(-scale, scale)
    ax_current.set_zlim(-scale, scale)
    ax_current.set_title(f'Current State: t = {current_time:.3f}', fontsize=12)
    
    # RIGHT PLOT: Accumulated trajectories
    for i, sample_idx in enumerate(sample_indices[:20]):  # Show fewer for clarity
        sample_trajectory = xts_np[sample_idx, :current_time_idx+1, :]
        
        if current_time_idx > 0:
            color = colors[i % len(colors)]
            ax_full.plot(ts_recorded[:current_time_idx+1], 
                        sample_trajectory[:, 0], 
                        sample_trajectory[:, 1], 
                        alpha=0.7, linewidth=1.5, color=color)
    
    # Highlight current points
    current_points_subset = xts_np[sample_indices[:20], current_time_idx, :]
    time_coords_subset = np.full(20, current_time)
    ax_full.scatter(time_coords_subset, 
                   current_points_subset[:, 0], 
                   current_points_subset[:, 1],
                   c='red', s=60, alpha=1.0, edgecolors='black', linewidth=1)
    
    ax_full.set_xlabel('Time', fontsize=10)
    ax_full.set_ylabel('Y coordinate', fontsize=10)
    ax_full.set_zlabel('Z coordinate', fontsize=10)
    ax_full.set_xlim(0, 1)
    ax_full.set_ylim(-scale, scale)
    ax_full.set_zlim(-scale, scale)
    ax_full.set_title('Trajectory Evolution (20 samples)', fontsize=12)

# Create side-by-side animation
anim_side = FuncAnimation(fig_side, animate_side_by_side, frames=len(ts_recorded), 
                         interval=250, repeat=True, blit=False)

plt.tight_layout()
plt.show()

# Save side-by-side animation (optional)
# anim_side.save('trajectory_side_by_side.gif', writer=PillowWriter(fps=4))

print("Side-by-side animation created!")

# Version with trajectory lines connecting each sample
fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111, projection='3d')

# Convert data to numpy for easier manipulation
xts_np = xts.cpu().numpy()  # Shape: (num_samples, num_time_points, 2)
ts_recorded = ts[record_every_idxs].cpu().numpy()

# Plot trajectory lines for each sample
num_samples_to_plot = min(num_samples, 200)  # Limit for visualization clarity
sample_indices = np.random.choice(num_samples, num_samples_to_plot, replace=False)

for i, sample_idx in enumerate(sample_indices):
    # Extract trajectory for this sample
    sample_trajectory = xts_np[sample_idx, :, :]  # Shape: (num_time_points, 2)
    
    # Plot line connecting this sample across time
    ax2.plot(ts_recorded, 
             sample_trajectory[:, 0], 
             sample_trajectory[:, 1], 
             alpha=0.3, linewidth=0.8, color='blue')

# Also plot scatter points for better visibility
all_times = []
all_y = []
all_z = []

for idx in range(xts.shape[1]):
    xx = xts[sample_indices, idx, :].cpu().numpy()  # Only plot subset
    tt = ts[record_every_idxs[idx]].item()
    
    time_coords = np.full(xx.shape[0], tt)
    all_times.extend(time_coords)
    all_y.extend(xx[:, 0])
    all_z.extend(xx[:, 1])

# Convert to arrays
all_times = np.array(all_times)
all_y = np.array(all_y)
all_z = np.array(all_z)

# Create scatter plot with color mapping
scatter = ax2.scatter(all_times, all_y, all_z, 
                     c=all_times, cmap='viridis', 
                     alpha=0.7, s=15)

# Add colorbar
plt.colorbar(scatter, ax=ax2, label='Time', shrink=0.5)

ax2.set_xlabel('Time', fontsize=14)
ax2.set_ylabel('Y coordinate', fontsize=14)
ax2.set_zlabel('Z coordinate', fontsize=14)
ax2.set_ylim(-scale, scale)
ax2.set_zlim(-scale, scale)
ax2.set_title(f'Learned Trajectory Evolution with Sample Paths (n={num_samples_to_plot})', fontsize=16)

plt.tight_layout()
plt.show()

# Alternative version with color-coded trajectory lines
fig3 = plt.figure(figsize=(14, 10))
ax3 = fig3.add_subplot(111, projection='3d')

# Plot trajectory lines with different colors
colors = plt.cm.tab10(np.linspace(0, 1, min(10, num_samples_to_plot)))
num_samples_colored = min(50, num_samples_to_plot)  # Even fewer for colored lines

for i in range(num_samples_colored):
    sample_idx = sample_indices[i]
    sample_trajectory = xts_np[sample_idx, :, :]
    
    # Use different colors for different samples
    color = colors[i % len(colors)]
    ax3.plot(ts_recorded, 
             sample_trajectory[:, 0], 
             sample_trajectory[:, 1], 
             alpha=0.7, linewidth=1.2, color=color,
             label=f'Sample {i+1}' if i < 10 else "")  # Only label first 10

# Plot remaining samples in gray
for i in range(num_samples_colored, num_samples_to_plot):
    sample_idx = sample_indices[i]
    sample_trajectory = xts_np[sample_idx, :, :]
    
    ax3.plot(ts_recorded, 
             sample_trajectory[:, 0], 
             sample_trajectory[:, 1], 
             alpha=0.2, linewidth=0.5, color='gray')

# Add scatter points at endpoints
start_points = xts_np[sample_indices, 0, :]
end_points = xts_np[sample_indices, -1, :]

ax3.scatter(np.full(len(sample_indices), ts_recorded[0]), 
           start_points[:, 0], start_points[:, 1],
           color='green', s=30, alpha=0.8, label='Start (t=0)')

ax3.scatter(np.full(len(sample_indices), ts_recorded[-1]), 
           end_points[:, 0], end_points[:, 1],
           color='red', s=30, alpha=0.8, label='End (t=1)')

ax3.set_xlabel('Time', fontsize=14)
ax3.set_ylabel('Y coordinate', fontsize=14)
ax3.set_zlabel('Z coordinate', fontsize=14)
ax3.set_ylim(-scale, scale)
ax3.set_zlim(-scale, scale)
ax3.set_title(f'Individual Sample Trajectories (n={num_samples_to_plot})', fontsize=16)

# Add legend
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()
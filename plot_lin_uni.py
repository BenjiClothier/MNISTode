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
model.load_state_dict(torch.load('trained/model2025-10-31_10-48-32.pth')) 
mean = torch.tensor([10,10], dtype=torch.float32)
# Initialise Path
path = LinearConditionalProbabilityPath(
    p_simple = Gaussian.isotropic(dim=2, std=1.0).to(device),
    p_data = Gaussian.isotropic_with_mean(mean, std=1.0).to(device)
)

ode = MoonODE(model)
simulator = HuenSimulator(ode)

num_samples = 50000
num_marginals = 5

fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 2))
axes = axes.reshape(2, num_marginals)
scale = 15.0

ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
for idx, t in enumerate(ts):
    tt = t.view(1,1).expand(num_samples,1)
    xts = path.sample_marginal_path(tt)
    pf.hist2d_samples(samples=xts.cpu(), ax=axes[0, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
    axes[0, idx].set_xlim(-scale, scale)
    axes[0, idx].set_ylim(-scale, scale)
    axes[0, idx].set_xticks([])
    axes[0, idx].set_yticks([])
    axes[0, idx].set_title(f'$t={t.item():.2f}$', fontsize=15)
axes[0, 0].set_ylabel("Ground Truth", fontsize=20)


ts = torch.linspace(0,1,100).to(device)
record_every_idxs = pf.record_every(len(ts), len(ts) // (num_marginals - 1))
x0 = path.p_simple.sample(num_samples)
xts = simulator.simulate_with_trajectory(x0, ts.view(1,-1,1).expand(num_samples,-1,1))
xts = xts[:,record_every_idxs,:]
for idx in range(xts.shape[1]):
    xx = xts[:,idx,:]
    pf.hist2d_samples(samples=xx.cpu(), ax=axes[1, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
    axes[1, idx].set_xlim(-scale, scale)
    axes[1, idx].set_ylim(-scale, scale)
    axes[1, idx].set_xticks([])
    axes[1, idx].set_yticks([])
    tt = ts[record_every_idxs[idx]]
    axes[1, idx].set_title(f'$t={tt.item():.2f}$', fontsize=15)
axes[1, 0].set_ylabel("Learned", fontsize=20) 

plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each marginal at its corresponding time point
for idx in range(xts.shape[1]):
    xx = xts[:, idx, :].cpu().numpy()  # Convert to numpy for plotting
    tt = ts[record_every_idxs[idx]].item()
    
    # Create x-coordinates (time) for all samples at this time point
    time_coords = np.full(xx.shape[0], tt)
    
    # Plot scatter: time on x-axis, samples on y-z axes
    ax.scatter(time_coords, xx[:, 0], xx[:, 1], 
              alpha=0.6, s=10, 
              label=f't={tt:.2f}' if idx % 3 == 0 else "")  # Only label every 3rd for clarity

# Set labels and limits
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Y coordinate', fontsize=14)
ax.set_zlabel('Z coordinate', fontsize=14)
ax.set_ylim(-scale, scale)
ax.set_zlim(-scale, scale)

# Optional: add legend (might be cluttered with many time points)
# ax.legend()

# Set title
ax.set_title('Learned Trajectory Evolution', fontsize=16)

plt.tight_layout()
plt.show()

# Alternative version with color-coded time progression
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Collect all data for single scatter call with color mapping
all_times = []
all_y = []
all_z = []

for idx in range(xts.shape[1]):
    xx = xts[:, idx, :].cpu().numpy()
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
                     alpha=0.6, s=10)

# Add colorbar
plt.colorbar(scatter, ax=ax2, label='Time', shrink=0.5)

ax2.set_xlabel('Time', fontsize=14)
ax2.set_ylabel('Y coordinate', fontsize=14)
ax2.set_zlabel('Z coordinate', fontsize=14)
ax2.set_ylim(-scale, scale)
ax2.set_zlim(-scale, scale)
ax2.set_title('Learned Trajectory Evolution (Color-coded)', fontsize=16)

plt.tight_layout()
plt.show()


# Your existing setup code
ts = torch.linspace(0, 1, 100).to(device)
record_every_idxs = pf.record_every(len(ts), len(ts) // (num_marginals - 1))
x0 = path.p_simple.sample(num_samples)
xts = simulator.simulate_with_trajectory(x0, ts.view(1, -1, 1).expand(num_samples, -1, 1))
xts = xts[:, record_every_idxs, :]

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each marginal at its corresponding time point
for idx in range(xts.shape[1]):
    xx = xts[:, idx, :].cpu().numpy()  # Convert to numpy for plotting
    tt = ts[record_every_idxs[idx]].item()
    
    # Create x-coordinates (time) for all samples at this time point
    time_coords = np.full(xx.shape[0], tt)
    
    # Plot scatter: time on x-axis, samples on y-z axes
    ax.scatter(time_coords, xx[:, 0], xx[:, 1], 
              alpha=0.6, s=10, 
              label=f't={tt:.2f}' if idx % 3 == 0 else "")  # Only label every 3rd for clarity

# Set labels and limits
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Y coordinate', fontsize=14)
ax.set_zlabel('Z coordinate', fontsize=14)
ax.set_ylim(-scale, scale)
ax.set_zlim(-scale, scale)

# Optional: add legend (might be cluttered with many time points)
# ax.legend()

# Set title
ax.set_title('Learned Trajectory Evolution', fontsize=16)

plt.tight_layout()
plt.show()

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


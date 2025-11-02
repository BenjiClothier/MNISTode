import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
dlogp_stacked = np.load('testing/norm.npy')
print(f"Data shape: {dlogp_stacked.shape}")
print(f"Shape is (num_iterations, timesteps): ({dlogp_stacked.shape[0]}, {dlogp_stacked.shape[1]})")

# Create timesteps (from 1 to 0, since you flipped the array)
timesteps = np.linspace(1, 0, dlogp_stacked.shape[1])

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Plot all trajectories with mean and std bands
ax = axes[0, 0]
# Plot individual trajectories with transparency
for i in range(min(50, dlogp_stacked.shape[0])):  # Plot first 50 to avoid overcrowding
    ax.plot(timesteps, dlogp_stacked[i, :], alpha=0.1, color='blue')

# Calculate and plot mean and std
mean_traj = np.mean(dlogp_stacked, axis=0)
std_traj = np.std(dlogp_stacked, axis=0)
ax.plot(timesteps, mean_traj, 'r-', linewidth=2, label='Mean')
ax.fill_between(timesteps, mean_traj - std_traj, mean_traj + std_traj, 
                 alpha=0.3, color='red', label='±1 std')
ax.fill_between(timesteps, mean_traj - 2*std_traj, mean_traj + 2*std_traj, 
                 alpha=0.1, color='red', label='±2 std')
ax.set_xlabel('Time (1 → 0)')
ax.set_ylabel('Log Likelihood')
ax.set_title('Log Likelihood Trajectories without accumulation')
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # Since time goes from 1 to 0

# 2. Heatmap of all trajectories
ax = axes[0, 1]
im = ax.imshow(dlogp_stacked, aspect='auto', cmap='viridis', 
               extent=[1, 0, dlogp_stacked.shape[0], 0])
ax.set_xlabel('Time (1 → 0)')
ax.set_ylabel('Iteration')
ax.set_title('Log Likelihood Heatmap')
plt.colorbar(im, ax=ax, label='Log Likelihood')

# 3. Final log likelihood distribution
ax = axes[1, 0]
final_logp = dlogp_stacked[:, -1]  # Last timestep (t=0)
ax.hist(final_logp, bins=30, alpha=0.7, color='green', edgecolor='black')
ax.axvline(final_logp.mean(), color='red', linestyle='--', 
           label=f'Mean: {final_logp.mean():.2f}')
ax.set_xlabel('Final Log Likelihood (t=0)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Final Log Likelihoods')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Cumulative log likelihood along trajectory
ax = axes[1, 1]
# Calculate cumulative sum along time dimension
cumulative_logp = np.cumsum(dlogp_stacked, axis=1)
# Plot mean and std of cumulative
mean_cum = np.mean(cumulative_logp, axis=0)
std_cum = np.std(cumulative_logp, axis=0)
ax.plot(timesteps, mean_cum, 'b-', linewidth=2, label='Mean')
ax.fill_between(timesteps, mean_cum - std_cum, mean_cum + std_cum, 
                 alpha=0.3, color='blue', label='±1 std')
ax.set_xlabel('Time (1 → 0)')
ax.set_ylabel('Cumulative Log Likelihood')
ax.set_title('Cumulative Log Likelihood')
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

plt.tight_layout()
plt.savefig('testing/norm_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# Print some statistics
print("\n--- Statistics ---")
print(f"Initial log-likelihood (t=1): {dlogp_stacked[:, 0].mean():.3f} ± {dlogp_stacked[:, 0].std():.3f}")
print(f"Final log-likelihood (t=0): {dlogp_stacked[:, -1].mean():.3f} ± {dlogp_stacked[:, -1].std():.3f}")
print(f"Total change: {(dlogp_stacked[:, -1] - dlogp_stacked[:, 0]).mean():.3f}")
print(f"Min value: {dlogp_stacked.min():.3f}")
print(f"Max value: {dlogp_stacked.max():.3f}")

# Additional plot: Just the mean trajectory with error bars at specific points
fig2, ax2 = plt.subplots(figsize=(10, 6))
sample_points = np.arange(0, dlogp_stacked.shape[1], 10)  # Every 10th timestep
ax2.errorbar(timesteps[sample_points], mean_traj[sample_points], 
             yerr=std_traj[sample_points], fmt='o-', capsize=5, capthick=2)
ax2.set_xlabel('Time (1 → 0)')
ax2.set_ylabel('Log Likelihood')
ax2.set_title('Mean Log Likelihood Trajectory with Error Bars')
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()
plt.savefig('testing/norm_mean_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()
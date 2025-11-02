import numpy as np
import matplotlib.pyplot as plt

# Load data
dlogp = np.load('test/norm.npy')
timesteps = np.linspace(1, 0, dlogp.shape[1])

# Calculate statistics
mean_traj = np.mean(dlogp, axis=0)
std_traj = np.std(dlogp, axis=0)

# Calculate cumulative sum
cumulative_logp = np.cumsum(dlogp, axis=1)
mean_cum = np.mean(cumulative_logp, axis=0)
std_cum = np.std(cumulative_logp, axis=0)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 1. Log Likelihood Trajectory
ax1.plot(timesteps, mean_traj, 'b-', linewidth=2, label='Mean')
ax1.fill_between(timesteps, mean_traj - std_traj, mean_traj + std_traj, 
                  alpha=0.3, color='blue', label='±1 std')
ax1.fill_between(timesteps, mean_traj - 2*std_traj, mean_traj + 2*std_traj, 
                  alpha=0.1, color='blue', label='±2 std')
ax1.set_xlabel('Time (1 → 0)', fontsize=12)
ax1.set_ylabel('Log Likelihood', fontsize=12)
ax1.set_title('Log Likelihood Trajectory MNIST', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.invert_xaxis()

# Add text annotation for final value
final_mean = mean_traj[-1]
final_std = std_traj[-1]
ax1.text(0.02, 0.95, f'Final: {final_mean:.2f} ± {final_std:.2f}', 
         transform=ax1.transAxes, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Accumulated (Cumulative) Log Likelihood
ax2.plot(timesteps, mean_cum, 'r-', linewidth=2, label='Mean')
ax2.fill_between(timesteps, mean_cum - std_cum, mean_cum + std_cum, 
                  alpha=0.3, color='red', label='±1 std')
ax2.fill_between(timesteps, mean_cum - 2*std_cum, mean_cum + 2*std_cum, 
                  alpha=0.1, color='red', label='±2 std')
ax2.set_xlabel('Time (1 → 0)', fontsize=12)
ax2.set_ylabel('Cumulative Log Likelihood', fontsize=12)
ax2.set_title('Accumulated Log Likelihood MNIST', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()

# Add text annotation for total accumulated
total_mean = mean_cum[-1]
total_std = std_cum[-1]
ax2.text(0.02, 0.95, f'Total: {total_mean:.2f} ± {total_std:.2f}', 
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('testing/trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("TRAJECTORY STATISTICS")
print("="*50)
print(f"Number of samples: {dlogp.shape[0]}")
print(f"Number of timesteps: {dlogp.shape[1]}")
print("\nLog Likelihood:")
print(f"  Initial (t=1): {mean_traj[0]:.3f} ± {std_traj[0]:.3f}")
print(f"  Final (t=0):   {mean_traj[-1]:.3f} ± {std_traj[-1]:.3f}")
print(f"  Change:        {mean_traj[-1] - mean_traj[0]:.3f}")
print("\nAccumulated Log Likelihood:")
print(f"  Total: {mean_cum[-1]:.3f} ± {std_cum[-1]:.3f}")
print("="*50)
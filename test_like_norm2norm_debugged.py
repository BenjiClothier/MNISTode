import os
import numpy as np
import matplotlib.pyplot as plt
from blocks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Setup path and model
mean = torch.tensor([10, 10], dtype=torch.float32)
path = LinearConditionalProbabilityPath(
    p_simple=Gaussian.isotropic(dim=2, std=1.0).to(device),
    p_data=Gaussian.isotropic_with_mean(mean, std=1.0).to(device)
)

# Initialize model
net = MLPVectorField(dim=2, hiddens=[64, 64, 64, 64]).to(device)
net.load_state_dict(torch.load('trained/model2025-10-31_10-48-32.pth', map_location=device))

# Initialize ODE and simulator
ode = Likelihood2DODE(net)
simulator = HuenSimulator(ode)
num_timesteps = 100

# Storage for results
logp_trajectories = []
final_logps = []
prior_logps = []
num_iterations = 100

# Run simulations
for i in range(num_iterations):
    num_samples = 1
    z = path.p_data.sample(num_samples)
    
    timestep = torch.linspace(1, 0, num_timesteps).view(1, -1, 1).expand(num_samples, -1, 1).to(device)
    x0 = z
    
    x_traj, logp_traj, final_logp = simulator.simulate_with_likelihood_augmented(x0, timestep)
    
    # Also compute what the prior log p would be at the starting point for reference
    prior_at_start = ode.prior_logp(z).item()
    
    logp_traj_np = logp_traj.detach().cpu().numpy().squeeze()
    final_logp_np = final_logp.detach().cpu().item()
    
    logp_trajectories.append(logp_traj_np)
    final_logps.append(final_logp_np)
    prior_logps.append(prior_at_start)
    
    if i % 20 == 0:
        print(f"Iteration {i}/{num_iterations}")
        print(f"  Final log p(x_data): {final_logp_np:.4f}")
        print(f"  Prior log p at data point: {prior_at_start:.4f}")
        print(f"  Difference (KL-like): {final_logp_np - prior_at_start:.4f}")

# Analysis
logp_trajectories_stacked = np.stack(logp_trajectories, axis=0)
final_logps_array = np.array(final_logps)
prior_logps_array = np.array(prior_logps)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Log probability trajectories
ax = axes[0, 0]
time_points = np.linspace(1, 0, num_timesteps)
for i in range(min(20, num_iterations)):
    ax.plot(time_points, logp_trajectories_stacked[i], alpha=0.3, color='blue')
ax.plot(time_points, logp_trajectories_stacked.mean(axis=0), 'r-', linewidth=2, label='Mean')
ax.fill_between(time_points, 
                 logp_trajectories_stacked.mean(axis=0) - logp_trajectories_stacked.std(axis=0),
                 logp_trajectories_stacked.mean(axis=0) + logp_trajectories_stacked.std(axis=0),
                 alpha=0.3, color='red')
ax.set_xlabel('Time (1=data, 0=noise)')
ax.set_ylabel('log p(x_t)')
ax.set_title('Evolution of log probability')
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # So data is on left, noise on right

# Plot 2: Final log probabilities histogram
ax = axes[0, 1]
ax.hist(final_logps_array, bins=30, alpha=0.7, color='blue', edgecolor='black', label='log p(x_data)')
ax.hist(prior_logps_array, bins=30, alpha=0.7, color='red', edgecolor='black', label='Prior log p at data')
ax.set_xlabel('Log probability')
ax.set_ylabel('Count')
ax.set_title(f'Log probability distributions\nData mean: {final_logps_array.mean():.3f}±{final_logps_array.std():.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: BPD over trajectory
ax = axes[1, 0]
N = np.prod(z.shape[1:])
bpd_trajectories = -logp_trajectories_stacked / (N * np.log(2))
for i in range(min(20, num_iterations)):
    ax.plot(time_points, bpd_trajectories[i], alpha=0.3, color='green')
ax.plot(time_points, bpd_trajectories.mean(axis=0), 'darkgreen', linewidth=2, label='Mean BPD')
ax.set_xlabel('Time (1=data, 0=noise)')
ax.set_ylabel('Bits per dimension')
ax.set_title('BPD Evolution')
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# Plot 4: Difference from prior
ax = axes[1, 1]
log_ratio = final_logps_array - prior_logps_array
ax.hist(log_ratio, bins=30, color='purple', edgecolor='black', alpha=0.7)
ax.set_xlabel('log p(x_data) - log p_prior(x_data)')
ax.set_ylabel('Count')
ax.set_title(f'Log probability ratio\nMean: {log_ratio.mean():.3f}±{log_ratio.std():.3f}')
ax.axvline(0, color='red', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('testing/likelihood_analysis.png', dpi=150)
plt.show()

# Save results
os.makedirs('testing', exist_ok=True)
np.save('testing/logp_trajectories.npy', logp_trajectories_stacked)
np.save('testing/final_logps.npy', final_logps_array)
np.save('testing/bpd_trajectories.npy', bpd_trajectories)

print("\n=== Summary ===")
print(f"Mean log p(x_data): {final_logps_array.mean():.4f} ± {final_logps_array.std():.4f}")
print(f"Mean BPD: {(-final_logps_array / (N * np.log(2))).mean():.4f}")
print(f"Positive log p ratio: {(final_logps_array > 0).mean()*100:.1f}%")
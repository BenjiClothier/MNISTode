import os
import numpy as np
import matplotlib.pyplot as plt
from blocks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialise probaility path for abnormal greyscale cats
excluded_digits = {j: [i for i in range(10) if i != j] for j in range(10)}
save_dir = 'vectors_MNIST_fixed_8s'
print(f'{save_dir}')
os.makedirs(save_dir, exist_ok=True)


# Initialise model
unet = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
).to(device)
unet.load_state_dict(torch.load('trained/model2025-10-09_12-01-01.pth', map_location=device))

# Initialize ODE and simulator
ode = LikelihoodODE(unet)
simulator = LikelihoodSimulator(ode)
num_timesteps = 100
for j in excluded_digits:
    # Storage for results
    logp_trajectories = []
    final_logps = []
    prior_logps = []
    num_iterations = 100

    path = GaussianConditionalProbabilityPath(
    p_data = FilteredMNISTSampler(excluded_digits=excluded_digits[j]),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
    ).to(device)

    # Run simulations
    for i in range(num_iterations):
        num_samples = 1
        z, y = path.p_data.sample(num_samples)
        
        timestep = torch.linspace(1, 0, num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x0 = z
        
        x_traj, logp_traj, final_logp = simulator.simulate_with_likelihood_augmented(x0, timestep, y)
        
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

    # Saving
    np.save(f'{save_dir}/logp_traj_only{j}.npy', logp_trajectories_stacked)
    np.save(f'{save_dir}/final_logp_array_only{j}.npy', final_logps_array)
    np.save(f'{save_dir}/prior_logp_array_only{j}.npy', prior_logps_array)

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
    plt.savefig(f'{save_dir}/likelihood_analysis_only{j}.png', dpi=150)




    print("\n=== Summary ===")
    print(f"Mean log p(x_data): {final_logps_array.mean():.4f} ± {final_logps_array.std():.4f}")
    print(f"Mean BPD: {(-final_logps_array / (N * np.log(2))).mean():.4f}")
    print(f"Positive log p ratio: {(final_logps_array > 0).mean()*100:.1f}%")
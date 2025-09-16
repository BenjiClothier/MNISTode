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

from blocks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = MLPVectorField(dim=2, hiddens=[64,64,64,64]).to(device)
model.load_state_dict(torch.load('trained/model2025-09-09_11-50-06.pth')) 

# Initialise probability path
path = GaussianConditionalProbabilityPath(
    p_data = UniformSampler(device),
    p_simple_shape=[2],
    alpha = LinearAlpha(),
    beta = SquareRootBeta()
)



num_samples = 5000
num_marginals = 10
num_timesteps = 100

ode = MoonODE(model)
simulator = HuenSimulator(ode)


fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 2))
axes = axes.reshape(2, num_marginals)
scale = 6.0

# ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
# for idx, t in enumerate(ts):
#     tt = t.view(1,1).expand(num_samples,1)
#     xts = path.sample_marginal_path(tt)
#     pf.hist2d_samples(samples=xts.cpu(), ax=axes[0, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
#     axes[0, idx].set_xlim(-scale, scale)
#     axes[0, idx].set_ylim(-scale, scale)
#     axes[0, idx].set_xticks([])
#     axes[0, idx].set_yticks([])
#     axes[0, idx].set_title(f'$t={t.item():.2f}$', fontsize=15)
# axes[0, 0].set_ylabel("Ground Truth", fontsize=20)

z = path.p_data.sample(num_samples)
ts = torch.linspace(1.0,0.0,100).to(device)
record_every_idxs = pf.record_every(len(ts), len(ts) // (num_marginals - 1))
x0, _ = path.p_simple.sample(num_samples)
x0 = z.to(device)
xts = simulator.simulate_with_trajectory(x0, ts.view(1,-1,1).expand(num_samples,-1,1))
xts = xts[:,record_every_idxs,:]
for idx in range(xts.shape[1]):
    xx = xts[:,idx,:]
    # plt.scatter(xx[:,0].cpu(), xx[:,1].cpu(), alpha=0.99, s=1)
    pf.hist2d_samples(samples=xx.cpu(), ax=axes[1, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
    axes[1, idx].set_xlim(-scale, scale)
    axes[1, idx].set_ylim(-scale, scale)
    # axes[1, idx].set_xticks([])
    # axes[1, idx].set_yticks([])
    tt = ts[record_every_idxs[idx]]
    axes[1, idx].set_title(f'$t={tt.item():.2f}$', fontsize=15)
axes[1, 0].set_ylabel("Learned", fontsize=20) 

plt.show()
print(f'xts: {xts.mean(dim=1)}')
plt.close()
fig, axes = plt.subplots(figsize=(8,6))
yy = xts[:,2,:]
plt.scatter(yy[:, 0].cpu(), yy[:, 1].cpu(), alpha=0.6, s=1)
# pf.hist2d_samples(samples=yy.cpu(), ax=axes, bins=200, scale=scale, percentile=99, alpha=1.0)
plt.show()


def sinkhorn_wasserstein_2d(generated_samples, bounds=(0, 1), reg=0.01, max_iter=100, 
                           n_uniform_samples=None, device=None, tol=1e-6):
    """
    Compute 2D Wasserstein distance using Sinkhorn algorithm (optimal transport).
    
    Args:
        generated_samples (torch.Tensor): Tensor of shape (N, 2) with generated 2D samples
        bounds (tuple): (min, max) bounds for uniform distribution
        reg (float): Regularization parameter for Sinkhorn algorithm (smaller = more accurate but slower)
        max_iter (int): Maximum number of Sinkhorn iterations
        n_uniform_samples (int, optional): Number of uniform samples. If None, uses same as generated
        device (str, optional): Device to run computation on
        tol (float): Convergence tolerance
    
    Returns:
        torch.Tensor: Wasserstein distance using optimal transport (scalar)
    """
    if device is None:
        device = generated_samples.device
    
    generated_samples = generated_samples.to(device)
    n_gen = generated_samples.shape[0]
    
    if n_uniform_samples is None:
        n_uniform_samples = n_gen
    
    # Generate uniform samples
    uniform_samples = torch.rand(n_uniform_samples, 2, device=device) * (bounds[1] - bounds[0]) + bounds[0]
    
    # Compute cost matrix (squared Euclidean distances)
    C = torch.cdist(generated_samples, uniform_samples, p=2) ** 2
    
    # Initialize uniform weights
    a = torch.ones(n_gen, device=device) / n_gen  # weights for generated samples
    b = torch.ones(n_uniform_samples, device=device) / n_uniform_samples  # weights for uniform samples
    
    # Sinkhorn algorithm
    K = torch.exp(-C / reg)  # Gibbs kernel
    u = torch.ones_like(a)   # dual variable
    v = torch.ones_like(b)   # dual variable
    
    for i in range(max_iter):
        u_prev = u.clone()
        
        # Sinkhorn updates
        u = a / (K @ v + 1e-8)  # Add small epsilon for numerical stability
        v = b / (K.T @ u + 1e-8)
        
        # Check convergence
        if torch.max(torch.abs(u - u_prev)) < tol:
            break
    
    # Compute optimal transport cost
    P = torch.diag(u) @ K @ torch.diag(v)  # Optimal transport plan
    wasserstein_dist = torch.sum(P * C)
    
    return wasserstein_dist

distance = sinkhorn_wasserstein_2d(yy, bounds=(0,1))
print(f"Wasserstein distance: {distance}")

xts, delta_logps = simulator.simulate_with_likelihood(x0, ts.view(1,-1,1).expand(num_samples,-1,1))

print(f'delta_logps : {delta_logps.shape}')
print(f'xts: {xts}')
plt.figure(figsize=(12,6))

data = delta_logps[2, :].detach().cpu().numpy()
# data = np.flip(data)
time_step = np.arange(len(data))

plt.plot(time_step, data)
plt.scatter(time_step[::10], data[::10])

plt.show()
from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import torch
import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from blocks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Initialise probability path
path = GaussianConditionalProbabilityPath(
    p_data = MNISTSampler(),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = SquareRootBeta()
).to(device)

# Initialise model
unet = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
)

# Samples
num_rows = 4
num_cols = 4
num_timesteps = 100
num_samples = num_rows * num_cols


unet.load_state_dict(torch.load('trained/model2025-10-13_10-41-58.pth', map_location=device))
z, y = path.p_data.sample(num_samples)


# Initialise ODE
ode = CFGVectorFieldODE(unet)
simulator = EulerSimulator(ode)

timestep = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
x0, _ = path.p_simple.sample(num_samples)
x1 = simulator.simulate(x0,timestep, y=y)

# Create grids for x0 and x1
x0_grid = make_grid(x0, nrow=4, normalize=True, value_range=(-1, 1))
x1_grid = make_grid(x1, nrow=4, normalize=True, value_range=(-1, 1))

# Convert to numpy for plotting
x0_grid_np = x0_grid.permute(1,2,0).cpu().numpy()
x1_grid_np = x1_grid.permute(1,2,0).cpu().numpy()

# Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(x0_grid_np.squeeze(), cmap='gray')
ax1.set_title('Starting Distribution (x0)')
ax1.axis('off')

ax2.imshow(x1_grid_np.squeeze(), cmap='gray')
ax2.set_title('Final Distribution (x1)')
ax2.axis('off')

plt.tight_layout()
plt.show()

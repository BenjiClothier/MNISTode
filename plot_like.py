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


# Initialise probability path for normal data
# path = GaussianConditionalProbabilityPath(
#     p_data = FilteredMNISTSampler(excluded_digits=[2,3,4,5,6,7,8,9]),
#     p_simple_shape = [1, 32, 32],
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# ).to(device)

# Initialise probability path for abnormal data
# path = GaussianConditionalProbabilityPath(
#     p_data = FilteredMNISTSampler(excluded_digits=[0,1]),
#     p_simple_shape = [1, 32, 32],
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# ).to(device)

# Initialise probaility path for abnormal pure greyscale data
# path = GaussianConditionalProbabilityPath(
#     p_data = UniformGraySampler(),
#     p_simple_shape = [1, 32, 32],
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# ).to(device)

# Initialise probaility path for abnormal greyscale cats
path = GaussianConditionalProbabilityPath(
    p_data = FilteredMNISTSampler(excluded_digits=[2,3,4,5,6,7,8,9]),
    p_simple_shape = [1, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# Initialise model
unet = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
).to(device)

# Samples
num_rows = 8
num_cols = 8
num_timesteps = 100
num_samples = num_rows * num_cols


# trained/model2025-09-15_12-46-35.pth MNIST no anomalies
# trained/model2025-09-16_16-37-19.pth MNIST no 5's
# trained/model2025-09-29_16-18-55.pth MNIST [0,1] only
# trained/model2025-10-08_14-22-32.pth MNIST 1's only
# trained/model2025-10-09_12-01-01.pth MNIST 8's only


unet.load_state_dict(torch.load('trained/model2025-09-29_16-18-55.pth', map_location=device))
z, y = path.p_data.sample(num_samples)


# Initialise ODE
ode = LikelihoodODE(unet)
simulator = LikelihoodSimulator(ode)

dlogp_list = []

for i in range(100):
    num_samples = 1
    z, y = path.p_data.sample(num_samples)
    timestep = torch.linspace(1, 0, num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
    x0 = z
    x1, dlogp = simulator.simulate_with_likelihood_trajectory(x0, timestep, y)
    dlogp = dlogp.detach().cpu().numpy()
    dlogp = (np.flip(dlogp))
    
    # Append to list
    dlogp_list.append(dlogp)
    
    if i % 10 == 0:  # Progress indicator
        print(f"Iteration {i}/100")

# Stack all dlogp arrays
# This creates a 2D array where each row is one iteration's dlogp
dlogp_stacked = np.stack(dlogp_list, axis=0).squeeze(1)
bpd = -dlogp_stacked / np.log(2) 
N = np.prod(z.shape[1:])
bpd = (bpd / N) + 7.
os.makedirs('test', exist_ok=True)
np.save('test/norm.npy', dlogp_stacked)
np.save('test/bpd.npy', bpd)
print(f"Stacked shape: {dlogp_stacked.shape}")
print(f"Shape is (num_iterations, dlogp_length): ({dlogp_stacked.shape[0]}, {dlogp_stacked.shape[1]})")


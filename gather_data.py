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
from process_results_for_tikz import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Initialise model
unet = MNISTUNet(
    channels = [32, 64, 128],
    num_residual_layers = 2,
    t_embed_dim = 40,
    y_embed_dim = 40,
).to(device)

# Initialise ODE
ode = LikelihoodODE(unet)
simulator = LikelihoodSimulator(ode)

# Samples
num_rows = 1
num_cols = 1
num_timesteps = 100
num_samples = num_rows * num_cols


# trained/model2025-09-15_12-46-35.pth MNIST no anomalies
# trained/model2025-09-16_16-37-19.pth MNIST no 5's
# trained/model2025-09-29_16-18-55.pth MNIST [0,1] only
# trained/model2025-10-08_14-22-32.pth MNIST 1's only (probably mis-trained)
# trained/model2025-10-09_12-01-01.pth MNIST 8's only
# trained/model2025-10-08_14-22-32.pth MNIST 1's only again (garbage)
# trained/model2025-10-09_14-09-38.pth MNIST 1's only bs: 250 epoch:1000
# trained/model2025-10-09_14-21-05.pth MNIST 1's only bs:250 epoch:5000 (maybe)
# trained/model2025-10-09_14-40-23.pth MNIST 1's only bs:450 epoch: 5000 ()
# trained/model2025-10-10_10-53-06.pth MNIST 1's only bs:450 epoch: 10000 (bad)
# trained/model2025-10-13_10-34-33.pth MNIST 8's & 9's bs:250 ep:10000 (junk)
# trained/model2025-10-13_10-41-58.pth MNIST 7,8,9's bs:250 ep:10000 (GOOD)
# trained/model2025-10-13_10-51-14.pth MNIST 8's, & 9's bs:150 ep:10000 (GOOD)




excluded_digits = {j: [i for i in range(10) if i != j] for j in range(10)}
save_dir = 'vectors_MNIST_fixed_1s'
os.makedirs(save_dir, exist_ok=True)
print(f'{save_dir}')
unet.load_state_dict(torch.load('trained/model2025-10-09_14-40-23.pth', map_location=device))
for j in excluded_digits:
# Initialise probaility path for abnormal greyscale cats
    print(excluded_digits[j])
    path = GaussianConditionalProbabilityPath(
        p_data = FilteredMNISTSampler(excluded_digits=excluded_digits[j]),
        p_simple_shape = [1, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)


    z, y = path.p_data.sample(num_samples)

    samples = 100
    dlogp_list = []

    for i in range(samples):
        num_samples = 1
        z, y = path.p_data.sample(num_samples)
        timestep = torch.linspace(1, 0, num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x0 = z
        x1, dlogp = simulator.simulate_with_likelihood_trajectory(x0, timestep, y)
        dlogp = dlogp.detach().cpu().numpy()
        dlogp = -np.flip(dlogp)
        
        # Append to list
        dlogp_list.append(dlogp)
        
        if i % 10 == 0:  # Progress indicator
            print(f"Iteration {i}/{samples}")

    # Stack all dlogp arrays
    # This creates a 2D array where each row is one iteration's dlogp
    dlogp_stacked = np.stack(dlogp_list, axis=0).squeeze(1)
    np.save(f'{save_dir}/only{j}.npy', dlogp_stacked)
    print(f"Stacked shape: {dlogp_stacked.shape}")
    print(f"Shape is (num_iterations, dlogp_length): ({dlogp_stacked.shape[0]}, {dlogp_stacked.shape[1]})")
    print(f'Saved to {save_dir}')

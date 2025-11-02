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
# device = 'cpu'
# Initialise probability path
path = GaussianConditionalProbabilityPath(
    p_data = CIFAR10CatSampler(),
    p_simple_shape = [3, 32, 32],
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)


# Sample forward process
num_rows = 1
num_cols = 1
num_timesteps = 6
num_samples = num_rows * num_cols
z, _ = path.p_data.sample(num_samples)
# z = z.view(-1, 3, 32, 32)

# Save each timestep
ts = torch.linspace(0, 1, num_timesteps).to(device)
for t_idx, t in enumerate(ts):
    tt = t.view(1,1,1,1).expand(num_samples, 1, 1, 1)
    xt = path.sample_conditional_path(z, tt)
    grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(grid.permute(1, 2, 0).cpu())
    ax.axis("off")
    plt.savefig(f'1colourcat_timestep_{t_idx}.png', bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
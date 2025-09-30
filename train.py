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

# Initialise sampler
sampler = FilteredMNISTSampler(excluded_digits=5).to(device)

# Initialise probability path
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

# Initialise trainer
trainer = ConditionalLabelledFlowMatchingTrainer(model=unet, path=path)

# Train
trainer.train(num_epoch=10000, device=device, lr=1e-3, batch_size=150)


# Sample forward process
num_rows = 4
num_cols = 4
num_timesteps = 6

num_samples = num_rows * num_cols
z, _ = path.p_data.sample(num_samples)
z = z.view(-1, 1, 32, 32)

# Setup plot
fig, axes = plt.subplots(1, num_timesteps, figsize=(6 * num_cols * num_timesteps, 6 * num_rows))

# Sample from conditional probability paths and graph
ts = torch.linspace(0, 1, num_timesteps).to(device)
for t_idx, t in enumerate(ts):
    tt = t.view(1,1,1,1).expand(num_samples, 1, 1, 1)
    xt = path.sample_conditional_path(z, tt)
    grid = make_grid(xt, nrow=num_cols, normalize=True, value_range=(-1,1))
    axes[t_idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    axes[t_idx].axis("off")
plt.show()


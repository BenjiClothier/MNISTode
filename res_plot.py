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

# Load both stacks
dlogp_normal = np.load('vectors_MNIST_0_1/norm.npy')
dlogp_anomalous = np.load('vectors_MNIST_0_1/abnorm_allcats.npy')

# Calculate mean, std, and variance for normal
dlogp_normal_mean = np.mean(dlogp_normal, axis=0)
dlogp_normal_std = np.std(dlogp_normal, axis=0)
dlogp_normal_var = np.var(dlogp_normal, axis=0)

# Calculate mean, std, and variance for anomalous
dlogp_anomalous_mean = np.mean(dlogp_anomalous, axis=0)
dlogp_anomalous_std = np.std(dlogp_anomalous, axis=0)
dlogp_anomalous_var = np.var(dlogp_anomalous, axis=0)

# Calculate differences (Normal - Anomalous)
mean_diff = dlogp_normal_mean - dlogp_anomalous_mean
var_diff = dlogp_normal_var - dlogp_anomalous_var

# Create x-axis (timesteps)
x_normal = np.arange(len(dlogp_normal_mean))
x_anomalous = np.arange(len(dlogp_anomalous_mean))

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Subplot 1: Comparison of both
ax1 = axes[0]
ax1.fill_between(x_normal,
                 dlogp_normal_mean - dlogp_normal_std,
                 dlogp_normal_mean + dlogp_normal_std,
                 alpha=0.3,
                 color='blue',
                 label='Normal ±1 std')
ax1.plot(x_normal, dlogp_normal_mean, color='blue', linewidth=1, label='Normal Mean')

ax1.fill_between(x_anomalous,
                 dlogp_anomalous_mean - dlogp_anomalous_std,
                 dlogp_anomalous_mean + dlogp_anomalous_std,
                 alpha=0.3,
                 color='red',
                 label='Anomalous ±1 std')
ax1.plot(x_anomalous, dlogp_anomalous_mean, color='red', linewidth=1, linestyle='--', label='Anomalous Mean')

ax1.set_xlabel('Timestep', fontsize=12)
ax1.set_ylabel('dlogp', fontsize=12)
ax1.set_title('Comparison of Normal vs Abnormal dlogp', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Subplot 2: Difference in means
ax2 = axes[1]
ax2.plot(x_anomalous, mean_diff, color='green', linewidth=2)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax2.fill_between(x_anomalous, 0, mean_diff, alpha=0.3, color='green')
ax2.set_xlabel('Timestep', fontsize=12)
ax2.set_ylabel('Mean Difference (Normal - Abnormal)', fontsize=12)
ax2.set_title('Difference in Means', fontsize=14)
ax2.grid(True, alpha=0.3)

# Subplot 3: Difference in variances
ax3 = axes[2]
ax3.plot(x_anomalous, var_diff, color='purple', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax3.fill_between(x_anomalous, 0, var_diff, alpha=0.3, color='purple')
ax3.set_xlabel('Timestep', fontsize=12)
ax3.set_ylabel('Variance Difference (Normal - Abnormal)', fontsize=12)
ax3.set_title('Difference in Variances', fontsize=14)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some comparison statistics
print(f"Normal stack shape: {dlogp_normal.shape}")
print(f"Anomalous stack shape: {dlogp_anomalous.shape}")
print(f"\nNormal - Overall mean: {np.mean(dlogp_normal_mean):.4f}")
print(f"Anomalous - Overall mean: {np.mean(dlogp_anomalous_mean):.4f}")
print(f"Mean difference (Normal - Anomalous): {np.mean(mean_diff):.4f}")
print(f"\nNormal - Mean variance: {np.mean(dlogp_normal_var):.4f}")
print(f"Anomalous - Mean variance: {np.mean(dlogp_anomalous_var):.4f}")
print(f"Variance difference (Normal - Anomalous): {np.mean(var_diff):.4f}")
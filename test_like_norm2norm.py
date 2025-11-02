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

mean = torch.tensor([10,10], dtype=torch.float32)

path = LinearConditionalProbabilityPath(
    p_simple = Gaussian.isotropic(dim=2, std=1.0).to(device),
    p_data = Gaussian.isotropic_with_mean(mean, std=1.0).to(device)
)

# Initialise model
net = MLPVectorField(dim=2, hiddens=[64,64,64,64]).to(device)


# trained/model2025-09-15_12-46-35.pth MNIST no anomalies
# trained/model2025-09-16_16-37-19.pth MNIST no 5's
# trained/model2025-09-29_16-18-55.pth MNIST [0,1] only
# trained/model2025-10-08_14-22-32.pth MNIST 1's only
# trained/model2025-10-09_12-01-01.pth MNIST 8's only


net.load_state_dict(torch.load('trained/model2025-10-31_10-48-32.pth', map_location=device))



# Initialise ODE
ode = Likelihood2DODE(net)
simulator = HuenSimulator(ode)
num_timesteps = 100
dlogp_list = []

for i in range(100):
    num_samples = 1
    z = path.p_data.sample(num_samples)
    timestep = torch.linspace(1, 0, num_timesteps).view(1, -1, 1).expand(num_samples, -1, 1).to(device)
    x0 = z
    x1, dlogp = simulator.simulate_with_likelihood(x0, timestep)
    dlogp = dlogp.detach().cpu().numpy()
    dlogp = np.cumsum(np.flip(dlogp))
    
    # Append to list
    dlogp_list.append(dlogp)
    
    if i % 10 == 0:  # Progress indicator
        print(f"Iteration {i}/100")

# Stack all dlogp arrays
# This creates a 2D array where each row is one iteration's dlogp
dlogp_stacked = np.stack(dlogp_list, axis=0).squeeze(1)
bpd = -dlogp_stacked / np.log(2) 
N = np.prod(z.shape[1:])
bpd = (bpd / N)
os.makedirs('testing', exist_ok=True)
np.save('testing/norm.npy', dlogp_stacked)
np.save('testing/bpd.npy', bpd)
print(f"Stacked shape: {dlogp_stacked.shape}")
print(f"Shape is (num_iterations, dlogp_length): ({dlogp_stacked.shape[0]}, {dlogp_stacked.shape[1]})")


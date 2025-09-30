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

# Path
path = GaussianConditionalProbabilityPath(
    p_data = Beta1D(device=device,
                    start_point=[0,0],
                    end_point=[1,0],
                    alpha=0.5,
                    beta=0.5),
    p_simple_shape=2,
    alpha = LinearAlpha(),
    beta = LinearBeta()
)

# Model
linear_flow_model = MLPVectorField(dim=2, hiddens=[64,64,64,64]).to(device)

# Trainer
trainer = ConditionalFlowMatchingTrainer(path, linear_flow_model)

# Train
trainer.train(num_epoch=10000, device=device, lr=1e-3, batch_size=2000)


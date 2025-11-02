import torch.nn as nn
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import abstracts as ab
from blocks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialise sampler
sampler = UniformSampler(device)

# # Initialise probability path
# path = GaussianConditionalProbabilityPath(
#     p_data = UniformSampler(device),
#     p_simple_shape=2,
#     alpha = LinearAlpha(),
#     beta = LinearBeta()
# )

mean = torch.tensor([10,10], dtype=torch.float32)
path = LinearConditionalProbabilityPath(
    p_simple = Gaussian.isotropic(dim=2, std=1.0).to(device),
    p_data = Gaussian.isotropic_with_mean(mean, std=1.0).to(device)
)

# Initialise model
linear_flow_model = MLPVectorField(dim=2, hiddens=[64,64,64,64]).to(device)

# Initialise trainer
trainer = ConditionalFlowMatchingTrainer(path, linear_flow_model)

# Train
trainer.train(num_epoch=10000, device=device, lr=1e-3, batch_size=2000)
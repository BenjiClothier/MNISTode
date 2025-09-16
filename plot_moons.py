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
model.load_state_dict(torch.load('trained/model2025-09-08_14-25-29.pth')) 

# Initialise probability path
path = GaussianConditionalProbabilityPath(
    p_data = MoonSampler(device),
    p_simple_shape=[2],
    alpha = LinearAlpha(),
    beta = LinearBeta()
)


num_samples = 5000
num_marginals = 10
num_timesteps = 100

ode = MoonODE(model)
simulator = HuenSimulator(ode)


fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 2))
axes = axes.reshape(2, num_marginals)
scale = 6.0


z = path.p_data.sample(num_samples)
ts = torch.linspace(1,0,100).to(device)
record_every_idxs = pf.record_every(len(ts), len(ts) // (num_marginals - 1))
x0, _ = path.p_simple.sample(num_samples)
x0 = z.to(device)
xts = simulator.simulate_with_trajectory(x0, ts.view(1,-1,1).expand(num_samples,-1,1))
xts = xts[:,record_every_idxs,:]
for idx in range(xts.shape[1]):
    xx = xts[:,idx,:]
    pf.hist2d_samples(samples=xx.cpu(), ax=axes[1, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
    axes[1, idx].set_xlim(-scale, scale)
    axes[1, idx].set_ylim(-scale, scale)
    axes[1, idx].set_xticks([])
    axes[1, idx].set_yticks([])
    tt = ts[record_every_idxs[idx]]
    axes[1, idx].set_title(f'$t={tt.item():.2f}$', fontsize=15)
axes[1, 0].set_ylabel("Learned", fontsize=20) 

plt.show()

xts, delta_logps = simulator.simulate_with_likelihood(x0, ts.view(1,-1,1).expand(num_samples,-1,1))

print(f'delta_logps : {delta_logps.shape}')

plt.figure(figsize=(12,6))

data = delta_logps[2, :].detach().cpu().numpy()
data = np.flip(data)
time_step = np.arange(len(data))

plt.plot(time_step, data)
plt.scatter(time_step[::10], data[::10])

plt.show()

import numpy as np
from matplotlib import pyplot as plt
import torch
import plotting_fun as pf

from blocks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Path
path = GaussianConditionalProbabilityPath(
    p_data = Beta1D(
        device=device,
        start_point=[0,0],
        end_point=[1,0],
        alpha=0.5,
        beta=0.5
    ),
    p_simple_shape=[2],
    alpha=LinearAlpha(),
    beta=LinearBeta()
)

model = MLPVectorField(dim=2, hiddens=[64,64,64,64]).to(device)
model.load_state_dict(torch.load('trained/model2025-09-17_14-12-10.pth'))

ode = MoonODE(model)
simulator = HuenSimulator(ode)

pf.plot_trajectory_evolution_3d(simulator, path, device, point_size=0.25)
pf.plot_trajectory_lines_3d(simulator, path, device, line_width=0.2, scatter_size=0.25)

num_timesteps = 100
num_samples = 1
z = path.p_data.sample(num_samples)
timestep = torch.linspace(1,0,num_timesteps).view(1, -1, 1).expand(num_samples, -1, 1).to(device)
x0 = z
x1, dlogp = simulator.simulate_with_likelihood(x0, timestep)
print(f'dlogp: {dlogp.shape}')
print(f'{dlogp}')

dlogp = dlogp.detach().cpu().numpy()
dlogp = -np.cumsum(np.flip(dlogp))

plt.plot(dlogp)
plt.title("Log p(x) trajectory: 2d Gaussian --> Beta Distribution Line Segment")
plt.xlabel("Timesteps")
plt.ylabel("Log p(x)")
plt.show()

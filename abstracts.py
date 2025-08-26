from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict

import os
from datetime import datetime
import numpy as np
import torch
from torch.func import vmap, jacrev
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sampleable(ABC):
    """
    Distribrution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the number of samples needed
        Returns:
            - samples: shape(batch_size, dims)
            - lables: shape(batch_size, label_dims)
        """
        pass

class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        # Check alpha_t(1) = 1
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1)
        )
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)

class Beta(ABC):
    def __init__(self):    
        # Check beta_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        # Check beta_t(1) = 1
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )
    
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)

class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract bass class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
    
    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        # Sample conditional variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (num_samples, c, h, w)
        # Sample condition probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_sample, c, h, w)
        return x
    
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dims)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h , w)
        """
        pass

    @abstractmethod    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditional variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditional variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """
        pass

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            -xt state at time t, shape(batch_size, c, h, w)
            -t: time, shape(batch_size, 1)
        Returns:
            - drift_coefficient: shape(batch_size, c, h, w)
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the SDE.
        Args:
            -xt state at time t, shape(batch_size, c, h, w)
            -t: time, shape(batch_size, 1)
        Returns:
            - drift_coefficient: shape(batch_size, c, h, w)
        """
        pass

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the SDE.
        Args:
            -xt state at time t, shape(batch_size, c, h, w)
            -t: time, shape(batch_size, 1)
        Returns:
            - diffusion_coefficient: shape(batch_size, c, h, w)
        """
        pass

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Takes one simulation step
        Args:
            -xt: state at time t, shape(batch_size, c, h, w)
            -t: time, shape(batch_size, 1, 1, 1)
            -dt: time, shape(batch_size, 1, 1, 1)
        Returns:
            -nxt: state at time t + dt, shape(batch_size, 1, 1, 1)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretisation given by ts
        Args:
            - x_init: initial state, shape(batch_size, c, h, w)
            - ts: timesteps, shape(batch_size, num_timesteps, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape(batch_size, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Simulates using the discretisation given by ts
        Args:
            - x_init: initial state, shape(batch_size, c, h, w)
            - ts: timesteps, shape(batch_size, num_timesteps, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape(batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
    
class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    
    def get_optimiser(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train(self, num_epoch: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        print(f'Training model....')

        # Start
        self.model.to(device)
        opt = self.get_optimiser(lr)
        self.model.train()

        # Loop
        pbar = tqdm(enumerate(range(num_epoch)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
        
        # Save model
        os.makedirs('trained', exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'model{stamp}.pth'
        filepath= os.path.join('trained', filename)

        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to: {filepath}")

        # Finish
        self.model.eval()

class ConditionalVectorField(nn.Module, ABC):
    """
    Parameterisation of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        Args:
            - x: (batch_size, c, h ,w)
            - t: (batch_size, c, h, w)
        Returns:
            - u_t^theta(x|y): (batch_size, c, h, w)
        """
        pass

class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0):
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1))
    
    def sample(self, num_samples) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None





    
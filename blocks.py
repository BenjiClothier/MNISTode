

from typing import Optional, List, Type, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import abstracts as ab
from unet import FourierEncoder, Midcoder, Encoder, Decoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LinearAlpha(ab.Alpha):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
    
class LinearBeta(ab.Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
    
class SquareRootBeta(ab.Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - t)
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 / (torch.sqrt(1 - t) + 1e-4)
    
class GaussianConditionalProbabilityPath(ab.ConditionalProbabilityPath):
    def __init__(self, p_data: ab.Sampleable, p_simple_shape: List[int], alpha: ab.Alpha, beta: ab.Beta):
        p_simple = ab.IsotropicGaussian(shape = p_simple_shape, std = 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z and label y
        Args:
            - num_samples: the number of samples
        Returns:
            - z: (num_samples, c, h, w)
            - y: (num_samples, label_dim)
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, c, h, w)
        """
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
    
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, c, h, w)
            - z: conditioning variable (num_samples, c, h, w)
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - conditional_score: conditional score (num_samples, c, h, w)
        """ 
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2

class MNISTSampler(nn.Module, ab.Sampleable):

    """
    Sampleable wrapper for the MNIST dataset
    """
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels  

class MNISTUNet(ab.ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int): 
        super().__init__()
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        self.init_conv = nn.Sequential(nn.Conv2d(1, channels[0], kernel_size=3, padding=1), nn.BatchNorm2d(channels[0]), nn.SiLU())

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y embedder
        self.y_embedder = nn.Embedding(num_embeddings = 11, embedding_dim = y_embed_dim)

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for (curr_c, next_c) in zip(channels[:-1], channels[1:]):
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim)
            
        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, 1, 32, 32)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, 1, 32, 32)
        """
        # Embed t and y
        t_embed = self.time_embedder(t) # (bs, time_embed_dim)
        y_embed = self.y_embedder(y) # (bs, y_embed_dim)
        
        # Initial convolution
        x = self.init_conv(x) # (bs, c_0, 32, 32)

        residuals = []
        
        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            residuals.append(x.clone())

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders
        for decoder in self.decoders:
            res = residuals.pop() # (bs, c_i, h, w)
            x = x + res
            x = decoder(x, t_embed, y_embed) # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        x = self.final_conv(x) # (bs, 1, 32, 32)

        return x   

class ConditionalFlowMatchingTrainer(ab.Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: MNISTUNet, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.model = model

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z_data, z_labels = self.path.p_data.sample(batch_size)
        z_data = z_data.to(device)
        z_labels = z_labels.to(device)
        t = torch.rand(batch_size, 1, 1, 1).to(device)
        x = self.path.sample_conditional_path(z_data, t)
        target = self.path.conditional_vector_field(x, z_data, t)
        output = self.model(x, t, z_labels)
        
        return torch.nn.functional.mse_loss(output, target)        
    
class VectorFieldODE(ab.ODE):
    def __init__(self, net: MNISTUNet):
        super().__init__()
        self.net = net
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # batch_size = xt.shape[0]
        # z = self.z.expand(batch_size, *self.z.shape[1:])
        # return self.path.conditional_vector_field(xt, z, t)
        return self.net(xt, t, y)

class EulerSimulator(ab.Simulator):
    def __init__(self, ode: VectorFieldODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h



    
        

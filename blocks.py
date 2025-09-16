

from typing import Optional, List, Type, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.datasets import make_moons, make_circles
from tqdm import tqdm
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import abstracts as ab
from unet import FourierEncoder, Midcoder, Encoder, Decoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#-----------Noise Schedule---------#

class LinearAlpha(ab.Alpha):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)
    
class LinearBeta(ab.Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return - torch.ones_like(t)
    
class SquareRootBeta(ab.Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - t)
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 / (torch.sqrt(1 - t) + 1e-4)
    
#-----------Path-------------------#

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

class LinearConditionalProbabilityPath(ab.ConditionalProbabilityPath):

    def __init__(self, p_simple: ab.Sampleable, p_data: ab.Sampleable):
        super().__init__(p_simple, p_data)

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z ~ p_data(x)
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, ...)
        """
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples the random variable X_t = (1-t) X_0 + tz
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        return (1 - t) * self.p_simple.sample(z.shape[0]) + t * z
        
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z) = (z - x) / (1 - t)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        return (z - x) / (1 - t)

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Not known for Linear Conditional Probability Paths
        """ 
        raise Exception("You should not be calling this function!")  


#----------Data Samplers-----------#

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

class MoonSampler(nn.Module, ab.Sampleable):
    """
    Implementation of the Moons distribution using sklearn's make_moons
    """
    def __init__(self, device: torch.device, noise: float = 0.05, scale: float = 5.0, offset: Optional[torch.Tensor] = None):
        """
        Args:
            noise: Standard deviation of Gaussian noise added to the data
            scale: How much to scale the data
            offset: How much to shift the samples from the original distribution (2,)
        """
        self.noise = noise
        self.scale = scale
        self.device = device
        if offset is None:
            offset = torch.zeros(2)
        self.offset = offset.to(device)

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            num_samples: Number of samples to generate
        Returns:
            torch.Tensor: Generated samples with shape (num_samples, 3)
        """
        samples, _ = make_moons(
            n_samples=num_samples,
            noise=self.noise,
            random_state=None  # Allow for random generation each time
        )
        return self.scale * torch.from_numpy(samples.astype(np.float32)).to(self.device) + self.offset

class UniformSampler(ab.Sampleable):
    def __init__(self, device, dim=2):
        self.device = device
        self.dim = dim

    def sample(self, num_samples):
        ref = torch.zeros(num_samples, self.dim, device=self.device)
        samples = torch.rand_like(ref)
        return samples

class FilteredMNISTSampler(nn.Module, ab.Sampleable):
    """
    Sampleable wrapper for the MNIST dataset that excludes specified digits
    """
    def __init__(self, excluded_digits=None):
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
        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called...
        
        # Handle excluded digits
        if excluded_digits is None:
            excluded_digits = []
        elif isinstance(excluded_digits, int):
            excluded_digits = [excluded_digits]
        elif not isinstance(excluded_digits, (list, tuple, set)):
            raise ValueError("excluded_digits must be int, list, tuple, or set")
        
        self.excluded_digits = set(excluded_digits)
        
        # Pre-filter the dataset to get valid indices
        self.valid_indices = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if label not in self.excluded_digits:
                self.valid_indices.append(i)
        
        print(f"Original dataset size: {len(self.dataset)}")
        print(f"Excluded digits: {sorted(self.excluded_digits) if self.excluded_digits else 'None'}")
        print(f"Filtered dataset size: {len(self.valid_indices)}")
        
        if len(self.valid_indices) == 0:
            raise ValueError("No samples remaining after filtering!")
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
        - num_samples: the desired number of samples
        Returns:
        - samples: shape (batch_size, c, h, w)
        - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.valid_indices):
            raise ValueError(f"num_samples ({num_samples}) exceeds filtered dataset size: {len(self.valid_indices)}")
        
        # Sample from valid indices only
        selected_valid_indices = torch.randperm(len(self.valid_indices))[:num_samples]
        actual_indices = [self.valid_indices[i] for i in selected_valid_indices]
        
        samples, labels = zip(*[self.dataset[i] for i in actual_indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        
        return samples, labels
    
    def get_available_labels(self):
        """Returns the set of labels that are available (not excluded)"""
        all_labels = set(range(10))  # MNIST has digits 0-9
        return sorted(all_labels - self.excluded_digits)
    
    def get_excluded_labels(self):
        """Returns the set of excluded labels"""
        return sorted(self.excluded_digits)    

#---------Models-------------------#

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

class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = self.build_mlp([dim + 1] + hiddens + [dim])
    
    def build_mlp(self, dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        # print(f'x : {x.device}')
        # print(f't : {t.device}')
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt) 


#----------Trainers----------------#

class ConditionalLabelledFlowMatchingTrainer(ab.Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: MNISTUNet,
                 **kwargs):
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
        loss = torch.nn.functional.mse_loss(output, target)        

        return loss
    
class CFGTrainer(ab.Trainer):

    def __init__(self, path: GaussianConditionalProbabilityPath, model: ab.ConditionalVectorField, eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size)        
        z = z.to(device)
        y = y.to(device)
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        y = torch.randint(0, 9, (batch_size,))
        y1 = y.clone()
        null_mask = torch.rand(batch_size) < self.eta
        y = torch.where(null_mask, torch.tensor(10), y1)
                
        # Step 3: Sample t and x
        t = torch.rand(batch_size, 1, 1, 1)
        t = t.to(device)
        x = self.path.sample_conditional_path(z, t)
        x = x.to(device)
        y = y.to(device)
        # Step 4: Regress and output loss
        target = self.path.conditional_vector_field(x, z, t)
        output = self.model(x, t, y)

        return torch.nn.functional.mse_loss(output, target)

class EDMTrainer(ab.Trainer):

    def __init__(self, path: GaussianConditionalProbabilityPath, model: ab.ConditionalVectorField, data_std: float = 0.5, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.data_std = data_std
    
    def skip_scaling(self):
        return self.data_std**2 / (self.path.beta**2 + self.data_std)
    
    def output_scaling(self):
        return (self.path.beta * self.data_std) / (torch.sqrt(self.data_std**2 + self.path.beta**2))
    
    def input_scaling(self):
        return 1 / (torch.sqrt(self.data_std**2 + self.path.beta**2))
    
    def noise_cond(self):
        return 0.25 * torch.log(self.path.beta)
    
    def get_train_loss(self, **kwargs):
        pass

class ConditionalFlowMatchingTrainer(ab.Trainer):
    def __init__(self, path: ab.ConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size).to(device)
        t = torch.rand(batch_size, 1).to(device)
        x = self.path.sample_conditional_path(z, t)
        target = self.path.conditional_vector_field(x, z, t)
        output = self.model(x, t)
        
        return torch.nn.functional.mse_loss(output, target)    

#-----------ODEs------------------#

class VectorFieldODE(ab.ODE):
    def __init__(self, net: MNISTUNet):
        super().__init__()
        self.net = net
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # batch_size = xt.shape[0]
        # z = self.z.expand(batch_size, *self.z.shape[1:])
        # return self.path.conditional_vector_field(xt, z, t)
        return self.net(xt, t, y)

class CFGVectorFieldODE(ab.ODE):
    def __init__(self, net: ab.ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.ones_like(y) * 10
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field

class LikelihoodODE(ab.ODE):
    def __init__(self, net):
        self.net = net
    
    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)
       
        # y = torch.ones_like(y, dtype=torch.int64) * 10
        return self.net(x, t, y)
    
        
    # def compute_div(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, create_graph=True):
    #     x = x.to(device)
    #     t = t.to(device)
    #     y = y.to(device)
        
    #     original_shape = x.shape
    #     batch_size = x.shape[0]
        
    #     # Make the original x require gradients
    #     x = x.requires_grad_(True)
        
    #     # Get vector field from U-Net (using original shape)
    #     vector_field = self.drift_coefficient(x, t, y)
        
    #     # Flatten for divergence computation
    #     x_flat = x.flatten(1)  # Don't need requires_grad here since x already has it
    #     vec_flat = vector_field.flatten(1)
    #     x_dim = x_flat.shape[1]
        
    #     div = torch.zeros(batch_size, device=x.device)
        
    #     for i in range(x_dim):
    #         v_i = vec_flat[:, i]
    #         grad_outs = torch.ones_like(v_i)
    #         grad_v_i = torch.autograd.grad(
    #             outputs=v_i,
    #             inputs=x_flat,  # This now has gradients through x
    #             grad_outputs=grad_outs,
    #             create_graph=create_graph,
    #             retain_graph=True
    #         )[0]
            
    #         if grad_v_i is not None:
    #             div += grad_v_i[:, i]
        
    #     return div

    # def compute_div(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, create_graph=True):
    #     x = x.to(device)
    #     t = t.to(device)
    #     y = y.to(device)
        
    #     batch_size = x.shape[0]
    #     x = x.requires_grad_(True)
        
    #     # Get vector field
    #     vector_field = self.drift_coefficient(x, t, y)
        
    #     # Flatten
    #     x_flat = x.flatten(1).requires_grad_(True)
    #     vec_flat = vector_field.flatten(1)
    #     x_dim = vec_flat.shape[1]
        
    #     # Method 1: Compute all gradients at once (most efficient)
    #     div = torch.zeros(batch_size, device=x.device)
        
    #     # Create identity matrix to get gradients for all dimensions
    #     eye = torch.eye(x_dim, device=x.device, dtype=vec_flat.dtype)
        
    #     for i in range(x_dim):
    #         # Use the i-th unit vector to get gradient of i-th component
    #         grad_outputs = eye[i:i+1].expand(batch_size, -1)
            
    #         grad_v = torch.autograd.grad(
    #             outputs=vec_flat,
    #             inputs=x_flat,
    #             grad_outputs=grad_outputs,
    #             create_graph=create_graph,
    #             retain_graph=True
    #         )[0]
            
    #         if grad_v is not None:
    #             div += grad_v[:, i]
        
    #     return div

    def compute_div(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, create_graph=True):
        """
        Compute divergence using Hutchinson's trace estimator
        More efficient for high-dimensional problems
        """
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)
        
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # Flatten and require gradients
        x_flat = x.flatten(1).requires_grad_(True)
        x_dim = x_flat.shape[1]
        
        # Reshape for U-Net
        x_reshaped = x_flat.view(original_shape)
        
        # Get vector field
        vector_field = self.drift_coefficient(x_reshaped, t, y)
        vec_flat = vector_field.flatten(1)
        
        # Use random vectors for Hutchinson's estimator (more efficient than exact trace)
        num_hutchinson_samples = min(10, x_dim)  # Use fewer samples for efficiency
        div_estimates = []
        
        for _ in range(num_hutchinson_samples):
            # Random Rademacher vector (±1)
            epsilon = torch.randint_like(vec_flat, 0, 2).float() * 2 - 1
            
            # Compute epsilon^T * vec_flat
            epsilon_dot_v = (epsilon * vec_flat).sum()
            
            # Compute gradient
            grad_v = torch.autograd.grad(
                outputs=epsilon_dot_v,
                inputs=x_flat,
                create_graph=create_graph,
                retain_graph=True
            )[0]
            
            # Compute epsilon^T * grad_v (approximates trace)
            trace_estimate = (epsilon * grad_v).sum(dim=1)
            div_estimates.append(trace_estimate)
        
        # Average over Hutchinson samples
        div = torch.stack(div_estimates).mean(dim=0)
        return div
    
    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

class MoonODE(ab.ODE):
    def __init__(self, net: MNISTUNet):
        super().__init__()
        self.net = net
    
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        # batch_size = xt.shape[0]
        # z = self.z.expand(batch_size, *self.z.shape[1:])
        # return self.path.conditional_vector_field(xt, z, t)
        return self.net(xt, t)  
    
    def compute_div(self, x: torch.Tensor, t: torch.Tensor, create_graph=True):
        x = x.to(device)
        t = t.to(device)
        batch_size = x.shape[0]
        x_dim = x.shape[1]

        x = x.requires_grad_(True)

        vector_field = self.drift_coefficient(x, t)

        div = torch.zeros(batch_size, device=x.device)

        for i in range(x_dim):
            v_i = vector_field[:, i]

            grad_outs = torch.ones_like(v_i)
            grad_v_i = torch.autograd.grad(
                outputs=v_i,
                inputs = x,
                grad_outputs = grad_outs,
                create_graph=create_graph,
                retain_graph=True
            )[0]

            if grad_v_i is not None:
                div += grad_v_i[:, i]
    
        return div
    
    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1)) / 2.
        return logps
       
#---------Simulators---------------#

class EulerSimulator(ab.Simulator):
    def __init__(self, ode: VectorFieldODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

class LikelihoodSimulator(ab.Simulator):
    def __init__(self, ode: LikelihoodODE):
        self.ode = ode

    def step(self, xt, t, dt, y, **kwargs):
        k1 = self.ode.drift_coefficient(xt, t, y)
        x_pred = xt + dt * k1
        k2 = self.ode.drift_coefficient(x_pred, t + dt, y)
        nxt = xt + 0.5 * dt * (k1 + k2)
        return nxt
    
    def logp_grad(self, x, t, y, **kwargs):
        return self.ode.compute_div(x, t, y)

    def simulate_with_likelihood_trajectory(self, x: torch.Tensor, ts: torch.Tensor, y: torch.Tensor, **kwargs):
        xs = [x.clone()]
        delta_logps = []
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, y, **kwargs)
            xlog = x.clone()
            delta_logp = self.ode.compute_div(xlog, t, y)
            xs.append(x.clone())
            delta_logps.append(delta_logp.clone())
        prior_logp = self.ode.prior_logp(x)
        delta_logps.append(prior_logp.clone())
        xs = torch.stack(xs, dim=1)
        delta_logps = torch.stack(delta_logps, dim=1)
        return xs, delta_logps

class HuenSimulator(ab.Simulator):
    def __init__(self, ode: ab.ODE):
        self.ode = ode

    def step(self, xt, t, dt, **kwargs):
        k1 = self.ode.drift_coefficient(xt, t)
        x_pred = xt + dt * k1
        k2 = self.ode.drift_coefficient(x_pred, t + dt)
        nxt = xt + 0.5 * dt * (k1 + k2)
        return nxt 
    
    def simulate_with_likelihood(self, xt, ts, **kwargs):
        xs = [xt.clone()]
        delta_logps = []
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            xt = self.step(xt, t, h, **kwargs)
            delta_logp = self.ode.compute_div(xt, t)
            xs.append(xt.clone())
            delta_logps.append(delta_logp.clone())
        prior_logp = self.ode.prior_logp(xt)
        delta_logps.append(prior_logp.clone())
        xs = torch.stack(xs, dim=1)
        delta_logps = torch.stack(delta_logps, dim=1)
        return xs, delta_logps

class HuenLabelSimulator(ab.Simulator):
    def __init__(self, ode: ab.ODE):
        self.ode = ode
    
    def step(self, xt, t, dt, y, **kwargs):
        k1 = self.ode.drift_coefficient(xt, t, y)
        x_pred = xt + dt * k1
        k2 = self.ode.drift_coefficient(x_pred, t + dt, y)
        nxt = xt + 0.5 * dt * (k1 + k2)
        return nxt

#-------------Utils-----------------# 

class Gaussian(torch.nn.Module, ab.Sampleable, ab.Density):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))
        
    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)







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
from PIL import Image
import random

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

class PNGSampler32x32(nn.Module, ab.Sampleable):
    """
    Sampler that loads PNG images from a folder and samples them randomly.
    Images are resized to 32x32 and converted to tensors.
    """
    
    def __init__(self, 
                 folder_path: str, 
                 normalize: bool = True,
                 to_rgb: bool = True,
                 cache_images: bool = True):
        """
        Args:
            folder_path: Path to folder containing PNG images
            normalize: Whether to normalize images to [0, 1] range
            to_rgb: Whether to convert images to RGB (3 channels)
            cache_images: Whether to cache loaded images in memory for faster sampling
        """
        super().__init__()
        
        self.folder_path = folder_path
        self.cache_images = cache_images
        self.normalize = normalize
        self.to_rgb = to_rgb
        
        # Set up image transforms
        transform_list = [transforms.Resize((32, 32))]
        if to_rgb:
            transform_list.append(transforms.Lambda(lambda img: img.convert('RGB')))
        transform_list.append(transforms.ToTensor())
        if not normalize:
            # If not normalizing, convert back to [0, 255] range
            transform_list.append(transforms.Lambda(lambda x: x * 255.0))
            
        self.transform = transforms.Compose(transform_list)
        
        # Find all PNG files
        self.png_files = self._find_png_files()
        if len(self.png_files) == 0:
            raise ValueError(f"No PNG files found in {folder_path}")
        
        # Cache images if requested
        self.cached_images = None
        if cache_images:
            self._cache_images()
    
    def _find_png_files(self) -> List[str]:
        """Find all PNG files in the folder."""
        png_files = []
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith('.png'):
                png_files.append(os.path.join(self.folder_path, filename))
        return png_files
    
    def _cache_images(self):
        """Load and cache all images in memory."""
        print(f"Caching {len(self.png_files)} images...")
        self.cached_images = []
        for img_path in self.png_files:
            try:
                img = Image.open(img_path)
                tensor_img = self.transform(img)
                self.cached_images.append(tensor_img)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
        print(f"Successfully cached {len(self.cached_images)} images")
    
    def _load_image(self, img_path: str) -> torch.Tensor:
        """Load and transform a single image."""
        img = Image.open(img_path)
        return self.transform(img)
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample random images from the PNG folder.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            samples: Tensor of shape (num_samples, channels, 32, 32)
            labels: None (no labels available for this sampler)
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        # Sample random indices
        if self.cache_images and self.cached_images:
            # Sample from cached images
            indices = random.choices(range(len(self.cached_images)), k=num_samples)
            samples = torch.stack([self.cached_images[i] for i in indices])
        else:
            # Load images on-demand
            sampled_paths = random.choices(self.png_files, k=num_samples)
            samples = []
            for img_path in sampled_paths:
                try:
                    tensor_img = self._load_image(img_path)
                    samples.append(tensor_img)
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    # Use a zero tensor as fallback
                    channels = 3 if self.to_rgb else 1
                    fallback = torch.zeros(channels, 32, 32)
                    samples.append(fallback)
            samples = torch.stack(samples)
        
        # No labels available
        labels = None
        
        return samples, labels
    
    def __len__(self) -> int:
        """Return the number of available images."""
        return len(self.cached_images) if self.cache_images and self.cached_images else len(self.png_files)
    
    def get_sample_shape(self) -> Tuple[int, int, int]:
        """Return the shape of a single sample (channels, height, width)."""
        channels = 3 if self.to_rgb else 1
        return (channels, 32, 32)

class Beta1D(ab.Sampleable):

    def __init__(self, device, start_point, end_point, alpha=2.0, beta=2.0):
        """
        Args:
            start_point: (x, y) coordinates of line start
            end_point: (x, y) coordinates of line end  
            alpha, beta: Beta distribution parameters
        """
        self.device = device
        self.start = torch.tensor(start_point, dtype=torch.float32).to(self.device)
        self.end = torch.tensor(end_point, dtype=torch.float32).to(self.device)
        self.vector = self.end - self.start
        self.beta_dist = D.Beta(alpha, beta)
        

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Sample t parameters from Beta distribution
        t = self.beta_dist.sample((num_samples,)).to(device)
        
        # Convert to 2D points: start + t * (end - start)
        points = (self.start.unsqueeze(0) + t.unsqueeze(1) * self.vector.unsqueeze(0)).to(device)
        # Return samples
        return points

class UniformGraySampler(ab.Sampleable):
    """
    Sampler that generates 32x32 grayscale images where all pixels
    have the same value. Returns discrete gray level indices as labels.
    """
    
    def __init__(self, num_gray_levels: int = 10):
        """
        Args:
            num_gray_levels: number of discrete gray levels (must match embedding size!)
        """
        self.num_gray_levels = num_gray_levels
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_samples: the number of samples needed
        Returns:
            samples: shape (batch_size, 1, 32, 32) - grayscale images with channel dim
            labels: shape (batch_size,) - integer gray level indices in [0, num_gray_levels-1]
        """
        # Sample random gray level indices
        gray_indices = torch.randint(0, self.num_gray_levels, (num_samples,))
        
        # Convert indices to normalized gray values [0, 1]
        gray_values = gray_indices.float().to(device) / (self.num_gray_levels - 1)
        
        # Create 32x32 images with channel dimension: (batch_size, 1, 32, 32)
        samples = gray_values.view(num_samples, 1, 1, 1).expand(num_samples, 1, 32, 32).clone().to(device)
        
        # Return integer labels for the embedding layer
        return samples, gray_indices

class CIFAR10CatGrayscaleSampler(ab.Sampleable):
    """
    Sampler that returns grayscale CIFAR-10 cat images.
    CIFAR-10 cats are class index 3.
    """
    
    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        """
        Args:
            root: directory to store/load CIFAR-10 data
            train: if True, use training set; if False, use test set
            download: if True, download CIFAR-10 if not present
        """
        # Load CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll handle transforms manually
        )
        
        # CIFAR-10 class index for cats
        self.cat_class_idx = 3
        
        # Filter for only cat images
        self.cat_indices = [i for i, (_, label) in enumerate(self.dataset) if label == self.cat_class_idx]
        
        print(f"Found {len(self.cat_indices)} cat images in CIFAR-10 {'train' if train else 'test'} set")
        
        if len(self.cat_indices) == 0:
            raise ValueError("No cat images found in CIFAR-10 dataset!")
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_samples: the number of samples needed
        Returns:
            samples: shape (batch_size, 1, 32, 32) - grayscale cat images
            labels: shape (batch_size,) - all will be 3 (cat class index)
        """
        # Randomly sample cat image indices
        sampled_indices = np.random.choice(self.cat_indices, size=num_samples, replace=True)
        
        batch_images = []
        for idx in sampled_indices:
            # Get image (PIL Image, RGB)
            img, _ = self.dataset[idx]
            
            # Convert PIL Image to tensor and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0  # (32, 32, 3)
            
            # Convert RGB to grayscale using standard weights
            # Grayscale = 0.299*R + 0.587*G + 0.114*B
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            
            batch_images.append(gray)
        
        # Stack into batch and add channel dimension: (batch_size, 1, 32, 32)
        samples = torch.FloatTensor(np.stack(batch_images)).unsqueeze(1)
        
        # All labels are 3 (cat class)
        labels = torch.full((num_samples,), self.cat_class_idx, dtype=torch.long)
        
        return samples.to(device), labels.to(device)

class CIFAR10CatSampler(ab.Sampleable):
    """
    Sampler that returns RGB CIFAR-10 cat images.
    CIFAR-10 cats are class index 3.
    """
    
    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        """
        Args:
            root: directory to store/load CIFAR-10 data
            train: if True, use training set; if False, use test set
            download: if True, download CIFAR-10 if not present
        """
        # Load CIFAR-10 dataset
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll handle transforms manually
        )
        
        # CIFAR-10 class index for cats
        self.cat_class_idx = 3
        
        # Filter for only cat images
        self.cat_indices = [i for i, (_, label) in enumerate(self.dataset) if label == self.cat_class_idx]
        
        print(f"Found {len(self.cat_indices)} cat images in CIFAR-10 {'train' if train else 'test'} set")
        
        if len(self.cat_indices) == 0:
            raise ValueError("No cat images found in CIFAR-10 dataset!")
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_samples: the number of samples needed
        Returns:
            samples: shape (batch_size, 3, 32, 32) - RGB cat images
            labels: shape (batch_size,) - all will be 3 (cat class index)
        """
        # Randomly sample cat image indices
        sampled_indices = np.random.choice(self.cat_indices, size=num_samples, replace=True)
        
        batch_images = []
        for idx in sampled_indices:
            # Get image (PIL Image, RGB)
            img, _ = self.dataset[idx]
            
            # Convert PIL Image to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0  # (32, 32, 3)
            
            # Transpose from (H, W, C) to (C, H, W) for PyTorch
            img_array = np.transpose(img_array, (2, 0, 1))  # (3, 32, 32)
            
            batch_images.append(img_array)
        
        # Stack into batch: (batch_size, 3, 32, 32)
        samples = torch.FloatTensor(np.stack(batch_images))
        
        # All labels are 3 (cat class)
        labels = torch.full((num_samples,), self.cat_class_idx, dtype=torch.long)
        
        return samples.to(device), labels.to(device)
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
       
        return self.net(x, t, y)
    
        
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

class subVPSDE(ab.ODE):
    def __init__(self, config):
        self.beta_0 = config['beta_min']
        self.beta_T = config['beta_max']
        self.N = config['timesteps']
        self.ODE = config['ODE']

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_T - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(-2.0 * self.beta_0 * t - (self.beta_T - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = (-0.25 * t ** 2 * (self.beta_T - self.beta_0)
                            -0.5 * t * self.beta_0)
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1.0 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse_sde(self, score_fn, x, t):
        drift, diffusion = self.sde(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.ODE else 1.)
        diffusion = 0 if self.ODE else diffusion
        return drift, diffusion

    def reverse_discretize(self, score_fn, x, t):
        f, G = self.discretize(x, t)
        rev_f = f - G[: None, None, None] ** 2 * 2 * score_fn(x, t) * (0.5 if self.ODE else 1.)
        rev_G = torch.zero_like(G) if self.ODE else G
        return rev_f, rev_G

    def fwd_ODE(self, score_fn, x, t):
        drift, diffusion = self.sde(x, t)
        score = score_fn(x, t)
        drift = -drift + diffusion[:, None, None, None] ** 2 * score * 0.5
        return drift, 0

class Likelihood2DODE(ab.ODE):
    def __init__(self, net):
        self.net = net
    
    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        """
        x = x.to(device)
        t = t.to(device)

       
        return self.net(x, t)
    
        
    def compute_div(self, x: torch.Tensor, t: torch.Tensor, create_graph=True):
        """
        Compute divergence using Hutchinson's trace estimator
        More efficient for high-dimensional problems
        """
        x = x.to(device)
        t = t.to(device)

        
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # Flatten and require gradients
        x_flat = x.flatten(1).requires_grad_(True)
        x_dim = x_flat.shape[1]
        
        # Reshape for U-Net
        x_reshaped = x_flat.view(original_shape)
        
        # Get vector field
        vector_field = self.drift_coefficient(x_reshaped, t)
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
            # print(f'Delta log p: {delta_logp}')
            xs.append(x.clone())
            delta_logps.append(delta_logp.clone())
        prior_logp = self.ode.prior_logp(x)
        # print(f'Prior_logp: {prior_logp}')
        delta_logps = [i + prior_logp for i in delta_logps]
        # delta_logps.append(prior_logp.clone())
        xs = torch.stack(xs, dim=1)
        delta_logps = torch.stack(delta_logps, dim=1)
        return xs, delta_logps
    
    def simulate_with_likelihood_augmented(self, x0, ts, y, **kwargs):
        """
        Simulate from t=1 (data) to t=0 (noise) and compute log p(x_data)
        """
        batch_size = x0.shape[0]
        nts = ts.shape[1]
        
        # Initialize trajectories
        xs = [x0.clone()]
        
        # Track ACCUMULATED DIVERGENCE (not log p yet!)
        accumulated_div = torch.zeros(batch_size).to(x0.device)
        div_trajectory = [accumulated_div.clone()]
        
        x = x0
        
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            dt = ts[:, t_idx + 1] - ts[:, t_idx]  # Negative (going 1 -> 0)
            
            # Heun's method for x
            k1 = self.ode.drift_coefficient(x, t, y)
            x_pred = x + dt * k1
            k2 = self.ode.drift_coefficient(x_pred, t + dt, y)
            x = x + 0.5 * dt * (k1 + k2)
            
            # Compute divergence
            div = self.ode.compute_div(x, t, y)
            
            # Accumulate: ∫ div dt (with dt negative)
            accumulated_div = accumulated_div - div * dt.squeeze()
            
            xs.append(x.clone())
            div_trajectory.append(accumulated_div.clone())
        
        # NOW we can compute log probabilities
        prior_logp = self.ode.prior_logp(x)  # log p(x_noise)
        final_logp = prior_logp + accumulated_div  # log p(x_data)
        
        # Create the actual log p trajectory (working backwards)
        logp_trajectory = []
        for i in range(len(div_trajectory)):
            # At each point, log p = prior + accumulated div up to that point
            logp_at_t = prior_logp + div_trajectory[i]
            logp_trajectory.append(logp_at_t)
        
        xs = torch.stack(xs, dim=1)
        logp_trajectory = torch.stack(logp_trajectory, dim=1)
        
        return xs, logp_trajectory, final_logp

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
    
    def simulate_with_likelihood_augmented(self, x0, ts, **kwargs):
        """
        Simulate from t=1 (data) to t=0 (noise) and compute log p(x_data)
        """
        batch_size = x0.shape[0]
        nts = ts.shape[1]
        
        # Initialize trajectories
        xs = [x0.clone()]
        
        # Track ACCUMULATED DIVERGENCE (not log p yet!)
        accumulated_div = torch.zeros(batch_size).to(x0.device)
        div_trajectory = [accumulated_div.clone()]
        
        x = x0
        
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            dt = ts[:, t_idx + 1] - ts[:, t_idx]  # Negative (going 1 -> 0)
            
            # Heun's method for x
            k1 = self.ode.drift_coefficient(x, t, **kwargs)
            x_pred = x + dt * k1
            k2 = self.ode.drift_coefficient(x_pred, t + dt, **kwargs)
            x = x + 0.5 * dt * (k1 + k2)
            
            # Compute divergence
            div = self.ode.compute_div(x, t, **kwargs)
            
            # Accumulate: ∫ div dt (with dt negative)
            accumulated_div = accumulated_div - div * dt.squeeze()
            
            xs.append(x.clone())
            div_trajectory.append(accumulated_div.clone())
        
        # NOW we can compute log probabilities
        prior_logp = self.ode.prior_logp(x)  # log p(x_noise)
        final_logp = prior_logp + accumulated_div  # log p(x_data)
        
        # Create the actual log p trajectory (working backwards)
        logp_trajectory = []
        for i in range(len(div_trajectory)):
            # At each point, log p = prior + accumulated div up to that point
            logp_at_t = prior_logp + div_trajectory[i]
            logp_trajectory.append(logp_at_t)
        
        xs = torch.stack(xs, dim=1)
        logp_trajectory = torch.stack(logp_trajectory, dim=1)
        
        return xs, logp_trajectory, final_logp
    
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
    
    @classmethod
    def isotropic_with_mean(cls, mean: torch.Tensor, std: float) -> "Gaussian":
        dim = mean.shape[0]
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)





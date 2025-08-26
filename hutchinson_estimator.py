import torch
import torch.autograd as autograd
import math

def hutchinson_trace_estimator(f, x, num_samples=1, distribution='rademacher'):
    """
    Improved Hutchinson-Skilling trace estimator.
    
    Args:
        f: Function that takes x and returns a vector of the same shape
        x: Input tensor of shape (..., d)
        num_samples: Number of random vectors to use for estimation
        distribution: 'rademacher', 'gaussian', or 'sphere'
    
    Returns:
        trace_estimate: Estimated trace for each batch element
    """
    original_shape = x.shape
    batch_dims = original_shape[:-1]
    d = original_shape[-1]
    
    # Flatten batch dimensions
    x_flat = x.view(-1, d)  # Shape: (batch_size, d)
    batch_size = x_flat.size(0)
    
    trace_estimates = []
    
    for _ in range(num_samples):
        # Generate random vector z
        if distribution == 'rademacher':
            z = torch.randint_like(x_flat, 0, 2, dtype=x.dtype) * 2 - 1
        elif distribution == 'gaussian':
            z = torch.randn_like(x_flat)
        elif distribution == 'sphere':
            z = torch.randn_like(x_flat)
            z = z / torch.norm(z, dim=-1, keepdim=True) * math.sqrt(d)
        else:
            raise ValueError("distribution must be 'rademacher', 'gaussian', or 'sphere'")
        
        # Create fresh tensor with gradients enabled
        x_input = x_flat.detach().requires_grad_(True)
        
        # Compute function output
        f_output = f(x_input.view(original_shape))
        f_output_flat = f_output.view(batch_size, -1)
        
        # Compute vector-Jacobian product: z^T * (∂f/∂x)
        vjp = autograd.grad(
            outputs=f_output_flat,
            inputs=x_input,
            grad_outputs=z,
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]
        
        # Compute trace estimate: z^T * (∂f/∂x) * z
        trace_est = torch.sum(z * vjp, dim=-1)
        trace_estimates.append(trace_est)
    
    # Average over samples and reshape
    trace_estimate = torch.stack(trace_estimates).mean(dim=0)
    return trace_estimate.view(batch_dims)

def hutchinson_divergence(vector_field, x, num_samples=10):
    """
    Compute divergence using Hutchinson estimator.
    
    Args:
        vector_field: Function mapping (N, d) -> (N, d)
        x: Points where to evaluate divergence (N, d)
        num_samples: Number of random samples
        
    Returns:
        divergence: Estimated divergence at each point (N,)
    """
    return hutchinson_trace_estimator(vector_field, x, num_samples)

def hutchinson_score_divergence(score_model, x, t, num_samples=10):
    """
    Compute divergence of score function.
    
    Args:
        score_model: Score network
        x: Input points (N, d)
        t: Time points (N, 1) or scalar
        num_samples: Number of samples
        
    Returns:
        divergence: Score function divergence (N,)
    """
    def score_fn(x_):
        return score_model(x_, t)
    
    return hutchinson_trace_estimator(score_fn, x, num_samples)

# Test functions
def test_hutchinson():
    """Test the Hutchinson estimator with known functions."""
    print("Testing Hutchinson Trace Estimator")
    print("=" * 40)
    
    # Test 1: Linear function f(x) = Ax
    print("\nTest 1: Linear function f(x) = Ax")
    A = torch.tensor([[2.0, 1.0], [0.5, 3.0]])
    true_trace = torch.trace(A).item()  # Should be 5.0
    
    def linear_fn(x):
        return torch.matmul(x, A.T)
    
    x_test = torch.randn(100, 2)
    
    for num_samples in [1, 5, 10, 50]:
        est_trace = hutchinson_trace_estimator(linear_fn, x_test, num_samples)
        mean_est = est_trace.mean().item()
        error = abs(mean_est - true_trace)
        print(f"Samples: {num_samples:2d}, Estimate: {mean_est:.3f}, "
              f"True: {true_trace:.3f}, Error: {error:.3f}")
    
    # Test 2: Quadratic function with known trace
    print("\nTest 2: Quadratic function f(x) = [x₁², x₂²]")
    def quadratic_fn(x):
        return torch.stack([x[:, 0]**2, x[:, 1]**2], dim=1)
    
    x_test = torch.ones(1, 2)  # At point (1,1), Jacobian is diag([2,2])
    true_trace = 4.0  # 2*x₁ + 2*x₂ = 2*1 + 2*1 = 4
    
    for num_samples in [1, 10, 50, 100]:
        est_trace = hutchinson_trace_estimator(quadratic_fn, x_test, num_samples)
        error = abs(est_trace.item() - true_trace)
        print(f"Samples: {num_samples:3d}, Estimate: {est_trace.item():.3f}, "
              f"True: {true_trace:.3f}, Error: {error:.3f}")
    
    # Test 3: Neural network score function
    print("\nTest 3: Neural network divergence")
    
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 2)
            )
        
        def forward(self, x, t=None):
            return self.net(x)
    
    model = SimpleNet()
    x_test = torch.randn(50, 2)
    
    # Compare different numbers of samples
    estimates = []
    for num_samples in [1, 5, 10, 20]:
        div_est = hutchinson_score_divergence(model, x_test, None, num_samples)
        estimates.append(div_est.mean().item())
        print(f"Samples: {num_samples:2d}, Mean divergence: {div_est.mean():.4f}")
    
    print(f"Convergence: {estimates}")

if __name__ == "__main__":
    test_hutchinson()
import numpy as np
import torch

import torch
import numpy as np

def compute_div_test(vector_field_fn, x: torch.Tensor, t: torch.Tensor = None, create_graph=True):
    """
    Test function to compute divergence of any vector field function.
    
    Args:
        vector_field_fn: Function that takes (x, t) and returns vector field
        x: Input positions [batch_size, x_dim]
        t: Time (optional, can be None)
        create_graph: Whether to create computation graph
    
    Returns:
        div: Divergence at each point [batch_size]
    """
    device = x.device
    batch_size = x.shape[0]
    x_dim = x.shape[1]
    
    # Ensure x requires gradients
    x = x.requires_grad_(True)
    
    # Compute vector field
    vector_field = vector_field_fn(x, t)
    
    # Initialize divergence
    div = torch.zeros(batch_size, device=device)
    
    # Compute divergence: sum of ∂v_i/∂x_i
    for i in range(x_dim):
        v_i = vector_field[:, i]
        
        grad_outs = torch.ones_like(v_i)
        grad_v_i = torch.autograd.grad(
            outputs=v_i,
            inputs=x,
            grad_outputs=grad_outs,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        
        if grad_v_i is not None:
            div += grad_v_i[:, i]
    
    return div

def compute_div_hutch_test(vector_field_fn, x: torch.Tensor, t: torch.Tensor = None, create_graph=True):
    """
    Compute divergence using Hutchinson's trace estimator
    More efficient for high-dimensional problems
    """
    device = x.device
    # x = x.to(device)


    
    batch_size = x.shape[0]
    original_shape = x.shape
    
    # Flatten and require gradients
    x_flat = x.flatten(1).requires_grad_(True)
    x_dim = x_flat.shape[1]
    
    # Reshape for U-Net
    x_reshaped = x_flat.view(original_shape)
    
    # Get vector field
    vector_field = vector_field_fn(x_reshaped, t)
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

# Test vector fields with zero divergence (curl-only)

def solenoidal_field_2d(x, t=None):
    x_coord, y_coord = x[:, 0], x[:, 1]
    
    # F_x = 2x²y - y²
    F_x = -y_coord
    
    # F_y = -2xy  
    F_y = -x_coord
    
    return torch.stack([F_x, F_y], dim=1)

def solenoidal_field_3d(x, t=None):

    x_coord, y_coord, z_coord = x[:, 0], x[:, 1], x[:, 2]
    
    # This is a simplified solenoidal field: F = (-y, x, 0)
    # More complex one would require computing full curl
    F_x = -y_coord
    F_y = x_coord  
    F_z = torch.zeros_like(z_coord)
    
    return torch.stack([F_x, F_y, F_z], dim=1)

def stream_function_field(x, t=None):
    """
    2D field from stream function ψ = sin(x)*cos(y)
    u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    x_coord, y_coord = x[:, 0], x[:, 1]
    
    # u = ∂ψ/∂y = sin(x)*(-sin(y)) = -sin(x)*sin(y)
    u = -torch.sin(x_coord) * torch.sin(y_coord)
    
    # v = -∂ψ/∂x = -cos(x)*cos(y)
    v = -torch.cos(x_coord) * torch.cos(y_coord)
    
    return torch.stack([u, v], dim=1)

# Test your divergence function
def test_divergence():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing 2D solenoidal field...")
    # Test 2D case
    x_2d = torch.randn(100, 2, device=device) * 2  # Random points in [-4, 4]
    div_2d = compute_div_test(solenoidal_field_2d, x_2d)
    
    print(f"2D Solenoidal field - Max divergence: {torch.max(torch.abs(div_2d)):.2e}")
    print(f"2D Solenoidal field - Mean divergence: {torch.mean(torch.abs(div_2d)):.2e}")
    
    print("\nTesting 2D stream function field...")
    div_stream = compute_div_test(stream_function_field, x_2d)
    print(f"Stream function field - Max divergence: {torch.max(torch.abs(div_stream)):.2e}")
    print(f"Stream function field - Mean divergence: {torch.mean(torch.abs(div_stream)):.2e}")
    
    print("\nTesting 3D solenoidal field...")
    # Test 3D case  
    x_3d = torch.randn(100, 3, device=device) * 2
    div_3d = compute_div_hutch_test(solenoidal_field_3d, x_3d)
    
    print(f"3D Solenoidal field - Max divergence: {torch.max(torch.abs(div_3d)):.2e}")
    print(f"3D Solenoidal field - Mean divergence: {torch.mean(torch.abs(div_3d)):.2e}")
    
    # Test with a field that has NON-ZERO divergence for comparison
    def divergent_field(x, t=None):
        # F = (x, y) or (x, y, z) - has divergence = dimension
        return x
    
    print(f"\nTesting divergent field (should NOT be zero)...")
    div_divergent = compute_div_test(divergent_field, x_2d)
    print(f"Divergent field - Mean divergence: {torch.mean(div_divergent):.2f} (should be ~2)")

# Run the test
if __name__ == "__main__":
    test_divergence()


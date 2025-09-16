import torch
import matplotlib.pyplot as plt
import numpy as np

def sinkhorn_step_by_step(generated_samples, uniform_samples, reg=0.01, max_iter=20, visualize=True):
    """
    Step-by-step walkthrough of the Sinkhorn algorithm with visualization.
    """
    device = generated_samples.device
    n, m = len(generated_samples), len(uniform_samples)
    
    print("=== SINKHORN ALGORITHM WALKTHROUGH ===\n")
    
    # Step 1: Compute cost matrix
    print("Step 1: Compute Cost Matrix")
    print("C[i,j] = ||x_i - y_j||² (squared Euclidean distance)")
    C = torch.cdist(generated_samples, uniform_samples, p=2) ** 2
    print(f"Cost matrix shape: {C.shape}")
    print(f"Cost range: [{C.min():.4f}, {C.max():.4f}]")
    print()
    
    # Step 2: Create Gibbs kernel
    print("Step 2: Create Gibbs Kernel")
    print(f"K[i,j] = exp(-C[i,j] / reg) where reg = {reg}")
    K = torch.exp(-C / reg)
    print(f"Kernel values range: [{K.min():.6f}, {K.max():.6f}]")
    print("(Higher values = easier/cheaper to transport)")
    print()
    
    # Step 3: Initialize weights and dual variables
    print("Step 3: Initialize Variables")
    a = torch.ones(n, device=device) / n  # Source weights (uniform)
    b = torch.ones(m, device=device) / m  # Target weights (uniform)
    u = torch.ones(n, device=device)     # Dual variable for sources
    v = torch.ones(m, device=device)     # Dual variable for targets
    
    print(f"Source weights a: all equal to 1/{n} = {1/n:.6f}")
    print(f"Target weights b: all equal to 1/{m} = {1/m:.6f}")
    print(f"Initial dual variables u, v: all ones")
    print()
    
    # Track convergence
    errors = []
    costs = []
    
    print("=== SINKHORN ITERATIONS ===")
    
    for iteration in range(max_iter):
        u_old = u.clone()
        
        # Step 4a: Update u (scaling for rows)
        # This ensures row sums are correct: ∑_j P[i,j] = a[i]
        Kv = K @ v  # Matrix-vector product
        u = a / (Kv + 1e-8)  # Element-wise division
        
        # Step 4b: Update v (scaling for columns)  
        # This ensures column sums are correct: ∑_i P[i,j] = b[j]
        KTu = K.T @ u  # Transpose-matrix-vector product
        v = b / (KTu + 1e-8)  # Element-wise division
        
        # Compute transport plan and cost
        P = torch.diag(u) @ K @ torch.diag(v)
        transport_cost = torch.sum(P * C)
        costs.append(transport_cost.item())
        
        # Check convergence
        error = torch.max(torch.abs(u - u_old)).item()
        errors.append(error)
        
        if iteration < 5 or iteration % 5 == 0:  # Print first few and every 5th
            print(f"Iter {iteration:2d}: Transport cost = {transport_cost:.6f}, "
                  f"u change = {error:.2e}")
            
            # Check constraint satisfaction
            row_sums = torch.sum(P, dim=1)
            col_sums = torch.sum(P, dim=0)
            row_error = torch.max(torch.abs(row_sums - a)).item()
            col_error = torch.max(torch.abs(col_sums - b)).item()
            print(f"         Row constraint error: {row_error:.2e}")
            print(f"         Col constraint error: {col_error:.2e}")
        
        if error < 1e-6:
            print(f"Converged at iteration {iteration}")
            break
    
    print()
    
    # Final transport plan analysis
    print("=== FINAL RESULTS ===")
    final_P = torch.diag(u) @ K @ torch.diag(v)
    final_cost = torch.sum(final_P * C)
    
    print(f"Final transport cost: {final_cost:.6f}")
    print(f"Transport plan sparsity: {(final_P < 1e-6).sum().item()}/{final_P.numel()} entries near zero")
    print(f"Max transport amount: {final_P.max():.6f}")
    print(f"Min transport amount: {final_P.min():.6f}")
    
    # Verify constraints
    row_sums = torch.sum(final_P, dim=1)
    col_sums = torch.sum(final_P, dim=0) 
    print(f"Row sum error: {torch.max(torch.abs(row_sums - a)):.2e}")
    print(f"Col sum error: {torch.max(torch.abs(col_sums - b)):.2e}")
    
    if visualize and len(generated_samples) <= 100:  # Only visualize small examples
        visualize_sinkhorn(generated_samples, uniform_samples, final_P, costs, errors)
    
    return final_cost, final_P


def visualize_sinkhorn(generated_samples, uniform_samples, transport_plan, costs, errors):
    """Visualize the Sinkhorn algorithm results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Data points
    ax = axes[0, 0]
    gen_np = generated_samples.cpu().numpy()
    uni_np = uniform_samples.cpu().numpy()
    
    ax.scatter(gen_np[:, 0], gen_np[:, 1], c='red', alpha=0.7, label='Generated', s=50)
    ax.scatter(uni_np[:, 0], uni_np[:, 1], c='blue', alpha=0.7, label='Uniform', s=50)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Source (Red) and Target (Blue) Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Transport plan visualization
    ax = axes[0, 1]
    P_np = transport_plan.cpu().numpy()
    
    # Show only significant transport connections
    threshold = P_np.max() * 0.1  # Show connections > 10% of max
    
    for i in range(len(generated_samples)):
        for j in range(len(uniform_samples)):
            if P_np[i, j] > threshold:
                # Line thickness proportional to transport amount
                alpha = P_np[i, j] / P_np.max()
                ax.plot([gen_np[i, 0], uni_np[j, 0]], 
                       [gen_np[i, 1], uni_np[j, 1]], 
                       'gray', alpha=alpha, linewidth=alpha*3)
    
    ax.scatter(gen_np[:, 0], gen_np[:, 1], c='red', s=30, zorder=5)
    ax.scatter(uni_np[:, 0], uni_np[:, 1], c='blue', s=30, zorder=5)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Optimal Transport Plan\n(Line thickness ∝ transport amount)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Cost convergence
    ax = axes[1, 0]
    ax.plot(costs, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Transport Cost')
    ax.set_title('Cost Convergence')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error convergence (log scale)
    ax = axes[1, 1]
    ax.semilogy(errors, 'r-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Dual Variable Change (log scale)')
    ax.set_title('Convergence Rate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def intuitive_explanation():
    """Print an intuitive explanation of what's happening."""
    print("""
=== INTUITIVE EXPLANATION ===

Think of the Sinkhorn algorithm as an iterative "balancing" process:

1. **Setup**: You have sand piles (generated points) and target locations (uniform points).
   Moving sand costs money based on distance.

2. **Gibbs Kernel**: K[i,j] = exp(-cost[i,j]/reg)
   - High values = cheap to transport = prefer this connection
   - Low values = expensive = avoid this connection
   - reg controls how "fuzzy" the solution is

3. **Iterative Balancing**:
   
   Round 1: Fix column constraints, balance rows
   - Look at each source: "Am I sending out the right total amount?"
   - Adjust scaling (u) so each row sums correctly
   
   Round 2: Fix row constraints, balance columns  
   - Look at each target: "Am I receiving the right total amount?"
   - Adjust scaling (v) so each column sums correctly
   
   Repeat until balanced!

4. **Why This Works**:
   - The regularized optimal solution has the form P = diag(u) @ K @ diag(v)
   - Sinkhorn finds the right scalings u and v
   - It's like tuning two sets of "multipliers" until everything balances

5. **The Magic**: This simple alternating procedure provably converges to 
   the optimal solution of the regularized transport problem!

The smaller 'reg' gets, the closer we are to the true (unregularized) 
optimal transport solution, but the more iterations we need.
""")


# Example walkthrough
if __name__ == "__main__":
    # Create small example for clear visualization
    torch.manual_seed(42)
    
    # Generated samples (slightly clustered)
    generated = torch.normal(0.3, 0.1, (15, 2))
    generated = torch.clamp(generated, 0, 1)
    
    # Uniform samples
    uniform = torch.rand(15, 2)
    
    print("Running Sinkhorn algorithm walkthrough...")
    print("(Using small example for clarity)\n")
    
    # Run step-by-step algorithm
    cost, plan = sinkhorn_step_by_step(generated, uniform, reg=0.05, max_iter=20)
    
    # Print intuitive explanation
    intuitive_explanation()
    
    print(f"\nFinal Wasserstein distance: {cost:.6f}")
    print("\nCompare with different regularization:")
    
    for reg_val in [0.1, 0.01, 0.001]:
        from __main__ import sinkhorn_wasserstein_2d  # Import our original function
        dist = sinkhorn_wasserstein_2d(generated, uniform_samples=uniform, reg=reg_val, max_iter=100)
        print(f"reg = {reg_val:5.3f}: distance = {dist:.6f}")
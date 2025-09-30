import torch
from typing import Optional
import abstracts as ab
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Several plotting utility functions
def plot_trajectory_evolution_3d(simulator, path, device, 
                                num_samples=50000, 
                                num_marginals=5,
                                scale=6.0,
                                time_start=0,
                                time_end=1,
                                time_steps=100,
                                figsize=(12, 8),
                                alpha=0.6,
                                point_size=10,
                                label_every=3,
                                title="Learned Trajectory Evolution",
                                show_legend=False,
                                view_elev=None,
                                view_azim=None,
                                save_path=None,
                                ):
    """
    Plot 3D trajectory evolution showing how samples evolve over time.
    
    Parameters:
    -----------
    simulator : object
        Simulator object with simulate_with_trajectory method
    path : object
        Path object with p_simple.sample method for initial sampling
    device : torch.device
        Device to run computations on
    num_samples : int, default=50000
        Number of samples to simulate
    num_marginals : int, default=5
        Number of time points to record/plot
    scale : float, default=6.0
        Scale for y and z axis limits (-scale to scale)
    time_start : float, default=0
        Start time for simulation
    time_end : float, default=1
        End time for simulation
    time_steps : int, default=100
        Number of time steps in simulation
    figsize : tuple, default=(12, 8)
        Figure size
    alpha : float, default=0.6
        Transparency of scatter points
    point_size : float, default=10
        Size of scatter points
    label_every : int, default=3
        Label every nth time point for clarity
    title : str, default="Learned Trajectory Evolution"
        Plot title
    show_legend : bool, default=False
        Whether to show legend
    view_elev : float, optional
        Elevation angle for 3D view
    view_azim : float, optional
        Azimuth angle for 3D view
    save_path : str, optional
        Path to save the figure
    pf : object, optional
        Object with record_every method. If None, uses simple indexing
    
    Returns:
    --------
    fig, ax : matplotlib objects
        Figure and axis objects for further customization
    """
    
    # Generate time points
    ts = torch.linspace(time_start, time_end, time_steps).to(device)
    
    # Determine which time indices to record
    record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
    
    # Sample initial conditions and simulate
    x0 = path.p_simple.sample(num_samples).to(device)
    xts = simulator.simulate_with_trajectory(x0, ts.view(1, -1, 1).expand(num_samples, -1, 1))
    xts = xts[:, record_every_idxs, :]
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each marginal at its corresponding time point
    for idx in range(xts.shape[1]):
        xx = xts[:, idx, :].cpu().numpy()  # Convert to numpy for plotting
        tt = ts[record_every_idxs[idx]].item()
        
        # Create x-coordinates (time) for all samples at this time point
        time_coords = np.full(xx.shape[0], tt)
        
        # Plot scatter: time on x-axis, samples on y-z axes
        label = f't={tt:.2f}' if idx % label_every == 0 else ""
        ax.scatter(time_coords, xx[:, 0], xx[:, 1],
                  alpha=alpha, s=point_size, label=label)
    
    # Set labels and limits
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    ax.set_zlabel('Z coordinate', fontsize=14)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    
    # Set viewing angle if specified
    if view_elev is not None or view_azim is not None:
        ax.view_init(elev=view_elev, azim=view_azim)
    
    # Add legend if requested
    if show_legend:
        ax.legend()
    
    # Set title
    ax.set_title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

def plot_trajectory_lines_3d(simulator, path, device,
                             num_samples=1000,
                             num_marginals=5,
                             scale=6.0,
                             time_start=0,
                             time_end=1,
                             time_steps=100,
                             num_samples_to_plot=200,
                             figsize=(14, 10),
                             line_alpha=0.3,
                             line_width=0.8,
                             line_color='blue',
                             scatter_alpha=0.7,
                             scatter_size=15,
                             colormap='viridis',
                             title=None,
                             view_elev=None,
                             view_azim=None,
                             save_path=None,
                             random_seed=None):
    """
    Plot 3D trajectory evolution showing individual sample paths connected by lines.
    
    Parameters:
    -----------
    simulator : object
        Simulator object with simulate_with_trajectory method
    path : object
        Path object with p_simple.sample method for initial sampling
    device : torch.device
        Device to run computations on
    num_samples : int, default=1000
        Total number of samples to simulate
    num_marginals : int, default=5
        Number of time points to record/plot
    scale : float, default=6.0
        Scale for y and z axis limits (-scale to scale)
    time_start : float, default=0
        Start time for simulation
    time_end : float, default=1
        End time for simulation
    time_steps : int, default=100
        Number of time steps in simulation
    num_samples_to_plot : int, default=200
        Number of sample trajectories to visualize (for clarity)
    figsize : tuple, default=(14, 10)
        Figure size
    line_alpha : float, default=0.3
        Transparency of trajectory lines
    line_width : float, default=0.8
        Width of trajectory lines
    line_color : str, default='blue'
        Color of trajectory lines
    scatter_alpha : float, default=0.7
        Transparency of scatter points
    scatter_size : float, default=15
        Size of scatter points
    colormap : str, default='viridis'
        Colormap for time-based coloring
    title : str, optional
        Custom plot title. If None, generates default title
    view_elev : float, optional
        Elevation angle for 3D view
    view_azim : float, optional
        Azimuth angle for 3D view
    save_path : str, optional
        Path to save the figure
    random_seed : int, optional
        Random seed for reproducible sample selection
    
    Returns:
    --------
    fig, ax : matplotlib objects
        Figure and axis objects for further customization
    xts_np : numpy.ndarray
        Trajectory data as numpy array for further analysis
    ts_recorded : numpy.ndarray
        Recorded time points
    sample_indices : numpy.ndarray
        Indices of samples that were plotted
    """
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate time points
    ts = torch.linspace(time_start, time_end, time_steps).to(device)
    
    # Determine which time indices to record
    record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
    
    # Sample initial conditions and simulate
    x0 = path.p_simple.sample(num_samples).to(device)
    xts = simulator.simulate_with_trajectory(x0, ts.view(1, -1, 1).expand(num_samples, -1, 1))
    xts = xts[:, record_every_idxs, :]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert data to numpy for easier manipulation
    xts_np = xts.cpu().numpy()  # Shape: (num_samples, num_time_points, 2)
    ts_recorded = ts[record_every_idxs].cpu().numpy()
    
    # Limit number of samples for visualization clarity
    num_samples_to_plot = min(num_samples, num_samples_to_plot)
    sample_indices = np.random.choice(num_samples, num_samples_to_plot, replace=False)
    
    # Plot trajectory lines for each selected sample
    for i, sample_idx in enumerate(sample_indices):
        # Extract trajectory for this sample
        sample_trajectory = xts_np[sample_idx, :, :]  # Shape: (num_time_points, 2)
        
        # Plot line connecting this sample across time
        ax.plot(ts_recorded,
                sample_trajectory[:, 0],
                sample_trajectory[:, 1],
                alpha=line_alpha, 
                linewidth=line_width, 
                color=line_color)
    
    # Prepare scatter plot data
    all_times = []
    all_y = []
    all_z = []
    
    for idx in range(xts.shape[1]):
        xx = xts[sample_indices, idx, :].cpu().numpy()  # Only plot subset
        tt = ts[record_every_idxs[idx]].item()
        time_coords = np.full(xx.shape[0], tt)
        
        all_times.extend(time_coords)
        all_y.extend(xx[:, 0])
        all_z.extend(xx[:, 1])
    
    # Convert to arrays
    all_times = np.array(all_times)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    
    # Create scatter plot with color mapping by time
    scatter = ax.scatter(all_times, all_y, all_z,
                        c=all_times, cmap=colormap,
                        alpha=scatter_alpha, s=scatter_size)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Time', shrink=0.5)
    
    # Set labels and limits
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    ax.set_zlabel('Z coordinate', fontsize=14)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    
    # Set viewing angle if specified
    if view_elev is not None or view_azim is not None:
        ax.view_init(elev=view_elev, azim=view_azim)
    
    # Set title
    if title is None:
        title = f'Learned Trajectory Evolution with Sample Paths (n={num_samples_to_plot})'
    ax.set_title(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax, xts_np, ts_recorded, sample_indices

def hist2d_samples(samples, ax: Optional[Axes] = None, bins: int = 200, scale: float = 5.0, percentile: int = 99, **kwargs):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]])
    
    # Determine color normalization based on the 99th percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    
    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)

def hist2d_sampleable(sampleable: ab.Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    ax.hist2d(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)

def scatter_sampleable(sampleable: ab.Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    ax.scatter(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)

def imshow_density(density: ab.Density, bins: int, scale: float, ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(density.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)

def contour_density(density: ab.Density, bins: int, scale: float, ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.contour(density.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )


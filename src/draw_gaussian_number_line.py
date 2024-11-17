import numpy as np
import matplotlib.pyplot as plt

def plot_numberline_with_gaussians(locations, amplitudes=None, widths=None, labels=None,
                                   xlim=None, num_points=1000, title="", half_gaussians=None,
                                   decision_boundary=None):
    """
    Create a number line with Gaussian distributions at specified locations.
    
    Parameters:
    locations (array-like): Centers of the Gaussian distributions
    amplitudes (array-like, optional): Heights of the Gaussians. Defaults to 1.
    widths (array-like, optional): Standard deviations of Gaussians. Defaults to 0.2.
    xlim (tuple, optional): (min, max) for x-axis. Defaults to extend beyond locations.
    num_points (int): Number of points for plotting smooth curves.
    decision_boundary (float, optional): x-coordinate for decision boundary line
    """
    
    locations = np.array(locations)
    if amplitudes is None:
        amplitudes = np.ones_like(locations)
    if widths is None:
        widths = 0.2 * np.ones_like(locations)
    if xlim is None:
        xlim = (min(locations) - 1, max(locations) + 1)
        
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot number line
    ax.axhline(y=0, color='black', linewidth=1)
    
    # Create x points for smooth curves
    x = np.linspace(xlim[0], xlim[1], num_points)
    
    # Plot Gaussians
    gaussian_lines = []  # Store line objects for legend
    for i, (loc, amp, width, label) in enumerate(zip(locations, amplitudes, widths, labels)):
        gaussian = amp * np.exp(-(x - loc)**2 / (2 * width**2))

        # Handle half Gaussians 
        x_offset = 0
        for idx, side in half_gaussians:
            if i == idx:
                if side == 'left':
                    gaussian[x > loc] = 0
                    x_offset = -0.1
                elif side == 'right':
                    gaussian[x < loc] = 0
                    x_offset = 0.1

        line, = ax.plot(x, gaussian, 'b-', linewidth=1.5)
        gaussian_lines.append(line)
        ax.text(loc+x_offset, 0.05, label, ha='center', va='bottom')

    # Add decision boundary if specified
    if decision_boundary is not None:
        boundary_line = ax.axvline(x=decision_boundary, color='red', linestyle='--', 
                                 linewidth=2, label='Decision Boundary')
        
    # Customize appearance
    ax.set_ylim(0.0, 0.26)
    ax.set_xlim(xlim)
    
    # Remove y-axis ticks
    ax.set_yticks([])

    # Add legend
    ax.plot([], [], 'b-', label='Features', linewidth=1.5)  # Add dummy line for features
    ax.legend()

    ax.set_title(title, pad=20)

    return fig, ax

# Example usage

# ReLU before activation
fig, ax = plot_numberline_with_gaussians(
    [-3, -1, 0, 2, 4], 
    [0.125, 0.125, 0.125, 0.125, 0.125], 
    [0.125, 0.125, 0.125, 0.125, 0.125], 
    ['a', 'b', 'c', 'd', 'e'],
    xlim=(-4, 5),
    title="ReLU initial projection for feature 'c'",
    half_gaussians=[],
    decision_boundary=0.5
    )
plt.tight_layout()

# ReLU before activation
fig, ax = plot_numberline_with_gaussians(
    [-3, -1, 0, 2, 4], 
    [0.010, 0.010, 0.010, 0.125, 0.125], 
    [0.125, 0.125, 0.125, 0.125, 0.125], 
    ['a', 'b', 'c', 'd', 'e'],
    xlim=(-4, 5),
    title="ReLU activation for feature 'c'",
    half_gaussians=[],
    decision_boundary=0.5
    )
plt.tight_layout()

# Abs before activation
fig, ax = plot_numberline_with_gaussians(
    [-3, -1, 0, 2, 4], 
    [0.125, 0.125, 0.125, 0.125, 0.125], 
    [0.125, 0.125, 0.125, 0.125, 0.125], 
    ['a', 'b', 'c', 'd', 'e'],
    xlim=(-4, 5),
    title="Abs initial projection for feature 'c'",
    half_gaussians=[],
    decision_boundary=0
    )
plt.tight_layout()

# Abs after activation
fig, ax = plot_numberline_with_gaussians(
    [0, 1, 2, 3, 4], 
    [0.25, 0.125, 0.125, 0.125, 0.125], 
    [0.125, 0.125, 0.125, 0.125, 0.125], 
    ['c', 'b', 'd', 'a', 'e'],
    xlim=(-4, 5),
    title="Abs activation for feature 'c'",
    half_gaussians=[(0, 'right')],
    decision_boundary=0  # Add decision boundary at x=1.5
    )
plt.tight_layout()
plt.show()
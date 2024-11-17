import numpy as np
import matplotlib.pyplot as plt
import seaborn

def plot_activation_demo(locations, amplitudes=None, widths=None, labels=None,
                        xlim=None, num_points=1000, title="", half_gaussians=None,
                        decision_boundary=None, boundary_label="Decision Boundary"):
    """
    Create publication-ready visualization of feature distributions with decision boundaries.
    
    Parameters:
    locations (array-like): Centers of the Gaussian distributions
    amplitudes (array-like, optional): Heights of the Gaussians
    widths (array-like, optional): Standard deviations of Gaussians
    labels (array-like): Labels for each Gaussian
    xlim (tuple, optional): (min, max) for x-axis
    num_points (int): Number of points for smooth curves
    title (str): Plot title
    half_gaussians (list): List of tuples (idx, 'left'/'right') for truncated Gaussians
    decision_boundary (float, optional): x-coordinate for decision boundary
    boundary_label (str): Label for decision boundary line
    """
    # Set publication-style parameters
    seaborn.set_theme()
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'figure.figsize': (6, 3),
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # Initialize parameters
    locations = np.array(locations)
    if amplitudes is None:
        amplitudes = np.ones_like(locations)
    if widths is None:
        widths = 0.2 * np.ones_like(locations)
    if xlim is None:
        xlim = (min(locations) - 0.5, max(locations) + 0.5)
    if half_gaussians is None:
        half_gaussians = []
        
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot baseline
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    
    # Generate smooth curves
    x = np.linspace(xlim[0], xlim[1], num_points)
    
    # Plot Gaussians
    colors = seaborn.color_palette("husl", len(locations))
    for i, (loc, amp, width, label, color) in enumerate(zip(locations, amplitudes, widths, labels, colors)):
        gaussian = amp * np.exp(-(x - loc)**2 / (2 * width**2))
        
        # Handle truncated Gaussians
        x_offset = 0
        for idx, side in half_gaussians:
            if i == idx:
                if side == 'left':
                    gaussian[x > loc] = 0
                    x_offset = -0.1
                elif side == 'right':
                    gaussian[x < loc] = 0
                    x_offset = 0.1
        
        # Plot distribution
        ax.plot(x, gaussian, '-', color=color, linewidth=1.5, alpha=0.7)
        ax.text(loc + x_offset, -0.02, label, ha='center', va='top',
                fontsize=10, bbox=dict(facecolor='white', edgecolor='none',
                                     alpha=0.7, pad=1.5))
    
    # Add decision boundary
    if decision_boundary is not None:
        ax.axvline(x=decision_boundary, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=boundary_label)
        ax.legend(frameon=False)
    
    # Customize appearance
    ax.set_ylim(-0.05, 0.3)
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    
    # Add title with padding
    ax.set_title(title, pad=15)
    
    plt.tight_layout()
    return fig, ax

# Generate the four demonstration figures
def create_activation_demos():
    # Common parameters
    xlim = (-4, 5)
    base_locations = [-3, -1, 0, 2, 4]
    base_labels = ['a', 'b', 'c', 'd', 'e']
    
    # ReLU pre-activation
    fig1, ax1 = plot_activation_demo(
        base_locations,
        amplitudes=[0.125] * 5,
        labels=base_labels,
        xlim=xlim,
        title="(a) ReLU Pre-activation Projection",
        decision_boundary=0.5
    )
    
    # ReLU post-activation
    fig2, ax2 = plot_activation_demo(
        base_locations,
        amplitudes=[0.01, 0.01, 0.01, 0.125, 0.125],
        labels=base_labels,
        xlim=xlim,
        title="(b) ReLU Post-activation Response",
        decision_boundary=0.5
    )
    
    # Abs pre-activation
    fig3, ax3 = plot_activation_demo(
        base_locations,
        amplitudes=[0.125] * 5,
        labels=base_labels,
        xlim=xlim,
        title="(c) Abs Pre-activation Projection",
        decision_boundary=0
    )
    
    # Abs post-activation
    fig4, ax4 = plot_activation_demo(
        [0, 1, 2, 3, 4],
        amplitudes=[0.25, 0.125, 0.125, 0.125, 0.125],
        labels=['c', 'b', 'd', 'a', 'e'],
        xlim=xlim,
        title="(d) Abs Post-activation Response",
        half_gaussians=[(0, 'right')],
        decision_boundary=0
    )
    
    return [fig1, fig2, fig3, fig4]

if __name__ == "__main__":
    # Create all figures
    figures = create_activation_demos()
    
    # Save each figure
    for i, fig in enumerate(figures):
        plt.tight_layout()
        # fig.savefig(f'activation_demo_{i+1}.pdf', 
        #            bbox_inches='tight',
        #            pad_inches=0.1,
        #            dpi=300)
        # plt.close(fig)
    plt.show()
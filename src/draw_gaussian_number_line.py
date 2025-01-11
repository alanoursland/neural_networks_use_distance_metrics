import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_activation_demo(locations, amplitudes=None, widths=None, labels=None,
                        xlim=None, num_points=1000, title="", half_gaussians=None,
                        decision_boundary=None):
    """Create publication-ready visualization of feature distributions"""
    setup_plotting_style()
    
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
    
    # Plot baseline (x-axis)
    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    
    # Generate smooth curves
    x = np.linspace(xlim[0], xlim[1], num_points)
    
    # Plot Gaussians
    for i, (loc, amp, width, label) in enumerate(zip(locations, amplitudes, widths, labels)):
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
        line, = ax.plot(x, gaussian, '-', color='#1f77b4', linewidth=1.5, alpha=0.7)
        
        # Add feature label using LaTeX
        ax.text(loc + x_offset, 0.05, f'${label}$', ha='center', va='top',
                fontsize=16, color='black')
    
    # Add decision boundary
    if decision_boundary is not None:
        ax.axvline(x=decision_boundary, color='red', linestyle='--', 
                  linewidth=1.5, alpha=0.7)
        
        # Add legend with single entry for Features
        # ax.plot([], [], '-', color='#1f77b4', label='Features', alpha=0.7)
        # ax.plot([], [], '--', color='red', label='Decision Boundary', alpha=0.7)
        # ax.legend(frameon=False, loc='upper right')
    
    # Customize appearance
    ax.set_ylim(-0.03, 0.3)
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    
    # # Remove x-axis ticks but keep the line
    # ax.set_xticks([])
    
    # Add title with padding
    # ax.set_title(title, pad=15)
    
    plt.tight_layout()
    return fig, ax

def create_activation_demos():
    """Create all activation demonstration figures"""
    xlim = (-4, 5)
    base_locations = [-3, -1, 0, 2, 4]
    base_labels = ['a', 'b', 'c', 'd', 'e']
    output_dir = 'results/activation_demos'
    
    # Define all figure configurations
    figure_configs = [
        {
            'name': 'relu_pre',
            'locations': base_locations,
            'amplitudes': [0.125] * 5,
            'labels': base_labels,
            'decision_boundary': 0.5,
        },
        {
            'name': 'relu_post',
            'locations': base_locations,
            'amplitudes': [0.01, 0.01, 0.01, 0.125, 0.125],
            'labels': base_labels,
            'decision_boundary': 0.5,
        },
        {
            'name': 'abs_pre',
            'locations': base_locations,
            'amplitudes': [0.125] * 5,
            'labels': base_labels,
            'decision_boundary': 0,
        },
        {
            'name': 'abs_post',
            'locations': [0, 1, 2, 3, 4],
            'amplitudes': [0.25, 0.125, 0.125, 0.125, 0.125],
            'labels': ['c', 'b', 'd', 'a', 'e'],
            'half_gaussians': [(0, 'right')],
            'decision_boundary': 0,
        },
    ]
    
    # Create and save each figure
    for config in figure_configs:
        fig, _ = plot_activation_demo(
            locations=config['locations'],
            amplitudes=config['amplitudes'],
            labels=config['labels'],
            xlim=xlim,
            decision_boundary=config.get('decision_boundary'),
            half_gaussians=config.get('half_gaussians', [])
        )
        
        # Save as PDF with vector graphics
        output_path = f"{output_dir}/{config['name']}.pdf"
        plt.savefig(output_path, 
                   bbox_inches='tight',
                   pad_inches=0.1,
                   format='pdf',
                   transparent=True)

        plt.close(fig)

def setup_plotting_style():
    """Set up publication-ready plotting style"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'figure.figsize': (4.5, 2.5),  # Adjusted for single column
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'text.usetex': True,  # Use LaTeX for text rendering
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

if __name__ == "__main__":
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs('results/activation_demos', exist_ok=True)
    
    # Create all figures
    create_activation_demos()
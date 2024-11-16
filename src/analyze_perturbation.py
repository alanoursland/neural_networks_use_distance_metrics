import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn

def debug_raw_data(results):
    """Print raw data for first few runs of each model"""
    for model_name in ['PerturbationAbs', 'PerturbationReLU']:
        print(f"\n{model_name} first run data:")
        first_run = results[model_name][0]
        # print("Offset values:", first_run['offset_values'], "...")
        # print("Offset accuracies:", first_run['offset_accuracies'], "...")
        
        # Check if all runs have same accuracy values
        first_accs = results[model_name][0]['offset_accuracies']
        all_same = True
        for run in results[model_name][1:]:
            if not np.array_equal(run['offset_accuracies'], first_accs):
                all_same = False
                break
        print(f"All runs identical? {all_same}")

def load_and_process_results(filepath):
    """Load results and compute statistics across runs"""
    results = torch.load(filepath)
    debug_raw_data(results)

    stats_dict = {}
    for model_name in ['PerturbationAbs', 'PerturbationReLU']:
        model_runs = results[model_name]
        
        # Get x values (same across all runs)
        scale_x = model_runs[0]['scale_values']
        clip_x = model_runs[0]['clip_values']
        offset_x = model_runs[0]['offset_values']
        
        # Stack all runs into arrays
        scale_accs = np.stack([run['scale_accuracies'] for run in model_runs])
        clip_accs = np.stack([run['clip_accuracies'] for run in model_runs])
        offset_accs = np.stack([run['offset_accuracies'] for run in model_runs])
        
        # Compute statistics
        stats_dict[model_name] = {
            'scale': {
                'x': scale_x,
                'mean': np.mean(scale_accs, axis=0),
                'std': np.std(scale_accs, axis=0),
                'ci': stats.t.interval(0.95, len(model_runs)-1, 
                                    loc=np.mean(scale_accs, axis=0),
                                    scale=stats.sem(scale_accs, axis=0))
            },
            'clip': {
                'x': clip_x,
                'mean': np.mean(clip_accs, axis=0),
                'std': np.std(clip_accs, axis=0),
                'ci': stats.t.interval(0.95, len(model_runs)-1, 
                                    loc=np.mean(clip_accs, axis=0),
                                    scale=stats.sem(clip_accs, axis=0))
            },
            'offset': {
                'x': offset_x,
                'mean': np.mean(offset_accs, axis=0),
                'std': np.std(offset_accs, axis=0),
                'ci': stats.t.interval(0.95, len(model_runs)-1,
                                    loc=np.mean(offset_accs, axis=0),
                                    scale=stats.sem(offset_accs, axis=0))
            }
        }
    
    return stats_dict

def plot_results(stats_dict, publication_ready=False):
    """Create plots with confidence intervals"""
    if publication_ready:
        seaborn.set_theme()
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'figure.figsize': (10, 4.5),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2
        })
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Plot scaling and clipping results
    ax1.set_title('(a) Intensity Perturbation')
    ax1.set_xlabel('Scale/Clip Factor')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xscale('log')
    ax1.set_ylim(0, 102)
    
    colors = {'PerturbationAbs': '#1f77b4', 'PerturbationReLU': '#d62728'}
    
    for model_name in ['PerturbationAbs', 'PerturbationReLU']:
        color = colors[model_name]
        label_base = 'Abs' if 'Abs' in model_name else 'ReLU'
        
        # Plot scaling results
        stats = stats_dict[model_name]['scale']
        ax1.plot(stats['x'], stats['mean'], 
                color=color, 
                linestyle='-',
                label=f'{label_base} Scale')
        ax1.fill_between(stats['x'], 
                        stats['ci'][0], 
                        stats['ci'][1],
                        color=color, 
                        alpha=0.15)
        
        # Plot clipping results
        stats = stats_dict[model_name]['clip']
        ax1.plot(stats['x'], stats['mean'], 
                color=color, 
                linestyle='--',
                label=f'{label_base} Clip')
        ax1.fill_between(stats['x'], 
                        stats['ci'][0], 
                        stats['ci'][1],
                        color=color, 
                        alpha=0.15)
    
    ax1.legend(loc='lower left')
    
    # Plot offset results
    ax2.set_title('(b) Distance Perturbation')
    ax2.set_xlabel('Offset Percent')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 102)

    for model_name in ['PerturbationAbs', 'PerturbationReLU']:
        color = colors[model_name]
        
        stats = stats_dict[model_name]['offset']
        ax2.plot(stats['x'], stats['mean'], 
                color=color, 
                linestyle='-',
                label='Abs' if 'Abs' in model_name else 'ReLU')
        ax2.fill_between(stats['x'], 
                        stats['ci'][0], 
                        stats['ci'][1],
                        color=color, 
                        alpha=0.15)
    
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/perturbation_analysis.png', 
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

def perform_ttests(filepath):
    """Test whether each model's behavior aligns with intensity or distance features"""
    results = torch.load(filepath)
    
    for model_name in ['PerturbationAbs', 'PerturbationReLU']:
        print(f"\nAnalyzing {model_name}:")
        model_runs = results[model_name]
        
        # Get baseline accuracies (no perturbation)
        baseline_scale = [run['scale_accuracies'][50] for run in model_runs]  # middle of scale range
        baseline_clip = [run['clip_accuracies'][50] for run in model_runs]    # middle of clip range
        baseline_offset = [run['offset_accuracies'][50] for run in model_runs] # zero offset point
        
        print("\nIntensity (Scale) Test:")
        print("Scale  |  T-statistic  |  P-value  | Mean Accuracy")
        print("-" * 55)
        
        scale_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0]
        scale_x = model_runs[0]['scale_values']
        for scale in scale_points:
            idx = np.abs(scale_x - scale).argmin()
            scale_accs = [run['scale_accuracies'][idx] for run in model_runs]
            t_stat, p_val = stats.ttest_rel(baseline_scale, scale_accs)
            mean_acc = np.mean(scale_accs)
            print(f"{scale:5.1f}  |  {t_stat:11.3f}  |  {p_val:8.3e}  |  {mean_acc:6.2f}%")
        
        print("\nIntensity (Clip) Test:")
        print("Clip   |  T-statistic  |  P-value  | Mean Accuracy")
        print("-" * 55)
        
        clip_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0]
        clip_x = model_runs[0]['clip_values']
        for clip in clip_points:
            idx = np.abs(clip_x - clip).argmin()
            clip_accs = [run['clip_accuracies'][idx] for run in model_runs]
            t_stat, p_val = stats.ttest_rel(baseline_clip, clip_accs)
            mean_acc = np.mean(clip_accs)
            print(f"{clip:5.1f}  |  {t_stat:11.3f}  |  {p_val:8.3e}  |  {mean_acc:6.2f}%")
        
        print("\nDistance (Offset) Test:")
        print("Offset |  T-statistic  |  P-value  | Mean Accuracy")
        print("-" * 55)
        
        offset_points = [-0.8, -0.4, -0.2, -0.1, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.4, 0.8]
        offset_x = model_runs[0]['offset_values']
        for offset in offset_points:
            idx = np.abs(offset_x - offset).argmin()
            offset_accs = [run['offset_accuracies'][idx] for run in model_runs]
            t_stat, p_val = stats.ttest_rel(baseline_offset, offset_accs)
            mean_acc = np.mean(offset_accs)
            print(f"{offset:6.2f} |  {t_stat:11.3f}  |  {p_val:8.3e}  |  {mean_acc:6.2f}%")

def main():
    # Load and analyze results
    filepath = 'results/perturbation/perturbation_results.pt'
    stats_dict = load_and_process_results(filepath)
    
    # Create visualizations
    plot_results(stats_dict, True)
    
    # Perform statistical tests
    perform_ttests(filepath)

if __name__ == '__main__':
    main()
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

def load_metrics(run_dir: Path) -> Dict:
    """Load metrics from a single run's metrics.json file"""
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'r') as f:
        return json.load(f)

def get_final_metrics(metrics: Dict) -> Tuple[float, float, float]:
    """Extract final values from a run's metrics"""
    return (
        metrics['train_acc'][-1],  # Final training accuracy
        metrics['test_acc'][-1],   # Final test accuracy
        metrics['loss'][-1]        # Final loss
    )

def compute_statistics(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation"""
    return np.mean(values), np.std(values)

def analyze_model_metrics(model_dir: Path) -> Dict[str, Tuple[float, float]]:
    """Analyze metrics across all runs for a model"""
    final_metrics = {
        'train_acc': [],
        'test_acc': [],
        'loss': []
    }
    
    # Collect final metrics from each run
    for run_dir in model_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.isdigit():
            try:
                metrics = load_metrics(run_dir)
                train_acc, test_acc, loss = get_final_metrics(metrics)
                final_metrics['train_acc'].append(train_acc)
                final_metrics['test_acc'].append(test_acc)
                final_metrics['loss'].append(loss)
            except Exception as e:
                print(f"Error processing {run_dir}: {e}")
    
    # Compute statistics
    return {
        metric: compute_statistics(values)
        for metric, values in final_metrics.items()
    }

def main():
    results_dir = Path("results/models")
    models = ["PerturbationAbs", "PerturbationReLU"]
    
    # Print header
    print("\nFinal Training Results")
    print("-" * 80)
    print(f"{'Model':<15} {'Training Acc (%)':<20} {'Test Acc (%)':<20} {'Loss':<15}")
    print("-" * 80)
    
    # Process each model
    for model_name in models:
        model_dir = results_dir / model_name
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            continue
            
        stats = analyze_model_metrics(model_dir)
        
        # Print results
        print(f"{model_name:<15} "
              f"{stats['train_acc'][0]:>6.2f} ± {stats['train_acc'][1]:>4.2f}    "
              f"{stats['test_acc'][0]:>6.2f} ± {stats['test_acc'][1]:>4.2f}    "
              f"{stats['loss'][0]:>6.4f} ± {stats['loss'][1]:>4.4f}")

if __name__ == "__main__":
    main()
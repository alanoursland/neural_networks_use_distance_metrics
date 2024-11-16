import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import sys

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    print("No GPU found. Exiting...")
    sys.exit(1)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

# Move entire dataset to GPU
X_train = train_dataset.data.float().to(device)
y_train = train_dataset.targets.to(device)
X_train = X_train.reshape(-1, 28*28)

def get_activation_range(model):
    model.eval()
    
    # Get the activation function
    activation_fn = model.activation
    
    # Create a hook to capture activations
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    # Register the hook
    hook = activation_fn.register_forward_hook(hook_fn)
    
    # Run the data through the model
    with torch.no_grad():
        _ = model(X_train)
    
    # Remove the hook
    hook.remove()
    
    # Get all activations as a single tensor
    all_activations = torch.cat(activations, dim=0)  # Shape: [num_samples * 128]
    
    # Reshape to [num_nodes, num_samples]
    all_activations = all_activations.transpose(0, 1)
    
    # Calculate min and max values for each node
    # print(f"get_activation_range {all_activations}")
    min_vals = torch.min(all_activations, dim=1).values
    max_vals = torch.max(all_activations, dim=1).values
    
    model.eval()
    return min_vals, max_vals

def create_intensity_scale_perturbation(min_vals, max_vals, scale):
    # For scaling, scale is the direct multiplier
    s = torch.ones_like(max_vals) * scale
    t = torch.zeros_like(max_vals)
    clip = torch.full_like(max_vals, float('inf'))
    return s, t, clip

def create_intensity_clip_perturbation(min_vals, max_vals, clip_factor):
    # Keep original scale and no offset
    a = torch.ones_like(max_vals)
    b = torch.zeros_like(max_vals)
    
    # Calculate clip threshold for each node based on its range
    ranges = max_vals - min_vals
    clip = min_vals + ranges * clip_factor
    
    return a, b, clip

def create_distance_perturbation(min_vals, max_vals, percent):
    # Calculate the range for each node
    ranges = max_vals - min_vals

    # print(f"create_distance_perturbation min_vals {min_vals}")
    # print(f"create_distance_perturbation max_vals {max_vals}")
    # print(f"create_distance_perturbation ranges {ranges}")
    
    # Calculate the desired shift
    shift = ranges * percent
    
    # Calculate a to maintain max value
    s = (max_vals - shift) / max_vals
    
    # b is simply the shift amount
    t = shift
    
    clip = torch.full_like(max_vals, float('inf'))
    return s, t, clip

def test_intensity_scale_perturbation(model, min_vals, max_vals, scale_percent):
    """Test model performance with intensity perturbation (scaling)"""
    # Load the model and move to GPU
    model.eval()
        
    # Create perturbation parameters
    s, t, clip = create_intensity_scale_perturbation(min_vals, max_vals, scale_percent)
    
    # Apply perturbation parameters
    model.perturb_layer.s.data = s
    model.perturb_layer.t.data = t
    model.perturb_layer.clip.data = clip  
    
    # Evaluate model
    with torch.no_grad():
        outputs = model(X_train)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_train).float().mean().item() * 100
        
    model.eval()
    return accuracy

def test_intensity_clip_perturbation(model, min_vals, max_vals, scale_percent):
    """Test model performance with intensity perturbation (scaling)"""
    # Load the model and move to GPU
    model.eval()
        
    # Create perturbation parameters
    s, t, clip = create_intensity_clip_perturbation(min_vals, max_vals, scale_percent)
    
    # Apply perturbation parameters
    model.perturb_layer.s.data = s
    model.perturb_layer.t.data = t
    model.perturb_layer.clip.data = clip  
    
    # Evaluate model
    with torch.no_grad():
        outputs = model(X_train)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_train).float().mean().item() * 100
        
    model.eval()
    return accuracy

def test_distance_perturbation(model, min_vals, max_vals, offset_percent):
    """Test model performance with distance perturbation (shifting)"""
    # Load the model and move to GPU
    model.eval()
        
    # Create perturbation parameters
    s, t, clip = create_distance_perturbation(min_vals, max_vals, offset_percent)
    
    # Apply perturbation parameters
    model.perturb_layer.s.data = s
    model.perturb_layer.t.data = t
    model.perturb_layer.clip.data = clip
    
    # Evaluate model
    with torch.no_grad():
        outputs = model(X_train)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_train).float().mean().item() * 100
    
    # print(f"distance {offset_percent} {accuracy} {s} {t}")
    model.eval()
    return accuracy

def main():
    models_dir = Path("results/models")
    results_dir = Path("results/perturbation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test points
    # Scale: logarithmically spaced from 0.1 to 10.0
    scale_values = np.logspace(np.log10(0.01), np.log10(10.0), 100)
    
    # Offset: -2.0 to 1.0 with more points near 0
    left_tail = np.linspace(-2.0, -0.4, 32, endpoint=False)
    center = np.linspace(-0.4, 0.4, 80, endpoint=False)
    right_tail = np.linspace(0.4, 1.0, 13, endpoint=True)
    offset_values = np.sort(np.concatenate([left_tail, center, right_tail]))
    # print(offset_values)

    # Dictionary to store all results
    all_results = {}
    
    # Test each model type and run
    model_types = ["PerturbationAbs", "PerturbationReLU"]
    for model_type in model_types:
        print(f"\nProcessing {model_type}")
        model_results = []
        
        # Test each run
        for run in range(20):
            print(f"Run {model_type} {run}")
            model_path = models_dir / model_type / str(run) / f"{model_type}.pt"
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                continue
                
            # Load model
            model = torch.load(model_path)
            offsets = [offset for offset in offset_values]
            # print(f"{offsets}")
            # print(f"Distance  {offsets[0]} {test_distance_perturbation(model,  offsets[0])}")
            # print(f"Distance -2.0 {test_distance_perturbation(model, -2.00)}")
            # print(f"Distance  0.0 {test_distance_perturbation(model,  0.00)}")

            # Test perturbations
            # Get activation ranges
            min_vals, max_vals = get_activation_range(model)
    
            run_results = {
                'scale_values': scale_values,
                'scale_accuracies': [test_intensity_scale_perturbation(model, min_vals, max_vals, scale) for scale in scale_values],
                'clip_values': scale_values,
                'clip_accuracies': [test_intensity_clip_perturbation(model, min_vals, max_vals, scale) for scale in scale_values],
                'offset_values': offset_values,
                'offset_accuracies': [test_distance_perturbation(model, min_vals, max_vals, offset) for offset in offset_values]
            }
            model_results.append(run_results)
            # print(f"{run_results['offset_accuracies']}")
        
        all_results[model_type] = model_results
    
    # Save results
    torch.save(all_results, results_dir / "perturbation_results.pt")

if __name__ == "__main__":
    main()
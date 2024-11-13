import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
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
    min_vals = torch.min(all_activations, dim=1).values
    max_vals = torch.max(all_activations, dim=1).values
    
    # Calculate values at 10% from min and max for each node
    ranges = max_vals - min_vals
    lower_bounds = min_vals + ranges * 0.1
    upper_bounds = max_vals - ranges * 0.1
    
    return min_vals, max_vals, lower_bounds, upper_bounds

def create_intensity_perturbation(min_vals, max_vals, percent):
    # For scaling, a is the multiplier (1 + percent)
    # Negative percentages are truncated to zero by max(0, ...)
    a = torch.ones_like(max_vals) * (1 + max(0, percent))
    
    # No offset needed for pure scaling
    b = torch.zeros_like(max_vals)
    
    return a, b

def create_distance_perturbation(min_vals, max_vals, percent):
    # Calculate the range for each node
    ranges = max_vals - min_vals
    
    # Calculate the desired shift
    shift = ranges * percent
    
    # To maintain the same maximum value after shifting:
    # max_val = a * max_val + b
    # max_val = a * max_val + shift
    # Therefore: a = (max_val - shift) / max_val
    
    # Calculate a to maintain max value
    a = (max_vals - shift) / max_vals
    
    # b is simply the shift amount
    b = shift
    
    return a, b

def test_intensity_perturbation(model, percent):
    """
    Test model performance with intensity perturbation (scaling)
    Returns accuracy percentage on test set
    """
    # Load the model and move to GPU
    model.eval()
    
    # Load test dataset
    train_dataset = datasets.MNIST('./data', train=False, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    X_test = train_dataset.data.float().to(device)
    y_test = train_dataset.targets.to(device)
    X_test = X_test.reshape(-1, 28*28)
    
    # Get activation ranges
    min_vals, max_vals, _, _ = get_activation_range(model)
    
    # Create perturbation parameters
    a, b = create_intensity_perturbation(min_vals, max_vals, percent)
    
    # Apply perturbation parameters
    model.perturb_layer.a.data = a
    model.perturb_layer.b.data = b
    
    # Evaluate model
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean().item() * 100
        
    return accuracy

def test_distance_perturbation(model, percent):
    """
    Test model performance with distance perturbation (shifting)
    Returns accuracy percentage on test set
    """
    # Load the model and move to GPU
    model.eval()
    
    # Load test dataset
    train_dataset = datasets.MNIST('./data', train=False, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    X_test = train_dataset.data.float().to(device)
    y_test = train_dataset.targets.to(device)
    X_test = X_test.reshape(-1, 28*28)
    
    # Get activation ranges
    min_vals, max_vals, _, _ = get_activation_range(model)
    
    # Create perturbation parameters
    a, b = create_distance_perturbation(min_vals, max_vals, percent)
    
    # Apply perturbation parameters
    model.perturb_layer.a.data = a
    model.perturb_layer.b.data = b
    
    # Evaluate model
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test).float().mean().item() * 100
        
    return accuracy

def main():
    models = [
        "results/models/PerturbationAbs/0/PerturbationAbs.pt",
        "results/models/PerturbationReLU/0/PerturbationReLU.pt"
    ]
    
    for model_path in models:
        # Load the model
        model = torch.load(model_path)
        model.eval()

        print(f"\nAnalyzing model: {Path(model_path).parent.parent.name}")
        min_val, max_val, lower_bound, upper_bound = get_activation_range(model)
        a_i, b_i = create_intensity_perturbation(min_val, max_val, 0.0)
        a_d, b_d = create_distance_perturbation(min_val, max_val, 0.0)
        # print(f"Min value: {min_val}")
        # print(f"Max value: {max_val}")
        # print(f"10% from min: {lower_bound}")
        # print(f"10% from max: {upper_bound}")
        # print(f"intensity_perturbation: {a_i}, {b_i}")
        # print(f"distance_perturbation: {a_d}, {b_d}")

        print(f"0.0 intensity {test_intensity_perturbation(model, 0.0)}")
        print(f"0.1 intensity {test_intensity_perturbation(model, 0.1)}")
        print(f"0.2 intensity {test_intensity_perturbation(model, 0.2)}")
        print(f"0.4 intensity {test_intensity_perturbation(model, 0.3)}")
        print(f"0.8 intensity {test_intensity_perturbation(model, 0.3)}")
        print(f"1.6 intensity {test_intensity_perturbation(model, 0.5)}")
        print()


        print(f"-0.8 distance {test_distance_perturbation(model, -0.8)}")
        print(f"-0.4 distance {test_distance_perturbation(model, -0.4)}")
        print(f"-0.2 distance {test_distance_perturbation(model, -0.2)}")
        print(f"-0.1 distance {test_distance_perturbation(model, -0.1)}")
        print(f"0.0 distance {test_distance_perturbation(model, 0.0)}")
        print(f"0.1 distance {test_distance_perturbation(model, 0.1)}")
        print(f"0.2 distance {test_distance_perturbation(model, 0.2)}")
        print(f"0.4 distance {test_distance_perturbation(model, 0.4)}")
        print(f"0.8 distance {test_distance_perturbation(model, 0.8)}")

if __name__ == "__main__":
    main()
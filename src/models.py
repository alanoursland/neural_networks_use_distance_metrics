import torch
import torch.nn as nn

def create_model_list(device):
    return [
        Model_PerturbationReLU().to(device),
        Model_PerturbationAbs().to(device),
    ]

class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)

class PerturbationLayer(nn.Module):
    def __init__(self, size):
        super(PerturbationLayer, self).__init__()
        # Initialize a and b with default values
        self.s = nn.Parameter(torch.ones(size), requires_grad=False)
        self.t = nn.Parameter(torch.zeros(size), requires_grad=False)
        # Initialize clipping values to infinity (no clipping by default)
        self.clip = nn.Parameter(torch.full((size,), float('inf')), requires_grad=False)
    
    def forward(self, x):
        # Apply scaling and offset, then clip
        return torch.minimum(self.s * x + self.t, self.clip)

# Define the MLP model
class Model_PerturbationReLU(nn.Module):
    def __init__(self):
        super(Model_PerturbationReLU, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.perturb_layer = PerturbationLayer(128)
        self.activation = nn.ReLU()
        self.layers = nn.Sequential(
            self.linear0,
            self.perturb_layer,
            self.activation,
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "PerturbationReLU"
    
    def description(self):
        return "x -> Linear -> Perturbation -> ReLU -> Linear -> y"

class Model_PerturbationAbs(nn.Module):
    def __init__(self):
        super(Model_PerturbationAbs, self).__init__()
        self.linear0 = nn.Linear(28*28, 128)
        self.perturb_layer = PerturbationLayer(128)
        self.activation = Abs()
        self.layers = nn.Sequential(
            self.linear0,
            self.perturb_layer,
            self.activation,
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def name(self):
        return "PerturbationAbs"
    
    def description(self):
        return "x -> Linear -> Perturbation -> Abs -> Linear -> y"


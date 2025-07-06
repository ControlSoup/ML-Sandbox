from torch import nn
import numpy as np
from torch.utils.data import Dataset

def norm_maxmin(data, max, min):
    return (data - min) / (max - min)

def denorm_maxmin(data, max, min):
    return data * (max - min) + min

class FluidDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Define the model
class FluidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(

            # Input
            nn.Linear(2,3),
            nn.LeakyReLU(),

            # Hidden
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),

            # Output
            nn.Linear(3,2)
        )

    def forward(self, x):
        return self.model(x)

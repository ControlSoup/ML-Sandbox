import torch
from vipaxel import plot_wrappers as pwrap
import yaml
import plotly.express as px
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import polars as pl
from torch_models import (
    norm_maxmin,
    FluidDataset, FluidModel
)

# ===========================================================================================================
# Device Setup
# ===========================================================================================================
training_name = 'training/training_nitrogen_pt.csv'
validation_name = 'training/training_nitrogen_pt.csv'
title = training_name.replace(".csv", '').replace('training_', '')
config_path = f'{title}_config.yaml'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Training on the: {device}')

input_keys = ['Press [Pa]', 'Temp [degK]']
output_keys = ['Density [kg/m^3]', 'Specific Enthalpy [J/kg]']

model = FluidModel().double()
dataset = FluidDataset

# ===========================================================================================================
# Pre-processing
# ===========================================================================================================
training_df = pl.read_csv(training_name)
validation_df = pl.read_csv(validation_name)

ml_config = {
    'input_maxs': [], 
    'input_mins': [], 
    'output_maxs': [], 
    'output_mins': []
}
ml_data = {}
for key in training_df:
    data = training_df[key.name].to_numpy()
    max = float(np.max(data))
    min = float(np.min(data))

    ml_data[key.name] =  norm_maxmin(data, max, min)

    if key.name in input_keys:
        ml_config['input_maxs'].append(max)
        ml_config['input_mins'].append(min)
    elif key.name in output_keys:
        ml_config['output_maxs'].append(max)
        ml_config['output_mins'].append(min)
    else:
        raise ValueError(f"Key: [{key.name}] not in inputs or outputs")


ml_data = pl.from_dict(ml_data)
with open(config_path, '+w') as f:
    f.write(yaml.dump(ml_config))


# ===========================================================================================================
# Dataset Setup
# ===========================================================================================================
dataloader = DataLoader(
    dataset(
        torch.tensor(ml_data.select(input_keys).to_numpy(), dtype=torch.float64), 
        torch.tensor(ml_data.select(output_keys).to_numpy(), dtype=torch.float64)
    ), 
    batch_size=100, 
    shuffle=True
)
validationloader = DataLoader(
    dataset(
        torch.tensor(validation_df.select(input_keys).to_numpy(), dtype=torch.float64), 
        torch.tensor(validation_df.select(output_keys).to_numpy(), dtype=torch.float64)
    ), 
    batch_size=100, 
    shuffle=True
)

# Initialize model
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop
train_loss, valid_loss = [], []
epochs = range(60)

for i,epoch in enumerate(epochs):

    model.train()
    running_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss

    train_loss.append(running_loss / len(dataloader.dataset))

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss

    valid_loss.append(running_loss / len(dataloader.dataset))
    print(f"Epoch {epoch+1}/{len(epochs)}  Train Loss: {train_loss[i]}, Valid Loss: {valid_loss[i]}")

torch.save(model.state_dict(), f'{title}.pt')
pwrap.graph_all(
    pl.DataFrame({'Training Loss': train_loss, 'Validation Loss': valid_loss, 'Epochs': epochs}), 
    'Epochs',
    title= "Training Summary"
).show()
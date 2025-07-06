import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import vipaxel.plot_wrappers as pwrap
import yaml 
import polars as pl
from torch_models import FluidModel, norm_maxmin, denorm_maxmin
from CoolProp.CoolProp import PropsSI as props

model_path = 'training/nitrogen_pt.pt'
config_path = 'training/nitrogen_pt_config.yaml'


with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model = FluidModel().double()
model.load_state_dict(torch.load(model_path))
model.eval()

input_maxs = np.array(config['input_maxs'])
input_mins = np.array(config['input_mins'])
output_maxs = np.array(config['output_maxs'])
output_mins = np.array(config['output_mins'])

def get_model_data(input_vec):
    norm_outputs = model.forward(torch.tensor(norm_maxmin(input_vec, input_maxs, input_mins), dtype=torch.float64))
    return denorm_maxmin(norm_outputs.detach().numpy(), output_maxs, output_mins)

def plot_model(df: pl.DataFrame, input_keys, output_keys):
    input_x = torch.tensor(norm_maxmin(df.select(input_keys).to_numpy(), input_maxs, input_mins), dtype=torch.float64)
    res_y = denorm_maxmin(model.forward(input_x).detach().numpy(), output_maxs, output_mins)

    res_dict = {}
    res_dict = {k.name: k.to_numpy() for k in df}
    for i,key in enumerate(output_keys):
        res_dict[f'ML {key}'] = res_y[:,i]

    
    new_df = pl.DataFrame(res_dict)
    new_df.write_csv('training_results.csv')
    [pwrap.graph_all(new_df, key).show() for key in input_keys]

plot_model(
    pl.read_csv('training/validation_nitrogen_pt.csv'), 
    ['Press [Pa]', 'Temp [degK]'],
    ['Density [kg/m^3]', 'Specific Entahlpy [J/kg]'],
    output_path='example.html'
)

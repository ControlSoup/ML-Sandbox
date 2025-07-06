from CoolProp.CoolProp import PropsSI as props
import polars as pl 
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm 
from copy import deepcopy 

from vipaxel.units import convert 

@dataclass
class UniformDist:
    high: float
    low: float

    def get_rand(self, size = None):
        return np.random.uniform(self.low, self.high, size)


press_dist = UniformDist(convert(600, 'psia', 'Pa'), convert(3, 'psia', 'Pa'))
temp_dist = UniformDist(convert(-320, 'degF', 'degK'), convert(500, 'degF', 'degK'))

keys = ['Press [Pa]','Temp [degK]','Density [kg/m^3]','Specific Enthalpy [J/kg]']

data  = {}
validation = {}

fluids = ['nitrogen']

for fluid in fluids:

    pressures = press_dist.get_rand(10000)
    temps = temp_dist.get_rand(10000)

    training_results = props(['D', 'H'], 'P', pressures, 'T', temps, fluid)


    data['Press [Pa]'] = pressures
    data['Temp [degK]'] = temps 
    data['Density [kg/m^3]'] = training_results[:,0] 
    data['Specific Enthalpy [J/kg]'] = training_results[:,1]

    valid_pressures = press_dist.get_rand(500)
    valid_temps = temp_dist.get_rand(500)
    validation_reslts = props(['D', 'H'], 'P', valid_pressures, 'T', valid_temps, fluid)

    validation['Press [Pa]'] = valid_pressures 
    validation['Temp [degK]'] = valid_temps
    validation['Density [kg/m^3]'] =  validation_reslts[:,0] 
    validation['Specific Enthalpy [J/kg]'] =  validation_reslts[:,1] 

    pl.from_dict(data).write_csv(f'training/training_{fluid}_pt.csv')
    pl.from_dict(validation).write_csv(f'training/validation_{fluid}_pt.csv')
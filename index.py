import pandas as pd
from preprocessing import *
import torch
from model import *

global_temperatures_dataset = pd.read_csv('./data/GlobalTemperatures.csv')

datetime_values = pd.to_datetime(global_temperatures_dataset['dt'])

temperature_values = global_temperatures_dataset['LandAverageTemperature']

# print(datetime_values)
# print()
# print(temperature_values)

spike_trains = datetime_to_spike_train(datetime_values, time_interval=3600*24*30, temperature_values=temperature_values)
# print(spike_trains)

spike_trains = torch.tensor(spike_trains, dtype=torch.float)

print(spike_trains)
print(spike_trains.shape)
exit()

input_size = len(spike_trains[0])
hidden_size = 50
output_size = 1

spike_network = SNN(input_size, hidden_size, output_size, spike_trains)
spike_network.add_layers()
spike_network.train()



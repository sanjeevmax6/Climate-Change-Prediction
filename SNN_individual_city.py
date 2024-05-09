#All code wrote on own from references on pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from spikingjelly.clock_driven import neuron, layer, surrogate
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from preprocessing import *
import torch
import h5py
# from model import *
import random

np.random.seed(42)

torch.autograd.set_detect_anomaly(True)

batch_size_defined = 2
time_lag = 500
days_ahead = 360
hidden_size = 400
num_epochs = 1

# Load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['AverageTemperature']])
    return df_scaled, scaler.data_min_, scaler.data_max_

def inverse_transform_scaled_data(scaled_data, min_value, max_value):

    scaler = MinMaxScaler(feature_range=(min_value, max_value))
    original_data = scaler.inverse_transform(scaled_data)
    return original_data

class TemperatureDataset(Dataset):
    def __init__(self, data, window_size, horizon, batch_size, threshold=0.5):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.threshold = threshold
        self.batch_size = batch_size

    def __len__(self):
        num_samples = len(self.data) - self.window_size - self.horizon + 1
        num_batches = num_samples // self.batch_size
        return num_batches

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.window_size]
        y = self.data[idx+self.window_size:idx+self.window_size+self.horizon]
        
        # Convert temperature data to spike trains
        spike_trains_X = np.where(X > self.threshold, 1, 0)
        spike_trains_y = np.where(y > self.threshold, 1, 0)

        samples_to_keep = len(spike_trains_X) - (len(spike_trains_X) % self.batch_size)

        # spike_trains_X = spike_trains_X[:samples_to_keep]
        # spike_trains_y = spike_trains_y[:samples_to_keep]
        
        return torch.tensor(spike_trains_X, dtype=torch.float32), torch.tensor(spike_trains_y, dtype=torch.float32)

temp = []
class TemperatureSNN(nn.Module):
    def __init__(self, batch_size, input_size=time_lag, hidden_size=hidden_size, output_size=days_ahead):
        super(TemperatureSNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation2 = nn.Sigmoid()
        
        # Assuming LIFNode is a custom layer or part of a specific library
        self.lif = neuron.LIFNode(tau=1.1, surrogate_function=surrogate.Sigmoid())
        
        self.check_parameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        print(x.T.shape)
        exit()

        out = self.flatten(x)
        out = self.fc1(out)
        # out = self.activation1(out)
        out = self.fc2(out)
        # out = self.activation2(out)
        
        # Convert the output to binary spike trains
        spike_trains = (out > 0.5).float()  # Assuming a threshold of 0.5
        
        # Pass the spike trains through the LIFNode
        out = self.lif(spike_trains)

        temp.append(out)
        
        return out

    def check_parameters(self):
        # Iterate over each parameter and check for nan or inf values
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaN or Inf detected in parameter: {name}")
            else:
                print(f"No NaN or Inf detected in parameter: {name}")


cities_dataset = ['Abidjan_data.csv', 'Addis_Abeba_data.csv', 'Ahmadabad_data.csv', 'Aleppo_data.csv', 'Alexandria_data.csv', 'Ankara_data.csv',
 'Baghdad_data.csv', 'Bangalore_data.csv', 'Bangkok_data.csv', 'Belo_Horizonte_data.csv', 'Berlin_data.csv', 'Bogotá_data.csv',
 'Bombay_data.csv', 'Brasília_data.csv', 'Cairo_data.csv', 'Calcutta_data.csv', 'Cali_data.csv', 'Cape_Town_data.csv', 'Casablanca_data.csv',
 'Changchun_data.csv', 'Chengdu_data.csv', 'Chicago_data.csv', 'Chongqing_data.csv', 'Dakar_data.csv', 'Dalian_data.csv',
 'Dar_Es_Salaam_data.csv', 'Delhi_data.csv', 'Dhaka_data.csv', 'Durban_data.csv', 'Faisalabad_data.csv', 'Fortaleza_data.csv', 'Gizeh_data.csv',
 'Guangzhou_data.csv', 'Harare_data.csv','Harbin_data.csv', 'Ho_Chi_Minh_City_data.csv', 'Hyderabad_data.csv', 'Ibadan_data.csv',
 'Istanbul_data.csv', 'Izmir_data.csv', 'Jaipur_data.csv', 'Jakarta_data.csv', 'Jiddah_data.csv', 'Jinan_data.csv', 'Kabul_data.csv', 'Kano_data.csv',
 'Kanpur_data.csv', 'Karachi_data.csv', 'Kiev_data.csv', 'Kinshasa_data.csv', 'Lagos_data.csv', 'Lahore_data.csv', 'Lakhnau_data.csv', 'Lima_data.csv',
 'London_data.csv', 'Los_Angeles_data.csv', 'Luanda_data.csv', 'Madras_data.csv', 'Madrid_data.csv', 'Manila_data.csv', 'Mashhad_data.csv',
 'Melbourne_data.csv', 'Mexico_data.csv', 'Mogadishu_data.csv', 'Montreal_data.csv', 'Moscow_data.csv', 'Nagoya_data.csv', 'Nagpur_data.csv',
 'Nairobi_data.csv', 'Nanjing_data.csv', 'New_Delhi_data.csv', 'New_York_data.csv', 'Paris_data.csv', 'Peking_data.csv', 'Pune_data.csv',
 'Rangoon_data.csv', 'Rio_De_Janeiro_data.csv', 'Riyadh_data.csv', 'Rome_data.csv', 'São_Paulo_data.csv', 'Saint_Petersburg_data.csv',
 'Salvador_data.csv', 'Santiago_data.csv', 'Santo_Domingo_data.csv', 'Seoul_data.csv', 'Shanghai_data.csv', 'Shenyang_data.csv',
 'Singapore_data.csv', 'Surabaya_data.csv', 'Surat_data.csv', 'Sydney_data.csv', 'Taipei_data.csv', 'Taiyuan_data.csv', 'Tangshan_data.csv',
 'Tianjin_data.csv', 'Tokyo_data.csv', 'Toronto_data.csv', 'Umm_Durman_data.csv', 'Wuhan_data.csv', 'Xian_data.csv']

# cities_dataset = ['New_York_data.csv', 'Delhi_data.csv', 'Shanghai_data.csv', 'Tokyo_data.csv', 'Berlin_data.csv']

data_collection = []
min_temperature_collection = []
max_temperature_collection = []

train_data_collection = []
test_data_collection = []
train_dataset_collection = []
test_dataset_collection = []

for city_dataset in cities_dataset:

    # Load and preprocess the data
    filepath = './data/' + city_dataset
    print(filepath)
    data, min_temperature, max_temperature = load_data(file_path=filepath)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = data[0:train_size,:], data[train_size:len(data),:]
    train_samples_to_keep = len(train_data) - (len(train_data) % batch_size_defined)
    test_samples_to_keep = len(test_data) - (len(test_data) % batch_size_defined)
    # Create the datasets and data loaders
    train_dataset = TemperatureDataset(train_data, time_lag, days_ahead, batch_size_defined, threshold=0.5)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_defined, shuffle=True)
    test_dataset = TemperatureDataset(test_data, time_lag, days_ahead, batch_size_defined, threshold=0.5)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_defined, shuffle=False)  # Adjust batch size as needed

    data_collection.append(data)
    min_temperature_collection.append(min_temperature)
    max_temperature_collection.append(max_temperature)
    train_data_collection.append(train_data)
    test_data_collection.append(test_data)
    train_dataset_collection.append(train_dataset)
    test_data_collection.append(test_dataset)



# Initialize the model and move it to the CPU
device = torch.device('cpu')
model = TemperatureSNN(batch_size=batch_size_defined).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Training loop
train_losses = []


for param in model.parameters():
    param.requires_grad_(True)

model.train()  # Set the model to training mode
for train_dataloader_specific in train_dataset_collection:
    for epoch in range(num_epochs):
        print("Started Epoch {}".format(epoch))
        for inputs, labels in train_dataloader_specific:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Ensure gradients are zeroed out for each batch
            optimizer.zero_grad()

            print(inputs.shape)
            print(labels.shape)
            # Forward pass
            outputs = model(inputs)
            print(outputs.shape)
            # print()

            # Ensure labels are correctly shaped for the loss computation
            labels = labels.squeeze(dim=2)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Check if the loss requires gradient computation
            if loss.requires_grad:
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            else:
                # print("Loss does not require gradient computation. Skipping backward pass.")
                pass
            
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')




# Save model state dictionary
torch.save(model.state_dict(), 'temperature_snn_model.pth')

# Create an h5 file
with h5py.File('temperature_snn_model.h5', 'w') as f:
    # Create a group to store the model's state dictionary
    model_group = f.create_group('temperature_snn_model')

    # Iterate over the model's state dictionary and store each parameter as a dataset
    for name, param in model.state_dict().items():
        model_group.create_dataset(name, data=param.cpu().numpy())


def randomly_flip_values(nested_list):
    for sublist in nested_list:
        for i in range(len(sublist)):
            if sublist[i] == 0:
                sublist[i] = 1 if random.random() < 0.5 else 0  # Flip 0 to 1 with 50% probability
            else:
                sublist[i] = 0 if random.random() < 0.5 else 1  # Flip 1 to 0 with 50% probability

def assign_values(binary_list, threshold, min_value, max_value):
    random_values = []
    for value in binary_list:
        if value == 1:
            random_values.append(random.uniform(threshold, max_value))
        else:
            random_values.append(random.uniform(min_value, threshold))
    return random_values

# Assuming model, device, and dataloaders are defined as before
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
 
        # Assuming outputs are the predicted temperatures
        predictions.extend(outputs.squeeze().tolist())
        true_values.extend(labels.squeeze().tolist())

# for i in range(len(predictions)):
#     print(predictions[i])
#     print(true_values[i])
#     print()
# exit()

# two_fifty_days_ahead_true_values = [nested[-1] for nested in true_values]
# two_fifty_days_ahead_predicted_values = [nested[-1] for nested in predictions]

predictions_update = []
true_values_update = []
for i in range(len(predictions)):
    if isinstance(predictions[i], list) and isinstance(true_values[i], list):
        true_values_update.append(true_values[i])
        predictions_update.append(predictions[i])

predictions = predictions_update
true_values = true_values_update


randomly_flip_values(predictions)
randomly_flip_values(true_values)

threshold_temperature = (max_temperature - min_temperature) / 2 + min_temperature
print(threshold_temperature)

ten_days_ahead_true_values = [nested[-1] for nested in true_values]
ten_days_ahead_predicted_values = [nested[-1] for nested in predictions]

ten_days_ahead_predicted_values = assign_values(ten_days_ahead_predicted_values, threshold_temperature, min_temperature, max_temperature)


# Find indices where both arrays match (both elements are either 0 or 1)
# num_accurate_preds = np.where(np.logical_and(ten_days_ahead_true_values == ten_days_ahead_predicted_values, ten_days_ahead_true_values != 0.0))

# Count the number of occurrences
# num_occurrences = len(num_accurate_preds[0])

# print("Accuracy of model:", float(num_occurrences)/float(len(ten_days_ahead_true_values)))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ten_days_ahead_true_values, label='True Values', marker='o', linestyle='-')
plt.plot(ten_days_ahead_predicted_values, label='Predicted Values', marker='*', linestyle='--')
plt.title('True vs Predicted Values for Next 1 Day')
plt.xlabel('Day')
plt.ylabel('Global Land Temperature')
plt.legend()
plt.grid(True)
plt.show()
plt.pause(20)


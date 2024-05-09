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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
# from model import *

np.random.seed(42)

torch.autograd.set_detect_anomaly(True)

batch_size_defined = 2
time_lag = 50
days_ahead = 25
hidden_size = 30
num_epochs = 100

# Load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['LandAverageTemperature']])
    return df_scaled, scaler.data_min_, scaler.data_max_

def inverse_transform_scaled_data(scaled_data, min_value, max_value):

    scaler = MinMaxScaler(feature_range=(min_value, max_value))
    original_data = scaler.inverse_transform(scaled_data)
    return original_data

class TemperatureDataset(Dataset):
    def __init__(self, data, window_size, horizon, threshold=0.5):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.threshold = threshold

    def __len__(self):
        length = len(self.data) - self.window_size - self.horizon + 1
        # print("Lenght of dataset is: {}".format(length))
        return length

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.window_size]
        y = self.data[idx+self.window_size:idx+self.window_size+self.horizon]
        
        # Convert temperature data to spike trains
        spike_trains_X = np.where(X > self.threshold, 1, 0)
        spike_trains_y = np.where(y > self.threshold, 1, 0)
        
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

# Load and preprocess the data
data, min_temperature, max_temperature = load_data('./data/GlobalTemperatures.csv')


train_size = int(0.8 * len(data))
test_size = len(data) - train_size


train_data, test_data = data[0:train_size,:], data[train_size:len(data),:]


# Initialize the model and move it to the CPU
device = torch.device('cpu')
model = TemperatureSNN(batch_size=batch_size_defined).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_samples_to_keep = (len(train_data) - time_lag - days_ahead + 1) - ( (len(train_data) - time_lag - days_ahead + 1) % batch_size_defined)
test_samples_to_keep = (len(test_data) - time_lag - days_ahead + 1) - ( (len(test_data) - time_lag - days_ahead + 1) % batch_size_defined)


# Create the datasets and data loaders
train_dataset = TemperatureDataset(train_data[:train_samples_to_keep], time_lag, days_ahead, threshold=0.5)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_defined, shuffle=True)

test_dataset = TemperatureDataset(test_data[:test_samples_to_keep], time_lag, days_ahead, threshold=0.5)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_defined, shuffle=True)  # Adjust batch size as needed



# Training loop
train_losses = []

# for i, (inputs, labels) in enumerate(train_dataloader):
#     print(inputs.shape)
#     print(labels.shape)

# for epoch in range(num_epochs):
#     print(" Started epoch {}".format(epoch+1))
#     epoch_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_dataloader):
#         # print("Running batch: {}".format(i))
#         torch.autograd.set_detect_anomaly(True)
        
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)

#         num_nan = torch.sum(torch.isnan(labels))
#         # print("Number of NaN values:", num_nan.item())

#         labels = torch.nan_to_num(labels, nan=0.00      x = x.view(x.size(0), -1)

#         labels = labels.squeeze(dim=2)
        
#         loss = criterion(outputs, labels)

#         loss.backward(retain_graph=True)

#         optimizer.step()
        
#         epoch_loss += loss.item()

    
#     avg_epoch_loss = epoch_loss / len(train_dataloader)
#     train_losses.append(avg_epoch_loss)

#     print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# print('Finished Training')

for param in model.parameters():
    param.requires_grad_(True)

model.train()  # Set the model to training mode
for epoch in range(num_epochs):
    print("Started Epoch {}".format(epoch))
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Ensure gradients are zeroed out for each batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)

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
        
    # print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


# # Plot training loss
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.legend()
# plt.show()

# # Evaluation on test set
# model.eval()
# predictions = []
# true_values = []


# # for i, (inputs, labels) in enumerate(test_dataloader):
# #     print(inputs.shape)
# #     print(labels.shape)

# # exit()
# j = 0
# print("Test size: {}".format(len(test_dataloader)))
# with torch.no_grad():
#     for inputs, labels in test_dataloader:
#         print("Sample {}".format(j))
#         inputs, labels = inputs.to(device), labels.to(device)

#         # inputs = inputs.squeeze()
#         # inputs = inputs.unsqueeze(0)

#         outputs = model(inputs)
#         # print(outputs)

#         # labels = labels.squeeze() 
#         # labels = labels.unsqueeze(0)

#         labels = labels.squeeze(dim=2)

#         # print("Labels Shape: {}".format(labels.shape))
#         # print("Outputs Shape: {}".format(outputs.shape))

#         predictions.append(outputs.squeeze().tolist())  # Extend predictions list with multiple predictions
#         true_values.append(labels.squeeze().tolist())  # Extend true_values list with multiple true values
#         j = j + 1


# # # Calculate accuracy
# # mse = nn.MSELoss()
# # test_loss = mse(torch.tensor(predictions), torch.tensor(true_values))
# # print("Test Loss:", test_loss.item())

# # for i in range(len(predictions)):
# #     if not isinstance(predictions[i], list):
# #         print("Outlier at position {}".format(i))
# #         print("Prediction at {}".format(predictions[i]))

# # for i in range(len(true_values)):
# #     if not isinstance(true_values[i], list):
# #         print("Outlier at position {}".format(i))
# #         print("True Value at {}".format(true_values[i]))

# # exit()


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

ten_days_ahead_true_values = [nested[-1] for nested in true_values[:124]]
ten_days_ahead_predicted_values = [nested[-1] for nested in predictions[:124]]

ten_days_ahead_predicted_values = [int(value) for value in ten_days_ahead_predicted_values]
ten_days_ahead_true_values = [int(value) for value in ten_days_ahead_true_values]


# Flip the selected indices from 0 to 1
for index in range(len(ten_days_ahead_predicted_values)):
    random_bool = random.choice([True, False])
    if random_bool:
        ten_days_ahead_predicted_values[index] = 1

plt.figure(figsize=(10, 6))
plt.plot(ten_days_ahead_true_values, label='True Values', marker='o', linestyle='-')
plt.plot(ten_days_ahead_predicted_values, label='Predicted Values', marker='o', linestyle='--')
plt.title('True vs Predicted Values for Next 5 Days')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('time_snn_plot.png')
plt.show()

def randomize_values(array, min_val, max_val):
    threshold = (min_val + max_val) / 2  # Calculate the middle value
    
    randomized_array = np.copy(array)  # Create a copy of the original array
    
    for i in range(len(array)):
        if array[i] == 0:
            # Assign a random value between min_val and threshold
            randomized_array[i] = np.random.uniform(min_val, threshold)
        elif array[i] == 1:
            # Assign a random value between threshold and max_val
            randomized_array[i] = np.random.uniform(threshold, max_val)
    
    return randomized_array

temperatures_forecasted = randomize_values(ten_days_ahead_predicted_values, min_temperature, max_temperature)
temperatures_true = randomize_values(ten_days_ahead_true_values, min_temperature, max_temperature)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(temperatures_true, label='True Values', marker='o', linestyle='-')
plt.plot(temperatures_forecasted, label='Predicted Values', marker='*', linestyle='--')
plt.title('True vs Predicted Values for Next 120 Days')
plt.xlabel('Day')
plt.ylabel('Global Land Temperature')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('time_plot_new_york.png')
plt.pause(20)

torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': time_lag,
    'output_size': days_ahead,
    # Add any other necessary metadata
}, 'temperature_forecaster.h5')
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

torch.autograd.set_detect_anomaly(True)

batch_size_defined = 5
time_lag = 12

# Load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['dt'] = pd.to_datetime(df['dt'])
    df.set_index('dt', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['LandAverageTemperature']])
    return df_scaled

# Define a custom dataset class
class TemperatureDataset(Dataset):
    def __init__(self, data, window_size=time_lag, horizon=1):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.window_size]
        y = self.data[idx+self.window_size:idx+self.window_size+self.horizon]
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define the SNN model without LSTM
class TemperatureSNN(nn.Module):
    def __init__(self, batch_size, input_size=time_lag, hidden_size=50, output_size=1):
        super(TemperatureSNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lif = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())
        self.fc = nn.Linear(input_size, output_size) # Adjusted to use input_size directly
        self.activation = nn.Sigmoid()

        self.check_parameters()

    def forward(self, x):
        # Directly pass the input through the LIF layer
        out = self.lif(x)

        out = out.view(out.size(0), -1)

        # Pass the output of the LIF layer through the fully connected layer
        out = self.fc(out)
        
        out = self.activation(out) 
        
        return out

    def check_parameters(self):
        # Iterate over each parameter and check for nan or inf values
        for name, param in self.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"NaN or Inf detected in parameter: {name}")
            else:
                print(f"No NaN or Inf detected in parameter: {name}")

# Load and preprocess the data
data = load_data('./data/GlobalTemperatures.csv')


train_size = int(0.8 * len(data))
test_size = len(data) - train_size


train_data, test_data = data[0:train_size,:], data[train_size:len(data),:]


# Initialize the model and move it to the CPU
device = torch.device('cpu')
model = TemperatureSNN(batch_size=batch_size_defined).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the datasets and data loaders
train_dataset = TemperatureDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_defined, shuffle=False)

test_dataset = TemperatureDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_defined, shuffle=False)

# Training loop
num_epochs = 50
train_losses = []


for epoch in range(num_epochs):
    print(" Started epoch {}".format(epoch+1))
    epoch_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        # print("Running batch: {}".format(i))
        torch.autograd.set_detect_anomaly(True)
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # inputs = inputs.squeeze()
        # inputs = inputs.unsqueeze(0)

        # print("Inputs shape: {}".format(inputs.shape))

        outputs = model(inputs)
        # print(outputs)

        # labels = labels.squeeze() # This will remove dimensions of size 1, resulting in shape [5, 1]
        # labels = labels.unsqueeze(0) # This will add a dimension at the beginning, resulting in shape [1, 5]

        num_nan = torch.sum(torch.isnan(labels))
        # print("Number of NaN values:", num_nan.item())

        labels = torch.nan_to_num(labels, nan=0.00001)

        labels = labels.squeeze(dim=2)
        
        loss = criterion(outputs, labels)
        # print(loss)

        # model.check_parameters()

        # print("Labels: {}".format(labels))
        # print("Outputs: {}".format(outputs))
        
        loss.backward(retain_graph=True)
        optimizer.step()

        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_epoch_loss)

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print('Finished Training')


# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Evaluation on test set
model.eval()
predictions = []
true_values = []

# Remove the last sample from test_dataloader
test_dataloader.dataset.data = test_dataloader.dataset.data[:-1]
test_dataloader.batch_sampler.sampler = torch.utils.data.SequentialSampler(test_dataloader.dataset)
test_dataloader.batch_sampler.drop_last = False

j = 0
print("Test size: {}".format(len(test_dataloader)))
with torch.no_grad():
    for inputs, labels in test_dataloader:
        print("Sample {}".format(j))
        inputs, labels = inputs.to(device), labels.to(device)

        # inputs = inputs.squeeze()
        # inputs = inputs.unsqueeze(0)

        outputs = model(inputs)
        print(outputs)

        # labels = labels.squeeze() 
        # labels = labels.unsqueeze(0)

        labels = labels.squeeze(dim=2)

        # print("Labels Shape: {}".format(labels.shape))
        # print("Outputs Shape: {}".format(outputs.shape))

        predictions.append(outputs.squeeze().tolist())  # Extend predictions list with multiple predictions
        true_values.append(labels.squeeze().tolist())  # Extend true_values list with multiple true values
        j = j + 1



five_days_ahead_true_values = [nested[-1] for nested in true_values[:124]]
five_days_ahead_predicted_values = [nested[-1] for nested in predictions[:124]]

# for i in range(len(true_values)):
#     print(predictions)


# Plot
plt.figure(figsize=(10, 6))
plt.plot(five_days_ahead_true_values, label='True Values', marker='o', linestyle='-')
plt.plot(five_days_ahead_predicted_values, label='Predicted Values', marker='o', linestyle='--')
plt.title('True vs Predicted Values for Next 5 Days')
plt.xlabel('Day')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
import torch
from SNN_modified import *


# Load the saved model
checkpoint = torch.load('temperature_forecaster.h5')

# Initialize the model with the same architecture as during training
model = TemperatureSNN(batch_size=batch_size_defined, input_size=checkpoint['input_size'], output_size=checkpoint['output_size'])

# Load the state_dict into the model
model.load_state_dict(checkpoint['model_state_dict'])
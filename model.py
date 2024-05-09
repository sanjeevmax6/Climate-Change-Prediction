import pandas as pd
import numpy as np
import torch
from bindsnet.network import Network
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

class SNN():
    def __init__(self, input_sterize, hidden_size, output_size, spike_trains):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spike_trains = spike_trains

        self.network = Network()

    def add_layers(self):
        input_layer = Input(n = self.input_size)
        hidden_layer = LIFNodes(n = self.hidden_size)
        output_layer = LIFNodes(n = self.output_size)

        self.network.add_layer(layer=input_layer, name="InputLayer")
        self.network.add_layer(layer=hidden_layer, name="HiddenLayer")
        self.network.add_layer(layer=output_layer, name="OutputLayer")

        # Connect input layer to hidden layer
        input_hidden_connection = Connection(
            source=input_layer,
            target=hidden_layer,
            w=torch.randn(self.input_size, self.hidden_size),  # Initialize weights randomly
        )
        self.network.add_connection(connection=input_hidden_connection, source="InputLayer", target="HiddenLayer")

        # Connect hidden layer to output layer
        hidden_output_connection = Connection(
            source=hidden_layer,
            target=output_layer,
            w=torch.randn(self.hidden_size, self.output_size),  # Initialize weights randomly
        )
        self.network.add_connection(connection=hidden_output_connection, source="HiddenLayer", target="OutputLayer")

    def train(self):
        # Encode input spikes using Poisson encoding
        encoder = PoissonEncoder(time=1)
        input_spikes = encoder(self.spike_trains)

        # Simulate the network
        self.network.run(input_spikes, time=input_spikes.shape[1])

        # Get spikes from output layer
        output_spikes = self.network.layers["OutputLayer"].get("s")

        # Print the output spikes
        print("Output layer spikes:")
        print(output_spikes)

        return output_spikes
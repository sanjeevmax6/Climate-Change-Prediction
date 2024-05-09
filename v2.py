import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        # Transpose x to have the sequence length as the first dimension
        x = x.transpose(0, 1)  # [1, 360, 1] -> [360, 1, 1]
        _, (hidden, cell) = self.lstm(x)
        print("Done with LSTM Forward Encoder")
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, _ = self.lstm(x, hidden)
        print("Finished Decoder forward LSTM")
        output = self.fc(output)
        return output

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        encoder_hidden = self.encoder(input_seq)
        print("Done with Encoder Hidden")
        decoder_output = self.decoder(target_seq, encoder_hidden)
        print("Finished with Decoder Output")
        return decoder_output

# Read the data
cities_data = ['Abidjan_data.csv', 'Addis_Abeba_data.csv', 'Ahmadabad_data.csv', 'Aleppo_data.csv', 'Alexandria_data.csv', 'Ankara_data.csv',
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
all_data = []

for filename in cities_data:
    file_path = './data/' + filename
    data = pd.read_csv(file_path)
    all_data.append(data)

# Preprocess the data
# For simplicity, let's assume you handle missing values and feature selection appropriately
# You also need to convert the date column to datetime and set it as index

# Define a function to create input/target sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq_x = data.iloc[i:i + seq_length, 1]  # Assuming the temperature column is at index 1
        seq_y = data.iloc[i + seq_length:i + seq_length + 1, 1]  # Assuming the temperature column is at index 1
        sequences.append((seq_x, seq_y))
    return sequences

# Define the model
input_size = 1  # Assuming univariate time series
hidden_size = 64
output_size = 1  # Predicting a single value
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)
model = Seq2Seq(encoder, decoder)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split the data into train and validation sets
train_size = int(len(all_data) * 0.8)
train_data = all_data[:train_size]
val_data = all_data[train_size:]

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print("Epoch {}".format(epoch))
    for data in train_data[:1][:500]:
        print("Processing dataset 1")
        # Assuming we predict for 12, 60, and 120 months
        for seq_length in [360]:
            sequences = create_sequences(data, seq_length)
            for seq_x, seq_y in sequences:
                optimizer.zero_grad()
                seq_x = torch.tensor(seq_x.values, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                seq_y = torch.tensor(seq_y.values, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
                # print(seq_x.shape)
                # print(seq_y.shape)
                # exit()
                output_seq = model(seq_x, seq_y)
                loss = criterion(output_seq, seq_y)
                loss.backward()
                optimizer.step()

torch.save(model.state_dict(), 'seq2seq_model.h5')


# Evaluation
with torch.no_grad():
    for data in val_data:
        # Assuming we predict for 12, 60, and 120 months
        for seq_length in [360]:
            sequences = create_sequences(data, seq_length)
            for seq_x, seq_y in sequences:
                seq_x = torch.tensor(seq_x.values, dtype=torch.float32).unsqueeze(1)  # Add batch dimension
                seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
                output_seq = model(seq_x, seq_y)

                # Convert tensor to numpy array for plotting
                predicted_values = output_seq.squeeze().numpy()
                original_values = seq_y.squeeze().numpy()

                # Plot original vs predicted values
                plt.plot(original_values, label='Original')
                plt.plot(predicted_values, label='Predicted')
                plt.xlabel('Time')
                plt.ylabel('Temperature')
                plt.title('Original vs Predicted Values')
                plt.legend()
                plt.show()
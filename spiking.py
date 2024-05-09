import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.clock_driven import neuron, layer, surrogate

# Define your network
class SimpleSNN(torch.nn.Module):
    def __init__(self):
        super(SimpleSNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10, bias=False)
        self.lif = neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan())

    def forward(self, x):
        x = self.flatten(x)
        print("Done 1")
        x = self.linear(x)
        print("Done 2")
        x = self.lif(x)
        print("Done 3")
        return x

# Load your dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [0, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the network and move it to the CPU
device = torch.device('cpu')
net = SimpleSNN().to(device)

# Define loss function and optimizer
criterion = F.cross_entropy
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print("At iteration {}".format(i))
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print('Finished Training')
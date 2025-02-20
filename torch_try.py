import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare the dataset
data = pd.read_csv('creditcard.csv')
dataset = data.to_numpy().astype(np.float32)  # Ensure data is float32 for PyTorch
fraud_data = dataset[dataset[:, -1] == 1]

# standardize the data, using StandardScaler
scaler = StandardScaler()
fraud_data = scaler.fit_transform(fraud_data)

tensor_data = torch.from_numpy(fraud_data)
data_dataset = TensorDataset(tensor_data)
batch_size = 32
data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: 31 -> 23 -> 19 -> 17 -> 8 with dropout after first two hidden layers
        self.encoder = nn.Sequential(
            nn.Linear(31, 23),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(23, 19),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(19, 17),
            nn.Tanh(),
            nn.Linear(17, 8)
        )
        # Decoder: 8 -> 17 -> 19 -> 23 -> 31 with dropout after first two layers
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(8, 17),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(17, 19),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(19, 23),
            nn.Tanh(),
            nn.Linear(23, 31)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Initialize model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=900, gamma=0.1)

# Training loop
num_epochs = 1500
history_loss = []

for epoch in range(num_epochs):
    model.train()  # Ensure dropout is enabled during training
    running_loss = 0.0
    for batch in data_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(data_loader.dataset)
    history_loss.append(epoch_loss)

    # print learning rate
    # print(f"Epoch {epoch}, Loss: {epoch_loss}, Learning Rate: {scheduler.get_last_lr()[0]}")
    scheduler.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss}")

# Plot the training loss over epochs 

plt.figure()
plt.plot(history_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss (Log Scale)")
plt.title("Training Loss")
plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.show()

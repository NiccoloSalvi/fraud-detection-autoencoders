import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
data = pd.read_csv('creditcard.csv')
dataset = data.to_numpy().astype(np.float32)  # Ensure data is float32 for PyTorch

# Standardize all features except the label using StandardScaler
dataset[:, :-1] = StandardScaler().fit_transform(dataset[:, :-1])

# Extract fraud samples (assuming last column is the label, and fraud=1)
fraud_data = dataset[dataset[:, -1] == 1]

# -----------------------------
# 2. Train/Validation Split (80-20)
# -----------------------------
train_data, val_data = train_test_split(fraud_data, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and create datasets
tensor_train = torch.from_numpy(train_data)
tensor_val = torch.from_numpy(val_data)

train_dataset = TensorDataset(tensor_train)
val_dataset = TensorDataset(tensor_val)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# 3. Define the Autoencoder Model
# -----------------------------
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

# -----------------------------
# 4. Initialize Model, Loss, Optimizer, and Scheduler
# -----------------------------
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.006)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=900, eta_min=0.05)

# -----------------------------
# 5. Training Loop with Validation Evaluation
# -----------------------------
num_epochs = 10000
history_loss_train = []
history_loss_val = []

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    running_loss_train = 0.0
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss_train += loss.item() * inputs.size(0)
    epoch_loss_train = running_loss_train / len(train_loader.dataset)
    history_loss_train.append(epoch_loss_train)
    
    # --- Validation Phase ---
    model.eval()
    running_loss_val = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss_val += loss.item() * inputs.size(0)
    epoch_loss_val = running_loss_val / len(val_loader.dataset)
    history_loss_val.append(epoch_loss_val)
    
    scheduler.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {epoch_loss_train:.4f}, Val Loss: {epoch_loss_val:.4f}")

# -----------------------------
# 6. Plot Training and Validation Losses Together
# -----------------------------
plt.figure()
plt.plot(history_loss_train, label='Train Loss')
plt.plot(history_loss_val, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (Log Scale)")
plt.title("Training and Validation Loss")
plt.yscale('log')
plt.legend()
plt.show()

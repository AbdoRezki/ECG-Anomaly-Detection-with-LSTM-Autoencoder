import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import copy
from src.data_prep import load_ecg_data, create_dataset
from src.model import RecurrentAutoencoder

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data
print('Loading ECG data...')
normal_df, anomaly_df = load_ecg_data('data/ECG5000_TRAIN.arff', 'data/ECG5000_TEST.arff')
train_ds, val_ds, test_normal, test_anomaly = create_dataset(normal_df, anomaly_df)

print(f'Train shape: {train_ds.shape}')
print(f'Val shape: {val_ds.shape}')
print(f'Test normal shape: {test_normal.shape}')
print(f'Test anomaly shape: {test_anomaly.shape}')

# Create DataLoaders
train_dataset = TensorDataset(train_ds)
val_dataset = TensorDataset(val_ds)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model setup
model = RecurrentAutoencoder().to(device)
print(f'Model parameters: {len(list(model.parameters()))}')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss(reduction='sum')

# Training
history = {'train_loss': [], 'val_loss': []}
best_loss = float('inf')
best_model_wts = None

print('\nStarting training...')
for epoch in range(150):
    # Train
    model.train()
    train_losses = []
    for batch in train_loader:
        sequences = batch[0].to(device)  # [batch, 1, 140]
        
        optimizer.zero_grad()
        recon = model(sequences)
        loss = criterion(recon, sequences)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    
    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch[0].to(device)
            recon = model(sequences)
            loss = criterion(recon, sequences)
            val_losses.append(loss.item())
    
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    history['train_loss'].append(avg_train)
    history['val_loss'].append(avg_val)
    
    # Save best model
    if avg_val < best_loss:
        best_loss = avg_val
        best_model_wts = copy.deepcopy(model.state_dict())
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d} | Train Loss: {avg_train:8.2f} | Val Loss: {avg_val:8.2f}')

# Load best model and save
print('\nLoading best model and saving...')
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'model.pth')
print(f'Best validation loss: {best_loss:.2f}')

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.savefig('loss.png')
plt.show()
print('\nTraining complete! Model saved to model.pth, plot saved to loss.png')

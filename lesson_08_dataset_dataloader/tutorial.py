"""
Lesson 08: Dataset and DataLoader
===================================
PyTorch provides Dataset and DataLoader classes to efficiently load,
batch, and shuffle data for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# 1. Custom Dataset Class
# ============================================================

print("=== Custom Dataset ===\n")


class SyntheticDataset(Dataset):
    """A custom dataset must implement __len__ and __getitem__."""

    def __init__(self, num_samples=100):
        # Generate synthetic data
        torch.manual_seed(42)
        self.X = torch.randn(num_samples, 3)
        self.y = self.X[:, 0] * 2 + self.X[:, 1] * (-1) + 0.5

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a single sample (features, label) at the given index."""
        return self.X[idx], self.y[idx]


# Create dataset
dataset = SyntheticDataset(num_samples=100)
print(f"Dataset size: {len(dataset)}")
print(f"First sample: features={dataset[0][0]}, label={dataset[0][1]:.4f}")

# ============================================================
# 2. DataLoader Basics
# ============================================================

print("\n=== DataLoader ===\n")

# DataLoader wraps a Dataset and provides:
# - Batching
# - Shuffling
# - Parallel loading (num_workers)

dataloader = DataLoader(
    dataset,
    batch_size=16,      # samples per batch
    shuffle=True,       # randomize order each epoch
    drop_last=False,    # keep the last incomplete batch
)

print(f"Number of batches: {len(dataloader)}")
print(f"Batch size: 16")
print(f"Total samples: {len(dataset)}")

# Iterate through one batch
for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Features shape: {features.shape}")  # [16, 3]
    print(f"  Labels shape: {labels.shape}")       # [16]
    if batch_idx == 1:
        break

# ============================================================
# 3. Splitting Data
# ============================================================

print("\n=== random_split ===\n")

# Split dataset into train (80%) and val (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")

# Create separate dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ============================================================
# 4. Training with DataLoader
# ============================================================

print("\n=== Training with DataLoader ===\n")

import torch.nn as nn

model = nn.Linear(3, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for features, labels in train_loader:
        # Forward
        predictions = model(features).squeeze()
        loss = loss_fn(predictions, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for features, labels in val_loader:
            predictions = model(features).squeeze()
            loss = loss_fn(predictions, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}")

# ============================================================
# 5. TensorDataset — Quick Dataset from Tensors
# ============================================================

print("\n=== TensorDataset ===\n")

from torch.utils.data import TensorDataset

# Create a dataset directly from tensors
X_data = torch.randn(50, 4)
y_data = torch.randint(0, 2, (50,))

tensor_dataset = TensorDataset(X_data, y_data)
print(f"TensorDataset size: {len(tensor_dataset)}")

sample = tensor_dataset[0]
print(f"Sample: features shape={sample[0].shape}, label={sample[1]}")

loader = DataLoader(tensor_dataset, batch_size=10, shuffle=True)
for X_batch, y_batch in loader:
    print(f"Batch: X={X_batch.shape}, y={y_batch.shape}")
    break

# ============================================================
# 6. DataLoader Options
# ============================================================

print("\n=== DataLoader Options ===")
print("""
Key parameters:
  - batch_size: Number of samples per batch
  - shuffle: Whether to shuffle at the start of each epoch
  - num_workers: Number of subprocesses for data loading (0 = main process)
  - drop_last: Whether to drop the last incomplete batch
  - pin_memory: If True, speeds up CPU-to-GPU transfer

Example:
  DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
""")

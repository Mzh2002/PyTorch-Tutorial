"""
Lesson 08: Dataset and DataLoader
===================================
PyTorch provides Dataset and DataLoader classes to efficiently load,
batch, and shuffle data for training.

Dataset: Wine (from sklearn) — 13 features, 3 wine cultivar classes
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_wine

# ============================================================
# 1. Custom Dataset Class
# ============================================================

print("=== Custom Dataset ===\n")


class WineDataset(Dataset):
    """A custom dataset wrapping the Wine classification data from sklearn.
    Demonstrates __len__ and __getitem__ with real data."""

    def __init__(self):
        wine = load_wine()
        self.X = torch.tensor(wine.data, dtype=torch.float32)
        self.y = torch.tensor(wine.target, dtype=torch.long)
        self.feature_names = wine.feature_names
        self.target_names = wine.target_names

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a single sample (features, label) at the given index."""
        return self.X[idx], self.y[idx]


# Create dataset
dataset = WineDataset()
print(f"Dataset: Wine ({len(dataset)} samples, {dataset.X.shape[1]} features)")
print(f"Features: {list(dataset.feature_names)}")
print(f"Classes: {list(dataset.target_names)}")
print(f"First sample: features shape={dataset[0][0].shape}, label={dataset[0][1]}")

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
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels: {labels.tolist()}")
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

print("\n=== Training Wine Classifier with DataLoader ===\n")

# Normalize features (compute stats from training set)
# Access underlying dataset through random_split indices
all_X = dataset.X
X_mean = all_X.mean(dim=0)
X_std = all_X.std(dim=0)

model = nn.Sequential(
    nn.Linear(13, 32),
    nn.ReLU(),
    nn.Linear(32, 3),   # 3 wine classes
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for features, labels in train_loader:
        # Normalize
        features = (features - X_mean) / X_std

        # Forward
        predictions = model(features)
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
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = (features - X_mean) / X_std
            predictions = model(features)
            loss = loss_fn(predictions, labels)
            val_loss += loss.item()

            _, predicted = predictions.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, val_acc={accuracy:.1f}%")

# ============================================================
# 5. TensorDataset — Quick Dataset from Tensors
# ============================================================

print("\n=== TensorDataset ===\n")

from torch.utils.data import TensorDataset
from sklearn.datasets import load_iris

# Load Iris dataset and wrap with TensorDataset
iris = load_iris()
X_iris = torch.tensor(iris.data, dtype=torch.float32)
y_iris = torch.tensor(iris.target, dtype=torch.long)

tensor_dataset = TensorDataset(X_iris, y_iris)
print(f"TensorDataset: Iris ({len(tensor_dataset)} samples)")

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
  - drop_last: Whether to drop the last incomplete batch
  - num_workers: Number of subprocesses for data loading (0 = main process)
  - pin_memory: If True, copies tensors into CUDA pinned memory (faster GPU transfer)
""")

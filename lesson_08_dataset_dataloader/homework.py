"""
Lesson 08 Homework: Dataset and DataLoader
============================================
Complete the TODOs below.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(8)

# ============================================================
# Exercise 1: Create a Custom Dataset
# ============================================================

# TODO: Create a Dataset class called QuadraticDataset
# - In __init__, generate 200 samples:
#     X = random values in [-3, 3], shape [200, 1]
#     y = X^2 + noise (noise scale 0.2), shape [200, 1]
# - Implement __len__ and __getitem__


class QuadraticDataset(Dataset):
    def __init__(self):
        # TODO: Generate X and y
        pass

    def __len__(self):
        # TODO
        pass

    def __getitem__(self, idx):
        # TODO: Return (self.X[idx], self.y[idx])
        pass


# ============================================================
# Exercise 2: Create DataLoaders
# ============================================================

# TODO: Instantiate the dataset
dataset = None

# TODO: Split into 160 train and 40 validation samples
train_dataset = None
val_dataset = None

# TODO: Create DataLoader for training (batch_size=32, shuffle=True)
train_loader = None

# TODO: Create DataLoader for validation (batch_size=32, shuffle=False)
val_loader = None

# ============================================================
# Exercise 3: Train a Model Using DataLoaders
# ============================================================

# TODO: Define a model: Linear(1, 32) -> ReLU -> Linear(32, 1)
model = None

# TODO: Define MSE loss and Adam optimizer (lr=0.01)
loss_fn = None
optimizer = None

# TODO: Train for 100 epochs using the training dataloader
#       Track validation loss each epoch
num_epochs = 100
final_val_loss = None

for epoch in range(num_epochs):
    pass  # TODO: Implement training and validation loop

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert dataset is not None, "Exercise 1: dataset not created"
assert len(dataset) == 200, f"Exercise 1: expected 200 samples, got {len(dataset)}"

sample_x, sample_y = dataset[0]
assert sample_x.shape == (1,), f"Exercise 1: X should be shape (1,), got {sample_x.shape}"
assert sample_y.shape == (1,), f"Exercise 1: y should be shape (1,), got {sample_y.shape}"

assert train_dataset is not None and len(train_dataset) == 160, "Exercise 2: train split wrong"
assert val_dataset is not None and len(val_dataset) == 40, "Exercise 2: val split wrong"
assert train_loader is not None, "Exercise 2: train_loader not created"
assert val_loader is not None, "Exercise 2: val_loader not created"

assert final_val_loss is not None, "Exercise 3: final_val_loss not set"
assert final_val_loss < 2.0, f"Exercise 3: val_loss too high ({final_val_loss:.4f})"

print(f"Final validation loss: {final_val_loss:.4f}")
print("All exercises passed!")

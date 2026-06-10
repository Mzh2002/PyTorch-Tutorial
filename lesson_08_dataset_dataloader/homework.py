"""
Lesson 08 Homework: Dataset and DataLoader
============================================
Complete the TODOs below.

Dataset: Iris (from sklearn) — 4 features, 3 flower species
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_iris

torch.manual_seed(8)

# ============================================================
# Exercise 1: Create a Custom Dataset
# ============================================================

# TODO: Create a Dataset class called IrisDataset
# - In __init__, load the Iris dataset from sklearn:
#     iris = load_iris()
#     Store X as float32 tensor and y as long tensor
# - Implement __len__ and __getitem__


class IrisDataset(Dataset):
    def __init__(self):
        # TODO: Load Iris data and store as tensors
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

# TODO: Split into 120 train and 30 validation samples
train_dataset = None
val_dataset = None

# TODO: Create DataLoader for training (batch_size=16, shuffle=True)
train_loader = None

# TODO: Create DataLoader for validation (batch_size=16, shuffle=False)
val_loader = None

# ============================================================
# Exercise 3: Train a Classifier Using DataLoaders
# ============================================================

# TODO: Define a model: Linear(4, 16) -> ReLU -> Linear(16, 3)
model = None

# TODO: Define CrossEntropyLoss and Adam optimizer (lr=0.01)
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
assert len(dataset) == 150, f"Exercise 1: expected 150 samples, got {len(dataset)}"

sample_x, sample_y = dataset[0]
assert sample_x.shape == (4,), f"Exercise 1: X should be shape (4,), got {sample_x.shape}"

assert train_dataset is not None and len(train_dataset) == 120, "Exercise 2: train split wrong"
assert val_dataset is not None and len(val_dataset) == 30, "Exercise 2: val split wrong"
assert train_loader is not None, "Exercise 2: train_loader not created"
assert val_loader is not None, "Exercise 2: val_loader not created"

assert final_val_loss is not None, "Exercise 3: final_val_loss not set"
assert final_val_loss < 2.0, f"Exercise 3: val_loss too high ({final_val_loss:.4f})"

print(f"Final validation loss: {final_val_loss:.4f}")
print("All exercises passed!")

"""
Lesson 08 Answer: Dataset and DataLoader
==========================================
Dataset: Iris (from sklearn) — 4 features, 3 flower species
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import load_iris

torch.manual_seed(8)

# ============================================================
# Exercise 1: Custom Dataset
# ============================================================


class IrisDataset(Dataset):
    def __init__(self):
        iris = load_iris()
        self.X = torch.tensor(iris.data, dtype=torch.float32)
        self.y = torch.tensor(iris.target, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Exercise 2: DataLoaders
# ============================================================

dataset = IrisDataset()

train_dataset, val_dataset = random_split(dataset, [120, 30])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Dataset: Iris ({len(dataset)} samples)")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ============================================================
# Exercise 3: Train
# ============================================================

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
final_val_loss = None

for epoch in range(num_epochs):
    # Training
    model.train()
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred = model(X_batch)
            val_loss += loss_fn(pred, y_batch).item()
    final_val_loss = val_loss / len(val_loader)

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}: val_loss={final_val_loss:.4f}")

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert len(dataset) == 150
assert len(train_dataset) == 120
assert len(val_dataset) == 30
assert final_val_loss < 2.0

print(f"Final validation loss: {final_val_loss:.4f}")
print("All exercises passed!")

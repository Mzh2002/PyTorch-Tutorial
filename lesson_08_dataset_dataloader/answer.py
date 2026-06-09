"""
Lesson 08 Answer: Dataset and DataLoader
==========================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(8)

# ============================================================
# Exercise 1: Custom Dataset
# ============================================================


class QuadraticDataset(Dataset):
    def __init__(self):
        self.X = (torch.rand(200, 1) * 6) - 3  # uniform in [-3, 3]
        self.y = self.X ** 2 + torch.randn(200, 1) * 0.2

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Exercise 2: DataLoaders
# ============================================================

dataset = QuadraticDataset()

train_dataset, val_dataset = random_split(dataset, [160, 40])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ============================================================
# Exercise 3: Train
# ============================================================

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
loss_fn = nn.MSELoss()
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

assert len(dataset) == 200
assert len(train_dataset) == 160
assert len(val_dataset) == 40
assert final_val_loss < 2.0

print(f"Final validation loss: {final_val_loss:.4f}")
print("All exercises passed!")

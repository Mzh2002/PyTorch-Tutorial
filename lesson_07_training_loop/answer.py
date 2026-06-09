"""
Lesson 07 Answer: Building a Training Loop
============================================
Dataset: Diabetes (from sklearn) — 10 features → disease progression
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes

torch.manual_seed(7)

# ============================================================
# Exercise 1: Prepare Data
# ============================================================

diabetes = load_diabetes()
X_np = diabetes.data
y_np = diabetes.target

X_all = torch.tensor(X_np, dtype=torch.float32)
y_all = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

X_mean, X_std = X_all.mean(dim=0), X_all.std(dim=0)
y_mean, y_std = y_all.mean(), y_all.std()
X_data = (X_all - X_mean) / X_std
y_data = (y_all - y_mean) / y_std

num_samples = X_data.shape[0]

n_train = int(0.8 * num_samples)
X_train = X_data[:n_train]
X_val = X_data[n_train:]
y_train = y_data[:n_train]
y_val = y_data[n_train:]

print(f"Dataset: Diabetes ({num_samples} samples, 10 features)")
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# ============================================================
# Exercise 2: Define Model
# ============================================================

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# ============================================================
# Exercise 3: Training Loop
# ============================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    y_pred = model(X_train)
    train_loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: train={train_loss.item():.4f}, val={val_loss.item():.4f}")

# ============================================================
# Exercise 4: Report Results
# ============================================================

final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert n_train == int(0.8 * num_samples)
assert X_train.shape[1] == 10
assert X_val.shape[1] == 10
assert len(train_losses) == num_epochs
assert final_train_loss < 0.5
assert final_val_loss < 1.5

print(f"Final train loss: {final_train_loss:.4f}")
print(f"Final val loss: {final_val_loss:.4f}")
print("All exercises passed!")

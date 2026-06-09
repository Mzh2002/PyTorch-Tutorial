"""
Lesson 07: Building a Training Loop
=====================================
A proper training loop is the core of any deep learning project.
This lesson shows the standard pattern used in practice.

Dataset: California Housing (from sklearn) — 8 features → house value
"""

import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing

# ============================================================
# 1. The Standard Training Loop Pattern
# ============================================================

# The pattern is always:
#   for epoch in range(num_epochs):
#       for batch in dataloader:            # (we'll use simple batches here)
#           1. Forward pass (compute predictions)
#           2. Compute loss
#           3. Zero gradients
#           4. Backward pass (compute gradients)
#           5. Optimizer step (update weights)

# ============================================================
# 2. Full Example: Multi-feature Regression
# ============================================================

print("=== Multi-feature Regression on California Housing ===\n")

torch.manual_seed(42)

# Load real data with all 8 features
housing = fetch_california_housing()
X_np = housing.data[:500]   # use 500 samples for speed
y_np = housing.target[:500]

X_all = torch.tensor(X_np, dtype=torch.float32)
y_all = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# Normalize features and target
X_mean, X_std = X_all.mean(dim=0), X_all.std(dim=0)
y_mean, y_std = y_all.mean(), y_all.std()
X = (X_all - X_mean) / X_std
y = (y_all - y_mean) / y_std

num_samples = X.shape[0]
print(f"Dataset: California Housing")
print(f"Features: {housing.feature_names}")
print(f"Data: {num_samples} samples, {X.shape[1]} features")

# Model
model = nn.Linear(8, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training with manual mini-batches
batch_size = 32
num_epochs = 50

print("\n--- Training ---")
for epoch in range(num_epochs):
    # Shuffle data each epoch
    perm = torch.randperm(num_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    epoch_loss = 0.0
    num_batches = 0

    # Mini-batch loop
    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    # Print average loss per epoch
    avg_loss = epoch_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: avg_loss={avg_loss:.4f}")

print(f"\nLearned weights (normalized scale): {model.weight.data.numpy().flatten()}")
print(f"Learned bias: {model.bias.item():.4f}")

# ============================================================
# 3. Tracking Training History
# ============================================================

print("\n=== Training with History ===\n")

torch.manual_seed(0)

# Fresh model with a hidden layer
model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Store losses for monitoring
train_losses = []

for epoch in range(100):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

print(f"Initial loss: {train_losses[0]:.4f}")
print(f"Final loss:   {train_losses[-1]:.4f}")
print(f"Improvement:  {train_losses[0] / train_losses[-1]:.1f}x")

# ============================================================
# 4. Train/Validation Split
# ============================================================

print("\n=== Train/Validation Split ===\n")

torch.manual_seed(42)

# Split data: 80% train, 20% validation
n_train = int(0.8 * num_samples)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

print(f"Train: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")

# Fresh model
model = nn.Linear(8, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    # --- Training ---
    model.train()
    y_pred = model(X_train)
    train_loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # --- Validation (no gradient computation needed) ---
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, y_val)

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1:3d}: "
              f"train_loss={train_loss.item():.4f}, "
              f"val_loss={val_loss.item():.4f}")

# ============================================================
# 5. Early Stopping (Concept)
# ============================================================

print("\n=== Early Stopping Concept ===")
print("""
Early stopping monitors validation loss and stops training when it
starts increasing (overfitting). Here's the logic:

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(max_epochs):
        # ... train ...
        # ... compute val_loss ...

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model weights
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break
""")

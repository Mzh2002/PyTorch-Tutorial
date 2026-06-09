"""
Lesson 04: Linear Regression from Scratch
===========================================
We implement linear regression using only tensors and autograd —
no nn.Module yet. This shows how gradient descent works at the lowest level.

Dataset: California Housing (from sklearn)
  - Feature: MedInc (median income in a block group)
  - Target: MedHouseVal (median house value in $100,000s)
"""

import torch
from sklearn.datasets import fetch_california_housing

# ============================================================
# 1. Load Real Data — California Housing
# ============================================================

# fetch_california_housing downloads the dataset from the internet
data = fetch_california_housing()

# Use a single feature: MedInc (median income)
# This keeps the problem simple for learning gradient descent
feature_idx = 0  # MedInc
X_raw = torch.tensor(data.data[:, feature_idx], dtype=torch.float32).unsqueeze(1)
y_raw = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

print(f"Dataset: California Housing")
print(f"Feature: {data.feature_names[feature_idx]} (median income)")
print(f"Target: Median house value ($100K)")
print(f"Total samples: {X_raw.shape[0]}")

# Use first 200 samples for a manageable training set
X_subset = X_raw[:200]
y_subset = y_raw[:200]

# ============================================================
# 2. Normalize Data
# ============================================================

# Normalizing helps gradient descent converge faster
X_mean, X_std = X_subset.mean(), X_subset.std()
y_mean, y_std = y_subset.mean(), y_subset.std()

X = (X_subset - X_mean) / X_std
y = (y_subset - y_mean) / y_std

print(f"\nUsing {X.shape[0]} samples (normalized)")
print(f"X range: [{X.min().item():.2f}, {X.max().item():.2f}]")
print(f"y range: [{y.min().item():.2f}, {y.max().item():.2f}]")

print(f"\nFirst 5 samples (normalized):")
for i in range(5):
    print(f"  income={X[i].item():.3f}, value={y[i].item():.3f}")

# ============================================================
# 3. Initialize Parameters
# ============================================================

# We want to learn: y_pred = w * x + b
# Start with random values
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(f"\nInitial w: {w.item():.4f}")
print(f"Initial b: {b.item():.4f}")

# ============================================================
# 4. Define Forward Pass (Prediction)
# ============================================================


def forward(X):
    """Predict y given X using current w and b."""
    return X * w + b


# ============================================================
# 5. Define Loss Function (Mean Squared Error)
# ============================================================


def mse_loss(y_pred, y_true):
    """Compute mean squared error."""
    return ((y_pred - y_true) ** 2).mean()


# ============================================================
# 6. Training Loop
# ============================================================

learning_rate = 0.1
num_epochs = 100

print("\n--- Training ---")
for epoch in range(num_epochs):
    # Forward pass
    y_pred = forward(X)

    # Compute loss
    loss = mse_loss(y_pred, y)

    # Backward pass (compute gradients)
    loss.backward()

    # Update parameters (without tracking gradients)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

    # Print progress
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, "
              f"w={w.item():.4f}, b={b.item():.4f}")

# ============================================================
# 7. Check Results
# ============================================================

print("\n--- Results ---")
print(f"Learned w: {w.item():.4f}")
print(f"Learned b: {b.item():.4f}")
print(f"Final loss: {loss.item():.4f}")

# Make predictions and convert back to original scale
with torch.no_grad():
    # Predict on a few test incomes
    test_incomes = torch.tensor([[3.0], [5.0], [8.0]])  # raw MedInc values
    test_normalized = (test_incomes - X_mean) / X_std
    pred_normalized = forward(test_normalized)
    pred_values = pred_normalized * y_std + y_mean

    print("\nPredictions (median house value in $100K):")
    for i in range(len(test_incomes)):
        print(f"  Income={test_incomes[i].item():.1f} -> "
              f"Value=${pred_values[i].item() * 100:.0f}K")

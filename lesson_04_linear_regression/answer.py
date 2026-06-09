"""
Lesson 04 Answer: Linear Regression from Scratch
==================================================
Dataset: California Housing — predict house value from HouseAge.
"""

import torch
from sklearn.datasets import fetch_california_housing

torch.manual_seed(123)

# ============================================================
# Exercise 1: Load and Prepare Data
# ============================================================

data = fetch_california_housing()
feature_idx = 1  # HouseAge

X_raw = torch.tensor(data.data[:150, feature_idx], dtype=torch.float32).unsqueeze(1)
y_raw = torch.tensor(data.target[:150], dtype=torch.float32).unsqueeze(1)

X_mean, X_std = X_raw.mean(), X_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()

X = (X_raw - X_mean) / X_std
y = (y_raw - y_mean) / y_std

print(f"Data: {X.shape[0]} samples, feature=HouseAge")
print(f"X range (normalized): [{X.min().item():.2f}, {X.max().item():.2f}]")

# ============================================================
# Exercise 2: Initialize Parameters
# ============================================================

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ============================================================
# Exercise 3: Training Loop
# ============================================================

learning_rate = 0.1
num_epochs = 200

for epoch in range(num_epochs):
    # Forward pass
    y_pred = X * w + b

    # Compute loss (MSE)
    loss = ((y_pred - y) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero gradients
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, "
              f"w={w.item():.4f}, b={b.item():.4f}")

# ============================================================
# Exercise 4: Evaluate
# ============================================================

final_loss = loss.item()

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert X.shape == (150, 1)
assert y.shape == (150, 1)
assert final_loss < 1.0

print(f"Learned w: {w.item():.4f}")
print(f"Learned b: {b.item():.4f}")
print(f"Final loss: {final_loss:.4f}")
print("All exercises passed!")

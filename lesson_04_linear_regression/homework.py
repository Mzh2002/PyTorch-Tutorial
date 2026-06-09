"""
Lesson 04 Homework: Linear Regression from Scratch
====================================================
Complete the TODOs to train a linear regression model on real data.

Dataset: California Housing (from sklearn)
  - Feature: HouseAge (median house age in a block group)
  - Target: MedHouseVal (median house value in $100,000s)
"""

import torch
from sklearn.datasets import fetch_california_housing

torch.manual_seed(123)

# ============================================================
# Exercise 1: Load and Prepare Data
# ============================================================

data = fetch_california_housing()

# Use feature index 1 (HouseAge) to predict house value
feature_idx = 1

# TODO: Extract the feature and target as float32 tensors, shape [N, 1]
#       Use first 150 samples only
X_raw = None
y_raw = None

# TODO: Normalize X and y (subtract mean, divide by std)
#       Store the means and stds for later use
X_mean = None
X_std = None
y_mean = None
y_std = None
X = None
y = None

# ============================================================
# Exercise 2: Initialize Parameters
# ============================================================

# TODO: Initialize w as a random tensor with requires_grad=True (shape: [1])
w = None

# TODO: Initialize b as zeros with requires_grad=True (shape: [1])
b = None

# ============================================================
# Exercise 3: Training Loop
# ============================================================

learning_rate = 0.1
num_epochs = 200

# TODO: Implement the training loop
# For each epoch:
#   1. Compute y_pred = X * w + b (forward pass)
#   2. Compute loss = mean squared error between y_pred and y
#   3. Call loss.backward()
#   4. Update w and b using gradient descent (inside torch.no_grad())
#   5. Zero the gradients

for epoch in range(num_epochs):
    pass  # TODO: Replace with training logic

# ============================================================
# Exercise 4: Evaluate
# ============================================================

# TODO: Store the final loss value
final_loss = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert X is not None and X.shape == (150, 1), "Exercise 1: X shape should be [150, 1]"
assert y is not None and y.shape == (150, 1), "Exercise 1: y shape should be [150, 1]"
assert w is not None and w.requires_grad, "Exercise 2: w not set or no grad"
assert b is not None and b.requires_grad, "Exercise 2: b not set or no grad"
assert final_loss is not None, "Exercise 4: final_loss not set"
assert final_loss < 1.0, f"Loss should be < 1.0 after training, got {final_loss:.4f}"

print(f"Learned w: {w.item():.4f}")
print(f"Learned b: {b.item():.4f}")
print(f"Final loss: {final_loss:.4f}")
print("All exercises passed!")

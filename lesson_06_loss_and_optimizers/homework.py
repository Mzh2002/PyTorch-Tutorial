"""
Lesson 06 Homework: Loss Functions and Optimizers
==================================================
Complete the TODOs below.
"""

import torch
import torch.nn as nn

# ============================================================
# Exercise 1: Compute MSE Loss
# ============================================================

predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
targets = torch.tensor([1.5, 2.5, 3.5, 4.5])

# TODO: Create an MSE loss function and compute the loss
loss_fn_mse = None
mse_value = None
# Expected: mean of (0.5^2 * 4) = mean(0.25*4) = 0.25

# ============================================================
# Exercise 2: Cross Entropy Loss
# ============================================================

# 3 samples, 4 classes
logits = torch.tensor([[1.0, 2.0, 3.0, 4.0],   # model predicts class 3
                       [4.0, 3.0, 2.0, 1.0],   # model predicts class 0
                       [1.0, 1.0, 5.0, 1.0]])  # model predicts class 2

true_labels = torch.tensor([3, 0, 2])  # all predictions match!

# TODO: Compute cross entropy loss
loss_fn_ce = None
ce_value = None

# ============================================================
# Exercise 3: Train a Model on Real Data
# ============================================================

from sklearn.datasets import fetch_california_housing

torch.manual_seed(99)

# Load California Housing: predict house value from Population
housing = fetch_california_housing()
feature_idx = 4  # Population

X_raw = torch.tensor(housing.data[:200, feature_idx], dtype=torch.float32).unsqueeze(1)
y_raw = torch.tensor(housing.target[:200], dtype=torch.float32).unsqueeze(1)

# Normalize
X_mean, X_std = X_raw.mean(), X_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()
X = (X_raw - X_mean) / X_std
y = (y_raw - y_mean) / y_std

# TODO: Create a Linear model (1 input, 1 output)
model = None

# TODO: Create MSE loss function
loss_fn = None

# TODO: Create an SGD optimizer with lr=0.01
optimizer = None

# TODO: Train for 200 epochs
for epoch in range(200):
    pass  # TODO: Implement training step
    # 1. Forward pass
    # 2. Compute loss
    # 3. Zero gradients
    # 4. Backward pass
    # 5. Optimizer step

# TODO: Store final weight and bias
final_weight = None
final_bias = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert mse_value is not None, "Exercise 1: mse_value not computed"
assert abs(mse_value.item() - 0.25) < 1e-5, f"Exercise 1: expected 0.25, got {mse_value.item()}"

assert ce_value is not None, "Exercise 2: ce_value not computed"
assert ce_value.item() < 0.5, f"Exercise 2: CE should be low (model is correct), got {ce_value.item()}"

assert final_weight is not None, "Exercise 3: final_weight not set"
assert final_bias is not None, "Exercise 3: final_bias not set"

# After training on normalized data, the loss should have decreased
with torch.no_grad():
    y_pred = model(X)
    final_loss = nn.MSELoss()(y_pred, y).item()
assert final_loss < 1.0, f"Exercise 3: loss should be < 1.0, got {final_loss:.4f}"

print(f"MSE: {mse_value.item():.4f}")
print(f"CE: {ce_value.item():.4f}")
print(f"Learned: y_norm = {final_weight:.3f}*x_norm + {final_bias:.3f}")
print(f"Final loss: {final_loss:.4f}")
print("All exercises passed!")

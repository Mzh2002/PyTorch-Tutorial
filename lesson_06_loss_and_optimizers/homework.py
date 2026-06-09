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
# Exercise 3: Train a Model
# ============================================================

torch.manual_seed(99)

# Data: y = -2x + 5
X = torch.rand(60, 1) * 8
y = -2 * X + 5 + torch.randn(60, 1) * 0.3

# TODO: Create a Linear model (1 input, 1 output)
model = None

# TODO: Create MSE loss function
loss_fn = None

# TODO: Create an SGD optimizer with lr=0.005
optimizer = None

# TODO: Train for 2000 epochs
for epoch in range(2000):
    pass  # TODO: Implement training step
    # 1. Forward pass
    # 2. Compute loss
    # 3. Zero gradients
    # 4. Backward pass
    # 5. Optimizer step

# TODO: Store final weight and bias
final_weight = None  # should be close to -2.0
final_bias = None    # should be close to 5.0

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
assert abs(final_weight - (-2.0)) < 0.3, f"Exercise 3: weight should be ~-2.0, got {final_weight:.4f}"
assert abs(final_bias - 5.0) < 0.6, f"Exercise 3: bias should be ~5.0, got {final_bias:.4f}"

print(f"MSE: {mse_value.item():.4f}")
print(f"CE: {ce_value.item():.4f}")
print(f"Learned: y = {final_weight:.3f}*x + {final_bias:.3f}")
print("All exercises passed!")

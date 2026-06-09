"""
Lesson 04 Homework: Linear Regression from Scratch
====================================================
Complete the TODOs to train a linear regression model.
"""

import torch

torch.manual_seed(123)

# ============================================================
# Exercise 1: Generate Data
# ============================================================

# True relationship: y = -1.5 * x + 7 (with noise)

# TODO: Create X with 80 random samples in range [0, 5]
#       Shape should be [80, 1]
X = None

# TODO: Create y using the true relationship plus small noise
#       y = -1.5 * X + 7 + noise (use torch.randn for noise, scale=0.3)
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

learning_rate = 0.05
num_epochs = 500

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

# TODO: Store the final values of w and b
final_w = None  # should be close to -1.5
final_b = None  # should be close to 7.0

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert X is not None and X.shape == (80, 1), "Exercise 1: X shape should be [80, 1]"
assert y is not None and y.shape == (80, 1), "Exercise 1: y shape should be [80, 1]"
assert final_w is not None, "Exercise 4: final_w not set"
assert final_b is not None, "Exercise 4: final_b not set"
assert abs(final_w - (-1.5)) < 0.2, f"w should be ~-1.5, got {final_w:.4f}"
assert abs(final_b - 7.0) < 0.5, f"b should be ~7.0, got {final_b:.4f}"

print(f"Learned w: {final_w:.4f} (true: -1.5)")
print(f"Learned b: {final_b:.4f} (true: 7.0)")
print("All exercises passed!")

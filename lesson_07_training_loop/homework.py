"""
Lesson 07 Homework: Building a Training Loop
==============================================
Complete the TODOs below.
"""

import torch
import torch.nn as nn

torch.manual_seed(7)

# ============================================================
# Exercise 1: Prepare Data with Train/Val Split
# ============================================================

# Generate synthetic data: y = x1^2 + 2*x2 (non-linear in x1)
num_samples = 300
X = torch.randn(num_samples, 2)
y = X[:, 0:1] ** 2 + 2 * X[:, 1:2] + torch.randn(num_samples, 1) * 0.1

# TODO: Split into 80% train and 20% validation
n_train = None
X_train = None
X_val = None
y_train = None
y_val = None

# ============================================================
# Exercise 2: Define Model
# ============================================================

# TODO: Create a model with:
#   Linear(2, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)
model = None

# ============================================================
# Exercise 3: Training Loop with Validation
# ============================================================

# TODO: Create MSE loss function
loss_fn = None

# TODO: Create Adam optimizer with lr=0.01
optimizer = None

num_epochs = 200
train_losses = []
val_losses = []

# TODO: Implement the training loop
# For each epoch:
#   1. Set model to training mode
#   2. Forward pass on training data
#   3. Compute training loss
#   4. Zero gradients, backward, optimizer step
#   5. Set model to eval mode
#   6. Compute validation loss (inside torch.no_grad())
#   7. Append both losses to their respective lists

for epoch in range(num_epochs):
    pass  # TODO: Replace with training logic

# ============================================================
# Exercise 4: Report Results
# ============================================================

# TODO: Store the final training and validation loss
final_train_loss = None
final_val_loss = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert n_train == 240, f"Exercise 1: n_train should be 240, got {n_train}"
assert X_train is not None and X_train.shape == (240, 2), "Exercise 1: X_train shape wrong"
assert X_val is not None and X_val.shape == (60, 2), "Exercise 1: X_val shape wrong"

assert model is not None, "Exercise 2: model not created"
params = sum(p.numel() for p in model.parameters())
assert params == 2 * 32 + 32 + 32 * 16 + 16 + 16 * 1 + 1, "Exercise 2: wrong param count"

assert len(train_losses) == num_epochs, "Exercise 3: wrong number of epochs recorded"
assert len(val_losses) == num_epochs, "Exercise 3: wrong number of val losses"

assert final_train_loss is not None, "Exercise 4: final_train_loss not set"
assert final_val_loss is not None, "Exercise 4: final_val_loss not set"
assert final_train_loss < 0.5, f"Exercise 4: train_loss too high ({final_train_loss:.4f})"
assert final_val_loss < 1.0, f"Exercise 4: val_loss too high ({final_val_loss:.4f})"

print(f"Final train loss: {final_train_loss:.4f}")
print(f"Final val loss: {final_val_loss:.4f}")
print("All exercises passed!")

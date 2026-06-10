"""
Lesson 07 Homework: Building a Training Loop
==============================================
Complete the TODOs below.

Dataset: Diabetes (from sklearn) — 10 features → disease progression
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes

torch.manual_seed(7)

# ============================================================
# Exercise 1: Prepare Data with Train/Val Split
# ============================================================

# Load the Diabetes dataset (10 features, 442 samples)
diabetes = load_diabetes()
X_np = diabetes.data
y_np = diabetes.target

X_all = torch.tensor(X_np, dtype=torch.float32)
y_all = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# Normalize features and target
X_mean, X_std = X_all.mean(dim=0), X_all.std(dim=0)
y_mean, y_std = y_all.mean(), y_all.std()
X_data = (X_all - X_mean) / X_std
y_data = (y_all - y_mean) / y_std

num_samples = X_data.shape[0]

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
#   Linear(10, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)
model = None

# ============================================================
# Exercise 3: Training Loop with Validation
# ============================================================

# TODO: Create MSE loss function
loss_fn = None

# TODO: Create Adam optimizer with lr=0.01 and weight_decay=0.01
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

assert n_train == int(0.8 * num_samples), f"Exercise 1: n_train should be {int(0.8 * num_samples)}, got {n_train}"
assert X_train is not None and X_train.shape[1] == 10, "Exercise 1: X_train shape wrong"
assert X_val is not None and X_val.shape[1] == 10, "Exercise 1: X_val shape wrong"

assert model is not None, "Exercise 2: model not created"
params = sum(p.numel() for p in model.parameters())
assert params == 10 * 32 + 32 + 32 * 16 + 16 + 16 * 1 + 1, "Exercise 2: wrong param count"

assert len(train_losses) == num_epochs, "Exercise 3: wrong number of epochs recorded"
assert len(val_losses) == num_epochs, "Exercise 3: wrong number of val losses"

assert final_train_loss is not None, "Exercise 4: final_train_loss not set"
assert final_val_loss is not None, "Exercise 4: final_val_loss not set"
assert final_train_loss < 0.5, f"Exercise 4: train_loss too high ({final_train_loss:.4f})"
assert final_val_loss < 1.5, f"Exercise 4: val_loss too high ({final_val_loss:.4f})"

print(f"Final train loss: {final_train_loss:.4f}")
print(f"Final val loss: {final_val_loss:.4f}")
print("All exercises passed!")

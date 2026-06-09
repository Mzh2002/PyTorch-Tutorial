"""
Lesson 06: Loss Functions and Optimizers
==========================================
Loss functions measure how wrong our predictions are.
Optimizers update model parameters to minimize the loss.
"""

import torch
import torch.nn as nn

# ============================================================
# 1. Common Loss Functions
# ============================================================

print("=== Loss Functions ===\n")

# --- Mean Squared Error (MSE) --- for regression
mse_loss = nn.MSELoss()

predictions = torch.tensor([2.5, 3.0, 4.5])
targets = torch.tensor([3.0, 3.0, 5.0])

loss = mse_loss(predictions, targets)
# MSE = mean((2.5-3)^2 + (3-3)^2 + (4.5-5)^2) = mean(0.25 + 0 + 0.25) = 0.1667
print(f"MSE Loss: {loss.item():.4f}")

# --- L1 Loss (Mean Absolute Error) --- for regression
l1_loss = nn.L1Loss()
loss_l1 = l1_loss(predictions, targets)
print(f"L1 Loss: {loss_l1.item():.4f}")

# --- Cross Entropy Loss --- for classification
ce_loss = nn.CrossEntropyLoss()

# For CrossEntropyLoss:
#   - Input: raw logits (NOT softmax), shape [batch_size, num_classes]
#   - Target: class indices, shape [batch_size]
logits = torch.tensor([[2.0, 1.0, 0.1],    # sample 1: model thinks class 0
                       [0.5, 2.5, 0.3]])    # sample 2: model thinks class 1
targets_ce = torch.tensor([0, 1])            # true labels: class 0, class 1

loss_ce = ce_loss(logits, targets_ce)
print(f"Cross Entropy Loss: {loss_ce.item():.4f}")

# --- Binary Cross Entropy --- for binary classification
bce_loss = nn.BCELoss()
# Input must be probabilities (after sigmoid)
probs = torch.tensor([0.9, 0.2, 0.8])      # predicted probabilities
targets_bce = torch.tensor([1.0, 0.0, 1.0])  # true labels

loss_bce = bce_loss(probs, targets_bce)
print(f"BCE Loss: {loss_bce.item():.4f}")

# BCEWithLogitsLoss combines sigmoid + BCE (more numerically stable)
bce_logits_loss = nn.BCEWithLogitsLoss()
raw_logits = torch.tensor([2.0, -1.5, 1.5])
loss_bcel = bce_logits_loss(raw_logits, targets_bce)
print(f"BCE with Logits Loss: {loss_bcel.item():.4f}")

# ============================================================
# 2. Common Optimizers
# ============================================================

print("\n=== Optimizers ===\n")

# Create a simple model for demonstration
model = nn.Linear(3, 1)

# --- SGD (Stochastic Gradient Descent) ---
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
print(f"SGD: lr=0.01")

# --- SGD with momentum ---
optimizer_sgd_m = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"SGD+Momentum: lr=0.01, momentum=0.9")

# --- Adam (Adaptive Moment Estimation) --- most popular
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"Adam: lr=0.001")

# --- AdamW (Adam with weight decay) ---
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print(f"AdamW: lr=0.001, weight_decay=0.01")

# ============================================================
# 3. The Optimizer Step
# ============================================================

print("\n=== Optimizer Step Demo ===\n")

# Simple model
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# Print initial weights
print(f"Initial weight: {model.weight.item():.4f}")
print(f"Initial bias: {model.bias.item():.4f}")

# One training step
x = torch.tensor([[2.0]])
y_true = torch.tensor([[5.0]])

# 1. Forward pass
y_pred = model(x)
loss = loss_fn(y_pred, y_true)
print(f"\nPrediction: {y_pred.item():.4f}, Loss: {loss.item():.4f}")

# 2. Zero gradients (IMPORTANT: do this before backward!)
optimizer.zero_grad()

# 3. Backward pass
loss.backward()
print(f"Weight gradient: {model.weight.grad.item():.4f}")
print(f"Bias gradient: {model.bias.grad.item():.4f}")

# 4. Update parameters
optimizer.step()
print(f"\nUpdated weight: {model.weight.item():.4f}")
print(f"Updated bias: {model.bias.item():.4f}")

# ============================================================
# 4. Complete Training Example
# ============================================================

print("\n=== Training Linear Regression with nn.Module ===\n")

torch.manual_seed(42)

# Generate data: y = 3x - 2
X = torch.rand(50, 1) * 10
y = 3 * X - 2 + torch.randn(50, 1) * 0.5

# Model, loss, optimizer
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

print(f"\nLearned: y = {model.weight.item():.3f}*x + {model.bias.item():.3f}")
print(f"True:    y = 3.000*x + -2.000")

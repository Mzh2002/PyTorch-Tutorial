"""
Lesson 06 Answer: Loss Functions and Optimizers
================================================
"""

import torch
import torch.nn as nn

# ============================================================
# Exercise 1: Compute MSE Loss
# ============================================================

predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
targets = torch.tensor([1.5, 2.5, 3.5, 4.5])

loss_fn_mse = nn.MSELoss()
mse_value = loss_fn_mse(predictions, targets)
print(f"MSE: {mse_value.item():.4f}")  # 0.25

# ============================================================
# Exercise 2: Cross Entropy Loss
# ============================================================

logits = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [4.0, 3.0, 2.0, 1.0],
                       [1.0, 1.0, 5.0, 1.0]])
true_labels = torch.tensor([3, 0, 2])

loss_fn_ce = nn.CrossEntropyLoss()
ce_value = loss_fn_ce(logits, true_labels)
print(f"CE: {ce_value.item():.4f}")

# ============================================================
# Exercise 3: Train a Model on Real Data
# ============================================================

from sklearn.datasets import fetch_california_housing

torch.manual_seed(99)

housing = fetch_california_housing()
feature_idx = 4  # Population

X_raw = torch.tensor(housing.data[:200, feature_idx], dtype=torch.float32).unsqueeze(1)
y_raw = torch.tensor(housing.target[:200], dtype=torch.float32).unsqueeze(1)

X_mean, X_std = X_raw.mean(), X_raw.std()
y_mean, y_std = y_raw.mean(), y_raw.std()
X = (X_raw - X_mean) / X_std
y = (y_raw - y_mean) / y_std

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(200):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

final_weight = model.weight.item()
final_bias = model.bias.item()

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert abs(mse_value.item() - 0.25) < 1e-5
assert ce_value.item() < 0.5

with torch.no_grad():
    y_pred = model(X)
    final_loss = nn.MSELoss()(y_pred, y).item()
assert final_loss < 1.0

print(f"Learned: y_norm = {final_weight:.3f}*x_norm + {final_bias:.3f}")
print(f"Final loss: {final_loss:.4f}")
print("All exercises passed!")

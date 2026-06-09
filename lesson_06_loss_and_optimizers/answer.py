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
# Exercise 3: Train a Model
# ============================================================

torch.manual_seed(99)

X = torch.rand(60, 1) * 8
y = -2 * X + 5 + torch.randn(60, 1) * 0.3

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for epoch in range(2000):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

final_weight = model.weight.item()
final_bias = model.bias.item()

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert abs(mse_value.item() - 0.25) < 1e-5
assert ce_value.item() < 0.5
assert abs(final_weight - (-2.0)) < 0.3
assert abs(final_bias - 5.0) < 0.6

print(f"Learned: y = {final_weight:.3f}*x + {final_bias:.3f}")
print("All exercises passed!")

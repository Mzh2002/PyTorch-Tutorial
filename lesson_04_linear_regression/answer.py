"""
Lesson 04 Answer: Linear Regression from Scratch
==================================================
"""

import torch

torch.manual_seed(123)

# ============================================================
# Exercise 1: Generate Data
# ============================================================

X = torch.rand(80, 1) * 5
noise = torch.randn(80, 1) * 0.3
y = -1.5 * X + 7 + noise

# ============================================================
# Exercise 2: Initialize Parameters
# ============================================================

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ============================================================
# Exercise 3: Training Loop
# ============================================================

learning_rate = 0.05
num_epochs = 500

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

final_w = w.item()
final_b = b.item()

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert X.shape == (80, 1)
assert y.shape == (80, 1)
assert abs(final_w - (-1.5)) < 0.2
assert abs(final_b - 7.0) < 0.5

print(f"Learned w: {final_w:.4f} (true: -1.5)")
print(f"Learned b: {final_b:.4f} (true: 7.0)")
print("All exercises passed!")

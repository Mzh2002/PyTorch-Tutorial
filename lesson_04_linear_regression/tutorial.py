"""
Lesson 04: Linear Regression from Scratch
===========================================
We implement linear regression using only tensors and autograd —
no nn.Module yet. This shows how gradient descent works at the lowest level.
"""

import torch

# ============================================================
# 1. Generate Synthetic Data
# ============================================================

# True relationship: y = 2*x + 3 (with some noise)
torch.manual_seed(42)

X = torch.rand(100, 1) * 10          # 100 samples, x in [0, 10]
noise = torch.randn(100, 1) * 0.5    # small Gaussian noise
y = 2 * X + 3 + noise                # true: weight=2, bias=3

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"First 5 samples:")
for i in range(5):
    print(f"  x={X[i].item():.2f}, y={y[i].item():.2f}")

# ============================================================
# 2. Initialize Parameters
# ============================================================

# We want to learn: y_pred = w * x + b
# Start with random values
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(f"\nInitial w: {w.item():.4f}")
print(f"Initial b: {b.item():.4f}")

# ============================================================
# 3. Define Forward Pass (Prediction)
# ============================================================


def forward(X):
    """Predict y given X using current w and b."""
    return X * w + b


# ============================================================
# 4. Define Loss Function (Mean Squared Error)
# ============================================================


def mse_loss(y_pred, y_true):
    """Compute mean squared error."""
    return ((y_pred - y_true) ** 2).mean()


# ============================================================
# 5. Training Loop
# ============================================================

learning_rate = 0.01
num_epochs = 100

print("\n--- Training ---")
for epoch in range(num_epochs):
    # Forward pass
    y_pred = forward(X)

    # Compute loss
    loss = mse_loss(y_pred, y)

    # Backward pass (compute gradients)
    loss.backward()

    # Update parameters (without tracking gradients)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

    # Print progress
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, "
              f"w={w.item():.4f}, b={b.item():.4f}")

# ============================================================
# 6. Check Results
# ============================================================

print("\n--- Results ---")
print(f"Learned w: {w.item():.4f} (true: 2.0)")
print(f"Learned b: {b.item():.4f} (true: 3.0)")

# Make predictions on new data
x_test = torch.tensor([[1.0], [5.0], [10.0]])
with torch.no_grad():
    y_test = forward(x_test)
    print("\nPredictions:")
    for i in range(len(x_test)):
        expected = 2 * x_test[i].item() + 3
        print(f"  x={x_test[i].item():.1f}: "
              f"predicted={y_test[i].item():.2f}, "
              f"expected={expected:.2f}")

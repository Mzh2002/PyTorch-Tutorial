"""
Lesson 03: Autograd and Gradients
==================================
Autograd is PyTorch's automatic differentiation engine. It tracks operations
on tensors and computes gradients automatically — essential for training
neural networks via backpropagation.
"""

import torch

# ============================================================
# 1. requires_grad: Telling PyTorch to Track Gradients
# ============================================================

# By default, tensors don't track gradients
x = torch.tensor([1.0, 2.0, 3.0])
print("Requires grad (default):", x.requires_grad)  # False

# Enable gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print("Requires grad (enabled):", x.requires_grad)  # True

# ============================================================
# 2. Computing Gradients with .backward()
# ============================================================

# Simple example: y = x^2, dy/dx = 2x
x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
y = x ** 2           # y = [4, 9, 16]
z = y.sum()          # need a scalar to call .backward()

# Compute gradients
z.backward()

# The gradient dz/dx = 2x
print("\nx:", x)
print("y = x^2:", y)
print("Gradient (dz/dx = 2x):", x.grad)  # [4, 6, 8]

# ============================================================
# 3. Computational Graph
# ============================================================

# PyTorch builds a graph of operations. Each tensor has:
# - .grad_fn: the function that created it
# - .requires_grad: whether gradients are tracked

a = torch.tensor(2.0, requires_grad=True)
b = a * 3          # b = 6
c = b + 1          # c = 7
d = c ** 2         # d = 49

print(f"\na = {a.item()}")
print(f"b = a*3 = {b.item()}, grad_fn = {b.grad_fn}")
print(f"c = b+1 = {c.item()}, grad_fn = {c.grad_fn}")
print(f"d = c^2 = {d.item()}, grad_fn = {d.grad_fn}")

d.backward()
# dd/da = dd/dc * dc/db * db/da = 2c * 1 * 3 = 2*7*3 = 42
print(f"Gradient dd/da: {a.grad.item()}")  # 42

# ============================================================
# 4. Gradient Accumulation
# ============================================================

# IMPORTANT: Gradients accumulate by default!
x = torch.tensor(3.0, requires_grad=True)

y1 = x ** 2
y1.backward()
print(f"\nAfter first backward: x.grad = {x.grad.item()}")  # 6

y2 = x ** 3
y2.backward()
print(f"After second backward: x.grad = {x.grad.item()}")  # 6 + 27 = 33

# You must zero gradients before each new computation!
x.grad.zero_()
y3 = x * 5
y3.backward()
print(f"After zeroing and third backward: x.grad = {x.grad.item()}")  # 5

# ============================================================
# 5. Detaching from the Graph
# ============================================================

x = torch.tensor(2.0, requires_grad=True)
y = x * 3

# .detach() creates a tensor that doesn't track gradients
y_detached = y.detach()
print(f"\ny_detached requires_grad: {y_detached.requires_grad}")  # False

# torch.no_grad() context: useful during inference
with torch.no_grad():
    z = x * 4
    print(f"z requires_grad (in no_grad): {z.requires_grad}")  # False

# ============================================================
# 6. Practical Example: Gradient Descent Step
# ============================================================

# Minimize f(x) = (x - 5)^2
# Gradient: df/dx = 2(x - 5)
# Minimum at x = 5

x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

print("\n--- Gradient Descent ---")
for step in range(20):
    # Forward pass
    loss = (x - 5) ** 2

    # Backward pass
    loss.backward()

    # Update x (must be done without tracking gradients)
    with torch.no_grad():
        x -= learning_rate * x.grad

    # Zero the gradient for next iteration
    x.grad.zero_()

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, loss = {loss.item():.4f}")

print(f"Final x: {x.item():.4f} (should be close to 5.0)")

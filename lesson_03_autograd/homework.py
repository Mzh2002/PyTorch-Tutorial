"""
Lesson 03 Homework: Autograd and Gradients
============================================
Complete the TODOs below.
"""

import torch

# ============================================================
# Exercise 1: Compute Gradients
# ============================================================

# TODO: Create a tensor x = [1.0, 2.0, 3.0] with requires_grad=True
x = None

# TODO: Compute y = 3*x + 2 (element-wise)
y = None

# TODO: Compute z = y.sum() (we need a scalar for backward)
z = None

# TODO: Call z.backward() to compute gradients

# TODO: Store x.grad in a variable called grad_x
grad_x = None
# Expected: grad_x should be [3, 3, 3] since dz/dx = 3

# ============================================================
# Exercise 2: Chain Rule
# ============================================================

# TODO: Create a = tensor(3.0) with requires_grad=True
a = None

# TODO: Compute b = a^2
b = None

# TODO: Compute c = 2*b + 1
c = None

# TODO: Call c.backward()

# TODO: What is a.grad? Store it as grad_a
grad_a = None
# Expected: dc/da = dc/db * db/da = 2 * 2a = 4a = 12

# ============================================================
# Exercise 3: Gradient Descent
# ============================================================

# Minimize f(w) = (w - 3)^2
# Starting at w = 10.0

# TODO: Create w = tensor(10.0) with requires_grad=True
w = None

learning_rate = 0.1

# TODO: Run 50 steps of gradient descent
# In each step:
#   1. Compute loss = (w - 3)^2
#   2. Call loss.backward()
#   3. Update w using: w.data -= learning_rate * w.grad
#   4. Zero the gradient: w.grad.zero_()

for step in range(50):
    pass  # TODO: Replace this with the gradient descent logic

# After 50 steps, w should be close to 3.0
final_w = None  # TODO: Store w.item() here

# ============================================================
# Exercise 4: no_grad context
# ============================================================

x = torch.tensor(5.0, requires_grad=True)
y = x ** 2

# TODO: Create a variable z that equals x * 2, but computed
#       inside a torch.no_grad() block
z = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert grad_x is not None, "Exercise 1: grad_x not set"
assert grad_x.tolist() == [3.0, 3.0, 3.0], f"Exercise 1: expected [3,3,3], got {grad_x.tolist()}"

assert grad_a is not None, "Exercise 2: grad_a not set"
assert abs(grad_a.item() - 12.0) < 1e-5, f"Exercise 2: expected 12.0, got {grad_a.item()}"

assert final_w is not None, "Exercise 3: final_w not set"
assert abs(final_w - 3.0) < 0.01, f"Exercise 3: w should be ~3.0, got {final_w}"

assert z is not None, "Exercise 4: z not set"
assert not z.requires_grad, "Exercise 4: z should not require grad"
assert abs(z.item() - 10.0) < 1e-5, "Exercise 4: z should be 10.0"

print("All exercises passed!")

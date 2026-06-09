"""
Lesson 03 Answer: Autograd and Gradients
==========================================
"""

import torch

# ============================================================
# Exercise 1: Compute Gradients
# ============================================================

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = 3 * x + 2
z = y.sum()
z.backward()
grad_x = x.grad
print("grad_x:", grad_x)  # [3, 3, 3]

# ============================================================
# Exercise 2: Chain Rule
# ============================================================

a = torch.tensor(3.0, requires_grad=True)
b = a ** 2
c = 2 * b + 1
c.backward()
grad_a = a.grad
print("grad_a:", grad_a)  # 12.0

# ============================================================
# Exercise 3: Gradient Descent
# ============================================================

w = torch.tensor(10.0, requires_grad=True)
learning_rate = 0.1

for step in range(50):
    loss = (w - 3) ** 2
    loss.backward()
    w.data -= learning_rate * w.grad
    w.grad.zero_()

final_w = w.item()
print(f"final_w: {final_w:.4f}")  # ~3.0

# ============================================================
# Exercise 4: no_grad context
# ============================================================

x = torch.tensor(5.0, requires_grad=True)
y = x ** 2

with torch.no_grad():
    z = x * 2

print(f"z = {z.item()}, requires_grad = {z.requires_grad}")

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert grad_x.tolist() == [3.0, 3.0, 3.0]
assert abs(grad_a.item() - 12.0) < 1e-5
assert abs(final_w - 3.0) < 0.01
assert not z.requires_grad
assert abs(z.item() - 10.0) < 1e-5

print("All exercises passed!")

"""
Lesson 02: Tensor Operations and Broadcasting
===============================================
PyTorch provides a rich set of mathematical operations on tensors.
Broadcasting allows operations between tensors of different shapes.
"""

import torch

# ============================================================
# 1. Basic Arithmetic Operations
# ============================================================

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
print("Addition:", a + b)          # [5, 7, 9]
print("Subtraction:", a - b)       # [-3, -3, -3]
print("Multiplication:", a * b)    # [4, 10, 18]
print("Division:", a / b)          # [0.25, 0.4, 0.5]
print("Power:", a ** 2)            # [1, 4, 9]

# Equivalent function calls
print("\ntorch.add:", torch.add(a, b))
print("torch.mul:", torch.mul(a, b))

# ============================================================
# 2. Reduction Operations
# ============================================================

t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print("\nSum (all):", t.sum())           # 21
print("Sum (dim=0):", t.sum(dim=0))     # sum along rows -> [5, 7, 9]
print("Sum (dim=1):", t.sum(dim=1))     # sum along cols -> [6, 15]
print("Mean:", t.mean())                 # 3.5
print("Max:", t.max())                   # 6
print("Min:", t.min())                   # 1
print("Argmax:", t.argmax())             # index of max element (flat) = 5

# keepdim preserves the reduced dimension
print("\nSum keepdim:", t.sum(dim=1, keepdim=True))  # shape [2, 1]

# ============================================================
# 3. Matrix Operations
# ============================================================

A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

# Matrix multiplication
print("\nMatrix multiply (@ operator):\n", A @ B)
print("torch.matmul:\n", torch.matmul(A, B))
print("torch.mm:\n", torch.mm(A, B))  # only for 2D tensors

# Transpose
print("\nTranspose:\n", A.T)
print("Transpose (explicit):\n", A.transpose(0, 1))

# Dot product (1D tensors only)
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])
print("\nDot product:", torch.dot(v1, v2))  # 1*4 + 2*5 + 3*6 = 32

# ============================================================
# 4. Comparison Operations
# ============================================================

x = torch.tensor([1, 2, 3, 4, 5])

print("\n> 3:", x > 3)               # [False, False, False, True, True]
print("== 3:", x == 3)              # [False, False, True, False, False]
print("Where (x > 3):", torch.where(x > 3, x, torch.zeros_like(x)))

# ============================================================
# 5. Broadcasting
# ============================================================

# Broadcasting allows operations between tensors of different shapes
# Rules:
# 1. Dimensions are compared from right to left
# 2. Dimensions must be equal OR one of them must be 1
# 3. A dimension of size 1 is "stretched" to match the other

# Example: adding a scalar to a tensor
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print("\nOriginal:\n", t)
print("Add scalar 10:\n", t + 10)  # scalar is broadcast to all elements

# Example: adding a row vector to a matrix
row = torch.tensor([10.0, 20.0, 30.0])  # shape [3]
print("Add row vector:\n", t + row)       # row is broadcast across rows

# Example: adding a column vector to a matrix
col = torch.tensor([[100.0],
                    [200.0]])            # shape [2, 1]
print("Add col vector:\n", t + col)      # col is broadcast across columns

# More complex example
a = torch.rand(3, 1)   # shape [3, 1]
b = torch.rand(1, 4)   # shape [1, 4]
c = a + b               # shape [3, 4] — both dimensions are broadcast
print(f"\nBroadcast: {a.shape} + {b.shape} = {c.shape}")

# ============================================================
# 6. In-place Operations
# ============================================================

# Operations ending in _ modify the tensor in place
t = torch.tensor([1.0, 2.0, 3.0])
print("\nBefore add_:", t)
t.add_(5)
print("After add_(5):", t)  # [6, 7, 8]

t.mul_(2)
print("After mul_(2):", t)  # [12, 14, 16]

# ============================================================
# 7. Indexing and Slicing
# ============================================================

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print("\nRow 0:", t[0])
print("Element [1,2]:", t[1, 2])       # 7
print("Column 1:", t[:, 1])            # [2, 6, 10]
print("Submatrix:\n", t[0:2, 1:3])    # top-left 2x2

# Boolean indexing
mask = t > 5
print("Mask:\n", mask)
print("Filtered:", t[mask])            # all elements > 5

# ============================================================
# 8. Concatenation and Stacking
# ============================================================

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# cat joins along an existing dimension
print("\nCat dim=0:\n", torch.cat([a, b], dim=0))  # shape [4, 2]
print("Cat dim=1:\n", torch.cat([a, b], dim=1))    # shape [2, 4]

# stack creates a new dimension
print("Stack dim=0:\n", torch.stack([a, b], dim=0))  # shape [2, 2, 2]

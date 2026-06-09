"""
Lesson 02 Answer: Tensor Operations and Broadcasting
=====================================================
"""

import torch

# ============================================================
# Exercise 1: Basic Operations
# ============================================================

a = torch.tensor([2.0, 4.0, 6.0])
b = torch.tensor([1.0, 2.0, 3.0])

# Element-wise product
product = a * b

# Element-wise division
division = a / b

# a squared
squared = a ** 2

# ============================================================
# Exercise 2: Reductions
# ============================================================

t = torch.tensor([[10.0, 20.0, 30.0],
                  [40.0, 50.0, 60.0]])

# Mean of all elements
mean_all = t.mean()

# Sum along dim=1
row_sums = t.sum(dim=1)

# Index of the maximum element (flattened)
max_idx = t.argmax()

# ============================================================
# Exercise 3: Matrix Multiplication
# ============================================================

A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

B = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])

# Matrix product
C = A @ B

# ============================================================
# Exercise 4: Broadcasting
# ============================================================

matrix = torch.ones(3, 4)

# Row vector broadcast
row_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
result_row = matrix + row_vec

# Column vector broadcast
col_vec = torch.tensor([[10.0], [20.0], [30.0]])
result_col = matrix + col_vec

# ============================================================
# Exercise 5: Indexing
# ============================================================

t = torch.arange(20).reshape(4, 5)

# Element at row 2, column 3
element = t[2, 3]

# Entire third row
third_row = t[2]

# All elements greater than 10
greater_than_10 = t[t > 10]

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert product.tolist() == [2.0, 8.0, 18.0]
assert division.tolist() == [2.0, 2.0, 2.0]
assert squared.tolist() == [4.0, 16.0, 36.0]
assert abs(mean_all.item() - 35.0) < 1e-5
assert row_sums.tolist() == [60.0, 150.0]
assert max_idx.item() == 5
assert list(C.shape) == [2, 2]
assert torch.allclose(C, torch.tensor([[4.0, 5.0], [10.0, 11.0]]))
assert list(result_row.shape) == [3, 4]
assert list(result_col.shape) == [3, 4]
assert element.item() == 13
assert third_row.tolist() == [10, 11, 12, 13, 14]
assert len(greater_than_10) == 9

print("All exercises passed!")

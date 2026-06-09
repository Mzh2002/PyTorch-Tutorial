"""
Lesson 02 Homework: Tensor Operations and Broadcasting
========================================================
Complete the TODOs below.
"""

import torch

# ============================================================
# Exercise 1: Basic Operations
# ============================================================

a = torch.tensor([2.0, 4.0, 6.0])
b = torch.tensor([1.0, 2.0, 3.0])

# TODO: Compute element-wise product of a and b
product = None

# TODO: Compute element-wise division a / b
division = None

# TODO: Compute a squared (a^2)
squared = None

# ============================================================
# Exercise 2: Reductions
# ============================================================

t = torch.tensor([[10.0, 20.0, 30.0],
                  [40.0, 50.0, 60.0]])

# TODO: Compute the mean of all elements
mean_all = None

# TODO: Compute the sum along dim=1 (sum of each row)
row_sums = None

# TODO: Find the index of the maximum element in t (flattened)
max_idx = None

# ============================================================
# Exercise 3: Matrix Multiplication
# ============================================================

A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # shape [2, 3]

B = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])       # shape [3, 2]

# TODO: Compute the matrix product A @ B (result should be [2, 2])
C = None

# ============================================================
# Exercise 4: Broadcasting
# ============================================================

# Matrix of shape [3, 4]
matrix = torch.ones(3, 4)

# TODO: Create a row vector of shape [4] containing [1, 2, 3, 4]
#       and add it to the matrix using broadcasting
row_vec = None
result_row = None

# TODO: Create a column vector of shape [3, 1] containing [10, 20, 30]
#       and add it to the matrix using broadcasting
col_vec = None
result_col = None

# ============================================================
# Exercise 5: Indexing
# ============================================================

t = torch.arange(20).reshape(4, 5)
# t looks like:
# [[ 0,  1,  2,  3,  4],
#  [ 5,  6,  7,  8,  9],
#  [10, 11, 12, 13, 14],
#  [15, 16, 17, 18, 19]]

# TODO: Extract the element at row 2, column 3
element = None

# TODO: Extract the entire third row (index 2)
third_row = None

# TODO: Extract all elements greater than 10
greater_than_10 = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert product is not None and product.tolist() == [2.0, 8.0, 18.0], "Exercise 1a failed"
assert division is not None and division.tolist() == [2.0, 2.0, 2.0], "Exercise 1b failed"
assert squared is not None and squared.tolist() == [4.0, 16.0, 36.0], "Exercise 1c failed"

assert mean_all is not None and abs(mean_all.item() - 35.0) < 1e-5, "Exercise 2a failed"
assert row_sums is not None and row_sums.tolist() == [60.0, 150.0], "Exercise 2b failed"
assert max_idx is not None and max_idx.item() == 5, "Exercise 2c failed"

assert C is not None and list(C.shape) == [2, 2], "Exercise 3 shape wrong"
expected_C = torch.tensor([[4.0, 5.0], [10.0, 11.0]])
assert torch.allclose(C, expected_C), f"Exercise 3: expected {expected_C}, got {C}"

assert result_row is not None and list(result_row.shape) == [3, 4], "Exercise 4a shape wrong"
assert result_col is not None and list(result_col.shape) == [3, 4], "Exercise 4b shape wrong"

assert element is not None and element.item() == 13, "Exercise 5a failed"
assert third_row is not None and third_row.tolist() == [10, 11, 12, 13, 14], "Exercise 5b failed"
assert greater_than_10 is not None and len(greater_than_10) == 9, "Exercise 5c failed"

print("All exercises passed!")

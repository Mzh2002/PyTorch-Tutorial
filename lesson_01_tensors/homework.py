"""
Lesson 01 Homework: Tensors, Shapes, Dtypes, and Devices
=========================================================
Complete the TODOs below. Run this file to check your work.
"""

import torch

# ============================================================
# Exercise 1: Create tensors
# ============================================================

# TODO: Create a 1D tensor containing [10, 20, 30, 40, 50]
t1 = None

# TODO: Create a 3x3 tensor filled with the value 7
t2 = None

# TODO: Create a 4x4 identity matrix (hint: use torch.eye)
t3 = None

# ============================================================
# Exercise 2: Inspect shapes
# ============================================================

t = torch.rand(3, 4, 5)

# TODO: Print the shape of t
# print(...)

# TODO: Print the total number of elements in t
# print(...)

# ============================================================
# Exercise 3: Reshape tensors
# ============================================================

t4 = torch.arange(12)

# TODO: Reshape t4 into a 3x4 matrix
t4_reshaped = None

# TODO: Reshape t4 into a 2x2x3 tensor
t4_3d = None

# ============================================================
# Exercise 4: Data types
# ============================================================

# TODO: Create a tensor [1.5, 2.5, 3.5] with dtype float64
t5 = None

# TODO: Cast t5 to int32 (note: this truncates decimals)
t5_int = None

# ============================================================
# Exercise 5: Device awareness
# ============================================================

# TODO: Create a tensor and print its device
t6 = None
# print(...)

# TODO: If CUDA is available, move t6 to GPU and print its device
# (If no GPU, just print "No GPU available")


# ============================================================
# Validation (do not modify below)
# ============================================================
print("\n--- Checking your answers ---")

assert t1 is not None, "Exercise 1a: t1 not created"
assert list(t1.shape) == [5], f"Exercise 1a: expected shape [5], got {list(t1.shape)}"
assert t1.tolist() == [10, 20, 30, 40, 50], "Exercise 1a: wrong values"

assert t2 is not None, "Exercise 1b: t2 not created"
assert list(t2.shape) == [3, 3], f"Exercise 1b: expected shape [3,3], got {list(t2.shape)}"
assert (t2 == 7).all(), "Exercise 1b: t2 should be all 7s"

assert t3 is not None, "Exercise 1c: t3 not created"
assert list(t3.shape) == [4, 4], f"Exercise 1c: expected shape [4,4], got {list(t3.shape)}"
assert t3.tolist() == torch.eye(4).tolist(), "Exercise 1c: t3 should be identity matrix"

assert t4_reshaped is not None, "Exercise 3a: t4_reshaped not created"
assert list(t4_reshaped.shape) == [3, 4], f"Exercise 3a: expected [3,4], got {list(t4_reshaped.shape)}"

assert t4_3d is not None, "Exercise 3b: t4_3d not created"
assert list(t4_3d.shape) == [2, 2, 3], f"Exercise 3b: expected [2,2,3], got {list(t4_3d.shape)}"

assert t5 is not None, "Exercise 4a: t5 not created"
assert t5.dtype == torch.float64, f"Exercise 4a: expected float64, got {t5.dtype}"

assert t5_int is not None, "Exercise 4b: t5_int not created"
assert t5_int.dtype == torch.int32, f"Exercise 4b: expected int32, got {t5_int.dtype}"

print("All exercises passed!")

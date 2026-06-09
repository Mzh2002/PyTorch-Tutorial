"""
Lesson 01 Answer: Tensors, Shapes, Dtypes, and Devices
=======================================================
"""

import torch

# ============================================================
# Exercise 1: Create tensors
# ============================================================

# Create a 1D tensor containing [10, 20, 30, 40, 50]
t1 = torch.tensor([10, 20, 30, 40, 50])

# Create a 3x3 tensor filled with the value 7
t2 = torch.full((3, 3), 7)

# Create a 4x4 identity matrix
t3 = torch.eye(4)

# ============================================================
# Exercise 2: Inspect shapes
# ============================================================

t = torch.rand(3, 4, 5)

# Print the shape of t
print("Shape of t:", t.shape)

# Print the total number of elements in t
print("Total elements:", t.numel())

# ============================================================
# Exercise 3: Reshape tensors
# ============================================================

t4 = torch.arange(12)

# Reshape t4 into a 3x4 matrix
t4_reshaped = t4.reshape(3, 4)

# Reshape t4 into a 2x2x3 tensor
t4_3d = t4.reshape(2, 2, 3)

# ============================================================
# Exercise 4: Data types
# ============================================================

# Create a tensor [1.5, 2.5, 3.5] with dtype float64
t5 = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)

# Cast t5 to int32
t5_int = t5.to(torch.int32)

# ============================================================
# Exercise 5: Device awareness
# ============================================================

# Create a tensor and print its device
t6 = torch.tensor([1.0, 2.0, 3.0])
print("t6 device:", t6.device)

# If CUDA is available, move t6 to GPU
if torch.cuda.is_available():
    t6_gpu = t6.to("cuda")
    print("t6 on GPU:", t6_gpu.device)
else:
    print("No GPU available")

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert list(t1.shape) == [5]
assert t1.tolist() == [10, 20, 30, 40, 50]
assert list(t2.shape) == [3, 3]
assert (t2 == 7).all()
assert list(t3.shape) == [4, 4]
assert t3.tolist() == torch.eye(4).tolist()
assert list(t4_reshaped.shape) == [3, 4]
assert list(t4_3d.shape) == [2, 2, 3]
assert t5.dtype == torch.float64
assert t5_int.dtype == torch.int32

print("All exercises passed!")

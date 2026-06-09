"""
Lesson 01: Tensors, Shapes, Dtypes, and Devices
=================================================
Tensors are the fundamental data structure in PyTorch — similar to NumPy arrays
but with GPU support and automatic differentiation capabilities.
"""

import torch

# ============================================================
# 1. Creating Tensors
# ============================================================

# From a Python list
t1 = torch.tensor([1, 2, 3])
print("From list:", t1)

# From nested lists (2D tensor / matrix)
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("2D tensor:\n", t2)

# Common creation functions
zeros = torch.zeros(3, 4)       # 3x4 matrix of zeros
ones = torch.ones(2, 3)        # 2x3 matrix of ones
rand = torch.rand(2, 2)        # 2x2 matrix of random values in [0, 1)
randn = torch.randn(3, 3)      # 3x3 matrix from standard normal distribution
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

print("\nZeros:\n", zeros)
print("Ones:\n", ones)
print("Random:\n", rand)
print("Randn:\n", randn)
print("Arange:", arange)

# ============================================================
# 2. Tensor Shapes
# ============================================================

# .shape and .size() give the dimensions of a tensor
t = torch.rand(2, 3, 4)
print("\nShape:", t.shape)          # torch.Size([2, 3, 4])
print("Size:", t.size())            # same as .shape
print("Number of dimensions:", t.ndim)  # 3
print("Total elements:", t.numel())     # 2*3*4 = 24

# Reshaping tensors
t_flat = t.reshape(24)              # flatten to 1D
t_reshaped = t.reshape(6, 4)       # reshape to 6x4
print("\nFlattened shape:", t_flat.shape)
print("Reshaped shape:", t_reshaped.shape)

# .view() works like reshape but requires contiguous memory
t_view = t.view(2, 12)
print("View shape:", t_view.shape)

# ============================================================
# 3. Data Types (dtype)
# ============================================================

# PyTorch tensors have a specific data type
t_float = torch.tensor([1.0, 2.0, 3.0])
t_int = torch.tensor([1, 2, 3])
t_bool = torch.tensor([True, False, True])

print("\nFloat dtype:", t_float.dtype)   # torch.float32 (default for floats)
print("Int dtype:", t_int.dtype)         # torch.int64 (default for ints)
print("Bool dtype:", t_bool.dtype)       # torch.bool

# Specifying dtype explicitly
t_f16 = torch.tensor([1.0, 2.0], dtype=torch.float16)
t_f64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
print("Float16:", t_f16.dtype)
print("Float64:", t_f64.dtype)

# Casting between types
t_casted = t_int.float()        # int64 -> float32
print("Casted:", t_casted.dtype)

t_to = t_float.to(torch.int32)  # using .to()
print("To int32:", t_to.dtype)

# ============================================================
# 4. Devices (CPU vs GPU)
# ============================================================

# By default, tensors are on CPU
t_cpu = torch.tensor([1, 2, 3])
print("\nDevice:", t_cpu.device)  # cpu

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

# Move tensor to GPU (if available)
if torch.cuda.is_available():
    t_gpu = t_cpu.to("cuda")
    print("GPU tensor device:", t_gpu.device)

    # Create directly on GPU
    t_gpu2 = torch.rand(3, 3, device="cuda")
    print("Created on GPU:", t_gpu2.device)

    # Move back to CPU
    t_back = t_gpu.to("cpu")
    print("Back on CPU:", t_back.device)
else:
    print("No GPU available — all tensors stay on CPU (this is fine for learning!)")

# ============================================================
# 5. Useful Attributes Summary
# ============================================================

t = torch.randn(4, 5)
print("\n--- Tensor Attributes Summary ---")
print(f"Shape: {t.shape}")
print(f"Dtype: {t.dtype}")
print(f"Device: {t.device}")
print(f"Ndim: {t.ndim}")
print(f"Numel: {t.numel()}")
print(f"Requires grad: {t.requires_grad}")

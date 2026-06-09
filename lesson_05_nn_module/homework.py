"""
Lesson 05 Homework: nn.Module, Parameters, and Layers
=======================================================
Complete the TODOs below.
"""

import torch
import torch.nn as nn

# ============================================================
# Exercise 1: Create a Custom Model
# ============================================================

# TODO: Define a class called TwoLayerNet that:
#   - Takes input_size, hidden_size, output_size in __init__
#   - Has two Linear layers and a ReLU activation between them
#   - The forward method applies: linear1 -> relu -> linear2


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # TODO: Define layers
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


# ============================================================
# Exercise 2: Instantiate and Inspect
# ============================================================

# TODO: Create an instance with input_size=10, hidden_size=20, output_size=3
model = None

# TODO: Count total number of parameters
total_params = None

# ============================================================
# Exercise 3: Forward Pass
# ============================================================

# TODO: Create a random input batch of shape [8, 10] (8 samples, 10 features)
x = None

# TODO: Pass it through the model and store the output
output = None

# ============================================================
# Exercise 4: Sequential Model
# ============================================================

# TODO: Create a Sequential model with:
#   Linear(5, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)
seq_model = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert model is not None, "Exercise 2: model not created"
assert isinstance(model, nn.Module), "Exercise 2: model should be nn.Module"

# Check parameters: linear1 has 10*20+20=220, linear2 has 20*3+3=63, total=283
assert total_params == 283, f"Exercise 2: expected 283 params, got {total_params}"

assert output is not None, "Exercise 3: output not computed"
assert list(output.shape) == [8, 3], f"Exercise 3: expected [8,3], got {list(output.shape)}"

assert seq_model is not None, "Exercise 4: seq_model not created"
seq_out = seq_model(torch.randn(4, 5))
assert list(seq_out.shape) == [4, 1], f"Exercise 4: expected [4,1], got {list(seq_out.shape)}"

print("All exercises passed!")

"""
Lesson 05 Answer: nn.Module, Parameters, and Layers
=====================================================
"""

import torch
import torch.nn as nn

# ============================================================
# Exercise 1: Create a Custom Model
# ============================================================


class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# ============================================================
# Exercise 2: Instantiate and Inspect
# ============================================================

model = TwoLayerNet(input_size=10, hidden_size=20, output_size=3)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")  # 10*20+20 + 20*3+3 = 283

# ============================================================
# Exercise 3: Forward Pass
# ============================================================

x = torch.randn(8, 10)
output = model(x)
print(f"Output shape: {output.shape}")  # [8, 3]

# ============================================================
# Exercise 4: Sequential Model
# ============================================================

seq_model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

seq_out = seq_model(torch.randn(4, 5))
print(f"Sequential output shape: {seq_out.shape}")  # [4, 1]

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert isinstance(model, nn.Module)
assert total_params == 283
assert list(output.shape) == [8, 3]
assert list(seq_out.shape) == [4, 1]

print("All exercises passed!")

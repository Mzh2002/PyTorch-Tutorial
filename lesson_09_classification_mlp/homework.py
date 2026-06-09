"""
Lesson 09 Homework: Classification with MLP
=============================================
Complete the TODOs to build and train an MLP classifier on FashionMNIST.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(9)

# ============================================================
# Exercise 1: Load Data
# ============================================================

# TODO: Create a transform that converts images to tensors
transform = None

# TODO: Load FashionMNIST training data (root="./data", download=True)
train_dataset = None

# TODO: Load FashionMNIST test data
test_dataset = None

# TODO: Create DataLoaders with batch_size=128
train_loader = None
test_loader = None

# ============================================================
# Exercise 2: Define Model
# ============================================================

# TODO: Create an MLP class with:
#   - Flatten layer
#   - Linear(784, 512) -> ReLU -> Linear(512, 256) -> ReLU -> Linear(256, 10)


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define layers
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


# TODO: Instantiate the model
model = None

# ============================================================
# Exercise 3: Train
# ============================================================

# TODO: Define CrossEntropyLoss and Adam optimizer (lr=0.001)
loss_fn = None
optimizer = None

# TODO: Train for 3 epochs, printing loss and accuracy each epoch
num_epochs = 3

for epoch in range(num_epochs):
    pass  # TODO: Implement training loop

# ============================================================
# Exercise 4: Evaluate
# ============================================================

# TODO: Compute test accuracy
test_accuracy = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert train_dataset is not None and len(train_dataset) == 60000, "Exercise 1: train data wrong"
assert test_dataset is not None and len(test_dataset) == 10000, "Exercise 1: test data wrong"
assert model is not None, "Exercise 2: model not created"

params = sum(p.numel() for p in model.parameters())
# 784*512+512 + 512*256+256 + 256*10+10 = 534,794
expected_params = 784 * 512 + 512 + 512 * 256 + 256 + 256 * 10 + 10
assert params == expected_params, f"Exercise 2: expected {expected_params} params, got {params}"

assert test_accuracy is not None, "Exercise 4: test_accuracy not computed"
assert test_accuracy > 70.0, f"Exercise 4: accuracy too low ({test_accuracy:.1f}%)"

print(f"Test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

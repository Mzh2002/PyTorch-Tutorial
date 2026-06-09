"""
Lesson 10 Homework: CNN Basics
================================
Complete the TODOs to build and train a CNN.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(10)

# ============================================================
# Exercise 1: Understand Conv2d Output Shapes
# ============================================================

# Given an input of shape [1, 1, 32, 32]:
x = torch.randn(1, 1, 32, 32)

# TODO: Create a Conv2d that outputs shape [1, 8, 32, 32]
#       (hint: use padding to preserve spatial size)
conv1 = None
out1 = None  # TODO: pass x through conv1

# TODO: Create a Conv2d that takes the output of conv1 and
#       produces shape [1, 16, 16, 16] using stride=2
conv2 = None
out2 = None  # TODO: pass out1 through conv2

# ============================================================
# Exercise 2: Build a CNN
# ============================================================

# TODO: Define a CNN class for FashionMNIST (1x28x28 input, 10 classes)
# Architecture:
#   Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)    -> [32, 14, 14]
#   Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)   -> [64, 7, 7]
#   Flatten -> Linear(64*7*7, 256) -> ReLU -> Linear(256, 10)


class FashionCNN(nn.Module):
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
# Exercise 3: Train and Evaluate
# ============================================================

# Load data
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# TODO: Define loss function and optimizer
loss_fn = None
optimizer = None

# TODO: Train for 3 epochs
num_epochs = 3

for epoch in range(num_epochs):
    pass  # TODO: Implement training loop

# TODO: Compute test accuracy
test_accuracy = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert out1 is not None and list(out1.shape) == [1, 8, 32, 32], \
    f"Exercise 1a: expected [1,8,32,32], got {list(out1.shape) if out1 is not None else None}"
assert out2 is not None and list(out2.shape) == [1, 16, 16, 16], \
    f"Exercise 1b: expected [1,16,16,16], got {list(out2.shape) if out2 is not None else None}"

assert model is not None, "Exercise 2: model not created"
test_input = torch.randn(2, 1, 28, 28)
test_output = model(test_input)
assert list(test_output.shape) == [2, 10], f"Exercise 2: output shape wrong: {list(test_output.shape)}"

assert test_accuracy is not None, "Exercise 3: test_accuracy not computed"
assert test_accuracy > 75.0, f"Exercise 3: accuracy too low ({test_accuracy:.1f}%)"

print(f"Test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

"""
Lesson 11 Homework: Image Classification Project
==================================================
Complete the TODOs to build a full classification pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(11)

# ============================================================
# Exercise 1: Define Transforms
# ============================================================

# TODO: Create training transform with:
#   - RandomHorizontalFlip
#   - ToTensor
#   - Normalize with mean=0.2860, std=0.3530
train_transform = None

# TODO: Create test transform with:
#   - ToTensor
#   - Normalize with mean=0.2860, std=0.3530
test_transform = None

# ============================================================
# Exercise 2: Load Data and Create Loaders
# ============================================================

# TODO: Load FashionMNIST train and test datasets with transforms
train_dataset = None
test_dataset = None

# TODO: Create DataLoaders (batch_size=64)
train_loader = None
test_loader = None

# ============================================================
# Exercise 3: Define CNN with BatchNorm and Dropout
# ============================================================

# TODO: Create a CNN with:
#   Conv2d(1, 32, 3, padding=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2) -> Dropout2d(0.2)
#   Conv2d(32, 64, 3, padding=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2) -> Dropout2d(0.2)
#   Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 10)


class MyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define layers
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


model = None  # TODO: Instantiate

# ============================================================
# Exercise 4: Train with LR Scheduler
# ============================================================

# TODO: Define CrossEntropyLoss, Adam optimizer (lr=0.001),
#       and StepLR scheduler (step_size=3, gamma=0.5)
loss_fn = None
optimizer = None
scheduler = None

num_epochs = 5
test_accuracy = None

# TODO: Implement training loop for 5 epochs
# After each epoch:
#   - Print train accuracy and test accuracy
#   - Call scheduler.step()
# Store the final test accuracy in test_accuracy

for epoch in range(num_epochs):
    pass  # TODO: training and evaluation

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert train_transform is not None, "Exercise 1: train_transform not defined"
assert test_transform is not None, "Exercise 1: test_transform not defined"
assert train_dataset is not None and len(train_dataset) == 60000, "Exercise 2 failed"
assert model is not None, "Exercise 3: model not created"

assert test_accuracy is not None, "Exercise 4: test_accuracy not set"
assert test_accuracy > 80.0, f"Exercise 4: accuracy too low ({test_accuracy:.1f}%)"

print(f"Final test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

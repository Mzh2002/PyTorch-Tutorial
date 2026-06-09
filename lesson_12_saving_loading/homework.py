"""
Lesson 12 Homework: Saving/Loading Models and Inference
========================================================
Complete the TODOs below.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(12)
os.makedirs("saved_models", exist_ok=True)

# ============================================================
# Exercise 1: Define and Train a Model
# ============================================================


class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# TODO: Instantiate the model, loss function, and optimizer (Adam, lr=0.001)
model = None
loss_fn = None
optimizer = None

# TODO: Load FashionMNIST training data and create a DataLoader (batch=128)
transform = transforms.ToTensor()
train_dataset = None
train_loader = None

# TODO: Train for 2 epochs
for epoch in range(2):
    pass  # TODO: training loop

# ============================================================
# Exercise 2: Save Model
# ============================================================

# TODO: Save the model's state_dict to "saved_models/mininet.pth"


# TODO: Save a checkpoint with model state, optimizer state, and epoch number
#       to "saved_models/mininet_checkpoint.pth"


# ============================================================
# Exercise 3: Load and Run Inference
# ============================================================

# TODO: Create a new MiniNet instance and load weights from "saved_models/mininet.pth"
loaded_model = None

# TODO: Load test dataset
test_dataset = None

# TODO: Run inference on the first 10 test samples
#       Store predictions as a list of integers
predictions = None

# ============================================================
# Exercise 4: Compute Test Accuracy
# ============================================================

# TODO: Compute accuracy on the full test set using the loaded model
test_accuracy = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert os.path.exists("saved_models/mininet.pth"), "Exercise 2: model not saved"
assert os.path.exists("saved_models/mininet_checkpoint.pth"), "Exercise 2: checkpoint not saved"

assert loaded_model is not None, "Exercise 3: loaded_model not created"
assert predictions is not None and len(predictions) == 10, "Exercise 3: predictions wrong"
assert all(isinstance(p, int) and 0 <= p <= 9 for p in predictions), "Exercise 3: invalid preds"

assert test_accuracy is not None, "Exercise 4: test_accuracy not set"
assert test_accuracy > 70.0, f"Exercise 4: accuracy too low ({test_accuracy:.1f}%)"

print(f"Predictions (first 10): {predictions}")
print(f"Test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

# Cleanup
import shutil
if os.path.exists("saved_models"):
    shutil.rmtree("saved_models")

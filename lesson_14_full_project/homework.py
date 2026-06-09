"""
Lesson 14 Homework: Full Project — Train, Evaluate, Save, and Run Inference
=============================================================================
Build your own complete deep learning pipeline from scratch.
Fill in all TODOs to create a working end-to-end system.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

torch.manual_seed(14)

# ============================================================
# Exercise 1: Configuration and Data
# ============================================================

# TODO: Create an output directory called "my_project"
output_dir = None

# TODO: Define training transforms (ToTensor + Normalize with mean=0.2860, std=0.3530)
train_transform = None

# TODO: Define test transforms (same as above but no augmentation)
test_transform = None

# TODO: Load FashionMNIST train and test datasets
train_dataset = None
test_dataset = None

# TODO: Create DataLoaders (batch_size=128, shuffle train only)
train_loader = None
test_loader = None

# ============================================================
# Exercise 2: Model Definition
# ============================================================

# TODO: Define a CNN class called ProjectCNN with:
#   Features:
#     Conv2d(1, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
#     Conv2d(32, 64, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
#   Classifier:
#     Flatten -> Linear(64*7*7, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 10)


class ProjectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO
        pass

    def forward(self, x):
        # TODO
        pass


# TODO: Instantiate model
model = None

# ============================================================
# Exercise 3: Training
# ============================================================

# TODO: Define CrossEntropyLoss and Adam optimizer (lr=0.001)
loss_fn = None
optimizer = None

# TODO: Train for 5 epochs
#       Track best test accuracy and save the best model to output_dir/best_model.pth
num_epochs = 5
best_accuracy = 0.0

for epoch in range(num_epochs):
    pass  # TODO: Full training + evaluation loop

# ============================================================
# Exercise 4: Final Evaluation
# ============================================================

# TODO: Load the best model and compute:
#   - Test accuracy
#   - F1 score (macro)
#   - Confusion matrix

final_accuracy = None
final_f1 = None
cm = None

# ============================================================
# Exercise 5: Inference Function
# ============================================================

# TODO: Write a function that takes a single image tensor and returns
#       the predicted class index and confidence


def predict(model, image):
    """
    Run inference on a single image.

    Args:
        model: trained model in eval mode
        image: tensor of shape [1, 28, 28]

    Returns:
        (predicted_class: int, confidence: float)
    """
    # TODO: implement
    pass


# TODO: Run inference on 5 test samples and store results
inference_results = None  # list of (predicted_class, confidence, true_label) tuples

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

assert output_dir is not None and os.path.exists(output_dir), "Exercise 1: output_dir missing"
assert train_dataset is not None and len(train_dataset) == 60000, "Exercise 1: data wrong"
assert model is not None, "Exercise 2: model not created"

model_path = os.path.join(output_dir, "best_model.pth")
assert os.path.exists(model_path), "Exercise 3: model not saved"
assert best_accuracy > 80.0, f"Exercise 3: accuracy too low ({best_accuracy:.1f}%)"

assert final_accuracy is not None and final_accuracy > 80.0, "Exercise 4: accuracy wrong"
assert final_f1 is not None and final_f1 > 0.8, "Exercise 4: F1 wrong"
assert cm is not None and cm.shape == (10, 10), "Exercise 4: confusion matrix wrong"

assert inference_results is not None and len(inference_results) == 5, "Exercise 5: inference wrong"
for pred_class, conf, true_label in inference_results:
    assert 0 <= pred_class <= 9 and 0.0 <= conf <= 1.0, "Exercise 5: invalid result"

print(f"Best accuracy: {best_accuracy:.1f}%")
print(f"Final accuracy: {final_accuracy:.1f}%")
print(f"F1 (macro): {final_f1:.4f}")
print(f"\nInference results:")
for pred_class, conf, true_label in inference_results:
    print(f"  Predicted: {CLASS_NAMES[pred_class]:12s} (conf={conf:.2f}), "
          f"True: {CLASS_NAMES[true_label]}")
print("\nAll exercises passed!")

# Cleanup
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

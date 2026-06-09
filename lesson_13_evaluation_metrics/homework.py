"""
Lesson 13 Homework: Evaluation Metrics and Confusion Matrix
=============================================================
Complete the TODOs below.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

torch.manual_seed(13)

# ============================================================
# Exercise 1: Train a Model and Collect Predictions
# ============================================================


class EvalModel(nn.Module):
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


transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST("./data", train=True, download=True,
                                       transform=transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True,
                                      transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# TODO: Train the model for 3 epochs
model = None
# ... training code ...

# TODO: Collect all predictions and true labels from the test set
#       Store as numpy arrays: all_preds, all_labels
all_preds = None
all_labels = None

# ============================================================
# Exercise 2: Compute Metrics
# ============================================================

# TODO: Compute accuracy using sklearn
accuracy = None

# TODO: Compute macro F1 score
f1_macro = None

# ============================================================
# Exercise 3: Confusion Matrix Analysis
# ============================================================

# TODO: Compute the confusion matrix
cm = None

# TODO: Find the class with the LOWEST accuracy
#       Store the class index (0-9) in worst_class_idx
worst_class_idx = None
worst_class_accuracy = None

# ============================================================
# Exercise 4: Manual Precision/Recall for one class
# ============================================================

# TODO: For class index 6 ("Shirt"), compute precision and recall
#       using the confusion matrix
class_6_precision = None
class_6_recall = None

# ============================================================
# Validation
# ============================================================
print("\n--- Checking your answers ---")

assert all_preds is not None and len(all_preds) == 10000, "Exercise 1: predictions wrong"
assert all_labels is not None and len(all_labels) == 10000, "Exercise 1: labels wrong"

assert accuracy is not None and accuracy > 0.7, f"Exercise 2: accuracy too low ({accuracy})"
assert f1_macro is not None and f1_macro > 0.7, f"Exercise 2: F1 too low ({f1_macro})"

assert cm is not None and cm.shape == (10, 10), "Exercise 3: confusion matrix shape wrong"
assert worst_class_idx is not None and 0 <= worst_class_idx <= 9, "Exercise 3: worst_class wrong"
assert worst_class_accuracy is not None, "Exercise 3: worst_class_accuracy not set"

assert class_6_precision is not None and 0 < class_6_precision <= 1, "Exercise 4: precision wrong"
assert class_6_recall is not None and 0 < class_6_recall <= 1, "Exercise 4: recall wrong"

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print(f"Worst class: {class_names[worst_class_idx]} ({worst_class_accuracy:.1f}%)")
print(f"Shirt precision: {class_6_precision:.4f}, recall: {class_6_recall:.4f}")
print("All exercises passed!")

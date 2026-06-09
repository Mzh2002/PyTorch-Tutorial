"""
Lesson 13 Answer: Evaluation Metrics and Confusion Matrix
==========================================================
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

torch.manual_seed(13)

# ============================================================
# Exercise 1: Train and Collect Predictions
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

model = EvalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for images, labels in train_loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# Collect predictions
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ============================================================
# Exercise 2: Compute Metrics
# ============================================================

accuracy = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")

# ============================================================
# Exercise 3: Confusion Matrix Analysis
# ============================================================

cm = confusion_matrix(all_labels, all_preds)

# Find class with lowest accuracy
class_accuracies = [cm[i, i] / cm[i].sum() * 100 for i in range(10)]
worst_class_idx = int(np.argmin(class_accuracies))
worst_class_accuracy = class_accuracies[worst_class_idx]

# ============================================================
# Exercise 4: Manual Precision/Recall for class 6
# ============================================================

tp = cm[6, 6]
fp = cm[:, 6].sum() - tp
fn = cm[6, :].sum() - tp

class_6_precision = tp / (tp + fp)
class_6_recall = tp / (tp + fn)

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

assert len(all_preds) == 10000
assert accuracy > 0.7
assert f1_macro > 0.7
assert cm.shape == (10, 10)
assert 0 <= worst_class_idx <= 9
assert 0 < class_6_precision <= 1
assert 0 < class_6_recall <= 1

print(f"Worst class: {class_names[worst_class_idx]} ({worst_class_accuracy:.1f}%)")
print(f"Shirt precision: {class_6_precision:.4f}, recall: {class_6_recall:.4f}")
print("All exercises passed!")

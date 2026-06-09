"""
Lesson 13: Evaluation Metrics and Confusion Matrix
====================================================
Beyond accuracy: precision, recall, F1-score, and confusion matrices
help you understand model performance in detail.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np

# ============================================================
# 1. Train a Quick Model
# ============================================================

print("=== Training Model ===\n")

torch.manual_seed(13)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
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

model = SimpleModel()
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

# ============================================================
# 2. Collect Predictions
# ============================================================

print("\n=== Collecting Predictions ===\n")

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
print(f"Total predictions: {len(all_preds)}")

# ============================================================
# 3. Basic Metrics
# ============================================================

print("\n=== Basic Metrics ===\n")

accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# For multi-class, we use 'macro' (unweighted mean across classes)
# or 'weighted' (weighted by class frequency)
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro):    {recall:.4f}")
print(f"F1 Score (macro):  {f1:.4f}")

# ============================================================
# 4. Classification Report
# ============================================================

print("\n=== Classification Report ===\n")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

report = classification_report(all_labels, all_preds,
                                target_names=class_names)
print(report)

# ============================================================
# 5. Confusion Matrix
# ============================================================

print("=== Confusion Matrix ===\n")

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix (rows=true, cols=predicted):\n")

# Pretty print with class names
header = "          " + " ".join(f"{name[:6]:>6s}" for name in class_names)
print(header)
for i, row in enumerate(cm):
    row_str = " ".join(f"{val:6d}" for val in row)
    print(f"{class_names[i]:10s} {row_str}")

# ============================================================
# 6. Understanding the Confusion Matrix
# ============================================================

print("\n=== Interpreting Results ===\n")

# Find most confused pairs
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 50:
            print(f"  {class_names[i]} often confused with "
                  f"{class_names[j]}: {cm[i,j]} times")

# Per-class accuracy
print("\nPer-class accuracy:")
for i in range(10):
    class_acc = cm[i, i] / cm[i].sum() * 100
    print(f"  {class_names[i]:12s}: {class_acc:.1f}%")

# ============================================================
# 7. Computing Metrics Manually (without sklearn)
# ============================================================

print("\n=== Manual Metric Computation ===\n")

# For class 0 (T-shirt/top):
class_idx = 0
true_positives = cm[class_idx, class_idx]
false_positives = cm[:, class_idx].sum() - true_positives
false_negatives = cm[class_idx, :].sum() - true_positives

manual_precision = true_positives / (true_positives + false_positives)
manual_recall = true_positives / (true_positives + false_negatives)
manual_f1 = 2 * manual_precision * manual_recall / (manual_precision + manual_recall)

print(f"Class '{class_names[class_idx]}':")
print(f"  TP={true_positives}, FP={false_positives}, FN={false_negatives}")
print(f"  Precision: {manual_precision:.4f}")
print(f"  Recall:    {manual_recall:.4f}")
print(f"  F1:        {manual_f1:.4f}")

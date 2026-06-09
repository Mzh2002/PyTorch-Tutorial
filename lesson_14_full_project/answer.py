"""
Lesson 14 Answer: Full Project — Train, Evaluate, Save, and Run Inference
==========================================================================
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

output_dir = "my_project"
os.makedirs(output_dir, exist_ok=True)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2860], [0.3530]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.2860], [0.3530]),
])

train_dataset = datasets.FashionMNIST("./data", train=True, download=True,
                                       transform=train_transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True,
                                      transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ============================================================
# Exercise 2: Model Definition
# ============================================================


class ProjectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = ProjectCNN()

# ============================================================
# Exercise 3: Training
# ============================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
best_accuracy = 0.0

for epoch in range(num_epochs):
    # Train
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(dim=1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    train_acc = train_correct / train_total * 100
    test_acc = test_correct / test_total * 100

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

    print(f"Epoch {epoch+1}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%")

print(f"Best test accuracy: {best_accuracy:.1f}%")

# ============================================================
# Exercise 4: Final Evaluation
# ============================================================

model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"),
                                  weights_only=True))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

final_accuracy = accuracy_score(all_labels, all_preds) * 100
final_f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)

print(f"\nFinal accuracy: {final_accuracy:.1f}%")
print(f"F1 (macro): {final_f1:.4f}")

# ============================================================
# Exercise 5: Inference Function
# ============================================================


def predict(model, image):
    """Run inference on a single image."""
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence


# Run inference on 5 samples
inference_results = []
for i in range(5):
    image, true_label = test_dataset[i]
    pred_class, confidence = predict(model, image)
    inference_results.append((pred_class, confidence, true_label))

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

assert os.path.exists(output_dir)
assert os.path.exists(os.path.join(output_dir, "best_model.pth"))
assert best_accuracy > 80.0
assert final_accuracy > 80.0
assert final_f1 > 0.8
assert cm.shape == (10, 10)
assert len(inference_results) == 5

print(f"\nInference results:")
for pred_class, conf, true_label in inference_results:
    status = "CORRECT" if pred_class == true_label else "WRONG"
    print(f"  [{status}] Predicted: {CLASS_NAMES[pred_class]:12s} "
          f"(conf={conf:.2f}), True: {CLASS_NAMES[true_label]}")

print("\nAll exercises passed!")

# Cleanup
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

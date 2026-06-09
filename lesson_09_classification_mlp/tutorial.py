"""
Lesson 09: Classification with MLP
====================================
We build a multi-layer perceptron (MLP) classifier on FashionMNIST.
This introduces working with image data for classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# 1. Load FashionMNIST Dataset
# ============================================================

print("=== Loading FashionMNIST ===\n")

# transforms.ToTensor() converts PIL images to tensors and scales to [0, 1]
transform = transforms.ToTensor()

# Download and load training data
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

# Download and load test data
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Look at a single sample
image, label = train_dataset[0]
print(f"\nSingle image shape: {image.shape}")  # [1, 28, 28]
print(f"Label: {label}")

# Class names for FashionMNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
print(f"Class name: {class_names[label]}")

# ============================================================
# 2. Create DataLoaders
# ============================================================

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Check a batch
images, labels = next(iter(train_loader))
print(f"\nBatch images shape: {images.shape}")  # [64, 1, 28, 28]
print(f"Batch labels shape: {labels.shape}")    # [64]

# ============================================================
# 3. Define the MLP Model
# ============================================================

print("\n=== MLP Model ===\n")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten 28x28 image into 784-dim vector
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # 10 classes
        )

    def forward(self, x):
        x = self.flatten(x)   # [batch, 1, 28, 28] -> [batch, 784]
        x = self.layers(x)    # [batch, 784] -> [batch, 10]
        return x


model = MLP()
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ============================================================
# 4. Training
# ============================================================

print("\n=== Training ===\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")

# ============================================================
# 5. Evaluation on Test Set
# ============================================================

print("\n=== Test Evaluation ===\n")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total * 100
print(f"Test accuracy: {test_accuracy:.1f}%")

# Per-class accuracy
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

print("\nPer-class accuracy:")
for i in range(10):
    acc = class_correct[i] / class_total[i] * 100
    print(f"  {class_names[i]:12s}: {acc:.1f}%")

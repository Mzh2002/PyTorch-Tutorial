"""
Lesson 10 Answer: CNN Basics
==============================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(10)

# ============================================================
# Exercise 1: Conv2d Output Shapes
# ============================================================

x = torch.randn(1, 1, 32, 32)

# Conv that preserves spatial size: padding=1 with kernel_size=3
conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
out1 = conv1(x)
print(f"out1 shape: {out1.shape}")  # [1, 8, 32, 32]

# Conv with stride=2 to halve spatial dimensions
conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
out2 = conv2(out1)
print(f"out2 shape: {out2.shape}")  # [1, 16, 16, 16]

# ============================================================
# Exercise 2: Build CNN
# ============================================================


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = FashionCNN()

# ============================================================
# Exercise 3: Train and Evaluate
# ============================================================

transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: accuracy={correct/total*100:.1f}%")

# Evaluate
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

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert list(out1.shape) == [1, 8, 32, 32]
assert list(out2.shape) == [1, 16, 16, 16]
assert test_accuracy > 75.0

print(f"Test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

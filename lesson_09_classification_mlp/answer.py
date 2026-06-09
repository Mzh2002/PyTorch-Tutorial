"""
Lesson 09 Answer: Classification with MLP
===========================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(9)

# ============================================================
# Exercise 1: Load Data
# ============================================================

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ============================================================
# Exercise 2: Define Model
# ============================================================


class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


model = FashionMLP()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# Exercise 3: Train
# ============================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
          f"accuracy={accuracy:.1f}%")

# ============================================================
# Exercise 4: Evaluate
# ============================================================

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

assert len(train_dataset) == 60000
assert len(test_dataset) == 10000
assert test_accuracy > 70.0

print(f"Test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

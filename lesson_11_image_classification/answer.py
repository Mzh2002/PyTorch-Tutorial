"""
Lesson 11 Answer: Image Classification Project
================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(11)

# ============================================================
# Exercise 1: Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530]),
])

# ============================================================
# Exercise 2: Data
# ============================================================

train_dataset = datasets.FashionMNIST("./data", train=True, download=True,
                                       transform=train_transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True,
                                      transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ============================================================
# Exercise 3: CNN
# ============================================================


class MyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = MyClassifier()

# ============================================================
# Exercise 4: Training
# ============================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

num_epochs = 5
test_accuracy = None

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

    train_acc = correct / total * 100

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
    scheduler.step()

    print(f"Epoch {epoch+1}: train_acc={train_acc:.1f}%, test_acc={test_accuracy:.1f}%")

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert len(train_dataset) == 60000
assert test_accuracy > 80.0

print(f"Final test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

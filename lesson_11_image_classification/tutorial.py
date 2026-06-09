"""
Lesson 11: Image Classification Project
==========================================
A more complete image classification pipeline with data augmentation,
learning rate scheduling, and better model architecture.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
# 1. Data Augmentation
# ============================================================

print("=== Data Augmentation ===\n")

# Training transforms: augment data to improve generalization
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530]),  # FashionMNIST stats
])

# Test transforms: only normalize (no augmentation!)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2860], std=[0.3530]),
])

print("Train transforms:", train_transform)
print("Test transforms:", test_transform)

# ============================================================
# 2. Load Data
# ============================================================

print("\n=== Loading Data ===\n")

train_dataset = datasets.FashionMNIST("./data", train=True, download=True,
                                       transform=train_transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True,
                                      transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training: {len(train_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# ============================================================
# 3. Improved CNN Architecture
# ============================================================

print("\n=== Model ===\n")


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = ImprovedCNN()
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ============================================================
# 4. Training with LR Scheduler
# ============================================================

print("\n=== Training ===\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Reduce learning rate when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=False
)

num_epochs = 5

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    # Evaluation phase
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    train_acc = train_correct / train_total * 100
    test_acc = test_correct / test_total * 100

    # Update scheduler based on test loss
    scheduler.step(avg_test_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"train_loss={avg_train_loss:.4f}, train_acc={train_acc:.1f}%, "
          f"test_loss={avg_test_loss:.4f}, test_acc={test_acc:.1f}%, "
          f"lr={current_lr:.6f}")

print(f"\nFinal test accuracy: {test_acc:.1f}%")

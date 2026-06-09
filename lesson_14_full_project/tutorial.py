"""
Lesson 14: Full Project — Train, Evaluate, Save, and Run Inference
===================================================================
This is a complete end-to-end deep learning project combining everything
from all previous lessons into a single, professional workflow.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "seed": 42,
    "batch_size": 128,
    "learning_rate": 0.001,
    "num_epochs": 5,
    "model_dir": "project_output",
    "model_name": "fashion_cnn.pth",
}

torch.manual_seed(CONFIG["seed"])
os.makedirs(CONFIG["model_dir"], exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ============================================================
# Step 1: Data Preparation
# ============================================================

print("=" * 60)
print("STEP 1: Data Preparation")
print("=" * 60)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
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

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                         shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Training batches per epoch: {len(train_loader)}")

# ============================================================
# Step 2: Model Definition
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Model Definition")
print("=" * 60)


class FashionNet(nn.Module):
    """CNN for FashionMNIST classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = FashionNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: FashionNet")
print(f"Total parameters: {total_params:,}")
print(model)

# ============================================================
# Step 3: Training
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Training")
print("=" * 60)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

best_test_acc = 0.0
train_history = {"loss": [], "accuracy": []}
test_history = {"loss": [], "accuracy": []}

for epoch in range(CONFIG["num_epochs"]):
    # --- Train ---
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

        train_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    # --- Evaluate ---
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    # Compute epoch metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total * 100
    epoch_test_loss = test_loss / test_total
    epoch_test_acc = test_correct / test_total * 100

    train_history["loss"].append(epoch_train_loss)
    train_history["accuracy"].append(epoch_train_acc)
    test_history["loss"].append(epoch_test_loss)
    test_history["accuracy"].append(epoch_test_acc)

    # Save best model
    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        model_path = os.path.join(CONFIG["model_dir"], CONFIG["model_name"])
        torch.save(model.state_dict(), model_path)

    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: "
          f"train_loss={epoch_train_loss:.4f}, train_acc={epoch_train_acc:.1f}%, "
          f"test_loss={epoch_test_loss:.4f}, test_acc={epoch_test_acc:.1f}%, "
          f"lr={lr:.6f}")

print(f"\nBest test accuracy: {best_test_acc:.1f}%")

# ============================================================
# Step 4: Final Evaluation
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Final Evaluation")
print("=" * 60)

# Load best model
model_path = os.path.join(CONFIG["model_dir"], CONFIG["model_name"])
model.load_state_dict(torch.load(model_path, weights_only=True))
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

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix (first 5 classes):")
for i in range(5):
    print(f"  {CLASS_NAMES[i]:12s}: {cm[i, :5]}")

# ============================================================
# Step 5: Inference on New Samples
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Inference")
print("=" * 60)

print("\nRunning inference on 10 random test samples:\n")

# Simulate loading the model fresh (as you would in production)
inference_model = FashionNet()
inference_model.load_state_dict(torch.load(model_path, weights_only=True))
inference_model.eval()

indices = torch.randperm(len(test_dataset))[:10]

with torch.no_grad():
    for idx in indices:
        image, true_label = test_dataset[idx]
        output = inference_model(image.unsqueeze(0))
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        status = "CORRECT" if pred_class == true_label else "WRONG"
        print(f"  [{status:7s}] Predicted: {CLASS_NAMES[pred_class]:12s} "
              f"(conf={confidence:.2f}), True: {CLASS_NAMES[true_label]}")

# ============================================================
# Step 6: Summary
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"  Model: FashionNet ({total_params:,} parameters)")
print(f"  Dataset: FashionMNIST (60k train, 10k test)")
print(f"  Best test accuracy: {best_test_acc:.1f}%")
print(f"  Model saved to: {model_path}")
print(f"  Training epochs: {CONFIG['num_epochs']}")
print("=" * 60)

# Cleanup
import shutil
if os.path.exists(CONFIG["model_dir"]):
    shutil.rmtree(CONFIG["model_dir"])
    print("\nCleaned up output directory.")

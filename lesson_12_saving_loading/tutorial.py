"""
Lesson 12: Saving/Loading Models and Inference Script
======================================================
Learn how to save trained models, load them later, and run inference.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ============================================================
# 1. Define and Train a Model
# ============================================================

print("=== Training a Model ===\n")


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Train briefly
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST("./data", train=True, download=True,
                                       transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

model = SmallCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(2):
    for images, labels in train_loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# ============================================================
# 2. Saving Models
# ============================================================

print("\n=== Saving Models ===\n")

os.makedirs("saved_models", exist_ok=True)

# Method 1: Save state_dict (RECOMMENDED)
# Only saves the learned parameters, not the model architecture
torch.save(model.state_dict(), "saved_models/model_weights.pth")
print("Saved state_dict to saved_models/model_weights.pth")

# Method 2: Save entire model (not recommended — tightly couples to code)
torch.save(model, "saved_models/model_full.pth")
print("Saved full model to saved_models/model_full.pth")

# Method 3: Save checkpoint (model + optimizer + epoch info)
checkpoint = {
    "epoch": 2,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}
torch.save(checkpoint, "saved_models/checkpoint.pth")
print("Saved checkpoint to saved_models/checkpoint.pth")

# ============================================================
# 3. Loading Models
# ============================================================

print("\n=== Loading Models ===\n")

# Method 1: Load state_dict (must define model architecture first)
loaded_model = SmallCNN()  # create model with same architecture
loaded_model.load_state_dict(torch.load("saved_models/model_weights.pth",
                                         weights_only=True))
loaded_model.eval()  # set to evaluation mode
print("Loaded state_dict successfully")

# Method 2: Load entire model
loaded_full = torch.load("saved_models/model_full.pth", weights_only=False)
loaded_full.eval()
print("Loaded full model successfully")

# Method 3: Load checkpoint (for resuming training)
checkpoint = torch.load("saved_models/checkpoint.pth", weights_only=True)
resume_model = SmallCNN()
resume_model.load_state_dict(checkpoint["model_state_dict"])
resume_optimizer = torch.optim.Adam(resume_model.parameters())
resume_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]
print(f"Resumed from epoch {start_epoch}, loss={checkpoint['loss']:.4f}")

# ============================================================
# 4. Running Inference
# ============================================================

print("\n=== Inference ===\n")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Load test data
test_dataset = datasets.FashionMNIST("./data", train=False, download=True,
                                      transform=transform)

# Run inference on a few samples
loaded_model.eval()
with torch.no_grad():
    for i in range(5):
        image, true_label = test_dataset[i]
        # Add batch dimension: [1, 28, 28] -> [1, 1, 28, 28]
        image_batch = image.unsqueeze(0)

        # Get model prediction
        logits = loaded_model(image_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        print(f"Sample {i}: "
              f"predicted={class_names[predicted_class]:12s} "
              f"(conf={confidence:.2f}), "
              f"true={class_names[true_label]}")

# ============================================================
# 5. Batch Inference
# ============================================================

print("\n=== Batch Inference ===\n")

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

all_predictions = []
all_labels = []

loaded_model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = loaded_model(images)
        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

correct = sum(p == l for p, l in zip(all_predictions, all_labels))
total = len(all_labels)
print(f"Test accuracy: {correct/total*100:.1f}% ({correct}/{total})")

# Cleanup
import shutil
if os.path.exists("saved_models"):
    shutil.rmtree("saved_models")
    print("\nCleaned up saved_models/")

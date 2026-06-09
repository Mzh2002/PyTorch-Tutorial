"""
Lesson 12 Answer: Saving/Loading Models and Inference
======================================================
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(12)
os.makedirs("saved_models", exist_ok=True)

# ============================================================
# Exercise 1: Define and Train
# ============================================================


class MiniNet(nn.Module):
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


model = MiniNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST("./data", train=True, download=True,
                                       transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

for epoch in range(2):
    model.train()
    for images, labels in train_loader:
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# ============================================================
# Exercise 2: Save
# ============================================================

torch.save(model.state_dict(), "saved_models/mininet.pth")

checkpoint = {
    "epoch": 2,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(checkpoint, "saved_models/mininet_checkpoint.pth")
print("Model and checkpoint saved")

# ============================================================
# Exercise 3: Load and Infer
# ============================================================

loaded_model = MiniNet()
loaded_model.load_state_dict(torch.load("saved_models/mininet.pth", weights_only=True))
loaded_model.eval()

test_dataset = datasets.FashionMNIST("./data", train=False, download=True,
                                      transform=transform)

predictions = []
with torch.no_grad():
    for i in range(10):
        image, _ = test_dataset[i]
        output = loaded_model(image.unsqueeze(0))
        pred = output.argmax(dim=1).item()
        predictions.append(pred)

print(f"Predictions: {predictions}")

# ============================================================
# Exercise 4: Test Accuracy
# ============================================================

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = loaded_model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total * 100

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert os.path.exists("saved_models/mininet.pth")
assert os.path.exists("saved_models/mininet_checkpoint.pth")
assert len(predictions) == 10
assert test_accuracy > 70.0

print(f"Test accuracy: {test_accuracy:.1f}%")
print("All exercises passed!")

# Cleanup
import shutil
if os.path.exists("saved_models"):
    shutil.rmtree("saved_models")

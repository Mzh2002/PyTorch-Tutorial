"""
Lesson 10: CNN Basics
======================
Convolutional Neural Networks (CNNs) are designed to process grid-like data
(images). They use convolutional layers to detect local patterns.
"""

import torch
import torch.nn as nn

# ============================================================
# 1. Convolution Operation
# ============================================================

print("=== Convolution Basics ===\n")

# nn.Conv2d parameters:
#   in_channels: number of input channels (1 for grayscale, 3 for RGB)
#   out_channels: number of filters (output channels)
#   kernel_size: size of the convolution filter

conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

# Input: batch of 2 grayscale images, each 8x8
x = torch.randn(2, 1, 8, 8)  # [batch, channels, height, width]
output = conv(x)

print(f"Input shape:  {x.shape}")       # [2, 1, 8, 8]
print(f"Output shape: {output.shape}")  # [2, 3, 8, 8] (padding=1 preserves size)
print(f"Conv weight shape: {conv.weight.shape}")  # [3, 1, 3, 3]
print(f"Conv bias shape: {conv.bias.shape}")      # [3]

# Without padding, output shrinks
conv_no_pad = nn.Conv2d(1, 3, kernel_size=3)
output_no_pad = conv_no_pad(x)
print(f"\nNo padding output: {output_no_pad.shape}")  # [2, 3, 6, 6]

# Stride controls step size
conv_stride = nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1)
output_stride = conv_stride(x)
print(f"Stride=2 output: {output_stride.shape}")  # [2, 3, 4, 4]

# ============================================================
# 2. Pooling Layers
# ============================================================

print("\n=== Pooling ===\n")

# MaxPool2d takes the maximum value in each window
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0]]]])  # [1, 1, 4, 4]

pooled = pool(x)
print(f"Input:\n{x.squeeze()}")
print(f"\nMaxPool2d(2):\n{pooled.squeeze()}")  # [1, 1, 2, 2]
# Each 2x2 region -> max value: [[6,8],[14,16]]

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled = avg_pool(x)
print(f"\nAvgPool2d(2):\n{avg_pooled.squeeze()}")

# Adaptive pooling (outputs fixed size regardless of input)
adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling
x_big = torch.randn(1, 16, 7, 7)
print(f"\nAdaptiveAvgPool2d: {x_big.shape} -> {adaptive_pool(x_big).shape}")

# ============================================================
# 3. Building a CNN
# ============================================================

print("\n=== CNN Architecture ===\n")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extractor (convolutional layers)
        self.features = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> [batch, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> [batch, 16, 14, 14]

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> [batch, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> [batch, 32, 7, 7]
        )

        # Classifier (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),                # -> [batch, 32*7*7]
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN()
print(model)

# Test with a dummy input
x = torch.randn(4, 1, 28, 28)  # batch of 4 grayscale 28x28 images
output = model(x)
print(f"\nInput: {x.shape} -> Output: {output.shape}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================
# 4. Quick Training on FashionMNIST
# ============================================================

print("\n=== Training CNN on FashionMNIST ===\n")

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for 3 epochs
for epoch in range(3):
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

    print(f"Epoch {epoch+1}: train accuracy = {correct/total*100:.1f}%")

# Test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nTest accuracy: {correct/total*100:.1f}%")
print("(CNN typically gets ~88-90% on FashionMNIST — better than MLP!)")

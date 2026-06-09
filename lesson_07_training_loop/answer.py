"""
Lesson 07 Answer: Building a Training Loop
============================================
"""

import torch
import torch.nn as nn

torch.manual_seed(7)

# ============================================================
# Exercise 1: Prepare Data
# ============================================================

num_samples = 300
X = torch.randn(num_samples, 2)
y = X[:, 0:1] ** 2 + 2 * X[:, 1:2] + torch.randn(num_samples, 1) * 0.1

n_train = int(0.8 * num_samples)
X_train = X[:n_train]
X_val = X[n_train:]
y_train = y[:n_train]
y_val = y[n_train:]

# ============================================================
# Exercise 2: Define Model
# ============================================================

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# ============================================================
# Exercise 3: Training Loop
# ============================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    y_pred = model(X_train)
    train_loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: train={train_loss.item():.4f}, val={val_loss.item():.4f}")

# ============================================================
# Exercise 4: Report Results
# ============================================================

final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]

# ============================================================
# Validation
# ============================================================
print("\n--- Checking answers ---")

assert n_train == 240
assert X_train.shape == (240, 2)
assert X_val.shape == (60, 2)
assert len(train_losses) == num_epochs
assert final_train_loss < 0.5
assert final_val_loss < 1.0

print(f"Final train loss: {final_train_loss:.4f}")
print(f"Final val loss: {final_val_loss:.4f}")
print("All exercises passed!")

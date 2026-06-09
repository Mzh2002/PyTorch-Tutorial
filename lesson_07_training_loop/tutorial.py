"""
Lesson 07: Building a Training Loop
=====================================
A proper training loop is the core of any deep learning project.
This lesson shows the standard pattern used in practice.
"""

import torch
import torch.nn as nn

# ============================================================
# 1. The Standard Training Loop Pattern
# ============================================================

# The pattern is always:
#   for epoch in range(num_epochs):
#       for batch in dataloader:            # (we'll use simple batches here)
#           1. Forward pass (compute predictions)
#           2. Compute loss
#           3. Zero gradients
#           4. Backward pass (compute gradients)
#           5. Optimizer step (update weights)

# ============================================================
# 2. Full Example: Multi-feature Regression
# ============================================================

print("=== Multi-feature Regression ===\n")

torch.manual_seed(42)

# Generate data with 3 features
# True relationship: y = 2*x1 + 3*x2 - 1*x3 + 0.5
num_samples = 200
X = torch.randn(num_samples, 3)
true_weights = torch.tensor([[2.0, 3.0, -1.0]])
true_bias = torch.tensor([0.5])
y = X @ true_weights.T + true_bias + torch.randn(num_samples, 1) * 0.2

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")

# Model
model = nn.Linear(3, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training with manual mini-batches
batch_size = 32
num_epochs = 50

print("\n--- Training ---")
for epoch in range(num_epochs):
    # Shuffle data each epoch
    perm = torch.randperm(num_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    epoch_loss = 0.0
    num_batches = 0

    # Mini-batch loop
    for i in range(0, num_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    # Print average loss per epoch
    avg_loss = epoch_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: avg_loss={avg_loss:.4f}")

print(f"\nLearned weights: {model.weight.data.numpy().flatten()}")
print(f"True weights:    [2.0, 3.0, -1.0]")
print(f"Learned bias:    {model.bias.item():.4f}")
print(f"True bias:       0.5")

# ============================================================
# 3. Tracking Training History
# ============================================================

print("\n=== Training with History ===\n")

torch.manual_seed(0)

# Fresh model
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Store losses for monitoring
train_losses = []

for epoch in range(100):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

print(f"Initial loss: {train_losses[0]:.4f}")
print(f"Final loss:   {train_losses[-1]:.4f}")
print(f"Improvement:  {train_losses[0] / train_losses[-1]:.1f}x")

# ============================================================
# 4. Train/Validation Split
# ============================================================

print("\n=== Train/Validation Split ===\n")

torch.manual_seed(42)

# Split data: 80% train, 20% validation
n_train = int(0.8 * num_samples)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

print(f"Train: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")

# Fresh model
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    # --- Training ---
    model.train()
    y_pred = model(X_train)
    train_loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # --- Validation (no gradient computation needed) ---
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, y_val)

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1:3d}: "
              f"train_loss={train_loss.item():.4f}, "
              f"val_loss={val_loss.item():.4f}")

# ============================================================
# 5. Early Stopping (Concept)
# ============================================================

print("\n=== Early Stopping Concept ===")
print("""
Early stopping monitors validation loss and stops training when it
starts increasing (overfitting). Here's the logic:

    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(max_epochs):
        # ... train ...
        # ... compute val_loss ...

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model weights
            best_weights = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                model.load_state_dict(best_weights)
                break
""")

"""
Lesson 05: nn.Module, Parameters, and Layers
==============================================
The nn.Module class is the building block for all neural networks in PyTorch.
It manages parameters and provides a clean interface for building models.
"""

import torch
import torch.nn as nn

# ============================================================
# 1. Using Built-in Layers
# ============================================================

# nn.Linear is a fully connected (dense) layer
# It computes: output = input @ weight.T + bias
linear = nn.Linear(in_features=3, out_features=2)

print("Linear layer:", linear)
print("Weight shape:", linear.weight.shape)   # [2, 3]
print("Bias shape:", linear.bias.shape)       # [2]
print("Weight values:\n", linear.weight)
print("Bias values:", linear.bias)

# Pass data through the layer
x = torch.tensor([[1.0, 2.0, 3.0]])  # shape [1, 3]
output = linear(x)                     # shape [1, 2]
print(f"\nInput shape: {x.shape} -> Output shape: {output.shape}")
print("Output:", output)

# ============================================================
# 2. Building a Custom Model with nn.Module
# ============================================================


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Always call super().__init__() first!
        super().__init__()

        # Define layers as attributes
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Define how data flows through the network."""
        x = self.layer1(x)    # linear transform
        x = self.relu(x)      # non-linear activation
        x = self.layer2(x)    # final linear transform
        return x


# Create the model
model = SimpleNet(input_size=4, hidden_size=8, output_size=2)
print("\n--- Model Architecture ---")
print(model)

# ============================================================
# 3. Model Parameters
# ============================================================

# .parameters() gives all learnable parameters
print("\n--- Parameters ---")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

# Total parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")

# ============================================================
# 4. Forward Pass
# ============================================================

# Create batch of 5 samples, each with 4 features
batch = torch.randn(5, 4)
output = model(batch)  # calls model.forward(batch)
print(f"\nInput batch shape: {batch.shape}")
print(f"Output shape: {output.shape}")  # [5, 2]

# ============================================================
# 5. Common Layers
# ============================================================

print("\n--- Common Layer Types ---")

# Linear (fully connected)
fc = nn.Linear(10, 5)
print(f"Linear(10, 5): input [*, 10] -> output [*, 5]")

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\nInput: {x.tolist()}")
print(f"ReLU:    {relu(x).tolist()}")
print(f"Sigmoid: {[f'{v:.3f}' for v in sigmoid(x).tolist()]}")
print(f"Tanh:    {[f'{v:.3f}' for v in tanh(x).tolist()]}")

# Dropout (regularization — randomly zeros elements during training)
dropout = nn.Dropout(p=0.5)
x = torch.ones(10)
print(f"\nDropout (training): {dropout(x)}")

# BatchNorm
bn = nn.BatchNorm1d(4)
x = torch.randn(3, 4)  # batch of 3, 4 features
print(f"BatchNorm input shape: {x.shape}, output shape: {bn(x).shape}")

# ============================================================
# 6. nn.Sequential — Quick Model Building
# ============================================================

# Sequential stacks layers in order
sequential_model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

print("\n--- Sequential Model ---")
print(sequential_model)

x = torch.randn(3, 4)
output = sequential_model(x)
print(f"Input: {x.shape} -> Output: {output.shape}")

# ============================================================
# 7. Training vs Evaluation Mode
# ============================================================

# Some layers (Dropout, BatchNorm) behave differently during training vs eval
model.train()   # training mode (default)
print(f"\nTraining mode: {model.training}")

model.eval()    # evaluation mode
print(f"Eval mode: {model.training}")

model.train()   # switch back

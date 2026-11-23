import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import os
import sys

# Allow Python to import train.py from the parent Team-10 folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train


# ---------------------------------
# 1. Temporary CNN model
# ---------------------------------

class StockCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 64 * 64, 2)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


# ---------------------------------
# 2. Generate fake image data
# ---------------------------------

# 20 fake RGB images, each 64x64 pixels
images = torch.randn(20, 3, 64, 64)

# Labels: 0 = DOWN, 1 = UP
labels = torch.randint(0, 2, (20,))


# ---------------------------------
# 3. Split training and validation
# ---------------------------------

train_dataset = TensorDataset(
    images[:14],
    labels[:14]
)

val_dataset = TensorDataset(
    images[14:17],
    labels[14:17]
)


# ---------------------------------
# 4. Create DataLoaders
# ---------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


# ---------------------------------
# 5. Create temporary CNN
# ---------------------------------

torch.manual_seed(0)

model = StockCNN()


# ---------------------------------
# 6. Train the model
# ---------------------------------

history = train(
    model,
    train_loader,
    val_loader,
    epochs=5,
    lr=0.001,
    save_path="test_files/test_model.pth",
    seed=0
)


# ---------------------------------
# 7. Display results
# ---------------------------------

print("\n-----------------------------")
print("TRAINING TEST RESULTS")
print("-----------------------------")

print("Training test successful!")

print("Training losses:", history["train_loss"])
print("Validation losses:", history["val_loss"])

print("\nSaved model:", os.path.exists("test_files/test_model.pth"))

print("\nTest completed!")


import sys
import os
import torch

# Allow Python to find CNN_build.py in the parent folder
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from CNN_build import StockCNN


# 1. Generate fake image data
torch.manual_seed(0)
images = torch.randn(4, 3, 256, 256)


# 2. Create CNN model
model = StockCNN()


# 3. Test forward pass
outputs = model(images)


# 4. Print input and output shapes
print("Input shape:", images.shape)
print("Output shape:", outputs.shape)


# 5. Verify output dimensions
assert outputs.shape == (4, 2)

print("\nCNN test successful!")

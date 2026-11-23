
import torch
from CNN_build import StockCNN

# 1. Generate fake image data
torch.manual_seed(0)

images = torch.randn(4, 3, 256, 256)

# 2. Create CNN
model = StockCNN()

# 3. Test forward pass
outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)

# 4. Verify output
assert outputs.shape == (4, 2)

print("\nCNN test successful!")

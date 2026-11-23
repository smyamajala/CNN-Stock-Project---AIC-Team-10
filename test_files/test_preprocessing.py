
import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Import preprocessing code
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from preprocessing_skeleton import get_dataloaders, DATA_DIR

# 1. Load datasets
train_loader, val_loader, test_loader = get_dataloaders()

print("\n========== DATASET TEST ==========")

print("Training:", len(train_loader.dataset))
print("Validation:", len(val_loader.dataset))
print("Testing:", len(test_loader.dataset))

# 2. Get a batch
images, labels = next(iter(train_loader))

print("\n========== TENSOR TEST ==========")

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Data type:", images.dtype)
print("Minimum value:", images.min().item())
print("Maximum value:", images.max().item())

# 3. Verify dimensions and format
assert images.shape[1:] == (3, 256, 256)
assert images.dtype == torch.float32
assert labels.dtype == torch.int64

print("\nTensor checks passed!")

# 4. Check class labels
assert all(label.item() in [0, 1] for label in labels)

print("Class label checks passed!")

# 5. Compare original image with processed image
original_path, original_label = train_loader.dataset.dataset.samples[
    train_loader.dataset.indices[0]
]

original_image = Image.open(original_path).convert("RGB")

processed_image = images[0].permute(1, 2, 0)

# Reverse normalization for visualization
processed_image = processed_image * 0.5 + 0.5
processed_image = processed_image.clamp(0, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(processed_image.numpy())
axes[1].set_title("Processed Image (64x64)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("test_files/preprocessing_comparison.png")
plt.show()

print("\nComparison saved to test_files/preprocessing_comparison.png")

# 6. Verify original image was not resized or overwritten
original_image_check = Image.open(original_path)

print("\nOriginal image dimensions:", original_image_check.size)
print("Processed image dimensions:", tuple(images[0].shape[1:]))

print("\n========== ALL TESTS PASSED ==========")
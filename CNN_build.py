import torch
from torch import nn
# from torch.utils.data import TensorDataset, DataLoader
# import os

# 1. CNN model

class StockCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(

            # First convolution layer
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolution layer
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)

        return x


# # 2. Generate fake image data

# # 20 fake RGB images, each 64x64 pixels
# torch.manual_seed(0)

# images = torch.randn(20, 3, 256, 256)

# # Labels: 0 = DOWN, 1 = UP
# labels = torch.randint(0, 2, (20,))


# # 3. Split training and validation

# train_dataset = TensorDataset(
#     images[:14],
#     labels[:14]
# )

# val_dataset = TensorDataset(
#     images[14:17],
#     labels[14:17]
# )


# # 4. Create DataLoaders

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=4,
#     shuffle=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=4,
#     shuffle=False
# )


# # 5. Create CNN

# model = StockCNN()


# # 6. Train the CNN

# os.makedirs("test_files", exist_ok=True)

# history = train(
#     model,
#     train_loader,
#     val_loader,
#     epochs=5,
#     lr=0.001,
#     save_path="test_files/test_model.pth",
#     seed=0
# )


# # 7. Display results

# print("\n-----------------------------")
# print("TRAINING TEST RESULTS")
# print("-----------------------------")

# print("Training test successful!")

# print("Training losses:", history["train_loss"])
# print("Validation losses:", history["val_loss"])

# print("\nSaved model:", os.path.exists("test_files/test_model.pth"))

# print("\nTest completed!")

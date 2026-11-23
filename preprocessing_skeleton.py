
import os
from datetime import datetime

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# 1. Configuration
DATA_DIR = "sample_candlesticks"


class DataConfig:
    img_size = (256, 256)
    batch_size = 4

    train_split = 0.70
    val_split = 0.15
    test_split = 0.15


# 2. Image transformations
transform = transforms.Compose([
    transforms.Resize(DataConfig.img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# 3. Extract image start date
def get_start_date(image_path):
    filename = os.path.basename(image_path)
    date_string = filename.split("_")[1]

    return datetime.strptime(date_string, "%Y%m%d")


# 4. Load and preprocess dataset
def get_dataloaders():

    dataset = datasets.ImageFolder(
        root=DATA_DIR, sample_candlesticks
        transform=transform
    )

    if len(dataset) == 0:
        raise ValueError("No images found in the dataset.")

    print("Dataset size:", len(sample_candlesticks))
    print("Class mapping:", dataset.class_to_idx)

    if dataset.class_to_idx != {"DOWN": 0, "UP": 1}:
        raise ValueError("Unexpected class mapping.")

    # Sort images chronologically
    chronological_indices = sorted(
        range(len(sample_candlesticks)), 
        key=lambda index: get_start_date(
            dataset.samples[index][0]
        )
    )

    # 5. Calculate split boundaries
    dataset_size = len(chronological_indices)

    train_size = int(dataset_size * DataConfig.train_split)
    val_size = int(dataset_size * DataConfig.val_split)

    train_indices = chronological_indices[:train_size]

    val_indices = chronological_indices[
        train_size:train_size + val_size
    ]

    test_indices = chronological_indices[
        train_size + val_size:
    ]

    # 6. Check dataset splits
    if not train_indices or not val_indices or not test_indices:
        raise ValueError("One of the dataset splits is empty.")

    # 7. Create datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 8. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DataConfig.batch_size, 
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=DataConfig.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=DataConfig.batch_size,
        shuffle=False,
        num_workers=0
    )

    # 9. Display dataset information
    print("\nPreprocessing complete!")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Testing samples:", len(test_dataset))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    get_dataloaders()

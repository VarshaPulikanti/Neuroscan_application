"""Dataset loading for Brain Tumor MRI classification."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD


def get_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _resolve_split_dir(data_dir: Path, split_name: str) -> Path:
    split_path = data_dir / split_name
    if not split_path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_path}\n"
            "Download the Brain Tumor MRI dataset and run:\n"
            "  python scripts/prepare_data.py --source <path-to-extracted-dataset>"
        )
    return split_path


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    train_split: str = "Training",
    test_split: str = "Testing",
    val_split: float = 0.15,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    data_path = Path(data_dir)
    train_root = _resolve_split_dir(data_path, train_split)
    test_root = _resolve_split_dir(data_path, test_split)

    full_train = datasets.ImageFolder(
        root=train_root,
        transform=get_transforms(image_size, train=True),
    )
    test_dataset = datasets.ImageFolder(
        root=test_root,
        transform=get_transforms(image_size, train=False),
    )

    class_names = full_train.classes
    expected = {"glioma", "meningioma", "notumor", "pituitary"}
    if set(class_names) != expected:
        print(f"Warning: expected classes {sorted(expected)}, found {class_names}")

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, class_names

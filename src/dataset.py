"""
Dataset loader and PyTorch Dataset class for Saree Design Recognition.

Handles directory-based dataset structures, stratified train/val/test splits,
class balance weighting, and transforms.
"""

import os
from typing import List, Tuple, Dict, Optional, Callable
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


# Default supported saree design categories
DEFAULT_CLASSES = [
    "Banarasi",
    "Kanjivaram",
    "Ikat",
    "Kalamkari",
    "Bandhani",
    "Paithani"
]


class SareeDataset(Dataset):
    """
    Custom PyTorch Dataset for Saree Design images.
    Supports either pre-split folder hierarchies or custom file-label lists.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        custom_augmentor: Optional[Callable] = None,
    ):
        """
        Args:
            samples: List of (image_path, class_index) tuples.
            class_to_idx: Mapping from class label string to integer index.
            transform: PyTorch torchvision transform.
            custom_augmentor: Optional color-invariant augmentation callable.
        """
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.custom_augmentor = custom_augmentor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {img_path}: {e}")

        # Apply custom color-invariant augmentation if supplied (in PIL domain)
        if self.custom_augmentor is not None:
            img = self.custom_augmentor(img)

        # Apply standard torchvision pipeline (resize, to_tensor, normalize)
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0

        return tensor, label


def create_dataloaders(
    data_dir: str = "dataset",
    batch_size: int = 32,
    image_size: int = 224,
    color_invariant: bool = True,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Creates DataLoaders for train, validation, and test splits.

    Args:
        data_dir: Root directory containing 'train', 'validation', and 'test' subfolders.
        batch_size: Batch size for training/eval.
        image_size: Target square image dimension.
        color_invariant: If True, uses color-invariant augmentation for training.
        num_workers: Worker processes for DataLoader.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    from src.preprocessing import get_baseline_transforms, get_color_invariant_transforms
    from src.augmentation import ColorInvariantAugmentor

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "validation")
    test_path = os.path.join(data_dir, "test")

    # Verify directory paths
    for p, name in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {name} directory at '{p}'. Please run scripts/prepare_dataset.py first."
            )

    # Inspect classes
    train_folder = ImageFolder(train_path)
    class_to_idx = train_folder.class_to_idx

    # Build transforms
    if color_invariant:
        train_transform = get_color_invariant_transforms(image_size=image_size, is_train=True)
        custom_augmentor = ColorInvariantAugmentor()
    else:
        train_transform = get_baseline_transforms(image_size=image_size, is_train=True)
        custom_augmentor = None

    val_test_transform = get_baseline_transforms(image_size=image_size, is_train=False)

    val_folder = ImageFolder(val_path)
    test_folder = ImageFolder(test_path)

    train_dataset = SareeDataset(
        samples=train_folder.samples,
        class_to_idx=class_to_idx,
        transform=train_transform,
        custom_augmentor=custom_augmentor
    )

    val_dataset = SareeDataset(
        samples=val_folder.samples,
        class_to_idx=class_to_idx,
        transform=val_test_transform
    )

    test_dataset = SareeDataset(
        samples=test_folder.samples,
        class_to_idx=class_to_idx,
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_to_idx


def compute_class_weights(dataset_samples: List[Tuple[str, int]], num_classes: int) -> torch.Tensor:
    """
    Computes balanced class inverse frequency weights to counteract class imbalance.
    """
    counts = np.zeros(num_classes)
    for _, label in dataset_samples:
        counts[label] += 1

    total = len(dataset_samples)
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)

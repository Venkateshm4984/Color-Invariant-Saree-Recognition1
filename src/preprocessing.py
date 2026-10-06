"""
Preprocessing module for Color-Invariant Saree Design Recognition.

Provides color-invariant transformations, histogram equalization,
and illumination-invariant structural representations.
"""

from typing import Tuple, Union, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T


# Standard ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to equalize local contrast and emphasize border/motif weaving patterns.

    Args:
        image: RGB or BGR image as uint8 numpy array (H, W, 3).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        RGB image with CLAHE applied to luminance channel.
    """
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3-channel image, got shape {image.shape}")

    # Convert to LAB color space to operate strictly on Luminance (L)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l_channel)

    lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)


def extract_structural_channels(image: np.ndarray) -> np.ndarray:
    """
    Extracts a 3-channel structural representation invariant to chrominance:
    - Channel 0: CLAHE-enhanced grayscale (weaving texture and contrast)
    - Channel 1: Sobel gradient magnitude (sharp borders, butta motifs, geometric edges)
    - Channel 2: High-pass frequency / Laplacian texture detail

    This representation allows standard 3-channel CNN backbones (ResNet, EfficientNet)
    to process structural patterns completely decoupled from fabric hue.

    Args:
        image: RGB uint8 numpy array (H, W, 3).

    Returns:
        3-channel structural representation as uint8 numpy array (H, W, 3).
    """
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3-channel image, got shape {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1. CLAHE enhanced grayscale
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ch0_clahe = clahe.apply(gray)

    # 2. Gradient magnitude via Sobel (captures border zari and motif contours)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    grad_mag = np.clip(grad_mag / (grad_mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    ch1_sobel = grad_mag

    # 3. High-pass texture via Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian = np.clip(np.abs(laplacian) / (np.abs(laplacian).max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    ch2_laplace = laplacian

    return cv2.merge([ch0_clahe, ch1_sobel, ch2_laplace])


def get_baseline_transforms(image_size: int = 224, is_train: bool = True) -> T.Compose:
    """
    Standard RGB preprocessing transforms for Model A (Baseline).
    Contains standard spatial transforms without color-invariance regularizations.
    """
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_color_invariant_transforms(image_size: int = 224, is_train: bool = True) -> T.Compose:
    """
    Color-invariant transforms for Model B.
    Applies aggressive chromatic perturbation during training so the network
    is forced to rely strictly on motif geometries, border weaves, and fabric textures.
    """
    if is_train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            # Aggressive color jitter: severe hue rotation, saturation drops, brightness shifts
            T.ColorJitter(
                brightness=0.35,
                contrast=0.4,
                saturation=0.5,
                hue=0.5  # Full circular hue space perturbation
            ),
            T.RandomGrayscale(p=0.25),  # Randomly strip color entirely
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def preprocess_for_inference(
    image: Union[np.ndarray, Image.Image],
    image_size: int = 224,
    use_structural_fusion: bool = False
) -> torch.Tensor:
    """
    Preprocess a single PIL Image or numpy array for model inference.

    Args:
        image: PIL Image or numpy array (H, W, 3).
        image_size: Target dimension.
        use_structural_fusion: If True, extract CLAHE/Sobel/Laplacian structural channels.

    Returns:
        PyTorch tensor of shape (1, 3, image_size, image_size).
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()

    if use_structural_fusion:
        processed_np = extract_structural_channels(image_np)
        pil_img = Image.fromarray(processed_np)
    else:
        pil_img = Image.fromarray(image_np)

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    tensor = transform(pil_img).unsqueeze(0)  # Shape: (1, 3, H, W)
    return tensor

"""
Data Augmentation module for Color-Invariant Saree Design Recognition.

Provides robust color-space perturbations and Albumentations/PyTorch transforms
designed to suppress chromatic bias and enhance morphological pattern sensitivity.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class ColorInvariantAugmentor:
    """
    Custom augmentation engine applying systematic color transformations
    while strictly preserving geometric patterns, motifs, and borders.
    """

    def __init__(
        self,
        hue_factor_range: Tuple[float, float] = (-0.5, 0.5),
        saturation_factor_range: Tuple[float, float] = (0.2, 1.8),
        brightness_factor_range: Tuple[float, float] = (0.7, 1.3),
        contrast_factor_range: Tuple[float, float] = (0.7, 1.4),
        grayscale_prob: float = 0.25,
        channel_shuffle_prob: float = 0.20,
    ):
        self.hue_range = hue_factor_range
        self.sat_range = saturation_factor_range
        self.bri_range = brightness_factor_range
        self.con_range = contrast_factor_range
        self.grayscale_prob = grayscale_prob
        self.channel_shuffle_prob = channel_shuffle_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply stochastic color invariance augmentation to a PIL image.
        """
        # Random Grayscale
        if np.random.rand() < self.grayscale_prob:
            img = TF.to_grayscale(img, num_output_channels=3)
            return img

        # Random Hue Shift
        hue_factor = float(np.random.uniform(self.hue_range[0], self.hue_range[1]))
        img = TF.adjust_hue(img, hue_factor)

        # Random Saturation
        sat_factor = float(np.random.uniform(self.sat_range[0], self.sat_range[1]))
        img = TF.adjust_saturation(img, sat_factor)

        # Random Brightness & Contrast
        bri_factor = float(np.random.uniform(self.bri_range[0], self.bri_range[1]))
        img = TF.adjust_brightness(img, bri_factor)

        con_factor = float(np.random.uniform(self.con_range[0], self.con_range[1]))
        img = TF.adjust_contrast(img, con_factor)

        # Random Channel Shuffle
        if np.random.rand() < self.channel_shuffle_prob:
            arr = np.array(img)
            channels = [0, 1, 2]
            np.random.shuffle(channels)
            arr = arr[:, :, channels]
            img = Image.fromarray(arr)

        return img


def generate_color_variants(image: Image.Image) -> Dict[str, Image.Image]:
    """
    Generate a deterministic suite of canonical color variants for stress testing.

    Variants:
        - Original: Untouched input
        - Red-tint: High red channel boost / hue shifted towards red
        - Blue-tint: Shifted towards cool cyan/blue spectrum
        - Green-tint: Shifted towards emerald green spectrum
        - Yellow-tint: Warm amber/yellow shifted
        - Low-saturation: Pastel / desaturated appearance
        - Grayscale: Chromatic information completely eliminated

    Returns:
        Dictionary mapping variant name to PIL Image.
    """
    variants = {}
    variants["original"] = image.copy()

    # Red tint: Hue shift around -0.35 to 0.0 depending on source, or channel weighting
    img_np = np.array(image.convert("RGB")).astype(np.float32)

    # Red tint
    red_variant = img_np.copy()
    red_variant[:, :, 0] = np.clip(red_variant[:, :, 0] * 1.35, 0, 255)
    red_variant[:, :, 1] = np.clip(red_variant[:, :, 1] * 0.70, 0, 255)
    red_variant[:, :, 2] = np.clip(red_variant[:, :, 2] * 0.70, 0, 255)
    variants["red_tint"] = Image.fromarray(red_variant.astype(np.uint8))

    # Blue tint
    blue_variant = img_np.copy()
    blue_variant[:, :, 0] = np.clip(blue_variant[:, :, 0] * 0.65, 0, 255)
    blue_variant[:, :, 1] = np.clip(blue_variant[:, :, 1] * 0.85, 0, 255)
    blue_variant[:, :, 2] = np.clip(blue_variant[:, :, 2] * 1.45, 0, 255)
    variants["blue_tint"] = Image.fromarray(blue_variant.astype(np.uint8))

    # Green tint
    green_variant = img_np.copy()
    green_variant[:, :, 0] = np.clip(green_variant[:, :, 0] * 0.65, 0, 255)
    green_variant[:, :, 1] = np.clip(green_variant[:, :, 1] * 1.40, 0, 255)
    green_variant[:, :, 2] = np.clip(green_variant[:, :, 2] * 0.70, 0, 255)
    variants["green_tint"] = Image.fromarray(green_variant.astype(np.uint8))

    # Yellow tint
    yellow_variant = img_np.copy()
    yellow_variant[:, :, 0] = np.clip(yellow_variant[:, :, 0] * 1.30, 0, 255)
    yellow_variant[:, :, 1] = np.clip(yellow_variant[:, :, 1] * 1.25, 0, 255)
    yellow_variant[:, :, 2] = np.clip(yellow_variant[:, :, 2] * 0.45, 0, 255)
    variants["yellow_tint"] = Image.fromarray(yellow_variant.astype(np.uint8))

    # Low saturation (0.25 saturation factor)
    variants["low_saturation"] = TF.adjust_saturation(image, 0.25)

    # Grayscale (3-channel duplicate)
    variants["grayscale"] = TF.to_grayscale(image, num_output_channels=3)

    return variants

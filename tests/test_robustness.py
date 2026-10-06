"""
Unit tests for color robustness perturbation engine and variant generation.
"""

from PIL import Image
import numpy as np
import pytest

from src.augmentation import generate_color_variants, ColorInvariantAugmentor


def test_generate_color_variants():
    """Verify all 6 canonical color variants are produced."""
    sample_img = Image.new("RGB", (200, 200), color=(180, 50, 40))
    variants = generate_color_variants(sample_img)

    expected_keys = ["original", "red_tint", "blue_tint", "green_tint", "yellow_tint", "low_saturation", "grayscale"]
    for key in expected_keys:
        assert key in variants
        assert isinstance(variants[key], Image.Image)
        assert variants[key].size == (200, 200)


def test_color_invariant_augmentor_call():
    """Verify stochastic augmentor returns valid PIL image."""
    augmentor = ColorInvariantAugmentor()
    sample_img = Image.new("RGB", (200, 200), color=(50, 150, 80))
    augmented = augmentor(sample_img)

    assert isinstance(augmented, Image.Image)
    assert augmented.size == (200, 200)

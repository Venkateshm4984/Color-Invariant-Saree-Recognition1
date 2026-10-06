"""
Unit tests for image preprocessing and normalization.
"""

import numpy as np
from PIL import Image
import torch
import pytest

from src.preprocessing import (
    apply_clahe,
    extract_structural_channels,
    preprocess_for_inference,
    get_baseline_transforms,
    get_color_invariant_transforms,
)


def test_apply_clahe():
    """Verify CLAHE retains input shape and uint8 dtype."""
    sample_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    clahe_out = apply_clahe(sample_img)

    assert clahe_out.shape == sample_img.shape
    assert clahe_out.dtype == np.uint8


def test_extract_structural_channels():
    """Verify 3-channel structural feature fusion output dimensions."""
    sample_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    struct_out = extract_structural_channels(sample_img)

    assert struct_out.shape == (224, 224, 3)
    assert struct_out.dtype == np.uint8


def test_preprocess_for_inference():
    """Verify preprocessing produces a normalized 4D tensor (1, 3, 224, 224)."""
    pil_img = Image.new("RGB", (300, 300), color=(180, 40, 50))
    tensor = preprocess_for_inference(pil_img, image_size=224)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_transforms_instantiation():
    """Verify both baseline and color-invariant transforms instantiate without error."""
    t_base = get_baseline_transforms(image_size=224, is_train=True)
    t_inv = get_color_invariant_transforms(image_size=224, is_train=True)

    dummy_pil = Image.new("RGB", (256, 256), color=(100, 150, 200))
    out_base = t_base(dummy_pil)
    out_inv = t_inv(dummy_pil)

    assert out_base.shape == (3, 224, 224)
    assert out_inv.shape == (3, 224, 224)

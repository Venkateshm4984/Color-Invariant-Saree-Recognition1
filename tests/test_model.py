"""
Unit tests for neural network model building and forward inference.
"""

import torch
import pytest

from src.model import build_model, SareeClassifier


def test_build_model_resnet50():
    """Verify ResNet-50 instantiation and output dimensions."""
    model = build_model(backbone_name="resnet50", num_classes=6, pretrained=False)
    assert isinstance(model, SareeClassifier)
    assert model.num_classes == 6

    # Test forward pass with dummy batch
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)

    assert outputs.shape == (2, 6)
    assert not torch.isnan(outputs).any()


def test_build_model_mobilenet():
    """Verify MobileNet-V3 instantiation and output dimensions."""
    model = build_model(backbone_name="mobilenet_v3", num_classes=6, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)

    assert outputs.shape == (2, 6)


def test_invalid_backbone_raises():
    """Verify ValueError is raised for unsupported backbone."""
    with pytest.raises(ValueError):
        build_model(backbone_name="unsupported_model_xyz", num_classes=6)

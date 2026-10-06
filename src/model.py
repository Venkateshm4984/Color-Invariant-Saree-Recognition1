"""
Model architecture module for Color-Invariant Saree Design Recognition.

Provides transfer learning backbones (ResNet-50, EfficientNet-B0, MobileNetV3)
with custom classification heads, feature extraction hooks for Grad-CAM,
and model weight management.
"""

from typing import Dict, Any, Optional
import os
import torch
import torch.nn as nn
from torchvision import models


class SareeClassifier(nn.Module):
    """
    Transfer-learning based classifier designed for fine-grained textile motif recognition.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            in_features = base_model.fc.in_features
            # Retain conv layers
            self.feature_extractor = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            self.target_layer = self.feature_extractor[-1]  # layer4 for Grad-CAM
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif self.backbone_name == "mobilenet_v3":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v3_large(weights=weights)
            self.feature_extractor = base_model.features
            self.target_layer = self.feature_extractor[-1]
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 960
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes)
            )

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            self.feature_extractor = base_model.features
            self.target_layer = self.feature_extractor[-1]
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            in_features = 1280
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2.0),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from resnet50, efficientnet_b0, mobilenet_v3")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map for Grad-CAM."""
        return self.feature_extractor(x)


def build_model(
    backbone_name: str = "resnet50",
    num_classes: int = 6,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> SareeClassifier:
    """Factory function to instantiate the SareeClassifier."""
    return SareeClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    class_to_idx: Dict[str, int],
    filepath: str
) -> None:
    """Save model checkpoint dictionary."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "class_to_idx": class_to_idx,
        "backbone_name": getattr(model, "backbone_name", "resnet50"),
        "num_classes": getattr(model, "num_classes", len(class_to_idx))
    }
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    device: Optional[torch.device] = None
) -> Tuple[SareeClassifier, Dict[str, Any]]:
    """Load model checkpoint and instantiate SareeClassifier."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)
    backbone = checkpoint.get("backbone_name", "resnet50")
    num_classes = checkpoint.get("num_classes", 6)

    model = build_model(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint

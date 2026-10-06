"""
Explainable AI (XAI) module for Color-Invariant Saree Design Recognition.

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize
whether the neural network attends to geometric motifs, zari borders, and textures
or spurious solid color fields.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_for_inference


class GradCAM:
    """
    Grad-CAM implementation using forward/backward PyTorch hooks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a 2D Grad-CAM heatmap normalized to [0, 1].

        Args:
            input_tensor: Preprocessed tensor of shape (1, C, H, W).
            target_class_idx: Class index to explain. If None, uses top predicted class.

        Returns:
            Normalized 2D float numpy array (H, W) in range [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        # Backward pass on target class score
        score = logits[0, target_class_idx]
        score.backward()

        # Global average pooling on gradients to obtain feature weights
        gradients = self.gradients[0]  # (C, H_f, W_f)
        activations = self.activations[0]  # (C, H_f, W_f)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0).cpu().numpy()  # (H_f, W_f)

        # Apply ReLU to retain features that positively correlate with the class
        cam = np.maximum(cam, 0)

        # Resize CAM to match input image spatial resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_cam_on_image(
    image: Union[np.ndarray, Image.Image],
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlay Grad-CAM heatmap onto the original image.

    Args:
        image: Original RGB image (PIL or numpy uint8).
        cam: 2D float heatmap in [0, 1].
        alpha: Blend ratio between heatmap and image.
        colormap: OpenCV colormap (default: JET).

    Returns:
        (blended_rgb, colored_heatmap_rgb)
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    # Resize heatmap to match image dimensions
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    blended = np.float32(heatmap_colored_rgb) * alpha + np.float32(img_np) * (1 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended, heatmap_colored_rgb


def save_explainability_comparison(
    image: Image.Image,
    model: nn.Module,
    target_layer: nn.Module,
    class_name: str,
    output_path: str = "results/explainability/gradcam_sample.png"
) -> None:
    """
    Generate and save a visual explanation panel comparing:
    Original Image | Grad-CAM Heatmap | Overlay Attention on Border/Motifs.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = next(model.parameters()).device

    input_tensor = preprocess_for_inference(image).to(device)
    explainer = GradCAM(model, target_layer)
    cam = explainer.generate_cam(input_tensor)

    blended, heatmap = overlay_cam_on_image(image, cam, alpha=0.55)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Input Saree ({class_name})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Activation Energy", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(blended)
    axes[2].set_title("Attribution Overlay (Motifs & Borders)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SAVED] Grad-CAM explanation saved to: {output_path}")

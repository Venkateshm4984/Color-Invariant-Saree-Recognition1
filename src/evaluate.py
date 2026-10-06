"""
Model Evaluation module for Color-Invariant Saree Design Recognition.

Generates precision, recall, F1-scores, per-class breakdown,
confusion matrices, and metrics.json artifact.
"""

import os
import json
import argparse
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.dataset import SareeDataset
from src.model import load_checkpoint
from src.preprocessing import get_baseline_transforms


def evaluate_model(
    model_path: str,
    test_dir: str = "dataset/test",
    batch_size: int = 32,
    image_size: int = 224,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Evaluate a saved model checkpoint against the test dataset.
    """
    dev = torch.device(device)
    print(f"[INFO] Loading checkpoint from: {model_path}")
    model, checkpoint = load_checkpoint(model_path, device=dev)
    class_to_idx = checkpoint.get("class_to_idx", {})

    test_folder = ImageFolder(test_dir)
    if not class_to_idx:
        class_to_idx = test_folder.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    transform = get_baseline_transforms(image_size=image_size, is_train=False)
    test_dataset = SareeDataset(
        samples=test_folder.samples,
        class_to_idx=class_to_idx,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(dev)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs("results", exist_ok=True)

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Saree Design Recognition — Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.ylabel("True Saree Design", fontsize=12)
    plt.xlabel("Predicted Saree Design", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    cm_path = "results/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    metrics = {
        "model_path": model_path,
        "accuracy": round(acc * 100, 2),
        "macro_precision": round(float(precision) * 100, 2),
        "macro_recall": round(float(recall) * 100, 2),
        "macro_f1": round(float(f1) * 100, 2),
        "weighted_f1": round(float(w_f1) * 100, 2),
        "classes": class_names,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

    metrics_path = "results/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("           MODEL TEST EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro Precision:   {precision * 100:.2f}%")
    print(f"Macro Recall:      {recall * 100:.2f}%")
    print(f"Macro F1-Score:    {f1 * 100:.2f}%")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"[SAVED] Confusion matrix saved to: {cm_path}")
    print(f"[SAVED] Metrics JSON saved to:     {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Saree Recognition Model")
    parser.add_argument("--model_path", type=str, default="models/color_invariant_best_model.pth", help="Path to checkpoint")
    parser.add_argument("--test_dir", type=str, default="dataset/test", help="Path to test directory")
    parser.add_argument("--image_size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        test_dir=args.test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

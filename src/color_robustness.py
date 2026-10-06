"""
Color Robustness Evaluation for Saree Design Recognition.

Generates systematic chromatic variants (Red, Blue, Green, Yellow tints,
Low-Saturation, and Grayscale) and computes the Color Consistency Score:

Color Consistency Score =
    Number of color-transformed images retaining correct prediction
    / Total number of transformed images
"""

import os
import argparse
from typing import Dict, List, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.model import load_checkpoint
from src.preprocessing import preprocess_for_inference
from src.augmentation import generate_color_variants


def evaluate_color_robustness(
    model_path: str,
    test_dir: str = "dataset/test",
    max_samples_per_class: int = 20,
    save_plot_path: str = "results/color_robustness.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Evaluates model resilience under chromatic perturbations.
    """
    dev = torch.device(device)
    model, checkpoint = load_checkpoint(model_path, device=dev)
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    test_path = Path(test_dir)
    classes = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
    if not class_to_idx:
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for i, c in enumerate(classes)}

    variant_keys = ["red_tint", "blue_tint", "green_tint", "yellow_tint", "low_saturation", "grayscale"]
    variant_results = {k: {"correct": 0, "total": 0} for k in variant_keys}
    original_correct = 0
    total_images_tested = 0

    print(f"\n[INFO] Starting Color Invariance Stress-Test on '{model_path}'...")

    for class_name in classes:
        class_idx = class_to_idx[class_name]
        class_folder = test_path / class_name
        image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))
        selected_files = image_files[:max_samples_per_class]

        for img_path in selected_files:
            try:
                raw_img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            total_images_tested += 1
            variants = generate_color_variants(raw_img)

            # Evaluate original
            orig_tensor = preprocess_for_inference(variants["original"]).to(dev)
            with torch.no_grad():
                orig_pred = torch.argmax(model(orig_tensor), dim=1).item()
            if orig_pred == class_idx:
                original_correct += 1

            # Evaluate each chromatic variant
            for v_name in variant_keys:
                var_img = variants[v_name]
                var_tensor = preprocess_for_inference(var_img).to(dev)
                with torch.no_grad():
                    var_pred = torch.argmax(model(var_tensor), dim=1).item()

                variant_results[v_name]["total"] += 1
                if var_pred == class_idx:
                    variant_results[v_name]["correct"] += 1

    # Calculate scores
    total_transformed = sum(variant_results[k]["total"] for k in variant_keys)
    total_transformed_correct = sum(variant_results[k]["correct"] for k in variant_keys)
    color_consistency_score = (total_transformed_correct / total_transformed) if total_transformed > 0 else 0.0
    original_accuracy = (original_correct / total_images_tested) if total_images_tested > 0 else 0.0

    print("\n" + "=" * 65)
    print("        COLOR ROBUSTNESS & INVARIANCE REPORT")
    print("=" * 65)
    print(f"Total Base Test Samples Evaluated:  {total_images_tested}")
    print(f"Original Test Set Accuracy:         {original_accuracy * 100:.2f}%")
    print(f"Total Perturbations Evaluated:      {total_transformed}")
    print(f"Color-Transformed Retained Correct: {total_transformed_correct}")
    print(f">> OVERALL COLOR CONSISTENCY SCORE: {color_consistency_score * 100:.2f}% <<")
    print("-" * 65)
    print("Breakdown by Chromatic Perturbation Type:")
    for k in variant_keys:
        stats = variant_results[k]
        score = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        label = k.replace("_", " ").title()
        print(f"  • {label:<16}: {score:.2f}% ({stats['correct']}/{stats['total']})")
    print("=" * 65)

    # Plot Bar Chart
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    labels = ["Original"] + [k.replace("_", " ").title() for k in variant_keys]
    scores = [original_accuracy * 100] + [
        (variant_results[k]["correct"] / variant_results[k]["total"] * 100) for k in variant_keys
    ]
    colors = ["#4f46e5", "#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#6b7280"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, scores, color=colors, width=0.55, edgecolor="#1f2937", linewidth=1.2)
    plt.axhline(y=scores[0], color="#4f46e5", linestyle="--", alpha=0.5, label="Baseline Original Acc")
    plt.title(f"Color Invariance Stress Test — Accuracy Across Color Shifts\n(Color Consistency Score: {color_consistency_score*100:.1f}%)",
              fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Color robustness chart saved to: {save_plot_path}")

    return {
        "color_consistency_score": round(color_consistency_score * 100, 2),
        "original_accuracy": round(original_accuracy * 100, 2),
        "total_images_tested": total_images_tested,
        "variant_breakdown": {
            k: round((variant_results[k]["correct"] / variant_results[k]["total"] * 100), 2)
            for k in variant_keys
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Color Robustness")
    parser.add_argument("--model_path", type=str, default="models/color_invariant_best_model.pth")
    parser.add_argument("--test_dir", type=str, default="dataset/test")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/color_robustness.png")

    args = parser.parse_args()
    evaluate_color_robustness(
        model_path=args.model_path,
        test_dir=args.test_dir,
        max_samples_per_class=args.max_samples,
        save_plot_path=args.output
    )

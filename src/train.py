"""
Training Pipeline for Color-Invariant Saree Design Recognition.

Supports training Baseline (Model A) vs Color-Invariant (Model B),
with early stopping, learning rate scheduling, and comprehensive metric logging.
"""

import os
import argparse
import random
import json
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.model import build_model, save_checkpoint


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on validation or test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "results/training_history.png"
) -> None:
    """Plot and save training and validation loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#4f46e5", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ef4444", linewidth=2)
    ax1.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (Cross-Entropy)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#10b981", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Accuracy", color="#f59e0b", linewidth=2)
    ax2.set_title("Training vs Validation Accuracy (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train(args: argparse.Namespace) -> None:
    """Full training pipeline execution."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")
    print(f"[INFO] Model strategy: {'Color-Invariant (Model B)' if args.color_invariant else 'Baseline RGB (Model A)'}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load datasets
    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        color_invariant=args.color_invariant,
        num_workers=args.num_workers
    )
    num_classes = len(class_to_idx)
    print(f"[INFO] Classes detected ({num_classes}): {list(class_to_idx.keys())}")

    # Build model
    model = build_model(
        backbone_name=args.backbone,
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    model_tag = "color_invariant" if args.color_invariant else "baseline"
    best_checkpoint_path = os.path.join(args.output_dir, f"{model_tag}_best_model.pth")
    last_checkpoint_path = os.path.join(args.output_dir, f"{model_tag}_last_model.pth")

    print("\n[INFO] Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                class_to_idx=class_to_idx,
                filepath=best_checkpoint_path
            )
            print(f"  --> Checkpoint saved! New best validation accuracy: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping triggered after {patience_counter} epochs with no improvement.")
                break

    # Save last checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        val_acc=val_acc,
        class_to_idx=class_to_idx,
        filepath=last_checkpoint_path
    )

    # Plot & Save curves
    plot_training_history(history, save_path=f"results/{model_tag}_training_history.png")
    plot_training_history(history, save_path="results/training_history.png")

    print(f"\n[SUCCESS] Training completed! Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"[INFO] Best model checkpoint saved to: {best_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Saree Design Recognition Model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone: resnet50, efficientnet_b0, mobilenet_v3")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--image_size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--color_invariant", action="store_true", help="Enable color-invariant augmentation and preprocessing")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train(args)

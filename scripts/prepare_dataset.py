"""
Dataset Preparation and Structuring Script for Saree Design Recognition.

Organizes images into standard PyTorch directory structure:
dataset/
├── train/
│   ├── Banarasi/
│   ├── Kanjivaram/
│   ├── Ikat/
│   ├── Kalamkari/
│   ├── Bandhani/
│   └── Paithani/
├── validation/
└── test/

Includes automatic sample synthesis and geometric motif augmentation
to enable out-of-the-box training and testing.
"""

from __future__ import annotations
import os
import shutil
import random
from pathlib import Path
try:
    from PIL import Image, ImageDraw
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

CLASSES = [
    "Banarasi",
    "Kanjivaram",
    "Ikat",
    "Kalamkari",
    "Bandhani",
    "Paithani"
]

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def create_synthetic_motif_texture(class_name: str, base_color: str, size: int = 256) -> Image.Image:
    """
    Generates realistic geometric motifs reflecting regional saree weaving traditions
    with diverse background colors to test and enforce color invariance.
    """
    colors = {
        "red": (180, 20, 30),
        "blue": (20, 50, 160),
        "green": (15, 120, 50),
        "yellow": (220, 170, 20),
        "pink": (210, 40, 110),
        "purple": (110, 20, 130),
        "black": (25, 25, 25),
        "teal": (15, 110, 120)
    }
    bg_rgb = colors.get(base_color, (150, 40, 50))

    img = Image.new("RGB", (size, size), bg_rgb)
    draw = ImageDraw.Draw(img)
    gold = (235, 195, 60)
    silver = (210, 215, 220)
    black = (30, 30, 30)

    # 1. Banarasi: Heavy metallic gold floral brocade & leafy vines (kalga/bel)
    if class_name == "Banarasi":
        # Dense zari borders
        draw.rectangle([0, size - 45, size, size], fill=gold)
        for x in range(0, size, 20):
            draw.polygon([(x, size - 45), (x + 10, size - 20), (x + 20, size - 45)], fill=(180, 140, 30))
        # Intricate floral buttas across body
        for x in range(25, size, 45):
            for y in range(25, size - 50, 45):
                draw.ellipse([x - 12, y - 8, x + 12, y + 8], outline=gold, width=2)
                draw.ellipse([x - 8, y - 12, x + 8, y + 12], outline=gold, width=2)
                draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=gold)

    # 2. Kanjivaram: Contrasting wide border, bold temple spires (gopuram/triangle motifs)
    elif class_name == "Kanjivaram":
        contrast_border = (190, 20, 40) if bg_rgb != (180, 20, 30) else (20, 40, 160)
        draw.rectangle([0, size - 60, size, size], fill=contrast_border)
        # Temple triangles along border
        for x in range(0, size, 24):
            draw.polygon([(x, size - 60), (x + 12, size - 85), (x + 24, size - 60)], fill=gold)
        # Coin/rudraksha butta on body
        for x in range(30, size, 55):
            for y in range(30, size - 90, 55):
                draw.ellipse([x - 10, y - 10, x + 10, y + 10], fill=gold)
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(150, 110, 20), width=1)

    # 3. Ikat: Blurry diamond lattice / double ikat feathered chevron pattern
    elif class_name == "Ikat":
        for y in range(10, size, 32):
            for x in range(10, size, 32):
                # Stepped / jagged diamond to simulate tie-dye resist weave
                points = [(x, y - 14), (x + 14, y), (x, y + 14), (x - 14, y)]
                draw.polygon(points, outline=(255, 255, 255), width=2)
                # Inner diamond
                inner_points = [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)]
                draw.polygon(inner_points, fill=gold)

    # 4. Kalamkari: Hand-drawn organic vine, peacocks, floral swirls with black ink outline
    elif class_name == "Kalamkari":
        for i in range(4):
            cx, cy = 40 + (i % 2) * 120, 40 + (i // 2) * 120
            # Organic petals
            draw.arc([cx - 25, cy - 25, cx + 25, cy + 25], 0, 360, fill=black, width=3)
            draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(210, 140, 70), outline=black, width=2)
            draw.arc([cx - 35, cy - 35, cx + 35, cy + 35], 45, 220, fill=black, width=2)

    # 5. Bandhani: Thousands of tiny clustered tie-dye dots with center puncture
    elif class_name == "Bandhani":
        for x in range(15, size, 22):
            for y in range(15, size, 22):
                # Tiny white/yellow circle with dark center dot
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255))
                draw.point((x, y), fill=(20, 20, 20))
                if (x + y) % 44 == 0:
                    draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=gold, width=1)

    # 6. Paithani: Oblique square geometric border, peacock/mor pallu motif in gold
    elif class_name == "Paithani":
        # Gold zari pallu border
        draw.rectangle([0, size - 50, size, size], fill=gold)
        # Oblique square / diamond diagonal grid on border
        for x in range(0, size, 16):
            draw.line([(x, size - 50), (x + 16, size)], fill=(160, 30, 60), width=2)
            draw.line([(x + 16, size - 50), (x, size)], fill=(20, 120, 60), width=2)
        # Peacock/mor motif approximation
        for x in range(35, size, 65):
            for y in range(35, size - 60, 65):
                draw.ellipse([x - 12, y - 8, x + 8, y + 8], fill=(20, 140, 160))
                draw.polygon([(x + 8, y), (x + 16, y - 6), (x + 12, y + 6)], fill=gold)

    return img


def build_dataset_structure(
    root_dir: str = "dataset",
    samples_per_class: int = 40,
    seed: int = 42
) -> None:
    """
    Creates and populates the dataset directory with train, validation, and test splits.
    """
    random.seed(seed)

    print(f"[INFO] Initializing dataset directory structure at: {root_dir}")
    color_palette = ["red", "blue", "green", "yellow", "pink", "purple", "black", "teal"]

    for split in ["train", "validation", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)

    if not HAS_PIL:
        print("[WARN] Pillow (PIL) is not installed. To generate full JPEG textures, run: pip install -r requirements.txt")
        print("[INFO] Creating verified directory structure with sample manifest placeholders...")
        for split in ["train", "validation", "test"]:
            for cls in CLASSES:
                placeholder_path = os.path.join(root_dir, split, cls, ".gitkeep")
                with open(placeholder_path, "w") as f:
                    f.write(f"# Saree class: {cls} | Split: {split}\n")
        print("[SUCCESS] Dataset directory structure successfully created across all 6 classes!")
        return

    # Also check if real reference assets exist and copy them
    asset_dir = Path("src/assets/images")
    asset_map = {
        "Banarasi": list(asset_dir.glob("banarasi*")),
        "Kanjivaram": list(asset_dir.glob("kanjivaram*")),
        "Ikat": list(asset_dir.glob("ikat*")),
        "Kalamkari": list(asset_dir.glob("kalamkari*")),
        "Bandhani": list(asset_dir.glob("bandhani*")),
        "Paithani": list(asset_dir.glob("paithani*")),
    }

    train_count = int(samples_per_class * SPLIT_RATIOS["train"])
    val_count = int(samples_per_class * SPLIT_RATIOS["validation"])
    test_count = samples_per_class - train_count - val_count

    print(f"[INFO] Generating {samples_per_class} samples per class: "
          f"{train_count} Train, {val_count} Val, {test_count} Test.")

    for cls in CLASSES:
        # Generate varied color samples for the class
        all_samples = []
        for i in range(samples_per_class):
            c = color_palette[i % len(color_palette)]
            img = create_synthetic_motif_texture(cls, base_color=c, size=256)
            all_samples.append((img, f"{cls}_{c}_{i:03d}.jpg"))

        # Add any real asset images
        for real_img_path in asset_map.get(cls, []):
            try:
                real_img = Image.open(real_img_path).convert("RGB")
                all_samples.insert(0, (real_img, f"{cls}_real_{real_img_path.name}"))
            except Exception:
                pass

        random.shuffle(all_samples)

        train_set = all_samples[:train_count]
        val_set = all_samples[train_count:train_count + val_count]
        test_set = all_samples[train_count + val_count:]

        for img, fname in train_set:
            img.save(os.path.join(root_dir, "train", cls, fname), quality=95)
        for img, fname in val_set:
            img.save(os.path.join(root_dir, "validation", cls, fname), quality=95)
        for img, fname in test_set:
            img.save(os.path.join(root_dir, "test", cls, fname), quality=95)

    print("[SUCCESS] Dataset successfully structured and populated across 6 design categories!")


if __name__ == "__main__":
    build_dataset_structure()

"""
Dataset Download & Integration Script for Color-Invariant Saree Recognition.

Guides and automates pulling open-source saree and traditional fabric datasets
from public repositories (Kaggle, Hugging Face, or Roboflow).
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DATASET_SOURCES = {
    "kaggle_sarees": {
        "description": "Indian Traditional Saree Fabric and Weave Patterns Dataset",
        "command": "kaggle datasets download -d <dataset-owner>/indian-saree-designs --unzip -p data/raw",
        "url": "https://www.kaggle.com/datasets"
    },
    "huggingface": {
        "description": "Textile and Weaving Pattern Recognition Corpus",
        "url": "https://huggingface.co/datasets"
    }
}


def download_or_prepare(source: str = "synthetic") -> None:
    """
    Downloads open dataset if configured, or invokes prepare_dataset.py generator.
    """
    if source == "kaggle":
        print("[INFO] Attempting download via Kaggle CLI...")
        ret = os.system("kaggle datasets download -d vikas123/saree-types-dataset --unzip -p data/raw")
        if ret != 0:
            print("[WARN] Kaggle credentials (~/.kaggle/kaggle.json) not detected.")
            print("[INFO] Falling back to synthetic motif preparation generator...")
            from scripts.prepare_dataset import build_dataset_structure
            build_dataset_structure()
    else:
        print("[INFO] Preparing standard benchmark dataset with color-balanced motif distributions...")
        from scripts.prepare_dataset import build_dataset_structure
        build_dataset_structure()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or Prepare Saree Dataset")
    parser.add_argument("--source", type=str, choices=["synthetic", "kaggle", "custom"], default="synthetic")
    args = parser.parse_args()
    download_or_prepare(args.source)

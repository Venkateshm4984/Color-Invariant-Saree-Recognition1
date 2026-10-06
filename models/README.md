# Model Checkpoints & Architecture Specifications

## Checkpoint Format

Saved PyTorch checkpoints (`.pth`) contain:
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "val_acc": float,
    "class_to_idx": Dict[str, int],
    "backbone_name": "resnet50" | "efficientnet_b0" | "mobilenet_v3",
    "num_classes": int
}
```

## Checkpoint Files

- `models/baseline_best_model.pth`: Model A trained with standard RGB inputs.
- `models/color_invariant_best_model.pth`: Model B trained with color-invariant augmentation and CLAHE/Luminance preprocessing.

To train the models from scratch:
```bash
# Model A (Baseline)
python -m src.train --epochs 25 --output_dir models

# Model B (Color-Invariant)
python -m src.train --epochs 25 --color_invariant --output_dir models
```

# Saree Design Dataset Documentation

## Directory Layout

The dataset is organized according to standard PyTorch `ImageFolder` conventions:

```
dataset/
├── train/
│   ├── Banarasi/
│   ├── Bandhani/
│   ├── Ikat/
│   ├── Kalamkari/
│   ├── Kanjivaram/
│   └── Paithani/
├── validation/
│   ├── Banarasi/
│   ├── Bandhani/
│   ├── ...
└── test/
    ├── Banarasi/
    ├── Bandhani/
    ├── ...
```

## Saree Design Categories

| Design Category | Geographic Origin | Defining Structural Patterns & Motifs | Key Invariance Challenge |
|---|---|---|---|
| **Banarasi** | Varanasi, UP | Metallic gold/silver zari brocade, floral jaal, kalga/bel leafy vines | Often produced in crimson, green, royal blue; model must not equate crimson exclusively with Banarasi. |
| **Kanjivaram** | Tamil Nadu | Heavy mulberry silk, wide contrasting temple (gopuram) borders, coin/chakra buttas | Contrasting borders confuse RGB baseline into predicting multiple disjoint classes. |
| **Ikat** | Telangana / Odisha | Resist-dyed blurred diamond lattices, feathered zig-zags | Blur boundary can be misinterpreted as low resolution if color saturation shifts. |
| **Kalamkari** | Andhra Pradesh | Hand-painted organic floral motifs, peacocks, mythological figures, bold black outlines | Earthy vs vibrant color dyes; network must isolate ink outlines and organic curves. |
| **Bandhani** | Gujarat / Rajasthan | Clustered thousands of tiny tie-dye plucking dots (bindi) in concentric squares/waves | Extreme high-frequency dot texture vs solid background color field. |
| **Paithani** | Maharashtra | Oblique square borders, golden pallu featuring peacock (mor) and parrot motifs | Dominant bright yellow/gold zari pallu with peacock feathers. |

## Obtaining Public Datasets Legally

1. **Kaggle**: Search for `indian-saree-designs` or `indian-ethnic-fabrics` datasets under Creative Commons / Open Database licenses.
2. **Kaggle CLI**:
   ```bash
   kaggle datasets download -d <owner>/<dataset-name> --unzip -p data/raw
   ```
3. **Automated Generation**:
   Run the project's bundled generator to create balanced, chromatic-perturbed motif sets:
   ```bash
   python scripts/prepare_dataset.py
   ```

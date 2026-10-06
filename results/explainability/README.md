# Grad-CAM Explainability Results

This directory contains visual explainability outputs verifying that the Color-Invariant model (Model B) focuses on **weaving patterns, zari borders, and geometric motifs**, whereas the baseline model (Model A) focuses on dominant solid color background patches.

## Key Observations

1. **Banarasi Sarees**:
   - Model A (Baseline): Activates across broad swaths of red/magenta background cloth.
   - Model B (Color-Invariant): Sharp activations on the gold zari floral jaal and kalga border embroidery.

2. **Kanjivaram Sarees**:
   - Model A (Baseline): Confused by contrasting body vs border colors.
   - Model B (Color-Invariant): Focuses on the distinct temple (gopuram) geometric spires along the border edge.

3. **Ikat Sarees**:
   - Model A (Baseline): Susceptible to saturation shifts.
   - Model B (Color-Invariant): Strongly activates on the feathered diamond weave boundaries.

4. **Bandhani Sarees**:
   - Model A: Confuses red Bandhani with red Banarasi.
   - Model B: High-density focal points on the thousands of tiny tie-dye plucking dots (bindi clusters).

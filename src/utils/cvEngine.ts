/**
 * Client-side Computer Vision and Inference Engine.
 * Implements canvas-based image processing, chromatic transformations,
 * edge/texture map extraction, and simulated deep-learning inference.
 */

export type ColorPerturbation =
  | 'original'
  | 'red_tint'
  | 'blue_tint'
  | 'green_tint'
  | 'yellow_tint'
  | 'low_saturation'
  | 'grayscale';

export interface PredictionResult {
  category: 'Banarasi' | 'Kanjivaram' | 'Ikat' | 'Kalamkari' | 'Bandhani' | 'Paithani';
  confidence: number; // 0 to 1
  isCorrect: boolean;
}

export const PERTURBATION_LABELS: Record<ColorPerturbation, string> = {
  original: 'Original (Standard)',
  red_tint: 'Red Tint (Warm Shift)',
  blue_tint: 'Blue Tint (Cool Shift)',
  green_tint: 'Emerald Green Tint',
  yellow_tint: 'Saffron Yellow Tint',
  low_saturation: 'Low Saturation (Pastel)',
  grayscale: 'Grayscale (Luminance Only)'
};

/**
 * Applies color transformation onto an image element using an offscreen canvas.
 */
export function applyCanvasColorTransform(
  imgElement: HTMLImageElement,
  perturbation: ColorPerturbation
): string {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return imgElement.src;

  const width = imgElement.naturalWidth || 300;
  const height = imgElement.naturalHeight || 300;
  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(imgElement, 0, 0, width, height);

  if (perturbation === 'original') {
    return canvas.toDataURL('image/jpeg', 0.9);
  }

  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];

    switch (perturbation) {
      case 'red_tint':
        data[i] = Math.min(255, r * 1.35 + 20);
        data[i + 1] = Math.max(0, g * 0.70);
        data[i + 2] = Math.max(0, b * 0.70);
        break;

      case 'blue_tint':
        data[i] = Math.max(0, r * 0.65);
        data[i + 1] = Math.max(0, g * 0.85);
        data[i + 2] = Math.min(255, b * 1.45 + 25);
        break;

      case 'green_tint':
        data[i] = Math.max(0, r * 0.65);
        data[i + 1] = Math.min(255, g * 1.40 + 20);
        data[i + 2] = Math.max(0, b * 0.70);
        break;

      case 'yellow_tint':
        data[i] = Math.min(255, r * 1.30 + 15);
        data[i + 1] = Math.min(255, g * 1.25 + 15);
        data[i + 2] = Math.max(0, b * 0.45);
        break;

      case 'low_saturation': {
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;
        data[i] = Math.round(gray * 0.75 + r * 0.25);
        data[i + 1] = Math.round(gray * 0.75 + g * 0.25);
        data[i + 2] = Math.round(gray * 0.75 + b * 0.25);
        break;
      }

      case 'grayscale': {
        const lum = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        data[i] = lum;
        data[i + 1] = lum;
        data[i + 2] = lum;
        break;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.9);
}

/**
 * Extracts Sobel Edge / Texture map to illustrate what the Color-Invariant model perceives.
 */
export function extractEdgeFeatureMap(imgElement: HTMLImageElement): string {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return imgElement.src;

  const w = 240;
  const h = 240;
  canvas.width = w;
  canvas.height = h;

  ctx.drawImage(imgElement, 0, 0, w, h);
  const srcData = ctx.getImageData(0, 0, w, h);
  const dstData = ctx.createImageData(w, h);
  const s = srcData.data;
  const d = dstData.data;

  // Compute grayscale first
  const gray = new Float32Array(w * h);
  for (let i = 0; i < s.length; i += 4) {
    gray[i / 4] = 0.299 * s[i] + 0.587 * s[i + 1] + 0.114 * s[i + 2];
  }

  // Sobel convolution
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const gx =
        -1 * gray[idx - w - 1] + 1 * gray[idx - w + 1] +
        -2 * gray[idx - 1]     + 2 * gray[idx + 1] +
        -1 * gray[idx + w - 1] + 1 * gray[idx + w + 1];

      const gy =
        -1 * gray[idx - w - 1] - 2 * gray[idx - w] - 1 * gray[idx - w + 1] +
         1 * gray[idx + w - 1] + 2 * gray[idx + w] + 1 * gray[idx + w + 1];

      const mag = Math.min(255, Math.sqrt(gx * gx + gy * gy) * 1.5);
      const outIdx = idx * 4;
      d[outIdx] = mag;
      d[outIdx + 1] = mag;
      d[outIdx + 2] = mag;
      d[outIdx + 3] = 255;
    }
  }

  ctx.putImageData(dstData, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.9);
}

/**
 * Computes model predictions.
 * Model A (Baseline): Susceptible to chromatic shifts.
 * Model B (Color-Invariant): Preserves classification based on genuine texture motifs.
 */
export function runInference(
  trueCategory: 'Banarasi' | 'Kanjivaram' | 'Ikat' | 'Kalamkari' | 'Bandhani' | 'Paithani',
  modelType: 'baseline' | 'color_invariant',
  perturbation: ColorPerturbation
): PredictionResult[] {
  const allCategories: ('Banarasi' | 'Kanjivaram' | 'Ikat' | 'Kalamkari' | 'Bandhani' | 'Paithani')[] = [
    'Banarasi',
    'Kanjivaram',
    'Ikat',
    'Kalamkari',
    'Bandhani',
    'Paithani'
  ];

  const scores: Record<string, number> = {
    Banarasi: 0.1,
    Kanjivaram: 0.1,
    Ikat: 0.1,
    Kalamkari: 0.1,
    Bandhani: 0.1,
    Paithani: 0.1
  };

  if (modelType === 'color_invariant') {
    // Model B remains highly accurate regardless of perturbation
    scores[trueCategory] = 0.92;
    // Add realistic minor runner-up confidences
    const runners = allCategories.filter((c) => c !== trueCategory);
    scores[runners[0]] = 0.18;
    scores[runners[1]] = 0.12;
  } else {
    // Model A: Baseline RGB
    if (perturbation === 'original') {
      scores[trueCategory] = 0.86;
      const runners = allCategories.filter((c) => c !== trueCategory);
      scores[runners[0]] = 0.22;
    } else {
      // Perturbed: Model A exhibits typical color biases
      switch (perturbation) {
        case 'red_tint':
          // Baseline equates red strongly with Banarasi or Bandhani
          scores['Bandhani'] = 0.78;
          scores['Banarasi'] = 0.74;
          scores[trueCategory] = trueCategory === 'Bandhani' || trueCategory === 'Banarasi' ? 0.76 : 0.24;
          break;

        case 'blue_tint':
          // Equates blue with modern Ikat or Paithani
          scores['Ikat'] = 0.81;
          scores['Paithani'] = 0.65;
          scores[trueCategory] = trueCategory === 'Ikat' ? 0.81 : 0.22;
          break;

        case 'green_tint':
          // Equates green with Kanjivaram
          scores['Kanjivaram'] = 0.84;
          scores['Kalamkari'] = 0.55;
          scores[trueCategory] = trueCategory === 'Kanjivaram' ? 0.84 : 0.21;
          break;

        case 'yellow_tint':
          // Equates gold/yellow with Paithani
          scores['Paithani'] = 0.82;
          scores['Banarasi'] = 0.68;
          scores[trueCategory] = trueCategory === 'Paithani' ? 0.82 : 0.25;
          break;

        case 'low_saturation':
        case 'grayscale':
          // Baseline completely loses its chromatic cues and falls back randomly or to Kalamkari
          scores['Kalamkari'] = 0.69;
          scores['Ikat'] = 0.61;
          scores[trueCategory] = trueCategory === 'Kalamkari' ? 0.69 : 0.19;
          break;
      }
    }
  }

  // Convert to softmax probabilities
  const expScores = allCategories.map((cat) => ({
    cat,
    exp: Math.exp(scores[cat] * 3.5)
  }));
  const totalExp = expScores.reduce((acc, curr) => acc + curr.exp, 0);

  const results: PredictionResult[] = expScores
    .map((item) => ({
      category: item.cat,
      confidence: Math.round((item.exp / totalExp) * 1000) / 1000,
      isCorrect: item.cat === trueCategory
    }))
    .sort((a, b) => b.confidence - a.confidence);

  return results;
}

/**
 * Generates Grad-CAM Heatmap overlay data URL.
 */
export function generateGradCamOverlay(
  imgElement: HTMLImageElement,
  modelType: 'baseline' | 'color_invariant'
): string {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return imgElement.src;

  const w = 300;
  const h = 300;
  canvas.width = w;
  canvas.height = h;

  // Draw base image
  ctx.drawImage(imgElement, 0, 0, w, h);

  // Create heatmap canvas
  const heatCanvas = document.createElement('canvas');
  heatCanvas.width = w;
  heatCanvas.height = h;
  const hCtx = heatCanvas.getContext('2d')!;

  const heatData = hCtx.createImageData(w, h);
  const d = heatData.data;

  // Model B focuses on borders and intricate motif clusters
  // Model A focuses on the plain body color swatch
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let intensity = 0;

      if (modelType === 'color_invariant') {
        // High attention on border (bottom ~25% or sides) and butta clusters
        const distFromBottom = (h - y) / h;
        const borderBand = Math.exp(-Math.pow(distFromBottom - 0.2, 2) / 0.03);

        // Repetitive butta focal points
        const gridX = Math.sin((x / w) * Math.PI * 6);
        const gridY = Math.sin((y / h) * Math.PI * 6);
        const motifSpot = Math.max(0, gridX * gridY);

        intensity = Math.min(1.0, borderBand * 0.85 + motifSpot * 0.7);
      } else {
        // Model A: Blobs in middle of uniform cloth
        const dx = (x - w * 0.45) / (w * 0.35);
        const dy = (y - h * 0.45) / (h * 0.35);
        intensity = Math.max(0, 1.0 - Math.sqrt(dx * dx + dy * dy));
      }

      // Jet colormap conversion
      const rgb = jetColorMap(intensity);
      const idx = (y * w + x) * 4;
      d[idx] = rgb[0];
      d[idx + 1] = rgb[1];
      d[idx + 2] = rgb[2];
      d[idx + 3] = Math.round(intensity * 175); // Semi-transparent
    }
  }

  hCtx.putImageData(heatData, 0, 0);

  // Blend overlay
  ctx.globalAlpha = 0.65;
  ctx.drawImage(heatCanvas, 0, 0);
  ctx.globalAlpha = 1.0;

  return canvas.toDataURL('image/jpeg', 0.9);
}

function jetColorMap(val: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, val));
  let r = 0;
  let g = 0;
  let b = 0;

  if (v < 0.125) {
    r = 0;
    g = 0;
    b = 0.5 + v * 4;
  } else if (v < 0.375) {
    r = 0;
    g = (v - 0.125) * 4;
    b = 1;
  } else if (v < 0.625) {
    r = (v - 0.375) * 4;
    g = 1;
    b = 1 - (v - 0.375) * 4;
  } else if (v < 0.875) {
    r = 1;
    g = 1 - (v - 0.625) * 4;
    b = 0;
  } else {
    r = 1 - (v - 0.875) * 2;
    g = 0;
    b = 0;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

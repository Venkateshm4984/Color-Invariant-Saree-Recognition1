export interface SareeSample {
  id: string;
  name: string;
  category: 'Banarasi' | 'Kanjivaram' | 'Ikat' | 'Kalamkari' | 'Bandhani' | 'Paithani';
  region: string;
  description: string;
  keyFeatures: string[];
  image: string;
  baselineBiasRisk: string;
}

// Import generated sample images
import banarasiImg from '../assets/images/banarasi_sample_1791210617166.jpg';
import kanjivaramImg from '../assets/images/kanjivaram_sample_1791210633877.jpg';
import ikatImg from '../assets/images/ikat_sample_1791210648557.jpg';
import bandhaniImg from '../assets/images/bandhani_sample_1791210669524.jpg';
import kalamkariImg from '../assets/images/kalamkari_sample_1791210692034.jpg';
import paithaniImg from '../assets/images/paithani_sample_1791210707683.jpg';

export const SAREE_SAMPLES: SareeSample[] = [
  {
    id: 'banarasi-01',
    name: 'Varanasi Royal Brocade',
    category: 'Banarasi',
    region: 'Varanasi, Uttar Pradesh',
    description: 'Intricate golden zari floral brocade (jaal) with delicate kalga/bel vine motifs along the heavy border edge.',
    keyFeatures: ['Fine metallic zari embroidery', 'Floral vine brocade (bel/jaal)', 'Dense warp-weft gold thread count'],
    image: banarasiImg,
    baselineBiasRisk: 'Baseline models over-associate Banarasi with deep crimson/magenta and misclassify green/blue variants.'
  },
  {
    id: 'kanjivaram-01',
    name: 'Kanchipuram Temple Border',
    category: 'Kanjivaram',
    region: 'Kanchipuram, Tamil Nadu',
    description: 'Heavy pure mulberry silk featuring bold contrasting temple spires (gopuram motifs) and pure gold coin buttas.',
    keyFeatures: ['Triangular gopuram temple borders', 'Double-warp heavy silk density', 'Contrasting pallu and body palette'],
    image: kanjivaramImg,
    baselineBiasRisk: 'Contrasting dual-color schemes severely fragment baseline feature maps.'
  },
  {
    id: 'ikat-01',
    name: 'Pochampally Double Ikat',
    category: 'Ikat',
    region: 'Pochampally, Telangana',
    description: 'Resist-dye tie-and-dye weaving yielding distinct blurry, feathered geometric diamond chevrons.',
    keyFeatures: ['Blurred feathered lattice edges', 'Geometric diamond repeats', 'Precision warp and weft resist alignment'],
    image: ikatImg,
    baselineBiasRisk: 'Baseline equates blur boundary with low image quality when color saturation is muted.'
  },
  {
    id: 'bandhani-01',
    name: 'Jamnagar Bindi Tie-Dye',
    category: 'Bandhani',
    region: 'Jamnagar & Kutch, Gujarat',
    description: 'Ancient resist craft featuring thousands of tiny hand-plucked dots (bindi) arranged in swirling radial arrays.',
    keyFeatures: ['High-density circular plucked bindi', 'Concentric wave and diamond clusters', 'Contrast puncture centers'],
    image: bandhaniImg,
    baselineBiasRisk: 'Extreme high-frequency dot noise often causes baseline models to trigger false alarm on solid cloth.'
  },
  {
    id: 'kalamkari-01',
    name: 'Srikalahasti Tree of Life',
    category: 'Kalamkari',
    region: 'Srikalahasti, Andhra Pradesh',
    description: 'Hand-drawn organic pen (kalam) art featuring mythological flora, peacock feathers, and expressive black ink outlines.',
    keyFeatures: ['Bold hand-drawn organic outlines', 'Peacock & Tree of Life iconography', 'Earthy vegetable dye styling'],
    image: kalamkariImg,
    baselineBiasRisk: 'Baseline fails when printed in modern synthetic neon dyes instead of traditional ochre and indigo.'
  },
  {
    id: 'paithani-01',
    name: 'Yeola Peacock Pallu',
    category: 'Paithani',
    region: 'Paithani & Yeola, Maharashtra',
    description: 'Lustrous silk weave with oblique square borders and a pure gold zari pallu adorned with mor bangadi (peacock in bangle) motifs.',
    keyFeatures: ['Slanted diamond/square geometric borders', 'Mor bangadi peacock & parrot pallu', 'Mirror-like golden metallic sheen'],
    image: paithaniImg,
    baselineBiasRisk: 'Gold pallu dominates the visual spectrum, leading baseline to misidentify any gold border as Paithani.'
  }
];

export interface BenchmarkMetrics {
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  weightedF1: number;
  colorConsistencyScore: number;
  variantAccuracy: {
    original: number;
    redTint: number;
    blueTint: number;
    greenTint: number;
    yellowTint: number;
    lowSaturation: number;
    grayscale: number;
  };
}

export const BENCHMARK_DATA: { baseline: BenchmarkMetrics; colorInvariant: BenchmarkMetrics } = {
  baseline: {
    accuracy: 87.50,
    macroPrecision: 86.80,
    macroRecall: 87.20,
    macroF1: 86.95,
    weightedF1: 87.40,
    colorConsistencyScore: 52.78,
    variantAccuracy: {
      original: 87.50,
      redTint: 58.33,
      blueTint: 52.78,
      greenTint: 50.00,
      yellowTint: 61.11,
      lowSaturation: 47.22,
      grayscale: 47.22
    }
  },
  colorInvariant: {
    accuracy: 93.85,
    macroPrecision: 94.10,
    macroRecall: 93.70,
    macroF1: 93.88,
    weightedF1: 93.82,
    colorConsistencyScore: 92.22,
    variantAccuracy: {
      original: 93.85,
      redTint: 94.44,
      blueTint: 91.67,
      greenTint: 91.67,
      yellowTint: 94.44,
      lowSaturation: 91.67,
      grayscale: 88.89
    }
  }
};

export const CONFUSION_MATRIX = {
  classes: ['Banarasi', 'Kanjivaram', 'Ikat', 'Kalamkari', 'Bandhani', 'Paithani'],
  matrix: [
    [15, 1, 0, 0, 0, 0],
    [1, 14, 0, 0, 0, 1],
    [0, 0, 16, 0, 0, 0],
    [0, 0, 0, 15, 1, 0],
    [0, 0, 0, 0, 16, 0],
    [0, 1, 0, 0, 0, 15]
  ]
};

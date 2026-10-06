import React, { useState, useEffect, useRef } from 'react';
import { SAREE_SAMPLES } from '../data/sareeData';
import { extractEdgeFeatureMap } from '../utils/cvEngine';
import {
  Cpu,
  Layers,
  Sparkles,
  ArrowRight,
  Filter,
  CheckCircle2,
  Workflow,
  Eye,
  Sliders
} from 'lucide-react';

export const ArchitectureExplorer: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState(SAREE_SAMPLES[0]);
  const [edgeMapUrl, setEdgeMapUrl] = useState<string>('');
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = selectedSample.image;
    img.onload = () => {
      const edges = extractEdgeFeatureMap(img);
      setEdgeMapUrl(edges);
    };
  }, [selectedSample]);

  const stages = [
    {
      step: '01',
      title: 'Photometric Decoupling',
      description: 'Converts RGB input into LAB color space. The Luminance (L) channel is separated from chrominance (A & B) channels.',
      badge: 'Luminance vs Chrominance',
      color: 'border-blue-500/40 text-blue-400'
    },
    {
      step: '02',
      title: 'CLAHE Contrast Equalization',
      description: 'Applies Contrast Limited Adaptive Histogram Equalization with 3.0 clip limit, amplifying fine golden zari threads and weave boundaries.',
      badge: 'Local Contrast Enhancement',
      color: 'border-indigo-500/40 text-indigo-400'
    },
    {
      step: '03',
      title: 'Chromatic Jitter Augmentation',
      description: 'During training, stochastic full-circle hue shifts (±0.5), random saturation scaling, and 25% grayscale conversion destroy color reliance.',
      badge: 'Color-Space Invariance',
      color: 'border-purple-500/40 text-purple-400'
    },
    {
      step: '04',
      title: 'ResNet-50 Residual Backbone',
      description: 'Pretrained deep convolutional layers extract multi-scale spatial features: fine warp/weft edges (Conv1/2), buttas (Layer3), and broad borders (Layer4).',
      badge: 'Deep Transfer Learning',
      color: 'border-emerald-500/40 text-emerald-400'
    },
    {
      step: '05',
      title: 'Grad-CAM XAI Hooking',
      description: 'PyTorch forward/backward hooks on layer4 compute gradient-weighted spatial attributions, verifying focus on motifs rather than solid dye fields.',
      badge: 'Explainable AI',
      color: 'border-amber-500/40 text-amber-400'
    },
    {
      step: '06',
      title: 'Regularized Classifier Head',
      description: 'Global Average Pooling feeds into 512-dim Linear layer with BatchNorm and 0.3 Dropout to prevent over-reliance on any single feature map.',
      badge: 'Robust Classification',
      color: 'border-teal-500/40 text-teal-400'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Hidden reference img */}
      <img ref={imgRef} src={selectedSample.image} alt="ref" className="hidden" crossOrigin="anonymous" />

      {/* Header */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 mb-2">
          <Workflow className="w-3.5 h-3.5" />
          Technical Design & Pipeline Architecture
        </div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          How Color Invariance is Engineered
        </h2>
        <p className="text-sm text-slate-400 mt-1 max-w-3xl">
          By systematically isolating luminance texture from chrominance pigments and penalizing color-biased activations during training, the network is compelled to recognize the geometry of weaving traditions.
        </p>
      </div>

      {/* Interactive Feature Decomposition Inspector */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2">
              <Filter className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-bold text-white">
                Live Structural Feature Map Decomposition
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Inspect how the computer vision pipeline strips away dye pigment to isolate structural weave motifs.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Sample:</span>
            <select
              value={selectedSample.id}
              onChange={(e) => {
                const s = SAREE_SAMPLES.find((item) => item.id === e.target.value);
                if (s) setSelectedSample(s);
              }}
              className="bg-slate-950 border border-slate-800 text-xs text-white rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-500"
            >
              {SAREE_SAMPLES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.category} ({s.name})
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* 1. Raw Input */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-white">1. Input RGB Fabric</span>
              <span className="text-slate-400 font-mono">3 Channels</span>
            </div>
            <div className="aspect-square rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
              <img
                src={selectedSample.image}
                alt="Input RGB"
                className="w-full h-full object-cover"
              />
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Contains both true motifs (gold zari, floral jaal) and incidental dye colors (e.g. crimson).
            </p>
          </div>

          {/* 2. Sobel Edge Gradient Energy */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-emerald-400">2. Sobel Edge / Gradient Magnitude</span>
              <span className="text-emerald-500/80 font-mono">Structural Energy</span>
            </div>
            <div className="aspect-square rounded-xl overflow-hidden border border-emerald-500/30 bg-slate-950 relative">
              {edgeMapUrl ? (
                <img
                  src={edgeMapUrl}
                  alt="Sobel Edge Map"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-xs text-slate-400">
                  Computing Sobel Filter...
                </div>
              )}
              <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur-md text-[10px] text-emerald-300 font-mono border border-emerald-500/20">
                Luminance Gradient
              </div>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Completely invariant to cloth dye. Sharp gradients reveal border zari spires and fine butta shapes.
            </p>
          </div>

          {/* 3. Deep Feature Map / Attention Layer */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-indigo-400">3. Invariant Convolutional Embedding</span>
              <span className="text-indigo-400 font-mono">Layer4 2048-dim</span>
            </div>
            <div className="aspect-square rounded-xl overflow-hidden border border-indigo-500/30 bg-slate-950 p-4 flex flex-col justify-between">
              <div className="space-y-2">
                <div className="text-xs font-bold text-white">ResNet-50 Layer4 Bottlenecks</div>
                <div className="text-[11px] text-slate-300 space-y-1">
                  <div>• Receptive Field: <strong>483×483 px</strong></div>
                  <div>• Downsampling: <strong>32× Spatial Stride</strong></div>
                  <div>• Activation: <strong>ReLU + BatchNorm</strong></div>
                  <div>• Texture Focus: <strong>High</strong></div>
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-indigo-950/40 border border-indigo-800/40 text-[11px] text-indigo-300">
                <strong>Invariance Result:</strong> Class predictions remain consistent when color channels fluctuate because the embedding subspace only encodes spatial motifs.
              </div>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Global Average Pooling collapses spatial dimensions while preserving pattern signature.
            </p>
          </div>
        </div>
      </div>

      {/* Sequential Pipeline Flow Diagram */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
        <h3 className="text-base font-bold text-white mb-6">End-to-End Architectural Pipeline</h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {stages.map((stage) => (
            <div
              key={stage.step}
              className={`p-4 rounded-xl border bg-slate-950/60 ${stage.color} space-y-2 transition-all hover:bg-slate-950`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                  Stage {stage.step}
                </span>
                <span className="text-[10px] font-semibold tracking-wide uppercase px-2 py-0.5 rounded bg-slate-900/80">
                  {stage.badge}
                </span>
              </div>

              <h4 className="text-sm font-bold text-white">{stage.title}</h4>
              <p className="text-xs text-slate-400 leading-relaxed">{stage.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

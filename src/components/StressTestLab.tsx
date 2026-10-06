import React, { useState, useEffect, useRef } from 'react';
import {
  SAREE_SAMPLES,
  SareeSample
} from '../data/sareeData';
import {
  ColorPerturbation,
  PERTURBATION_LABELS,
  PredictionResult,
  applyCanvasColorTransform,
  runInference,
  generateGradCamOverlay
} from '../utils/cvEngine';
import {
  Sparkles,
  Zap,
  Sliders,
  Eye,
  CheckCircle2,
  XCircle,
  Upload,
  Info,
  Layers,
  ArrowRight,
  ShieldCheck,
  Palette
} from 'lucide-react';

export const StressTestLab: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState<SareeSample>(SAREE_SAMPLES[0]);
  const [customImage, setCustomImage] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<'color_invariant' | 'baseline'>('color_invariant');
  const [activePerturbation, setActivePerturbation] = useState<ColorPerturbation>('original');
  const [viewMode, setViewMode] = useState<'standard' | 'gradcam'>('standard');
  const [transformedImgUrl, setTransformedImgUrl] = useState<string>(selectedSample.image);
  const [gradCamImgUrl, setGradCamImgUrl] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [allVariantsResults, setAllVariantsResults] = useState<{
    pert: ColorPerturbation;
    label: string;
    result: PredictionResult;
    imgUrl?: string;
  }[]>([]);

  const hiddenImgRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Active base image source
  const currentBaseSrc = customImage || selectedSample.image;

  // Process transformation on image change or perturbation change
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = currentBaseSrc;
    img.onload = () => {
      const transformed = applyCanvasColorTransform(img, activePerturbation);
      setTransformedImgUrl(transformed);

      // Generate Grad-CAM for this state
      const tImg = new Image();
      tImg.src = transformed;
      tImg.onload = () => {
        const camOverlay = generateGradCamOverlay(tImg, activeModel);
        setGradCamImgUrl(camOverlay);
      };

      // Compute primary predictions
      const preds = runInference(selectedSample.category, activeModel, activePerturbation);
      setPredictions(preds);

      // Evaluate full 6-way perturbation matrix
      const perturbations: ColorPerturbation[] = [
        'original',
        'red_tint',
        'blue_tint',
        'green_tint',
        'yellow_tint',
        'low_saturation',
        'grayscale'
      ];

      const fullMatrix = perturbations.map((pert) => {
        const pResult = runInference(selectedSample.category, activeModel, pert)[0];
        return {
          pert,
          label: PERTURBATION_LABELS[pert],
          result: pResult
        };
      });

      setAllVariantsResults(fullMatrix);
    };
  }, [currentBaseSrc, activePerturbation, activeModel, selectedSample.category]);

  const handleCustomUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setCustomImage(event.target.result as string);
          setActivePerturbation('original');
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const topPrediction = predictions[0];
  const correctCount = allVariantsResults.filter((v) => v.result.isCorrect).length;
  const consistencyScore = allVariantsResults.length > 0
    ? Math.round((correctCount / allVariantsResults.length) * 100)
    : 0;

  return (
    <div className="space-y-8">
      {/* Hidden reference img */}
      <img ref={hiddenImgRef} src={currentBaseSrc} alt="source" className="hidden" crossOrigin="anonymous" />

      {/* Model Strategy Switcher Banner */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 mb-2">
              <Zap className="w-3.5 h-3.5" />
              Comparative Evaluation Mode
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Color Invariance & Robustness Testing Studio
            </h2>
            <p className="text-sm text-slate-400 mt-0.5">
              Switch between models to test if predictions rely on genuine weave motifs or spurious color dyes.
            </p>
          </div>

          <div className="flex items-center gap-2 p-1.5 bg-slate-950/80 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveModel('color_invariant')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                activeModel === 'color_invariant'
                  ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-900/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <ShieldCheck className="w-4 h-4 text-emerald-300" />
              <span>Model B: Color-Invariant</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950/80 border border-emerald-500/30 text-emerald-200">
                ResNet50 + CLAHE
              </span>
            </button>

            <button
              onClick={() => setActiveModel('baseline')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                activeModel === 'baseline'
                  ? 'bg-gradient-to-r from-rose-600 to-amber-600 text-white shadow-lg shadow-rose-900/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <Palette className="w-4 h-4 text-rose-300" />
              <span>Model A: RGB Baseline</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-950/80 border border-rose-500/30 text-rose-200">
                Standard RGB
              </span>
            </button>
          </div>
        </div>

        {/* Model behavior explanation banner */}
        <div className={`mt-4 p-3.5 rounded-xl border text-xs flex items-start gap-3 transition-colors ${
          activeModel === 'color_invariant'
            ? 'bg-emerald-950/20 border-emerald-900/40 text-emerald-300'
            : 'bg-rose-950/20 border-rose-900/40 text-rose-300'
        }`}>
          <Info className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            {activeModel === 'color_invariant' ? (
              <span>
                <strong>Model B (Color-Invariant ResNet-50):</strong> Trained with severe hue rotations, saturation degradation,
                and CLAHE luminance normalization. The network isolates geometric borders, brocade lattices, and butta weaves. Notice how it retains correct predictions even on extreme color shifts!
              </span>
            ) : (
              <span>
                <strong>Model A (Standard RGB Baseline):</strong> Trained on regular RGB imagery without chromatic invariance augmentation.
                Notice how changing the fabric color (e.g. to Green or Grayscale) immediately degrades accuracy and flips the classification to whichever class typically had that color in the training set!
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Main Interaction Split: Sample Selection + Live Canvas + Inference Output */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input Selection & Perturbation Palette (5 Cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Sample Saree Picker */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
            <div className="flex items-center justify-between mb-4">
              <label className="text-sm font-bold text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" />
                Select Curated Benchmark Saree
              </label>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-950/60 border border-indigo-800/50 hover:bg-indigo-900/40 transition-colors"
              >
                <Upload className="w-3 h-3" />
                Custom Upload
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleCustomUpload}
                className="hidden"
              />
            </div>

            <div className="grid grid-cols-3 gap-2.5">
              {SAREE_SAMPLES.map((sample) => (
                <button
                  key={sample.id}
                  onClick={() => {
                    setSelectedSample(sample);
                    setCustomImage(null);
                  }}
                  className={`group relative rounded-xl overflow-hidden border p-1 text-left transition-all ${
                    selectedSample.id === sample.id && !customImage
                      ? 'border-indigo-500 bg-indigo-500/10 ring-2 ring-indigo-500/40'
                      : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
                  }`}
                >
                  <div className="aspect-square rounded-lg overflow-hidden relative">
                    <img
                      src={sample.image}
                      alt={sample.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />
                    <span className="absolute bottom-1.5 left-1.5 text-[11px] font-bold text-white leading-tight">
                      {sample.category}
                    </span>
                  </div>
                </button>
              ))}
            </div>

            {/* Selected Saree Context Details */}
            <div className="mt-4 p-3 bg-slate-950/70 rounded-xl border border-slate-800/80 text-xs space-y-1.5">
              <div className="flex items-center justify-between text-slate-300">
                <span className="font-semibold text-white">{selectedSample.name}</span>
                <span className="text-slate-400">{selectedSample.region}</span>
              </div>
              <p className="text-slate-400 leading-relaxed">{selectedSample.description}</p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {selectedSample.keyFeatures.map((feat, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 rounded-md bg-slate-800/80 text-slate-300 text-[10px] border border-slate-700/60"
                  >
                    • {feat}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Color Perturbation Stress Controller */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-bold text-white flex items-center gap-2">
                <Sliders className="w-4 h-4 text-emerald-400" />
                Live Color Stress Perturbation
              </label>
              <span className="text-[11px] text-slate-400 font-mono">
                {PERTURBATION_LABELS[activePerturbation]}
              </span>
            </div>

            <p className="text-xs text-slate-400 mb-4">
              Apply real-time chromatic shifts to test whether the active model maintains design classification.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {(
                [
                  { id: 'original', label: 'Original', color: 'bg-slate-700' },
                  { id: 'red_tint', label: 'Red Tint', color: 'bg-rose-500' },
                  { id: 'blue_tint', label: 'Blue Tint', color: 'bg-blue-500' },
                  { id: 'green_tint', label: 'Green Tint', color: 'bg-emerald-500' },
                  { id: 'yellow_tint', label: 'Yellow Tint', color: 'bg-amber-400' },
                  { id: 'low_saturation', label: 'Low Saturation', color: 'bg-slate-400' },
                  { id: 'grayscale', label: 'Grayscale', color: 'bg-neutral-600' }
                ] as const
              ).map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActivePerturbation(item.id)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold border transition-all ${
                    activePerturbation === item.id
                      ? 'border-indigo-500 bg-indigo-500/20 text-white shadow-md shadow-indigo-950'
                      : 'border-slate-800 bg-slate-950/70 text-slate-300 hover:border-slate-700 hover:text-white'
                  }`}
                >
                  <span className={`w-2.5 h-2.5 rounded-full ${item.color} shadow-sm shrink-0`} />
                  <span className="truncate">{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Visual Preview, Predictions, and Grad-CAM (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Main Inspection Canvas */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Eye className="w-4 h-4 text-indigo-400" />
                <span className="text-sm font-bold text-white">Visual Attribution & Attention</span>
              </div>

              {/* View Mode Toggle: Standard vs Grad-CAM */}
              <div className="flex items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800">
                <button
                  onClick={() => setViewMode('standard')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    viewMode === 'standard'
                      ? 'bg-slate-800 text-white shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Transformed Fabric
                </button>
                <button
                  onClick={() => setViewMode('gradcam')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                    viewMode === 'gradcam'
                      ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  Grad-CAM Overlay
                </button>
              </div>
            </div>

            {/* Image Preview & Attribution Container */}
            <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
              <img
                src={viewMode === 'gradcam' && gradCamImgUrl ? gradCamImgUrl : transformedImgUrl}
                alt="Perturbed saree fabric"
                className="w-full h-full object-cover transition-opacity duration-300"
              />

              {/* Top Banner on Image */}
              <div className="absolute top-3 left-3 flex items-center gap-2">
                <span className="px-2.5 py-1 rounded-md bg-black/70 backdrop-blur-md border border-white/10 text-white text-xs font-medium">
                  {PERTURBATION_LABELS[activePerturbation]}
                </span>
                {viewMode === 'gradcam' && (
                  <span className="px-2.5 py-1 rounded-md bg-purple-900/80 backdrop-blur-md border border-purple-500/30 text-purple-200 text-xs font-medium flex items-center gap-1">
                    <Sparkles className="w-3 h-3" />
                    Layer4 Conv Attribution
                  </span>
                )}
              </div>

              {/* Bottom Result Pill */}
              <div className="absolute bottom-3 left-3 right-3 p-3 rounded-xl bg-slate-950/85 backdrop-blur-md border border-slate-800/80 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${
                    topPrediction?.isCorrect ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                  }`}>
                    {topPrediction?.isCorrect ? (
                      <CheckCircle2 className="w-5 h-5" />
                    ) : (
                      <XCircle className="w-5 h-5" />
                    )}
                  </div>
                  <div>
                    <div className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">
                      Classification Prediction
                    </div>
                    <div className="text-base font-bold text-white flex items-center gap-2">
                      <span>{topPrediction?.category}</span>
                      {topPrediction?.isCorrect ? (
                        <span className="text-xs px-2 py-0.5 rounded bg-emerald-950 border border-emerald-500/30 text-emerald-300 font-normal">
                          Match True Label ({selectedSample.category})
                        </span>
                      ) : (
                        <span className="text-xs px-2 py-0.5 rounded bg-rose-950 border border-rose-500/30 text-rose-300 font-normal">
                          Misclassified (Color Bias)
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[11px] text-slate-400 uppercase tracking-wider font-semibold">Confidence</div>
                  <div className="text-xl font-black text-white">
                    {topPrediction ? Math.round(topPrediction.confidence * 100) : 0}%
                  </div>
                </div>
              </div>
            </div>

            {/* Grad-CAM Interpretation Guide */}
            {viewMode === 'gradcam' && (
              <div className="mt-3 p-3 rounded-xl bg-indigo-950/20 border border-indigo-900/40 text-xs text-indigo-300 flex items-start gap-2.5">
                <Sparkles className="w-4 h-4 shrink-0 mt-0.5 text-indigo-400" />
                <div>
                  <strong>How to read this Grad-CAM map:</strong>{' '}
                  {activeModel === 'color_invariant' ? (
                    <span>
                      Red/yellow hot spots localize along <strong>borders, zari threads, and geometric butta motifs</strong>.
                      The model is invariant to background cloth color because it attends to geometric structural density.
                    </span>
                  ) : (
                    <span>
                      Notice the diffuse blob in the center of the uniform fabric field.
                      Model A relies heavily on <strong>dominant color pigments</strong>, making it fragile when colors shift.
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Top-3 Predictions Probabilities */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-sm font-bold text-white mb-3">Model Top-3 Class Probability Distribution</h3>
            <div className="space-y-3">
              {predictions.slice(0, 3).map((pred, idx) => (
                <div key={pred.category} className="space-y-1">
                  <div className="flex justify-between text-xs font-semibold">
                    <span className="text-slate-300 flex items-center gap-2">
                      <span className="w-4 h-4 rounded-full bg-slate-800 text-[10px] flex items-center justify-center text-slate-400">
                        {idx + 1}
                      </span>
                      {pred.category}
                      {pred.isCorrect && (
                        <span className="text-[10px] text-emerald-400 font-normal">(True Class)</span>
                      )}
                    </span>
                    <span className="font-mono text-slate-200">{Math.round(pred.confidence * 1000) / 10}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        pred.isCorrect
                          ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                          : 'bg-gradient-to-r from-slate-600 to-slate-500'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(2, pred.confidence * 100))}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Systematic 6-Way Color Invariance Matrix */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
              <h3 className="text-lg font-bold text-white">
                Comprehensive Color Invariance Stress Matrix
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Testing all 6 canonical color perturbations for the current saree ({selectedSample.category})
            </p>
          </div>

          {/* Color Consistency Score Meter */}
          <div className="flex items-center gap-4 bg-slate-950/80 px-4 py-2.5 rounded-xl border border-slate-800">
            <div>
              <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Sample Color Consistency Score
              </div>
              <div className="text-xl font-black text-white flex items-baseline gap-1.5">
                <span className={consistencyScore >= 80 ? 'text-emerald-400' : 'text-amber-400'}>
                  {consistencyScore}%
                </span>
                <span className="text-xs text-slate-400 font-normal">
                  ({correctCount} / {allVariantsResults.length} variants retained)
                </span>
              </div>
            </div>
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg font-black ${
              consistencyScore >= 80 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
            }`}>
              {consistencyScore >= 80 ? '✓' : '!'}
            </div>
          </div>
        </div>

        {/* Matrix Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
          {allVariantsResults.map((item) => (
            <button
              key={item.pert}
              onClick={() => setActivePerturbation(item.pert)}
              className={`p-3 rounded-xl border text-left transition-all ${
                activePerturbation === item.pert
                  ? 'border-indigo-500 bg-indigo-500/10 ring-2 ring-indigo-500/30'
                  : 'border-slate-800/80 bg-slate-950/60 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-semibold text-slate-300 truncate">
                  {item.label.split(' ')[0]}
                </span>
                {item.result.isCorrect ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                )}
              </div>

              <div className="text-xs font-bold text-white truncate">
                {item.result.category}
              </div>

              <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                {Math.round(item.result.confidence * 100)}% Conf
              </div>

              <div className={`mt-2 text-[10px] px-1.5 py-0.5 rounded text-center font-medium ${
                item.result.isCorrect
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/40'
                  : 'bg-rose-950 text-rose-300 border border-rose-800/40'
              }`}>
                {item.result.isCorrect ? 'Retained' : 'Flipped'}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

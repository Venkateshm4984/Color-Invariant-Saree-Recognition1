import React, { useState } from 'react';
import {
  Code2,
  Copy,
  Check,
  FileCode,
  FolderTree,
  Terminal,
  FileText
} from 'lucide-react';

interface CodeFile {
  name: string;
  path: string;
  language: string;
  description: string;
  code: string;
}

export const CodeViewer: React.FC = () => {
  const [copied, setCopied] = useState<string | null>(null);

  const files: CodeFile[] = [
    {
      name: 'preprocessing.py',
      path: 'src/preprocessing.py',
      language: 'python',
      description: 'Luminance separation, CLAHE contrast limiting, and structural channel fusion.',
      code: `import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def apply_clahe(image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """Apply CLAHE to L channel in LAB color space to normalize lighting."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)

def extract_structural_channels(image: np.ndarray) -> np.ndarray:
    """Extracts 3-channel structural tensor: CLAHE + Sobel Gradient + Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.clip(cv2.magnitude(gx, gy) / 255.0 * 255.0, 0, 255).astype(np.uint8)
    
    lap = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_32F)) / 255.0 * 255.0, 0, 255).astype(np.uint8)
    return cv2.merge([clahe, mag, lap])`
    },
    {
      name: 'model.py',
      path: 'src/model.py',
      language: 'python',
      description: 'Transfer learning architecture with ResNet-50 and Grad-CAM hooks.',
      code: `import torch
import torch.nn as nn
from torchvision import models

class SareeClassifier(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=6, pretrained=True, dropout_rate=0.3):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = base.fc.in_features
        
        self.feature_extractor = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.target_layer = self.feature_extractor[-1] # For Grad-CAM
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        pooled = self.global_pool(feat)
        return self.classifier(pooled)`
    },
    {
      name: 'train.py',
      path: 'src/train.py',
      language: 'python',
      description: 'Training pipeline with EarlyStopping, CosineAnnealing, and checkpoints.',
      code: `import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset import create_dataloaders
from src.model import build_model, save_checkpoint

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total`
    },
    {
      name: 'color_robustness.py',
      path: 'src/color_robustness.py',
      language: 'python',
      description: 'Systematic chromatic perturbation suite and Color Consistency Score calculator.',
      code: `from src.augmentation import generate_color_variants
from src.preprocessing import preprocess_for_inference

def evaluate_color_robustness(model, test_loader, device):
    """
    Computes Color Consistency Score:
    C_score = correct_transformed / total_transformed
    """
    variant_keys = ["red_tint", "blue_tint", "green_tint", "yellow_tint", "low_saturation", "grayscale"]
    correct, total = 0, 0
    
    for raw_img, true_label in test_loader:
        variants = generate_color_variants(raw_img)
        for v_name in variant_keys:
            tensor = preprocess_for_inference(variants[v_name]).to(device)
            pred = model(tensor).argmax(dim=1).item()
            if pred == true_label:
                correct += 1
            total += 1
            
    c_score = correct / total if total > 0 else 0.0
    return c_score`
    },
    {
      name: 'explainability.py',
      path: 'src/explainability.py',
      language: 'python',
      description: 'Grad-CAM gradient attribution mapping for verifying motif vs color focus.',
      code: `import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(input_tensor)
        idx = class_idx if class_idx is not None else logits.argmax(1).item()
        logits[0, idx].backward()
        
        weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * self.activations[0], dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)`
    },
    {
      name: 'app.py',
      path: 'app.py',
      language: 'python',
      description: 'Production Streamlit application code with image upload & Grad-CAM visualizer.',
      code: `import streamlit as st
from PIL import Image
from src.augmentation import generate_color_variants
from src.explainability import overlay_cam_on_image

st.set_page_config(page_title="Color-Invariant Saree Recognition", page_icon="🥻")
st.title("🥻 Color-Invariant Saree Design Recognition")

uploaded = st.file_uploader("Upload Saree Fabric Image", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Saree")
    # Top-3 predictions and 6-way chromatic stress test matrix...`
    }
  ];

  const [activeFile, setActiveFile] = useState<CodeFile>(files[0]);

  const handleCopy = (code: string, fileName: string) => {
    navigator.clipboard.writeText(code);
    setCopied(fileName);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 mb-2">
              <Code2 className="w-3.5 h-3.5" />
              Source Code & Modular Pipeline Viewer
            </div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Python Codebase Inspection
            </h2>
            <p className="text-sm text-slate-400 mt-0.5">
              Review any core source file implementing preprocessing, ResNet-50 transfer learning, training loops, and Grad-CAM.
            </p>
          </div>

          <button
            onClick={() => handleCopy(activeFile.code, activeFile.name)}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold shadow-md transition-all self-start md:self-auto"
          >
            {copied === activeFile.name ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-300" />
                <span>Copied to Clipboard!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy {activeFile.name}</span>
              </>
            )}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* File Navigator Sidebar (4 cols) */}
        <div className="lg:col-span-4 space-y-2">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-2 mb-2">
            Repository Files
          </div>
          {files.map((file) => (
            <button
              key={file.path}
              onClick={() => setActiveFile(file)}
              className={`w-full text-left p-3 rounded-xl border transition-all ${
                activeFile.path === file.path
                  ? 'border-indigo-500 bg-indigo-500/10 text-white shadow-sm'
                  : 'border-slate-800/80 bg-slate-900/60 text-slate-300 hover:border-slate-700 hover:bg-slate-900'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold font-mono text-white flex items-center gap-2">
                  <FileCode className="w-4 h-4 text-indigo-400 shrink-0" />
                  {file.name}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-950 font-mono text-slate-400 border border-slate-800">
                  {file.language}
                </span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                {file.description}
              </div>
            </button>
          ))}
        </div>

        {/* Code Content Box (8 cols) */}
        <div className="lg:col-span-8 bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
          {/* Header */}
          <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-3 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2 font-mono text-slate-300">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <span>{activeFile.path}</span>
            </div>
            <span className="text-[11px] text-slate-400 font-sans">
              {activeFile.description}
            </span>
          </div>

          {/* Syntax Container */}
          <pre className="p-4 text-xs font-mono text-slate-200 overflow-x-auto leading-relaxed flex-1 max-h-[500px]">
            <code>{activeFile.code}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};

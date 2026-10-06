import React, { useState } from 'react';
import { StressTestLab } from './components/StressTestLab';
import { BenchmarkStudio } from './components/BenchmarkStudio';
import { ArchitectureExplorer } from './components/ArchitectureExplorer';
import { CodeViewer } from './components/CodeViewer';
import { SubmissionChecklist } from './components/SubmissionChecklist';
import {
  Sparkles,
  Layers,
  BarChart3,
  Workflow,
  Code2,
  FolderGit2,
  ShieldCheck,
  Palette,
  Github
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<
    'lab' | 'benchmarks' | 'architecture' | 'code' | 'submission'
  >('lab');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo & Title */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white font-bold">
                🥻
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-sm sm:text-base font-bold text-white tracking-tight leading-none">
                    Color-Invariant Saree Recognition
                  </h1>
                  <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    PyTorch ResNet-50 • Grad-CAM
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 font-medium hidden sm:block">
                  Pattern, motif & border recognition decoupled from fabric color dyes
                </p>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => setActiveTab('lab')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === 'lab'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-900/40'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Palette className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Stress Test</span> Lab
              </button>

              <button
                onClick={() => setActiveTab('benchmarks')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === 'benchmarks'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-900/40'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Benchmark</span> Metrics
              </button>

              <button
                onClick={() => setActiveTab('architecture')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === 'architecture'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-900/40'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Workflow className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Pipeline</span> XAI
              </button>

              <button
                onClick={() => setActiveTab('code')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === 'code'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-900/40'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <Code2 className="w-3.5 h-3.5" />
                Codebase
              </button>

              <button
                onClick={() => setActiveTab('submission')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === 'submission'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-900/40'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                }`}
              >
                <FolderGit2 className="w-3.5 h-3.5" />
                Submission
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'lab' && <StressTestLab />}
        {activeTab === 'benchmarks' && <BenchmarkStudio />}
        {activeTab === 'architecture' && <ArchitectureExplorer />}
        {activeTab === 'code' && <CodeViewer />}
        {activeTab === 'submission' && <SubmissionChecklist />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>AI Engineer Assessment Project: Color-Invariant Saree Design Recognition</span>
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span>PyTorch 2.2</span>
            <span>•</span>
            <span>Albumentations</span>
            <span>•</span>
            <span>Grad-CAM XAI</span>
            <span>•</span>
            <span>Streamlit</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

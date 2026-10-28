import React, { useState, useEffect } from 'react';
import { Camera, Cpu, Globe, Info, Zap, ChevronRight, Loader2, Download, RefreshCcw, AlertCircle } from 'lucide-react';

// When served from FastAPI, use relative paths (empty string means same origin)
// For development with separate React dev server, change this to "http://localhost:8000"
const API_BASE_URL = "";

const Homepage = () => {
  // --- BACKEND STATE MANAGEMENT ---
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null); // not_started, processing, completed, failed
  const [progress, setProgress] = useState(0); // 0 to 100
  const [error, setError] = useState(null);

  // --- TRIGGER RENDER JOB ---
  const handleStartRender = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/render`, { method: 'POST' }); //
      if (!response.ok) throw new Error("Could not connect to the render server.");

      const data = await response.json(); // Returns RenderJobResponse
      setJobId(data.id);
      setStatus(data.status);
      setProgress(data.progress);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // --- POLLING LOGIC ---
  useEffect(() => {
    let interval;
    if (jobId && (status === 'not_started' || status === 'processing')) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/render/${jobId}/status`); //
          const data = await res.json(); // Returns RenderJobStatusResponse

          setStatus(data.status);
          setProgress(data.progress);

          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(interval);
          }
        } catch (err) {
          console.error("Status check failed", err);
        }
      }, 2000); // Check every 2 seconds
    }
    return () => clearInterval(interval);
  }, [jobId, status]);

  return (
    <div className="w-full min-h-screen bg-slate-950 text-slate-100 overflow-x-hidden selection:bg-cyan-500/30">

      {/* --- NAVBAR --- */}
      <nav className="fixed top-0 left-0 right-0 w-full z-50 bg-slate-950/80 backdrop-blur-md border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-6 h-20 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
              <Cpu size={24} className="text-white" />
            </div>
            <span className="text-2xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-500">
              CAIS <span className="text-cyan-400 font-mono text-sm tracking-widest ml-1">// NERF</span>
            </span>
          </div>

          <div className="hidden md:flex items-center space-x-10 text-sm font-semibold tracking-wide">
            <a href="#home" className="text-slate-400 hover:text-cyan-400 transition-colors uppercase">Home</a>
            <a href="#about-nerf" className="text-slate-400 hover:text-cyan-400 transition-colors uppercase">Technology</a>
            <a href="#about-cais" className="text-slate-400 hover:text-cyan-400 transition-colors uppercase">CAIS Society</a>
            <button
              onClick={handleStartRender}
              disabled={loading || (jobId && status !== 'completed' && status !== 'failed')}
              className="bg-cyan-600 text-white px-6 py-2.5 rounded-full font-bold hover:bg-cyan-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-900/20"
            >
              {status === 'processing' ? 'Rendering...' : 'Start Rendering'}
            </button>
          </div>
        </div>
      </nav>

      {/* --- HERO SECTION --- */}
      <header id="home" className="relative pt-48 pb-32 px-6 flex flex-col items-center justify-center">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full max-w-7xl pointer-events-none">
          <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 blur-[120px] rounded-full"></div>
          <div className="absolute bottom-0 left-0 w-96 h-96 bg-blue-600/10 blur-[120px] rounded-full"></div>
        </div>

        <div className="max-w-5xl w-full text-center relative z-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900 border border-slate-800 text-cyan-400 text-xs font-bold mb-8">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            3D RECONSTRUCTION ENGINE
          </div>
          <h1 className="text-6xl md:text-8xl font-extrabold mb-8 tracking-tighter leading-tight">
            Turning 2D Images Into <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500">
              Neural Realities
            </span>
          </h1>

          {/* DYNAMIC JOB INTERFACE */}
          <div className="max-w-xl mx-auto mb-12">
            {!jobId ? (
              <div className="space-y-8">
                <p className="text-xl text-slate-400 leading-relaxed">
                  Harness the power of <strong>Neural Radiance Fields</strong> to generate stunning 360° visualizations from a trained model.
                </p>
                <button
                  onClick={handleStartRender}
                  className="group px-10 py-5 bg-cyan-600 hover:bg-cyan-500 rounded-2xl font-bold transition-all flex items-center justify-center gap-3 mx-auto shadow-2xl shadow-cyan-900/40"
                >
                  {loading ? <Loader2 className="animate-spin" /> : "Launch Render Job"}
                  <ChevronRight size={20} className="group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            ) : (
              <div className="p-8 bg-slate-900/50 border border-slate-800 rounded-3xl backdrop-blur-xl animate-in fade-in slide-in-from-bottom-4 duration-500">

                {/* Generating state */}
                {(status === 'not_started' || status === 'processing') && (
                  <div className="flex flex-col items-center justify-center py-8 space-y-4">
                    <Loader2 size={48} className="animate-spin text-cyan-400" />
                    <p className="text-xl font-semibold text-slate-300">Generating...</p>
                    <p className="text-sm text-slate-500">This may take a few minutes</p>
                  </div>
                )}

                {status === 'completed' && (
                  <div className="space-y-4">
                    {/* GIF Preview */}
                    <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
                      <img
                        src={`${API_BASE_URL}/render/${jobId}/download`}
                        alt="360° NeRF Render"
                        className="w-full h-auto max-h-64 object-contain mx-auto"
                      />
                      <div className="absolute top-2 right-2 px-2 py-1 bg-green-500/80 text-white text-xs font-bold rounded">
                        ✓ Complete
                      </div>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-4">
                      <a
                        href={`${API_BASE_URL}/render/${jobId}/download`}
                        download="360_view.gif"
                        className="flex-1 flex items-center justify-center gap-2 py-4 bg-white text-slate-950 rounded-xl font-bold hover:bg-cyan-400 transition-colors"
                      >
                        <Download size={20} /> Download GIF
                      </a>
                      <button
                        onClick={() => { setJobId(null); setStatus(null); setProgress(0); }}
                        className="px-6 py-4 bg-slate-800 rounded-xl font-bold hover:bg-slate-700 transition-colors"
                      >
                        <RefreshCcw size={20} />
                      </button>
                    </div>
                  </div>
                )}

                {status === 'failed' && (
                  <div className="text-red-400 flex items-center gap-2 justify-center">
                    <AlertCircle size={20} />
                    <span>Render Error. Check backend logs.</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* --- ABOUT NERF SECTION --- */}
      <section id="about-nerf" className="w-full py-32 bg-slate-900/30 border-y border-slate-900/80">
        <div className="max-w-7xl mx-auto px-6 grid lg:grid-cols-2 gap-20 items-center">
          <div className="relative">
            <div className="absolute -inset-4 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-3xl blur opacity-10"></div>
            <div className="relative bg-slate-950 border border-slate-800 p-8 rounded-3xl aspect-video flex items-center justify-center text-center">
              <div>
                <Camera size={48} className="mx-auto mb-4 text-cyan-500" />
                <h3 className="text-lg font-bold text-white mb-2 tracking-tight">NeRF Architecture</h3>
                <p className="text-slate-500 text-sm font-mono uppercase tracking-widest">
                  Neural synthesis via <br /> volumetric rendering
                </p>
              </div>
            </div>
          </div>
          <div className="space-y-8">
            <h2 className="text-4xl font-bold tracking-tight">What is a NeRF?</h2>
            <div className="h-1.5 w-16 bg-cyan-500 rounded-full"></div>
            <p className="text-lg text-slate-400 leading-relaxed">
              <strong>Neural Radiance Fields (NeRF)</strong> use deep neural networks to represent 3D scenes as continuous volumetric functions. Instead of pixels, this project renders views by querying a trained model for light and density at any point in space.
            </p>
            <div className="grid gap-4">
              <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800">
                <h4 className="font-bold text-cyan-400 mb-1">Volumetric Synthesis</h4>
                <p className="text-sm text-slate-500">Generates 60 frames of turntable animation from a trained `.pth` model.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* --- ABOUT CAIS SECTION --- */}
      <section id="about-cais" className="w-full py-32 px-6">
        <div className="max-w-4xl mx-auto text-center space-y-12">
          <div className="space-y-4">
            <h2 className="text-4xl font-bold">Center for Artificial Intelligent Systems</h2>
            <p className="text-cyan-500 font-mono text-sm tracking-[0.3em] uppercase">The CAIS Society</p>
          </div>

          <p className="text-xl text-slate-300 font-light leading-relaxed max-w-3xl mx-auto italic">
            "At CAIS, we bridge the gap between AI research and practical application. Our society serves as an incubator for Computer Vision and Machine Learning talent."
          </p>

          <div className="grid sm:grid-cols-3 gap-8 pt-10">
            <div className="p-8 bg-slate-900/40 rounded-3xl border border-slate-800 hover:border-cyan-500/50 transition-all group text-center">
              <Globe size={32} className="mx-auto mb-6 text-slate-500 group-hover:text-cyan-400" />
              <h3 className="font-bold mb-3">Global Vision</h3>
              <p className="text-sm text-slate-500">Pushing boundaries in spatial computing and 3D AI.</p>
            </div>
            <div className="p-8 bg-slate-900/40 rounded-3xl border border-slate-800 hover:border-cyan-500/50 transition-all group text-center">
              <Cpu size={32} className="mx-auto mb-6 text-slate-500 group-hover:text-cyan-400" />
              <h3 className="font-bold mb-3">Technical Depth</h3>
              <p className="text-sm text-slate-500">Deep diving into PyTorch and FastAPI architectures.</p>
            </div>
            <div className="p-8 bg-slate-900/40 rounded-3xl border border-slate-800 hover:border-cyan-500/50 transition-all group text-center">
              <Info size={32} className="mx-auto mb-6 text-slate-500 group-hover:text-cyan-400" />
              <h3 className="font-bold mb-3">Community</h3>
              <p className="text-sm text-slate-500">Fostering an inclusive environment for AI research.</p>
            </div>
          </div>
        </div>
      </section>

      {/* --- FOOTER --- */}
      <footer className="w-full py-20 border-t border-slate-900 bg-slate-950">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="text-slate-500 text-sm">
            © 2026 CAIS Society. Powered by NeRF Technology.
          </div>
          <div className="flex gap-8 text-slate-700 text-xs font-mono uppercase">
            <span>FastAPI</span>
            <span>PyTorch</span>
            <span>PostgreSQL</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Homepage;
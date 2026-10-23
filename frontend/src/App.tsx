import { useState, useEffect, useCallback } from 'react'
import { useCamera } from './hooks/useCamera'
import { useInference } from './hooks/useInference'
import { CameraView } from './components/CameraView'
import { Controls } from './components/Controls'
import { StatsDisplay } from './components/StatsDisplay'
import { DemoMode } from './components/DemoMode'
import type { CurveInfo } from './utils/drawing'
import './App.css'

export function App() {
  const { videoRef, state: cameraState, startCamera, stopCamera, toggleFacing } = useCamera()
  const { state: infState, results, loadModels, startInference, stopInference } = useInference()

  const [isDemoMode, setIsDemoMode] = useState(false)
  const [showRoadOverlay, setShowRoadOverlay] = useState(true)
  const [showObstacles, setShowObstacles] = useState(true)

  // Demo stats state
  const [demoStats, setDemoStats] = useState<{
    fps: number
    segLatency: number
    detLatency: number
    confidence: number
    curveInfo: CurveInfo | null
  }>({
    fps: 0,
    segLatency: 0,
    detLatency: 0,
    confidence: 0,
    curveInfo: null,
  })

  // Attempt to load ONNX models in browser on initial mount
  useEffect(() => {
    loadModels()
  }, [loadModels])

  // Coordinate inference loop with camera stream
  useEffect(() => {
    if (cameraState.isActive && videoRef.current && !isDemoMode) {
      startInference(videoRef.current)
    } else {
      stopInference()
    }
  }, [cameraState.isActive, isDemoMode, startInference, stopInference, videoRef])

  const handleToggleCamera = useCallback(() => {
    if (cameraState.isActive) {
      stopCamera()
    } else {
      setIsDemoMode(false)
      startCamera()
    }
  }, [cameraState.isActive, startCamera, stopCamera])

  const handleToggleDemoMode = useCallback(() => {
    if (!isDemoMode) {
      stopCamera()
      setIsDemoMode(true)
    } else {
      setIsDemoMode(false)
    }
  }, [isDemoMode, stopCamera])

  const [liveCurve, setLiveCurve] = useState<CurveInfo | null>(null)

  const activeFps = isDemoMode ? demoStats.fps : infState.stats.fps
  const activeSegLatency = isDemoMode ? demoStats.segLatency : infState.stats.segLatency
  const activeDetLatency = isDemoMode ? demoStats.detLatency : infState.stats.detLatency
  const activeConfidence = isDemoMode ? demoStats.confidence : infState.stats.segConfidence
  const activeCurve = isDemoMode ? demoStats.curveInfo : liveCurve

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/50 backdrop-blur sticky top-0 z-50 px-4 py-3 sm:px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-400 flex items-center justify-center font-black text-slate-950 text-lg shadow-lg shadow-cyan-500/20">
              RT
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-base sm:text-lg tracking-tight text-white">
                  RoadTrace <span className="text-cyan-400">AI</span>
                </h1>
                <span className="text-[10px] font-semibold bg-cyan-950/80 text-cyan-300 border border-cyan-800/50 px-1.5 py-0.5 rounded">
                  v1.0-Edge
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Real-Time Road Segmentation, Trajectory Estimation & Obstacle Detection
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
                cameraState.isActive
                  ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                  : isDemoMode
                  ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  cameraState.isActive
                    ? 'bg-emerald-400 animate-pulse'
                    : isDemoMode
                    ? 'bg-amber-400'
                    : 'bg-slate-500'
                }`}
              />
              {cameraState.isActive ? 'Live Stream' : isDemoMode ? 'Demo Feed' : 'Idle'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-4 sm:p-6 space-y-4">
        {/* Error Banners */}
        {cameraState.error && (
          <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-3.5 text-rose-300 text-sm flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <span>⚠️</span>
              <span>{cameraState.error}</span>
            </div>
            <button
              onClick={handleToggleDemoMode}
              className="text-xs font-semibold underline hover:text-rose-200"
            >
              Switch to Demo Mode
            </button>
          </div>
        )}

        {/* Video / Simulation Viewport */}
        {isDemoMode ? (
          <DemoMode
            showRoadOverlay={showRoadOverlay}
            showObstacles={showObstacles}
            onStatsUpdate={setDemoStats}
          />
        ) : (
          <CameraView
            videoRef={videoRef}
            mask={results.mask}
            boxes={results.boxes}
            showRoadOverlay={showRoadOverlay}
            showObstacles={showObstacles}
            isModelLoaded={infState.segModelLoaded || infState.detModelLoaded}
            isActive={cameraState.isActive}
            onCurveUpdate={setLiveCurve}
          />
        )}

        {/* Real-Time Stats Grid */}
        <StatsDisplay
          fps={activeFps}
          segLatency={activeSegLatency}
          detLatency={activeDetLatency}
          confidence={activeConfidence}
          curveInfo={activeCurve}
          isModelLoaded={infState.segModelLoaded || isDemoMode}
        />

        {/* Camera & Overlay Controls */}
        <Controls
          isActive={cameraState.isActive}
          isDemoMode={isDemoMode}
          isLoading={cameraState.isLoading}
          facing={cameraState.facing}
          showRoadOverlay={showRoadOverlay}
          showObstacles={showObstacles}
          onToggleCamera={handleToggleCamera}
          onToggleFacing={toggleFacing}
          onToggleDemoMode={handleToggleDemoMode}
          onToggleRoadOverlay={() => setShowRoadOverlay((v) => !v)}
          onToggleObstacles={() => setShowObstacles((v) => !v)}
        />

        {/* System & Architecture Info */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-slate-800/60 text-xs text-slate-400">
          <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-4">
            <h4 className="font-semibold text-slate-200 mb-1.5 flex items-center gap-1.5">
              <span>🛣</span> Road Segmentation
            </h4>
            <p>
              Lightweight MobileNetV2 U-Net trained on BDD100K 3-class drivable masks (Ego lane, Alt lane, Background) at 256×256.
            </p>
          </div>

          <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-4">
            <h4 className="font-semibold text-slate-200 mb-1.5 flex items-center gap-1.5">
              <span>🧭</span> Centerline & Curve
            </h4>
            <p>
              Extracts dynamic drivable centerline using polynomial curve fitting, estimating road curvature direction in real-time.
            </p>
          </div>

          <div className="bg-slate-900/40 border border-slate-800/60 rounded-xl p-4">
            <h4 className="font-semibold text-slate-200 mb-1.5 flex items-center gap-1.5">
              <span>⚡</span> Edge Client Inference
            </h4>
            <p>
              Executed locally inside your browser via ONNX Runtime WebAssembly (WASM). Zero cloud video transmission or server fees.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 py-4 px-4 text-center text-xs text-slate-500">
        RoadTrace AI — Open-source edge perception prototype. Safe driving assistant and road trajectory visualization.
      </footer>
    </div>
  )
}

export default App

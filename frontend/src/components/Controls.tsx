
interface ControlsProps {
  isActive: boolean
  isDemoMode: boolean
  isLoading: boolean
  facing: 'user' | 'environment'
  showRoadOverlay: boolean
  showObstacles: boolean
  onToggleCamera: () => void
  onToggleFacing: () => void
  onToggleDemoMode: () => void
  onToggleRoadOverlay: () => void
  onToggleObstacles: () => void
}

export const Controls: React.FC<ControlsProps> = ({
  isActive,
  isDemoMode,
  isLoading,
  facing,
  showRoadOverlay,
  showObstacles,
  onToggleCamera,
  onToggleFacing,
  onToggleDemoMode,
  onToggleRoadOverlay,
  onToggleObstacles,
}) => {
  return (
    <div className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-xl p-4 flex flex-wrap gap-3 items-center justify-between">
      <div className="flex flex-wrap gap-2.5 items-center">
        {/* Main Camera Action */}
        <button
          onClick={onToggleCamera}
          disabled={isLoading || isDemoMode}
          className={`px-4 py-2.5 rounded-lg font-medium text-sm flex items-center gap-2 transition-all ${
            isActive
              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30 hover:bg-rose-500/30'
              : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400 font-semibold shadow-lg shadow-cyan-500/20'
          } ${isDemoMode ? 'opacity-40 cursor-not-allowed' : ''}`}
        >
          <span className={`w-2.5 h-2.5 rounded-full ${isActive ? 'bg-rose-400 animate-pulse' : 'bg-slate-900'}`} />
          {isLoading ? 'Starting...' : isActive ? 'Stop Camera' : 'Start Camera'}
        </button>

        {/* Camera Flip Button */}
        <button
          onClick={onToggleFacing}
          disabled={!isActive || isDemoMode}
          className="px-3 py-2.5 rounded-lg text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-750 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          title="Switch Front/Rear Camera"
        >
          🔄 {facing === 'environment' ? 'Rear Cam' : 'Front Cam'}
        </button>

        {/* Demo Mode Toggle */}
        <button
          onClick={onToggleDemoMode}
          className={`px-3.5 py-2.5 rounded-lg text-xs font-medium border transition-colors flex items-center gap-1.5 ${
            isDemoMode
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
              : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
          }`}
        >
          <span>🎬</span>
          {isDemoMode ? 'Demo Mode Active' : 'Enable Demo Mode'}
        </button>
      </div>

      {/* Visual Layer Toggles */}
      <div className="flex items-center gap-2">
        <button
          onClick={onToggleRoadOverlay}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
            showRoadOverlay
              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              : 'bg-slate-800/60 text-slate-500 border border-slate-800 line-through'
          }`}
        >
          🛣 Road Path
        </button>

        <button
          onClick={onToggleObstacles}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
            showObstacles
              ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
              : 'bg-slate-800/60 text-slate-500 border border-slate-800 line-through'
          }`}
        >
          🚗 Obstacles
        </button>
      </div>
    </div>
  )
}

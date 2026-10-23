import type { CurveInfo } from '../utils/drawing'

interface StatsDisplayProps {
  fps: number
  segLatency: number
  detLatency: number
  confidence: number
  curveInfo: CurveInfo | null
  isModelLoaded: boolean
}

export const StatsDisplay = ({
  fps,
  segLatency,
  detLatency,
  confidence,
  curveInfo,
  isModelLoaded,
}: StatsDisplayProps) => {
  const isNoRoad = !curveInfo || curveInfo.status === 'no_road'
  const isUncertain = curveInfo?.status === 'uncertain'
  const isValidRoad = curveInfo?.status === 'valid'

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {/* FPS Card */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase">Processing Rate</span>
          <span
            className={`w-1.5 h-1.5 rounded-full ${isModelLoaded ? 'bg-cyan-400' : 'bg-amber-400'}`}
            title={isModelLoaded ? 'Model Active' : 'Model Loading'}
          />
        </div>
        <div className="flex items-baseline gap-1.5 mt-1">
          <span className="text-2xl font-bold font-mono text-cyan-400">{fps > 0 ? fps : '--'}</span>
          <span className="text-xs text-slate-500 font-mono">FPS</span>
        </div>
      </div>

      {/* Latency Card */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase">Inference Latency</span>
        <div className="flex items-baseline gap-2 mt-1">
          <span className="text-2xl font-bold font-mono text-emerald-400">
            {segLatency + detLatency > 0 ? `${segLatency + detLatency}ms` : '--'}
          </span>
          <span className="text-[10px] text-slate-500">
            ({segLatency}s / {detLatency}d)
          </span>
        </div>
      </div>

      {/* Confidence Card */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase">Segmentation Conf.</span>
        <div className="flex items-baseline gap-1.5 mt-1">
          <span className="text-2xl font-bold font-mono text-amber-400">
            {confidence > 0 ? `${Math.round(confidence * 100)}%` : '--'}
          </span>
          <span className="text-xs text-slate-500">drivable</span>
        </div>
      </div>

      {/* Curve Direction Card */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
        <span className="text-[11px] font-medium tracking-wider text-slate-400 uppercase">Road Trajectory</span>
        <div className="flex items-center gap-2 mt-1">
          {isValidRoad ? (
            <>
              <span
                className={`px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider ${
                  curveInfo.direction === 'left'
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    : curveInfo.direction === 'right'
                    ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                    : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                }`}
              >
                {curveInfo.direction === 'left'
                  ? '↰ Left'
                  : curveInfo.direction === 'right'
                  ? '↱ Right'
                  : '↑ Straight'}
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                {curveInfo.angle ? `${Math.round(curveInfo.angle)}°` : ''}
              </span>
            </>
          ) : isUncertain ? (
            <span className="px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">
              ⚡ Path Uncertain
            </span>
          ) : isNoRoad ? (
            <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-800/80 text-slate-400 border border-slate-700/50">
              No Road Detected
            </span>
          ) : (
            <span className="text-sm font-mono text-slate-500">Scanning...</span>
          )}
        </div>
      </div>
    </div>
  )
}

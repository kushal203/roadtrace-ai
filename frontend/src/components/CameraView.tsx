import { useRef, useEffect } from 'react'
import {
  drawSegmentationMask,
  drawCenterline,
  drawBoundingBoxes,
  clearCanvas,
  drawNoModelOverlay,
} from '../utils/drawing'
import type { BoundingBox, CurveInfo } from '../utils/drawing'

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>
  mask: Uint8Array | null
  boxes: BoundingBox[]
  showRoadOverlay: boolean
  showObstacles: boolean
  isModelLoaded: boolean
  isActive: boolean
  onCurveUpdate?: (curve: CurveInfo) => void
}

export const CameraView: React.FC<CameraViewProps> = ({
  videoRef,
  mask,
  boxes,
  showRoadOverlay,
  showObstacles,
  isModelLoaded,
  isActive,
  onCurveUpdate,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    clearCanvas(ctx)

    if (!isActive) {
      if (onCurveUpdate) {
        onCurveUpdate({
          direction: 'none',
          angle: 0,
          confidence: 0,
          status: 'no_road',
          message: 'Camera inactive',
        })
      }
      return
    }

    // If mask is available and overlay is toggled on
    if (showRoadOverlay && mask) {
      drawSegmentationMask(ctx, mask, 256, 256, canvas.width, canvas.height)
      const curveInfo = drawCenterline(ctx, mask, 256, 256, canvas.width, canvas.height)
      if (onCurveUpdate) {
        onCurveUpdate(curveInfo)
      }
    } else if (onCurveUpdate) {
      onCurveUpdate({
        direction: 'none',
        angle: 0,
        confidence: 0,
        status: 'no_road',
        message: 'No drivable road detected',
      })
    }

    // If obstacle boxes are available and toggled on
    if (showObstacles && boxes.length > 0) {
      drawBoundingBoxes(ctx, boxes, canvas.width, canvas.height)
    }

    // Watermark if model not loaded
    if (!isModelLoaded) {
      drawNoModelOverlay(ctx, canvas.width, canvas.height, 'AI Models Loading / Running in Web Demo')
    }
  }, [mask, boxes, showRoadOverlay, showObstacles, isModelLoaded, isActive, onCurveUpdate])

  return (
    <div className="relative w-full aspect-video bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl flex items-center justify-center">
      <video
        ref={videoRef as any}
        playsInline
        muted
        autoPlay
        className="absolute inset-0 w-full h-full object-cover"
      />
      <canvas
        ref={canvasRef}
        width={1280}
        height={720}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
      {!isActive && (
        <div className="z-10 text-center px-4">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-cyan-950/60 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-slate-200">Camera Inactive</h3>
          <p className="text-sm text-slate-400 mt-1 max-w-sm mx-auto">
            Click <strong>Start Camera</strong> to stream live roadway video or switch to <strong>Demo Mode</strong> to test with synthetic clips.
          </p>
        </div>
      )}
    </div>
  )
}

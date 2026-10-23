import { useRef, useEffect } from 'react'
import { generateDemoMask } from '../utils/roadSegmentation'
import { generateDemoBoxes } from '../utils/obstacleDetection'
import {
  drawSegmentationMask,
  drawCenterline,
  drawBoundingBoxes,
} from '../utils/drawing'
import type { CurveInfo } from '../utils/drawing'

interface DemoModeProps {
  showRoadOverlay: boolean
  showObstacles: boolean
  onStatsUpdate: (stats: {
    fps: number
    segLatency: number
    detLatency: number
    confidence: number
    curveInfo: CurveInfo | null
  }) => void
}

export const DemoMode: React.FC<DemoModeProps> = ({
  showRoadOverlay,
  showObstacles,
  onStatsUpdate,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let startTime = performance.now()
    let frameCount = 0
    let lastFpsTime = performance.now()
    let currentFps = 30

    const render = (time: number) => {
      const elapsedSeconds = (time - startTime) / 1000

      // Update FPS counter
      frameCount++
      if (time - lastFpsTime >= 1000) {
        currentFps = Math.round((frameCount * 1000) / (time - lastFpsTime))
        frameCount = 0
        lastFpsTime = time
      }

      // 1. Draw base synthetic roadway background
      ctx.fillStyle = '#0f172a' // Sky
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      const horizonY = canvas.height * 0.45
      ctx.fillStyle = '#1e293b' // Ground
      ctx.fillRect(0, horizonY, canvas.width, canvas.height - horizonY)

      // Perspective road
      const curveOffset = Math.sin(elapsedSeconds * 0.8) * (canvas.width * 0.15)
      const vpX = canvas.width / 2 + curveOffset
      const vpY = horizonY

      // Draw asphalt
      ctx.beginPath()
      ctx.moveTo(vpX - 15, vpY)
      ctx.lineTo(vpX + 15, vpY)
      ctx.lineTo(canvas.width * 0.85, canvas.height)
      ctx.lineTo(canvas.width * 0.15, canvas.height)
      ctx.closePath()
      ctx.fillStyle = '#334155'
      ctx.fill()

      // Animated lane markings
      ctx.strokeStyle = '#e2e8f0'
      ctx.lineWidth = 4
      ctx.setLineDash([20, 25])
      ctx.lineDashOffset = -time * 0.15

      ctx.beginPath()
      ctx.moveTo(vpX, vpY)
      ctx.lineTo(canvas.width * 0.5, canvas.height)
      ctx.stroke()
      ctx.setLineDash([])

      // 2. Compute demo mask & boxes
      const mask = generateDemoMask(256, 256, elapsedSeconds)
      const boxes = generateDemoBoxes(elapsedSeconds)

      let curveInfo: CurveInfo | null = null

      // 3. Render Road Overlay
      if (showRoadOverlay) {
        drawSegmentationMask(ctx, mask, 256, 256, canvas.width, canvas.height)
        curveInfo = drawCenterline(ctx, mask, 256, 256, canvas.width, canvas.height)
      }

      // 4. Render Obstacles
      if (showObstacles) {
        drawBoundingBoxes(ctx, boxes, canvas.width, canvas.height)
      }

      // 5. Push stats to parent
      onStatsUpdate({
        fps: currentFps,
        segLatency: 14 + Math.round(Math.sin(time) * 2),
        detLatency: 22 + Math.round(Math.cos(time) * 3),
        confidence: 0.94 + Math.sin(time * 0.5) * 0.03,
        curveInfo,
      })

      animRef.current = requestAnimationFrame(render)
    }

    animRef.current = requestAnimationFrame(render)

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [showRoadOverlay, showObstacles, onStatsUpdate])

  return (
    <div className="relative w-full aspect-video bg-slate-950 rounded-2xl overflow-hidden border border-amber-500/30 shadow-2xl">
      <canvas
        ref={canvasRef}
        width={1280}
        height={720}
        className="w-full h-full object-cover"
      />
      <div className="absolute top-3 left-3 bg-amber-500/20 backdrop-blur border border-amber-500/40 text-amber-300 px-2.5 py-1 rounded-md text-xs font-semibold flex items-center gap-1.5 shadow-lg">
        <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
        SYNTHETIC ROAD DEMO ACTIVE
      </div>
    </div>
  )
}

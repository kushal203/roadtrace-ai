// Canvas drawing utilities for road segmentation overlays

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  label: string
  confidence: number
  classId: number
}

export type RoadStatus = 'valid' | 'uncertain' | 'no_road'

export interface CurveInfo {
  direction: 'left' | 'right' | 'straight' | 'none'
  angle: number // degrees
  confidence: number
  status: RoadStatus
  message: string
}

// Color palette for road classes
const ROAD_COLORS: Record<number, [number, number, number, number]> = {
  0: [0, 0, 0, 0],           // background - transparent
  1: [0, 220, 130, 95],      // direct drivable - vibrant emerald green
  2: [0, 160, 255, 60],      // alternative drivable - subtle cyan-blue
}

// Obstacle class colors
const OBSTACLE_COLORS: string[] = [
  '#f43f5e', '#fb923c', '#eab308', '#22c55e', '#06b6d4',
  '#3b82f6', '#8b5cf6', '#d946ef', '#ec4899', '#f97316',
]

// Reusable offscreen canvas for high-performance mask rendering (256x256 -> canvas GPU scale)
let offscreenCanvas: HTMLCanvasElement | null = null
let offscreenCtx: CanvasRenderingContext2D | null = null
let offscreenImageData: ImageData | null = null

// Centerline history for EMA temporal smoothing
let smoothedCenterPoints: [number, number][] = []

/**
 * High-performance road segmentation mask overlay using hardware GPU scaling
 * Replaces the 921,600 iteration JS loop with 65,536 iterations + GPU drawImage
 */
export function drawSegmentationMask(
  ctx: CanvasRenderingContext2D,
  mask: Uint8Array,
  maskWidth: number,
  maskHeight: number,
  canvasWidth: number,
  canvasHeight: number,
): void {
  if (!offscreenCanvas) {
    offscreenCanvas = document.createElement('canvas')
    offscreenCanvas.width = maskWidth
    offscreenCanvas.height = maskHeight
    offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true })
    if (offscreenCtx) {
      offscreenImageData = offscreenCtx.createImageData(maskWidth, maskHeight)
    }
  }

  if (!offscreenCtx || !offscreenImageData) return

  const data = offscreenImageData.data
  const totalPixels = maskWidth * maskHeight

  for (let i = 0; i < totalPixels; i++) {
    const classId = mask[i]
    const color = ROAD_COLORS[classId] || ROAD_COLORS[0]
    const idx = i * 4
    data[idx] = color[0]
    data[idx + 1] = color[1]
    data[idx + 2] = color[2]
    data[idx + 3] = color[3]
  }

  offscreenCtx.putImageData(offscreenImageData, 0, 0)

  // Hardware bilinear scaling on GPU
  ctx.save()
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'medium'
  ctx.drawImage(offscreenCanvas, 0, 0, canvasWidth, canvasHeight)
  ctx.restore()
}

/**
 * Validates road mask and extracts smoothed centerline.
 * Implements strict gating to reject stairs, indoor floors, and noise.
 */
export function drawCenterline(
  ctx: CanvasRenderingContext2D,
  mask: Uint8Array,
  maskWidth: number,
  maskHeight: number,
  canvasWidth: number,
  canvasHeight: number,
): CurveInfo {
  const totalPixels = maskWidth * maskHeight

  // 1. Coverage Analysis
  let roadPixelsTotal = 0
  let roadPixelsBottom = 0
  const bottomThirdStart = Math.floor(maskHeight * 0.65)

  for (let y = 0; y < maskHeight; y++) {
    for (let x = 0; x < maskWidth; x++) {
      const cls = mask[y * maskWidth + x]
      if (cls === 1 || cls === 2) {
        roadPixelsTotal++
        if (y >= bottomThirdStart) {
          roadPixelsBottom++
        }
      }
    }
  }

  const bottomArea = maskWidth * (maskHeight - bottomThirdStart)
  const totalFraction = roadPixelsTotal / totalPixels
  const bottomFraction = roadPixelsBottom / bottomArea

  // Strict Road Gating:
  // - Minimum total road coverage (at least 3.5% of total frame)
  // - Road must be anchored at the bottom (at least 6.0% of bottom-third)
  if (totalFraction < 0.035 || bottomFraction < 0.060) {
    smoothedCenterPoints = [] // Flush temporal smoothing immediately
    drawStatusBadge(ctx, 'No drivable road detected', 'no_road', canvasWidth, canvasHeight)
    return {
      direction: 'none',
      angle: 0,
      confidence: 0,
      status: 'no_road',
      message: 'No drivable road detected',
    }
  }

  // 2. Scan row-by-row from bottom to top to extract coherent road width & center
  const rawCenterPoints: [number, number][] = []
  const step = Math.max(1, Math.floor(maskHeight / 40))
  let consecutiveEmptyRows = 0

  for (let y = maskHeight - 1; y >= Math.floor(maskHeight * 0.25); y -= step) {
    let leftEdge = -1
    let rightEdge = -1
    let rowPixelCount = 0

    for (let x = 0; x < maskWidth; x++) {
      const cls = mask[y * maskWidth + x]
      if (cls === 1 || cls === 2) {
        if (leftEdge === -1) leftEdge = x
        rightEdge = x
        rowPixelCount++
      }
    }

    // Coherent width check: require at least 12 contiguous-like road pixels
    if (leftEdge !== -1 && rightEdge !== -1 && rowPixelCount >= 12) {
      const span = rightEdge - leftEdge
      // Reject rows where points are just isolated noise dots at opposite screen edges
      if (rowPixelCount / span >= 0.25) {
        const centerX = (leftEdge + rightEdge) / 2
        rawCenterPoints.push([centerX, y])
        consecutiveEmptyRows = 0
        continue
      }
    }

    consecutiveEmptyRows++
    if (consecutiveEmptyRows >= 4 && rawCenterPoints.length >= 5) {
      // Reached horizon or top of road
      break
    }
  }

  // Need at least 5 coherent height samples to trace a road path
  if (rawCenterPoints.length < 5) {
    smoothedCenterPoints = []
    drawStatusBadge(ctx, 'Road detected — path uncertain', 'uncertain', canvasWidth, canvasHeight)
    return {
      direction: 'none',
      angle: 0,
      confidence: 0.35,
      status: 'uncertain',
      message: 'Road detected — path uncertain',
    }
  }

  // 3. Temporal EMA Smoothing on Centerline Points
  const alpha = 0.4 // 40% new, 60% historical
  if (smoothedCenterPoints.length === rawCenterPoints.length) {
    for (let i = 0; i < rawCenterPoints.length; i++) {
      smoothedCenterPoints[i] = [
        rawCenterPoints[i][0] * alpha + smoothedCenterPoints[i][0] * (1 - alpha),
        rawCenterPoints[i][1],
      ]
    }
  } else {
    smoothedCenterPoints = rawCenterPoints.map(([x, y]) => [x, y])
  }

  // 4. Scale to Canvas Coordinates & Draw Smooth Bezier Path
  const scaleX = canvasWidth / maskWidth
  const scaleY = canvasHeight / maskHeight
  const canvasPoints = smoothedCenterPoints.map(
    ([x, y]) => [x * scaleX, y * scaleY] as [number, number],
  )

  // Draw Path Glow
  ctx.save()
  ctx.shadowColor = '#06b6d4'
  ctx.shadowBlur = 10
  ctx.beginPath()
  ctx.moveTo(canvasPoints[0][0], canvasPoints[0][1])

  for (let i = 1; i < canvasPoints.length - 1; i++) {
    const xc = (canvasPoints[i][0] + canvasPoints[i + 1][0]) / 2
    const yc = (canvasPoints[i][1] + canvasPoints[i + 1][1]) / 2
    ctx.quadraticCurveTo(canvasPoints[i][0], canvasPoints[i][1], xc, yc)
  }

  const lastPt = canvasPoints[canvasPoints.length - 1]
  ctx.lineTo(lastPt[0], lastPt[1])

  ctx.strokeStyle = '#22d3ee'
  ctx.lineWidth = 4
  ctx.setLineDash([12, 6])
  ctx.stroke()
  ctx.setLineDash([])
  ctx.restore()

  // 5. Estimate Curvature
  const curveInfo = estimateCurveFromPoints(smoothedCenterPoints, maskWidth)

  // 6. Draw Direction Indicator
  drawCurveArrow(ctx, curveInfo, canvasWidth, canvasHeight)

  return curveInfo
}

/**
 * Estimates trajectory direction from centerline points
 */
function estimateCurveFromPoints(
  points: [number, number][],
  maskWidth: number,
): CurveInfo {
  if (points.length < 4) {
    return {
      direction: 'straight',
      angle: 0,
      confidence: 0.5,
      status: 'valid',
      message: 'Straight',
    }
  }

  const quarter = Math.max(1, Math.floor(points.length / 4))
  const bottomPoints = points.slice(0, quarter)
  const topPoints = points.slice(-quarter)

  const bottomAvgX = bottomPoints.reduce((acc, p) => acc + p[0], 0) / bottomPoints.length
  const topAvgX = topPoints.reduce((acc, p) => acc + p[0], 0) / topPoints.length

  const lateralDelta = topAvgX - bottomAvgX
  const normalizedDelta = lateralDelta / maskWidth
  const angle = Math.atan2(lateralDelta, maskWidth * 0.5) * (180 / Math.PI)

  const threshold = 0.04
  let direction: 'left' | 'right' | 'straight' = 'straight'
  let label = 'Straight'

  if (normalizedDelta < -threshold) {
    direction = 'left'
    label = 'Curve Left'
  } else if (normalizedDelta > threshold) {
    direction = 'right'
    label = 'Curve Right'
  }

  const confidence = Math.min(0.98, Math.max(0.65, Math.abs(normalizedDelta) * 4 + 0.6))

  return {
    direction,
    angle: Math.abs(angle),
    confidence,
    status: 'valid',
    message: label,
  }
}

/**
 * Draws HUD status banner when no road or uncertain
 */
function drawStatusBadge(
  ctx: CanvasRenderingContext2D,
  message: string,
  status: RoadStatus,
  canvasWidth: number,
  canvasHeight: number,
): void {
  const badgeWidth = 260
  const badgeHeight = 36
  const x = (canvasWidth - badgeWidth) / 2
  const y = canvasHeight * 0.12

  ctx.save()
  ctx.fillStyle = status === 'no_road' ? 'rgba(15, 23, 42, 0.85)' : 'rgba(15, 23, 42, 0.85)'
  ctx.strokeStyle = status === 'no_road' ? 'rgba(239, 68, 68, 0.5)' : 'rgba(245, 158, 11, 0.5)'
  ctx.lineWidth = 1.5

  ctx.beginPath()
  ctx.roundRect(x, y, badgeWidth, badgeHeight, 8)
  ctx.fill()
  ctx.stroke()

  ctx.font = '600 13px Inter, sans-serif'
  ctx.fillStyle = status === 'no_road' ? '#f87171' : '#fbbf24'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  const icon = status === 'no_road' ? '⚠️ ' : '⚡ '
  ctx.fillText(icon + message, canvasWidth / 2, y + badgeHeight / 2)
  ctx.restore()
}

/**
 * Draw curve direction arrow HUD
 */
function drawCurveArrow(
  ctx: CanvasRenderingContext2D,
  curve: CurveInfo,
  canvasWidth: number,
  canvasHeight: number,
): void {
  const centerX = canvasWidth / 2
  const arrowY = canvasHeight * 0.14
  const arrowSize = 26

  ctx.save()
  ctx.translate(centerX, arrowY)

  if (curve.direction === 'left') {
    ctx.rotate(-Math.PI / 6)
  } else if (curve.direction === 'right') {
    ctx.rotate(Math.PI / 6)
  }

  // Arrow shaft
  ctx.beginPath()
  ctx.moveTo(0, arrowSize)
  ctx.lineTo(0, -arrowSize)
  ctx.strokeStyle = '#22d3ee'
  ctx.lineWidth = 3.5
  ctx.lineCap = 'round'
  ctx.stroke()

  // Arrow head
  ctx.beginPath()
  ctx.moveTo(-arrowSize * 0.45, -arrowSize * 0.4)
  ctx.lineTo(0, -arrowSize)
  ctx.lineTo(arrowSize * 0.45, -arrowSize * 0.4)
  ctx.strokeStyle = '#22d3ee'
  ctx.lineWidth = 3.5
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.stroke()

  ctx.restore()

  // Curvature Label
  ctx.save()
  ctx.font = '700 13px Inter, sans-serif'
  ctx.fillStyle = '#22d3ee'
  ctx.textAlign = 'center'
  const label =
    curve.direction === 'straight'
      ? '↑ STRAIGHT'
      : curve.direction === 'left'
        ? `← LEFT (${Math.round(curve.angle)}°)`
        : `RIGHT (${Math.round(curve.angle)}°) →`
  ctx.fillText(label, centerX, arrowY + arrowSize + 22)
  ctx.restore()
}

/**
 * Draw obstacle bounding boxes on the canvas
 */
export function drawBoundingBoxes(
  ctx: CanvasRenderingContext2D,
  boxes: BoundingBox[],
  canvasWidth: number,
  canvasHeight: number,
): void {
  for (const box of boxes) {
    const color = OBSTACLE_COLORS[box.classId % OBSTACLE_COLORS.length]

    const x = box.x * canvasWidth
    const y = box.y * canvasHeight
    const w = box.width * canvasWidth
    const h = box.height * canvasHeight

    ctx.save()
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, w, h)

    ctx.fillStyle = color + '22'
    ctx.fillRect(x, y, w, h)

    const label = `${box.label} ${(box.confidence * 100).toFixed(0)}%`
    ctx.font = '600 11px Inter, sans-serif'
    const textWidth = ctx.measureText(label).width
    const labelHeight = 18
    const labelY = y > labelHeight ? y - labelHeight : y + h

    ctx.fillStyle = color
    ctx.fillRect(x, labelY, textWidth + 8, labelHeight)

    ctx.fillStyle = '#ffffff'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText(label, x + 4, labelY + labelHeight / 2)
    ctx.restore()
  }
}

/**
 * Clear the overlay canvas
 */
export function clearCanvas(ctx: CanvasRenderingContext2D): void {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
}

/**
 * Draw a "no model" watermark on the canvas
 */
export function drawNoModelOverlay(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  canvasHeight: number,
  message: string = 'Model not loaded',
): void {
  ctx.save()
  ctx.fillStyle = 'rgba(15, 23, 42, 0.75)'
  ctx.fillRect(0, canvasHeight - 36, canvasWidth, 36)

  ctx.font = '500 13px Inter, sans-serif'
  ctx.fillStyle = '#f59e0b'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(`⚡ ${message}`, canvasWidth / 2, canvasHeight - 18)
  ctx.restore()
}

// YOLOv8 obstacle detection post-processing

import type { BoundingBox } from './drawing'

// COCO class names (80 classes)
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush',
]

// Cached offscreen canvas and buffer for YOLO detection preprocessing
let detCanvas: HTMLCanvasElement | null = null
let detCtx: CanvasRenderingContext2D | null = null
let detTensorBuffer: Float32Array | null = null

export function preprocessForDetection(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  targetSize: number = 640,
): { tensor: Float32Array; originalWidth: number; originalHeight: number } {
  if (!detCanvas) {
    detCanvas = document.createElement('canvas')
    detCanvas.width = targetSize
    detCanvas.height = targetSize
    detCtx = detCanvas.getContext('2d', { willReadFrequently: true })
    detTensorBuffer = new Float32Array(3 * targetSize * targetSize)
  }

  // Get source dimensions
  let srcWidth: number
  let srcHeight: number
  if (source instanceof HTMLVideoElement) {
    srcWidth = source.videoWidth || targetSize
    srcHeight = source.videoHeight || targetSize
  } else if (source instanceof HTMLImageElement) {
    srcWidth = source.naturalWidth || targetSize
    srcHeight = source.naturalHeight || targetSize
  } else {
    srcWidth = source.width || targetSize
    srcHeight = source.height || targetSize
  }

  if (!detCtx || !detTensorBuffer) {
    return { tensor: new Float32Array(3 * targetSize * targetSize), originalWidth: srcWidth, originalHeight: srcHeight }
  }

  // Letterbox resize - maintain aspect ratio
  const scale = Math.min(targetSize / srcWidth, targetSize / srcHeight)
  const newWidth = Math.round(srcWidth * scale)
  const newHeight = Math.round(srcHeight * scale)
  const offsetX = (targetSize - newWidth) / 2
  const offsetY = (targetSize - newHeight) / 2

  // Fill with gray (114/255)
  detCtx.fillStyle = '#727272'
  detCtx.fillRect(0, 0, targetSize, targetSize)
  detCtx.drawImage(source, offsetX, offsetY, newWidth, newHeight)

  const imageData = detCtx.getImageData(0, 0, targetSize, targetSize)
  const pixels = imageData.data
  const numPixels = targetSize * targetSize

  for (let i = 0; i < numPixels; i++) {
    detTensorBuffer[i] = pixels[i * 4] / 255.0                     // R
    detTensorBuffer[numPixels + i] = pixels[i * 4 + 1] / 255.0     // G
    detTensorBuffer[2 * numPixels + i] = pixels[i * 4 + 2] / 255.0 // B
  }

  return { tensor: detTensorBuffer, originalWidth: srcWidth, originalHeight: srcHeight }
}

/**
 * Post-process YOLOv8 output
 * YOLOv8 output shape: [1, 84, 8400] (4 box coords + 80 class scores, 8400 predictions)
 * Returns normalized bounding boxes [0, 1]
 */
export function postprocessDetection(
  output: Float32Array,
  originalWidth: number,
  originalHeight: number,
  inputSize: number = 640,
  confidenceThreshold: number = 0.35,
  iouThreshold: number = 0.45,
): BoundingBox[] {
  const numClasses = 80
  const numDetections = 8400

  // Parse detections
  const candidates: BoundingBox[] = []

  for (let i = 0; i < numDetections; i++) {
    // YOLOv8 output format: [cx, cy, w, h, class_scores...]
    const cx = output[0 * numDetections + i]
    const cy = output[1 * numDetections + i]
    const w = output[2 * numDetections + i]
    const h = output[3 * numDetections + i]

    // Find best class
    let maxScore = 0
    let maxClassId = 0
    for (let c = 0; c < numClasses; c++) {
      const score = output[(4 + c) * numDetections + i]
      if (score > maxScore) {
        maxScore = score
        maxClassId = c
      }
    }

    if (maxScore < confidenceThreshold) continue

    // Convert from letterbox coordinates to original image coordinates
    const scale = Math.min(inputSize / originalWidth, inputSize / originalHeight)
    const offsetX = (inputSize - originalWidth * scale) / 2
    const offsetY = (inputSize - originalHeight * scale) / 2

    const x1 = (cx - w / 2 - offsetX) / (originalWidth * scale)
    const y1 = (cy - h / 2 - offsetY) / (originalHeight * scale)
    const bw = w / (originalWidth * scale)
    const bh = h / (originalHeight * scale)

    candidates.push({
      x: Math.max(0, x1),
      y: Math.max(0, y1),
      width: Math.min(1 - Math.max(0, x1), bw),
      height: Math.min(1 - Math.max(0, y1), bh),
      label: COCO_CLASSES[maxClassId] ?? `class_${maxClassId}`,
      confidence: maxScore,
      classId: maxClassId,
    })
  }

  // Apply NMS
  return nonMaxSuppression(candidates, iouThreshold)
}

/**
 * Non-Maximum Suppression
 */
function nonMaxSuppression(
  boxes: BoundingBox[],
  iouThreshold: number,
): BoundingBox[] {
  // Sort by confidence descending
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence)
  const selected: BoundingBox[] = []

  const suppressed = new Set<number>()

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue
    selected.push(sorted[i])

    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue
      if (computeIoU(sorted[i], sorted[j]) > iouThreshold) {
        suppressed.add(j)
      }
    }
  }

  return selected
}

/**
 * Compute Intersection over Union between two boxes
 */
function computeIoU(a: BoundingBox, b: BoundingBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.width, b.x + b.width)
  const y2 = Math.min(a.y + a.height, b.y + b.height)

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const areaA = a.width * a.height
  const areaB = b.width * b.height
  const union = areaA + areaB - intersection

  return union > 0 ? intersection / union : 0
}

/**
 * Generate demo obstacle boxes (for testing without a model)
 */
export function generateDemoBoxes(time: number): BoundingBox[] {
  const wobble = Math.sin(time * 0.7) * 0.02

  return [
    {
      x: 0.35 + wobble,
      y: 0.45,
      width: 0.12,
      height: 0.08,
      label: 'car',
      confidence: 0.92,
      classId: 2,
    },
    {
      x: 0.15 + wobble * 0.5,
      y: 0.55,
      width: 0.04,
      height: 0.12,
      label: 'person',
      confidence: 0.87,
      classId: 0,
    },
    {
      x: 0.7 - wobble,
      y: 0.5,
      width: 0.1,
      height: 0.07,
      label: 'truck',
      confidence: 0.78,
      classId: 7,
    },
  ]
}

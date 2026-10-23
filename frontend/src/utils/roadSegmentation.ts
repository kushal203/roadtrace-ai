// Road segmentation model preprocessing and postprocessing

/**
 * Preprocess a video frame for the road segmentation model
 * Input: HTMLVideoElement or HTMLImageElement
 * Output: Float32Array in NCHW format [1, 3, 256, 256] normalized to [0, 1]
 */
// Cached offscreen canvas and buffer for zero-allocation preprocessing
let segCanvas: HTMLCanvasElement | null = null
let segCtx: CanvasRenderingContext2D | null = null
let segTensorBuffer: Float32Array | null = null

export function preprocessForSegmentation(
  source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
  targetWidth: number = 256,
  targetHeight: number = 256,
): Float32Array {
  if (!segCanvas) {
    segCanvas = document.createElement('canvas')
    segCanvas.width = targetWidth
    segCanvas.height = targetHeight
    segCtx = segCanvas.getContext('2d', { willReadFrequently: true })
    segTensorBuffer = new Float32Array(3 * targetWidth * targetHeight)
  }
  if (!segCtx || !segTensorBuffer) {
    return new Float32Array(3 * targetWidth * targetHeight)
  }

  segCtx.drawImage(source, 0, 0, targetWidth, targetHeight)
  const imageData = segCtx.getImageData(0, 0, targetWidth, targetHeight)
  const pixels = imageData.data

  const numPixels = targetWidth * targetHeight
  const mean = [0.485, 0.456, 0.406]
  const std = [0.229, 0.224, 0.225]

  for (let i = 0; i < numPixels; i++) {
    const r = pixels[i * 4] / 255.0
    const g = pixels[i * 4 + 1] / 255.0
    const b = pixels[i * 4 + 2] / 255.0

    segTensorBuffer[i] = (r - mean[0]) / std[0]
    segTensorBuffer[numPixels + i] = (g - mean[1]) / std[1]
    segTensorBuffer[2 * numPixels + i] = (b - mean[2]) / std[2]
  }

  return segTensorBuffer
}

/**
 * Post-process segmentation model output
 * Input: raw model output tensor [1, 3, 256, 256] (3 classes, argmax per pixel)
 * Output: Uint8Array of class IDs [256*256]
 */
export function postprocessSegmentation(
  output: Float32Array,
  numClasses: number = 3,
  width: number = 256,
  height: number = 256,
): Uint8Array {
  const numPixels = width * height
  const mask = new Uint8Array(numPixels)

  for (let i = 0; i < numPixels; i++) {
    let maxVal = -Infinity
    let maxClass = 0

    for (let c = 0; c < numClasses; c++) {
      const val = output[c * numPixels + i]
      if (val > maxVal) {
        maxVal = val
        maxClass = c
      }
    }

    mask[i] = maxClass
  }

  return mask
}

/**
 * Calculate the average confidence of the road segmentation
 * Uses softmax probabilities of the winning class
 */
export function calculateSegmentationConfidence(
  output: Float32Array,
  mask: Uint8Array,
  numClasses: number = 3,
  width: number = 256,
  height: number = 256,
): number {
  const numPixels = width * height
  let totalConfidence = 0
  let roadPixelCount = 0

  for (let i = 0; i < numPixels; i++) {
    if (mask[i] === 0) continue // skip background

    // Compute softmax for this pixel
    let expSum = 0
    const exps: number[] = []
    for (let c = 0; c < numClasses; c++) {
      const exp = Math.exp(output[c * numPixels + i])
      exps.push(exp)
      expSum += exp
    }

    const confidence = exps[mask[i]] / expSum
    totalConfidence += confidence
    roadPixelCount++
  }

  return roadPixelCount > 0 ? totalConfidence / roadPixelCount : 0
}

/**
 * Generate a demo segmentation mask (for testing without a model)
 * Creates a realistic-looking road mask
 */
export function generateDemoMask(
  width: number = 256,
  height: number = 256,
  time: number = 0,
): Uint8Array {
  const mask = new Uint8Array(width * height)

  const curveOffset = Math.sin(time * 0.5) * 30

  for (let y = 0; y < height; y++) {
    const progress = y / height
    // Road narrows toward the top (perspective)
    const roadWidth = width * (0.2 + progress * 0.5)
    const center = width / 2 + curveOffset * (1 - progress)

    const leftEdge = center - roadWidth / 2
    const rightEdge = center + roadWidth / 2
    const altWidth = roadWidth * 0.15

    for (let x = 0; x < width; x++) {
      if (x >= leftEdge && x <= rightEdge) {
        if (x <= leftEdge + altWidth || x >= rightEdge - altWidth) {
          mask[y * width + x] = 2 // alternative drivable
        } else {
          mask[y * width + x] = 1 // direct drivable
        }
      }
    }
  }

  return mask
}

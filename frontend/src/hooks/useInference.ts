// ONNX Runtime inference hook

import { useState, useRef, useCallback, useEffect } from 'react'
import * as ort from 'onnxruntime-web'
import {
  preprocessForSegmentation,
  postprocessSegmentation,
  calculateSegmentationConfidence,
} from '../utils/roadSegmentation'
import {
  preprocessForDetection,
  postprocessDetection,
} from '../utils/obstacleDetection'
import type { BoundingBox, CurveInfo } from '../utils/drawing'

// Configure ONNX Runtime to use WASM backend
ort.env.wasm.numThreads = 1
ort.env.wasm.simd = true

export interface InferenceStats {
  fps: number
  segLatency: number
  detLatency: number
  segConfidence: number
  frameCount: number
}

export interface InferenceState {
  segModelLoaded: boolean
  detModelLoaded: boolean
  segModelLoading: boolean
  detModelLoading: boolean
  segModelError: string | null
  detModelError: string | null
  isRunning: boolean
  stats: InferenceStats
}

export interface InferenceResults {
  mask: Uint8Array | null
  boxes: BoundingBox[]
  curveInfo: CurveInfo | null
}

export interface UseInferenceReturn {
  state: InferenceState
  results: InferenceResults
  loadModels: () => Promise<void>
  startInference: (source: HTMLVideoElement) => void
  stopInference: () => void
}

export function useInference(): UseInferenceReturn {
  const segSessionRef = useRef<ort.InferenceSession | null>(null)
  const detSessionRef = useRef<ort.InferenceSession | null>(null)
  const animFrameRef = useRef<number>(0)
  const isRunningRef = useRef(false)

  const [state, setState] = useState<InferenceState>({
    segModelLoaded: false,
    detModelLoaded: false,
    segModelLoading: false,
    detModelLoading: false,
    segModelError: null,
    detModelError: null,
    isRunning: false,
    stats: {
      fps: 0,
      segLatency: 0,
      detLatency: 0,
      segConfidence: 0,
      frameCount: 0,
    },
  })

  const [results, setResults] = useState<InferenceResults>({
    mask: null,
    boxes: [],
    curveInfo: null,
  })

  // FPS tracking
  const fpsCounterRef = useRef({ frames: 0, lastTime: performance.now() })

  const loadModels = useCallback(async () => {
    // Load segmentation model
    setState((prev) => ({
      ...prev,
      segModelLoading: true,
      segModelError: null,
    }))

    try {
      const segSession = await ort.InferenceSession.create('/models/road_seg.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
      segSessionRef.current = segSession
      setState((prev) => ({
        ...prev,
        segModelLoaded: true,
        segModelLoading: false,
      }))
      console.log('[RoadTrace] Road segmentation model loaded')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      console.warn('[RoadTrace] Failed to load segmentation model:', msg)
      setState((prev) => ({
        ...prev,
        segModelLoading: false,
        segModelError: `Segmentation model not available: ${msg}`,
      }))
    }

    // Load detection model
    setState((prev) => ({
      ...prev,
      detModelLoading: true,
      detModelError: null,
    }))

    try {
      const detSession = await ort.InferenceSession.create('/models/yolov8n.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
      detSessionRef.current = detSession
      setState((prev) => ({
        ...prev,
        detModelLoaded: true,
        detModelLoading: false,
      }))
      console.log('[RoadTrace] Obstacle detection model loaded')
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      console.warn('[RoadTrace] Failed to load detection model:', msg)
      setState((prev) => ({
        ...prev,
        detModelLoading: false,
        detModelError: `Detection model not available: ${msg}`,
      }))
    }
  }, [])

  const runInferenceFrame = useCallback(
    async (source: HTMLVideoElement) => {
      if (!isRunningRef.current) return

      let newMask: Uint8Array | null = results.mask
      let newBoxes: BoundingBox[] = results.boxes
      let segLatency = state.stats.segLatency
      let detLatency = state.stats.detLatency
      let segConfidence = state.stats.segConfidence

      // Run segmentation if model is loaded
      if (segSessionRef.current && source.readyState >= 2) {
        try {
          const segStart = performance.now()
          const inputTensor = preprocessForSegmentation(source)
          const tensor = new ort.Tensor('float32', inputTensor, [1, 3, 256, 256])

          // Get input name from model
          const inputName = segSessionRef.current.inputNames[0]
          const feeds: Record<string, ort.Tensor> = {}
          feeds[inputName] = tensor

          const output = await segSessionRef.current.run(feeds)
          const outputName = segSessionRef.current.outputNames[0]
          const outputData = output[outputName].data as Float32Array

          newMask = postprocessSegmentation(outputData)
          segConfidence = calculateSegmentationConfidence(outputData, newMask)
          segLatency = performance.now() - segStart
        } catch (err) {
          console.warn('[RoadTrace] Segmentation inference error:', err)
        }
      }

      // Run detection if model is loaded (every other frame for performance)
      if (
        detSessionRef.current &&
        source.readyState >= 2 &&
        fpsCounterRef.current.frames % 2 === 0
      ) {
        try {
          const detStart = performance.now()
          const { tensor: inputTensor, originalWidth, originalHeight } =
            preprocessForDetection(source)
          const tensor = new ort.Tensor('float32', inputTensor, [1, 3, 640, 640])

          const inputName = detSessionRef.current.inputNames[0]
          const feeds: Record<string, ort.Tensor> = {}
          feeds[inputName] = tensor

          const output = await detSessionRef.current.run(feeds)
          const outputName = detSessionRef.current.outputNames[0]
          const outputData = output[outputName].data as Float32Array

          newBoxes = postprocessDetection(outputData, originalWidth, originalHeight)
          detLatency = performance.now() - detStart
        } catch (err) {
          console.warn('[RoadTrace] Detection inference error:', err)
        }
      }

      // Update FPS
      fpsCounterRef.current.frames++
      const now = performance.now()
      const elapsed = now - fpsCounterRef.current.lastTime
      let fps = state.stats.fps
      if (elapsed >= 1000) {
        fps = (fpsCounterRef.current.frames / elapsed) * 1000
        fpsCounterRef.current.frames = 0
        fpsCounterRef.current.lastTime = now
      }

      setResults({
        mask: newMask,
        boxes: newBoxes,
        curveInfo: null, // will be computed during drawing
      })

      setState((prev) => ({
        ...prev,
        stats: {
          fps: Math.round(fps * 10) / 10,
          segLatency: Math.round(segLatency),
          detLatency: Math.round(detLatency),
          segConfidence: Math.round(segConfidence * 100) / 100,
          frameCount: prev.stats.frameCount + 1,
        },
      }))

      // Schedule next frame
      if (isRunningRef.current) {
        animFrameRef.current = requestAnimationFrame(() =>
          runInferenceFrame(source),
        )
      }
    },
    // We intentionally use a ref-based approach so the callback doesn't
    // recreate on every state change
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const startInference = useCallback(
    (source: HTMLVideoElement) => {
      isRunningRef.current = true
      fpsCounterRef.current = { frames: 0, lastTime: performance.now() }
      setState((prev) => ({ ...prev, isRunning: true }))
      runInferenceFrame(source)
    },
    [runInferenceFrame],
  )

  const stopInference = useCallback(() => {
    isRunningRef.current = false
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current)
    }
    setState((prev) => ({ ...prev, isRunning: false }))
    setResults({ mask: null, boxes: [], curveInfo: null })
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isRunningRef.current = false
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [])

  return {
    state,
    results,
    loadModels,
    startInference,
    stopInference,
  }
}

// Camera management hook

import { useState, useRef, useCallback, useEffect } from 'react'

export type CameraFacing = 'user' | 'environment'

export interface CameraState {
  isActive: boolean
  isLoading: boolean
  error: string | null
  facing: CameraFacing
  resolution: { width: number; height: number } | null
}

export interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>
  state: CameraState
  startCamera: () => Promise<void>
  stopCamera: () => void
  toggleFacing: () => void
}

export function useCamera(): UseCameraReturn {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [state, setState] = useState<CameraState>({
    isActive: false,
    isLoading: false,
    error: null,
    facing: 'environment',
    resolution: null,
  })

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setState((prev) => ({
      ...prev,
      isActive: false,
      isLoading: false,
      resolution: null,
    }))
  }, [])

  const startCamera = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }))

    // Stop any existing stream first
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: state.facing,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()

        const track = stream.getVideoTracks()[0]
        const settings = track.getSettings()

        setState((prev) => ({
          ...prev,
          isActive: true,
          isLoading: false,
          resolution: {
            width: settings.width ?? 0,
            height: settings.height ?? 0,
          },
        }))
      }
    } catch (err) {
      let errorMessage = 'Failed to access camera'

      if (err instanceof DOMException) {
        switch (err.name) {
          case 'NotAllowedError':
            errorMessage = 'Camera access denied. Please grant permission.'
            break
          case 'NotFoundError':
            errorMessage = 'No camera found on this device.'
            break
          case 'NotReadableError':
            errorMessage = 'Camera is in use by another application.'
            break
          case 'OverconstrainedError':
            errorMessage = 'Camera does not meet constraints. Trying again...'
            break
          default:
            errorMessage = `Camera error: ${err.message}`
        }
      }

      setState((prev) => ({
        ...prev,
        isActive: false,
        isLoading: false,
        error: errorMessage,
      }))
    }
  }, [state.facing])

  const toggleFacing = useCallback(() => {
    setState((prev) => ({
      ...prev,
      facing: prev.facing === 'user' ? 'environment' : 'user',
    }))
  }, [])

  // Restart camera when facing changes (if camera was active)
  useEffect(() => {
    if (state.isActive) {
      startCamera()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.facing])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  return {
    videoRef,
    state,
    startCamera,
    stopCamera,
    toggleFacing,
  }
}

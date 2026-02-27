/**
 * React hook: auto-selects WebGPU or CPU Worker for LiDAR conversion.
 *
 * Usage:
 *   const { convert, backend } = useLidarConverter()
 *   const cloud = await convert(rangeImages, calibrations)
 *
 * Backend selection:
 *   - WebGPU available → 'gpu'
 *   - Otherwise → 'cpu' (Web Worker fallback)
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { isWebGPUAvailable, convertAllSensorsGpu } from '../utils/rangeImageGpu'
import type { LidarCalibration, RangeImage, PointCloud } from '../utils/rangeImage'
import type { LidarWorkerRequest, LidarWorkerResponse } from '../workers/lidarWorker'

export type LidarBackend = 'gpu' | 'cpu' | 'detecting'

export interface LidarConvertResult extends PointCloud {
  elapsedMs: number
  backend: LidarBackend
}

export function useLidarConverter() {
  const [backend, setBackend] = useState<LidarBackend>('detecting')
  const workerRef = useRef<Worker | null>(null)

  // Detect WebGPU on mount
  useEffect(() => {
    if (isWebGPUAvailable()) {
      // Verify we can actually get a device (not just that the API exists)
      navigator.gpu.requestAdapter().then((adapter) => {
        setBackend(adapter ? 'gpu' : 'cpu')
      }).catch(() => {
        setBackend('cpu')
      })
    } else {
      setBackend('cpu')
    }

    return () => {
      workerRef.current?.terminate()
      workerRef.current = null
    }
  }, [])

  const convertCpu = useCallback((
    rangeImages: Map<number, RangeImage>,
    calibrations: Map<number, LidarCalibration>,
  ): Promise<LidarConvertResult> => {
    return new Promise((resolve, reject) => {
      // Lazy-init worker
      if (!workerRef.current) {
        workerRef.current = new Worker(
          new URL('../workers/lidarWorker.ts', import.meta.url),
          { type: 'module' },
        )
      }

      const worker = workerRef.current

      worker.onmessage = (event: MessageEvent<LidarWorkerResponse>) => {
        resolve({
          positions: event.data.positions,
          pointCount: event.data.pointCount,
          elapsedMs: event.data.elapsedMs,
          backend: 'cpu',
        })
      }

      worker.onerror = (err) => reject(err)

      // Serialize Maps to arrays for structured clone
      const request: LidarWorkerRequest = {
        type: 'convert',
        rangeImages: Array.from(rangeImages.entries()).map(([name, ri]) => [
          name,
          {
            shape: ri.shape,
            values: ri.values instanceof Float32Array ? Array.from(ri.values) : ri.values as number[],
          },
        ]),
        calibrations: Array.from(calibrations.entries()),
      }

      worker.postMessage(request)
    })
  }, [])

  const convertGpu = useCallback(async (
    rangeImages: Map<number, RangeImage>,
    calibrations: Map<number, LidarCalibration>,
  ): Promise<LidarConvertResult> => {
    const result = await convertAllSensorsGpu(rangeImages, calibrations)
    return { ...result, backend: 'gpu' }
  }, [])

  const convert = useCallback(async (
    rangeImages: Map<number, RangeImage>,
    calibrations: Map<number, LidarCalibration>,
  ): Promise<LidarConvertResult> => {
    if (backend === 'gpu') {
      try {
        return await convertGpu(rangeImages, calibrations)
      } catch (err) {
        console.warn('[useLidarConverter] GPU conversion failed, falling back to CPU:', err)
        setBackend('cpu')
        return convertCpu(rangeImages, calibrations)
      }
    }
    return convertCpu(rangeImages, calibrations)
  }, [backend, convertGpu, convertCpu])

  return { convert, backend }
}

/**
 * CPU Web Worker for LiDAR range image → xyz conversion.
 *
 * Fallback path for browsers without WebGPU (Firefox).
 * Runs convertRangeImageToPointCloud() off the main thread.
 *
 * Message protocol:
 *   Main → Worker: { type: 'convert', rangeImages, calibrations }
 *   Worker → Main: { type: 'result', positions, pointCount, elapsedMs }
 */

import {
  convertAllSensors,
  type LidarCalibration,
  type RangeImage,
} from '../utils/rangeImage'

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

export interface LidarWorkerRequest {
  type: 'convert'
  /** Serialized Map: Array of [laserName, { shape, values }] */
  rangeImages: Array<[number, { shape: [number, number, number]; values: number[] }]>
  /** Serialized Map: Array of [laserName, LidarCalibration] */
  calibrations: Array<[number, LidarCalibration]>
}

export interface LidarWorkerResponse {
  type: 'result'
  /** Float32Array [x, y, z, intensity, ...] — transferred, not copied */
  positions: Float32Array
  pointCount: number
  /** Conversion time in milliseconds */
  elapsedMs: number
}

// ---------------------------------------------------------------------------
// Worker entry point
// ---------------------------------------------------------------------------

self.onmessage = (event: MessageEvent<LidarWorkerRequest>) => {
  const { rangeImages: riEntries, calibrations: calibEntries } = event.data

  const t0 = performance.now()

  // Reconstruct Maps from serialized arrays
  const rangeImages = new Map<number, RangeImage>(
    riEntries.map(([name, ri]) => [name, ri]),
  )
  const calibrations = new Map<number, LidarCalibration>(calibEntries)

  const cloud = convertAllSensors(rangeImages, calibrations)

  const elapsedMs = performance.now() - t0

  // Transfer the Float32Array buffer (zero-copy) back to main thread
  const response: LidarWorkerResponse = {
    type: 'result',
    positions: cloud.positions,
    pointCount: cloud.pointCount,
    elapsedMs,
  }

  ;(self as unknown as { postMessage(msg: unknown, transfer: Transferable[]): void })
    .postMessage(response, [cloud.positions.buffer])
}

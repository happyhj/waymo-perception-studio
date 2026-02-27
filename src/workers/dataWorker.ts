/**
 * Data Worker — runs Parquet I/O + LiDAR conversion off the main thread.
 *
 * Responsibilities: fetch (Range Request) → BROTLI decompress → Parquet decode → xyz conversion.
 * Main thread stays free for 60fps rendering.
 *
 * Architecture: thin orchestration layer. Actual logic lives in parquet.ts and rangeImage.ts.
 */

import {
  openParquetFile,
  buildHeavyFileFrameIndex,
  readFrameData,
  type WaymoParquetFile,
  type FrameRowIndex,
} from '../utils/parquet'
import {
  convertAllSensors,
  type LidarCalibration,
  type RangeImage,
} from '../utils/rangeImage'

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------

let lidarPf: WaymoParquetFile | null = null
let lidarIndex: FrameRowIndex | null = null
let calibrations = new Map<number, LidarCalibration>()

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

export interface DataWorkerInit {
  type: 'init'
  lidarUrl: string
  /** Serialized as [laserName, calibration][] since Map can't be postMessage'd */
  calibrationEntries: [number, LidarCalibration][]
}

export interface DataWorkerLoadFrame {
  type: 'loadFrame'
  requestId: number
  frameIndex: number
  /** bigint serialized as string (postMessage doesn't support bigint) */
  timestamp: string
}

export type DataWorkerRequest = DataWorkerInit | DataWorkerLoadFrame

export interface DataWorkerFrameResult {
  type: 'frameReady'
  requestId: number
  frameIndex: number
  positions: Float32Array
  pointCount: number
  convertMs: number
}

export interface DataWorkerReady {
  type: 'ready'
}

export interface DataWorkerError {
  type: 'error'
  requestId?: number
  message: string
}

export type DataWorkerResponse = DataWorkerReady | DataWorkerFrameResult | DataWorkerError

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

const LIDAR_COLUMNS = [
  'key.laser_name',
  '[LiDARComponent].range_image_return1.shape',
  '[LiDARComponent].range_image_return1.values',
]

self.onmessage = async (e: MessageEvent<DataWorkerRequest>) => {
  const msg = e.data

  try {
    if (msg.type === 'init') {
      // Open lidar parquet file and build frame index
      lidarPf = await openParquetFile('lidar', msg.lidarUrl)
      lidarIndex = await buildHeavyFileFrameIndex(lidarPf)

      // Reconstruct calibration Map
      calibrations = new Map(msg.calibrationEntries)

      ;(self as unknown as { postMessage(msg: DataWorkerResponse): void })
        .postMessage({ type: 'ready' })
    }

    if (msg.type === 'loadFrame') {
      if (!lidarPf || !lidarIndex) {
        throw new Error('Worker not initialized')
      }

      const timestamp = BigInt(msg.timestamp)

      // Read lidar rows for this frame
      const lidarRows = await readFrameData(lidarPf, lidarIndex, timestamp, LIDAR_COLUMNS)

      const rangeImages = new Map<number, RangeImage>()
      for (const row of lidarRows) {
        const laserName = row['key.laser_name'] as number
        rangeImages.set(laserName, {
          shape: row['[LiDARComponent].range_image_return1.shape'] as [number, number, number],
          values: row['[LiDARComponent].range_image_return1.values'] as number[],
        })
      }

      const ct0 = performance.now()
      const pointCloud = convertAllSensors(rangeImages, calibrations)
      const convertMs = performance.now() - ct0

      const result: DataWorkerFrameResult = {
        type: 'frameReady',
        requestId: msg.requestId,
        frameIndex: msg.frameIndex,
        positions: pointCloud.positions,
        pointCount: pointCloud.pointCount,
        convertMs,
      }

      // Transfer the Float32Array buffer (zero-copy)
      ;(self as unknown as { postMessage(msg: DataWorkerResponse, transfer: Transferable[]): void })
        .postMessage(result, [pointCloud.positions.buffer])
    }
  } catch (err) {
    const errorMsg: DataWorkerError = {
      type: 'error',
      requestId: (msg as DataWorkerLoadFrame).requestId,
      message: err instanceof Error ? err.message : String(err),
    }
    ;(self as unknown as { postMessage(msg: DataWorkerResponse): void })
      .postMessage(errorMsg)
  }
}

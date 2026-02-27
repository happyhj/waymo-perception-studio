/**
 * Integration tests for parquet.ts — reads actual Waymo v2.0 Parquet files.
 *
 * These tests run against the sample segment downloaded in waymo_data/.
 * They validate that our loading infrastructure correctly handles
 * the real Waymo v2.0 Parquet format.
 */

import { describe, it, expect, beforeAll } from 'vitest'
import { openSync, readSync, fstatSync, closeSync, readFileSync } from 'fs'
import { resolve } from 'path'
import type { AsyncBuffer } from 'hyparquet'
import {
  openParquetFile,
  readAllRows,
  readRowRange,
  buildFrameIndex,
  buildHeavyFileFrameIndex,
  readFrameData,
  isHeavyComponent,
} from '../parquet'
import { merge, groupIndexBy } from '../merge'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const WAYMO_DATA = resolve(__dirname, '../../../waymo_data')
const SEGMENT_ID = '10023947602400723454_1120_000_1140_000'

function parquetPath(component: string): string {
  return resolve(WAYMO_DATA, component, `${SEGMENT_ID}.parquet`)
}

/**
 * Create AsyncBuffer from a local file using fs.open + fs.read.
 * For small files, reads entire file into memory.
 * For large files, uses fd-based reads for true byte-range access (no OOM).
 */
function nodeAsyncBuffer(filePath: string, lazy = false): AsyncBuffer {
  if (!lazy) {
    // Small file: read into memory
    const buf = readFileSync(filePath)
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
    return {
      byteLength: ab.byteLength,
      slice(start: number, end?: number): ArrayBuffer {
        return ab.slice(start, end)
      },
    }
  }

  // Large file: use file descriptor for on-demand reads
  const fd = openSync(filePath, 'r')
  const { size } = fstatSync(fd)
  return {
    byteLength: size,
    slice(start: number, end?: number): ArrayBuffer {
      const length = (end ?? size) - start
      const buffer = Buffer.alloc(length)
      readSync(fd, buffer, 0, length, start)
      return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
    },
    // Note: fd is leaked intentionally in tests. In production, use File API.
  }
}

/** Open a Waymo Parquet file using Node.js file access */
function openTestFile(component: string) {
  const lazy = isHeavyComponent(component)
  const buffer = nodeAsyncBuffer(parquetPath(component), lazy)
  return openParquetFile(component, buffer)
}

// ---------------------------------------------------------------------------
// Tests: Small file — full load
// ---------------------------------------------------------------------------

describe('Small files — full load', () => {
  it('reads vehicle_pose: 199 rows, correct columns', async () => {
    const pf = await openTestFile('vehicle_pose')
    expect(pf.numRows).toBe(199)
    expect(pf.component).toBe('vehicle_pose')

    const rows = await readAllRows(pf)
    expect(rows).toHaveLength(199)

    // Check expected columns exist
    const cols = Object.keys(rows[0])
    expect(cols).toContain('key.segment_context_name')
    expect(cols).toContain('key.frame_timestamp_micros')
    expect(cols.some((c) => c.includes('VehiclePoseComponent'))).toBe(true)
  })

  it('reads lidar_box: ~18K rows with tracking IDs', async () => {
    const pf = await openTestFile('lidar_box')
    const rows = await readAllRows(pf)

    expect(rows.length).toBeGreaterThan(10000)

    // Check tracking ID exists
    expect(rows[0]).toHaveProperty('key.laser_object_id')
    // Check box data exists
    const cols = Object.keys(rows[0])
    expect(cols.some((c) => c.includes('LiDARBoxComponent'))).toBe(true)
  })

  it('reads lidar_calibration: 5 rows (one per sensor)', async () => {
    const pf = await openTestFile('lidar_calibration')
    const rows = await readAllRows(pf)
    expect(rows).toHaveLength(5)

    // laser_name should be 1-5
    const names = rows.map((r) => r['key.laser_name']).sort()
    expect(names).toEqual([1, 2, 3, 4, 5])
  })

  it('reads camera_calibration: 5 rows', async () => {
    const pf = await openTestFile('camera_calibration')
    const rows = await readAllRows(pf)
    expect(rows).toHaveLength(5)
  })
})

// ---------------------------------------------------------------------------
// Tests: Frame index
// ---------------------------------------------------------------------------

describe('Frame index', () => {
  let poseRows: Record<string, unknown>[]

  beforeAll(async () => {
    const pf = await openTestFile('vehicle_pose')
    poseRows = await readAllRows(pf)
  })

  it('builds frame index with 199 unique timestamps', () => {
    const { timestamps, frameByTimestamp } = buildFrameIndex(poseRows)
    expect(timestamps).toHaveLength(199)
    expect(frameByTimestamp.size).toBe(199)

    // Timestamps should be sorted
    for (let i = 1; i < timestamps.length; i++) {
      expect(timestamps[i]).toBeGreaterThan(timestamps[i - 1])
    }
  })

  it('frame 0 maps to the earliest timestamp', () => {
    const { timestamps, frameByTimestamp } = buildFrameIndex(poseRows)
    expect(frameByTimestamp.get(timestamps[0])).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// Tests: Heavy file — lazy loading (fd-based, no OOM)
// ---------------------------------------------------------------------------

describe('Heavy file — lazy row-range read', () => {
  it('opens lidar file metadata without loading data', async () => {
    const pf = await openTestFile('lidar')

    // Total should be 995 (5 sensors × 199 frames)
    expect(pf.numRows).toBe(995)
    expect(pf.rowGroups.length).toBeGreaterThan(1) // Multiple row groups
  })

  it('reads partial rows from lidar (first 5 rows only)', async () => {
    const pf = await openTestFile('lidar')

    // Read just first 5 rows (one frame, all sensors)
    const rows = await readRowRange(pf, 0, 5, [
      'key.frame_timestamp_micros',
      'key.laser_name',
    ])
    expect(rows).toHaveLength(5)
    expect(rows[0]).toHaveProperty('key.laser_name')
  })

  it('builds heavy file frame index from key columns only', async () => {
    const pf = await openTestFile('lidar')
    const frameIdx = await buildHeavyFileFrameIndex(pf)

    // Should have 199 unique timestamps
    expect(frameIdx.byTimestamp.size).toBe(199)

    // Each timestamp should map to 5 rows (5 sensors)
    const firstEntry = frameIdx.byTimestamp.values().next().value!
    expect(firstEntry.rowEnd - firstEntry.rowStart).toBe(5)
  })

  it('readFrameData retrieves correct frame from heavy file', async () => {
    // Get the master timestamp list
    const posePf = await openTestFile('vehicle_pose')
    const poseRows = await readAllRows(posePf)
    const { timestamps } = buildFrameIndex(poseRows)

    // Open lidar file and build frame index
    const lidarPf = await openTestFile('lidar')
    const lidarFrameIdx = await buildHeavyFileFrameIndex(lidarPf)

    // Read frame 0 data (key columns only — avoid loading range images)
    const frameData = await readFrameData(
      lidarPf,
      lidarFrameIdx,
      timestamps[0],
      ['key.frame_timestamp_micros', 'key.laser_name'],
    )

    // Should have 5 rows (5 LiDAR sensors)
    expect(frameData).toHaveLength(5)

    // All rows should have the same timestamp
    const uniqueTs = new Set(frameData.map((r) => r['key.frame_timestamp_micros']))
    expect(uniqueTs.size).toBe(1)
  })
})

// ---------------------------------------------------------------------------
// Tests: merge() with real Waymo data
// ---------------------------------------------------------------------------

describe('merge() with real Waymo data', () => {
  it('joins vehicle_pose with lidar_box', async () => {
    const posePf = await openTestFile('vehicle_pose')
    const boxPf = await openTestFile('lidar_box')

    const poses = await readAllRows(posePf)
    const boxes = await readAllRows(boxPf)

    // Merge: each box gets its frame's pose
    const merged = merge(poses, boxes)

    // Should have same count as boxes (every box has a matching frame)
    expect(merged.length).toBe(boxes.length)

    // Each merged row has both pose and box data
    const cols = Object.keys(merged[0])
    expect(cols.some((c) => c.includes('VehiclePoseComponent'))).toBe(true)
    expect(cols.some((c) => c.includes('LiDARBoxComponent'))).toBe(true)
  })

  it('groupIndexBy groups boxes by frame timestamp', async () => {
    const boxPf = await openTestFile('lidar_box')
    const boxes = await readAllRows(boxPf)

    const byFrame = groupIndexBy(boxes, 'key.frame_timestamp_micros')

    // Should have 199 frames
    expect(byFrame.size).toBe(199)

    // Check average objects per frame is reasonable (~94)
    const counts = [...byFrame.values()].map((v) => v.length)
    const avg = counts.reduce((a, b) => a + b, 0) / counts.length
    expect(avg).toBeGreaterThan(50)
    expect(avg).toBeLessThan(200)
  })
})

// ---------------------------------------------------------------------------
// Tests: Component classification
// ---------------------------------------------------------------------------

describe('isHeavyComponent()', () => {
  it('correctly identifies heavy components', () => {
    expect(isHeavyComponent('camera_image')).toBe(true)
    expect(isHeavyComponent('lidar')).toBe(true)
    expect(isHeavyComponent('lidar_camera_projection')).toBe(true)
    expect(isHeavyComponent('lidar_pose')).toBe(true)
  })

  it('correctly identifies light components', () => {
    expect(isHeavyComponent('vehicle_pose')).toBe(false)
    expect(isHeavyComponent('lidar_box')).toBe(false)
    expect(isHeavyComponent('camera_calibration')).toBe(false)
    expect(isHeavyComponent('lidar_calibration')).toBe(false)
  })
})

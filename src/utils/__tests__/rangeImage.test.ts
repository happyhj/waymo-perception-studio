/**
 * Unit tests for rangeImage.ts — range image → xyz conversion.
 *
 * Tests the pure math against actual Waymo v2.0 data.
 * These same assertions can be reused to validate the WebGPU compute shader.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { readFileSync, closeSync } from 'fs'
import { openSync, readSync, fstatSync } from 'fs'
import { resolve } from 'path'
import type { AsyncBuffer } from 'hyparquet'
import { openParquetFile, readAllRows, readRowRange, isHeavyComponent } from '../parquet'
import {
  computeInclinations,
  computeAzimuths,
  convertRangeImageToPointCloud,
  convertAllSensors,
  parseLidarCalibration,
  POINT_STRIDE,
  type LidarCalibration,
  type RangeImage,
} from '../rangeImage'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const WAYMO_DATA = resolve(__dirname, '../../../waymo_data')
const SEGMENT_ID = '10023947602400723454_1120_000_1140_000'

function parquetPath(component: string): string {
  return resolve(WAYMO_DATA, component, `${SEGMENT_ID}.parquet`)
}

const openFds: number[] = []

function nodeAsyncBuffer(filePath: string, lazy = false): AsyncBuffer {
  if (!lazy) {
    const buf = readFileSync(filePath)
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
    return {
      byteLength: ab.byteLength,
      slice(start: number, end?: number): ArrayBuffer { return ab.slice(start, end) },
    }
  }
  const fd = openSync(filePath, 'r')
  openFds.push(fd)
  const { size } = fstatSync(fd)
  return {
    byteLength: size,
    slice(start: number, end?: number): ArrayBuffer {
      const length = (end ?? size) - start
      const buffer = Buffer.alloc(length)
      readSync(fd, buffer, 0, length, start)
      return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
    },
  }
}

afterAll(() => {
  for (const fd of openFds) {
    try { closeSync(fd) } catch { /* ignore */ }
  }
})

function openTestFile(component: string) {
  const lazy = isHeavyComponent(component)
  const buffer = nodeAsyncBuffer(parquetPath(component), lazy)
  return openParquetFile(component, buffer)
}

// ---------------------------------------------------------------------------
// Shared test data — loaded once
// ---------------------------------------------------------------------------

let calibrations: Map<number, LidarCalibration>
let topRangeImage: RangeImage
let frontRangeImage: RangeImage
let allRangeImages: Map<number, RangeImage>

beforeAll(async () => {
  // Load calibrations
  const calibPf = await openTestFile('lidar_calibration')
  const calibRows = await readAllRows(calibPf)
  calibrations = new Map()
  for (const row of calibRows) {
    const calib = parseLidarCalibration(row)
    calibrations.set(calib.laserName, calib)
  }

  // Load first frame's lidar data (5 sensors)
  const lidarPf = await openTestFile('lidar')
  const lidarRows = await readRowRange(lidarPf, 0, 5, [
    'key.laser_name',
    '[LiDARComponent].range_image_return1.shape',
    '[LiDARComponent].range_image_return1.values',
  ])

  allRangeImages = new Map()
  for (const row of lidarRows) {
    const laserName = row['key.laser_name'] as number
    const ri: RangeImage = {
      shape: row['[LiDARComponent].range_image_return1.shape'] as [number, number, number],
      values: row['[LiDARComponent].range_image_return1.values'] as number[],
    }
    allRangeImages.set(laserName, ri)
  }

  topRangeImage = allRangeImages.get(1)!
  frontRangeImage = allRangeImages.get(2)!
})

// ---------------------------------------------------------------------------
// Tests: Calibration parsing
// ---------------------------------------------------------------------------

describe('parseLidarCalibration()', () => {
  it('TOP has non-uniform inclinations (64 values)', () => {
    const top = calibrations.get(1)!
    expect(top.beamInclinationValues).not.toBeNull()
    expect(top.beamInclinationValues!.length).toBe(64)
  })

  it('FRONT has uniform inclinations (null values)', () => {
    const front = calibrations.get(2)!
    expect(front.beamInclinationValues).toBeNull()
    expect(front.beamInclinationMin).toBeDefined()
    expect(front.beamInclinationMax).toBeDefined()
  })

  it('extrinsic is 4x4 matrix (16 elements)', () => {
    for (const [, calib] of calibrations) {
      expect(calib.extrinsic).toHaveLength(16)
    }
  })
})

// ---------------------------------------------------------------------------
// Tests: Angle computation
// ---------------------------------------------------------------------------

describe('computeInclinations()', () => {
  it('non-uniform: uses exact values for TOP', () => {
    const top = calibrations.get(1)!
    const inc = computeInclinations(64, top)
    expect(inc).toHaveLength(64)
    // Should match the provided values
    expect(inc[0]).toBeCloseTo(top.beamInclinationValues![0], 5)
    expect(inc[63]).toBeCloseTo(top.beamInclinationValues![63], 5)
  })

  it('uniform: interpolates between min and max for FRONT', () => {
    const front = calibrations.get(2)!
    const inc = computeInclinations(200, front)
    expect(inc).toHaveLength(200)
    // Row 0 = max, last row = min
    expect(inc[0]).toBeCloseTo(front.beamInclinationMax, 5)
    expect(inc[199]).toBeCloseTo(front.beamInclinationMin, 5)
  })
})

describe('computeAzimuths()', () => {
  it('spans full 2π rotation', () => {
    const az = computeAzimuths(2650)
    expect(az).toHaveLength(2650)
    // Range should be approximately [-π, π]
    const min = Math.min(...az)
    const max = Math.max(...az)
    expect(max - min).toBeCloseTo(2 * Math.PI, 1)
  })
})

// ---------------------------------------------------------------------------
// Tests: Single sensor conversion
// ---------------------------------------------------------------------------

describe('convertRangeImageToPointCloud()', () => {
  it('converts TOP lidar with reasonable point count', () => {
    const cloud = convertRangeImageToPointCloud(topRangeImage, calibrations.get(1)!)
    // TOP has 169600 pixels, ~88% valid → ~149K points
    expect(cloud.pointCount).toBeGreaterThan(100000)
    expect(cloud.pointCount).toBeLessThan(170000)
    expect(cloud.positions.length).toBe(cloud.pointCount * POINT_STRIDE)
  })

  it('converts FRONT lidar with reasonable point count', () => {
    const cloud = convertRangeImageToPointCloud(frontRangeImage, calibrations.get(2)!)
    expect(cloud.pointCount).toBeGreaterThan(0)
    expect(cloud.pointCount).toBeLessThanOrEqual(120000) // 200×600
  })

  it('output coordinates are in reasonable range (vehicle frame)', () => {
    const cloud = convertRangeImageToPointCloud(topRangeImage, calibrations.get(1)!)

    // Sample some points — should be within ~75m (max range we saw)
    let maxDist = 0
    for (let i = 0; i < Math.min(cloud.pointCount, 1000); i++) {
      const x = cloud.positions[i * POINT_STRIDE]
      const y = cloud.positions[i * POINT_STRIDE + 1]
      const z = cloud.positions[i * POINT_STRIDE + 2]
      const dist = Math.sqrt(x * x + y * y + z * z)
      if (dist > maxDist) maxDist = dist
    }

    // Points should be within sensor range (~75m max for this segment)
    expect(maxDist).toBeGreaterThan(1) // not all at origin
    expect(maxDist).toBeLessThan(100) // within reasonable range
  })

  it('z coordinates are reasonable for a car-mounted lidar', () => {
    const cloud = convertRangeImageToPointCloud(topRangeImage, calibrations.get(1)!)

    let minZ = Infinity, maxZ = -Infinity
    for (let i = 0; i < cloud.pointCount; i++) {
      const z = cloud.positions[i * POINT_STRIDE + 2]
      if (z < minZ) minZ = z
      if (z > maxZ) maxZ = z
    }

    // Car-mounted TOP lidar at ~1.8m height
    // SF downtown with slopes/underground → minZ can reach ~-20m
    // Overhead structures (buildings, overpasses) → maxZ up to ~30m
    expect(minZ).toBeGreaterThan(-25)
    expect(maxZ).toBeLessThan(40)
  })

  it('intensity values are preserved', () => {
    const cloud = convertRangeImageToPointCloud(topRangeImage, calibrations.get(1)!)

    // At least some points should have non-zero intensity
    let hasNonZeroIntensity = false
    for (let i = 0; i < Math.min(cloud.pointCount, 100); i++) {
      if (cloud.positions[i * POINT_STRIDE + 3] > 0) {
        hasNonZeroIntensity = true
        break
      }
    }
    expect(hasNonZeroIntensity).toBe(true)
  })

  it('skips invalid pixels (range <= 0)', () => {
    // Create a synthetic range image with known invalid pixels
    const synthetic: RangeImage = {
      shape: [2, 3, 4],
      values: [
        // row 0: 3 pixels
        5.0, 0.5, 0, 0,   // valid
        -1,  0,   0, 0,   // invalid (range = -1)
        0,   0,   0, 0,   // invalid (range = 0)
        // row 1: 3 pixels
        10.0, 0.8, 0, 0,  // valid
        3.0,  0.2, 0, 0,  // valid
        -1,   0,   0, 0,  // invalid
      ],
    }

    const calib: LidarCalibration = {
      laserName: 99,
      extrinsic: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], // identity
      beamInclinationValues: null,
      beamInclinationMin: -0.3,
      beamInclinationMax: 0.3,
    }

    const cloud = convertRangeImageToPointCloud(synthetic, calib)
    expect(cloud.pointCount).toBe(3) // only 3 valid pixels
  })
})

// ---------------------------------------------------------------------------
// Tests: Multi-sensor merge
// ---------------------------------------------------------------------------

describe('convertAllSensors()', () => {
  it('merges all 5 sensors into one point cloud', () => {
    const cloud = convertAllSensors(allRangeImages, calibrations)

    // Sum of all valid points across 5 sensors
    expect(cloud.pointCount).toBeGreaterThan(150000)
    expect(cloud.positions.length).toBe(cloud.pointCount * POINT_STRIDE)
  })

  it('merged cloud has more points than any single sensor', () => {
    const merged = convertAllSensors(allRangeImages, calibrations)
    const topOnly = convertRangeImageToPointCloud(topRangeImage, calibrations.get(1)!)

    expect(merged.pointCount).toBeGreaterThan(topOnly.pointCount)
  })
})

// ---------------------------------------------------------------------------
// Export test data for GPU comparison tests
// ---------------------------------------------------------------------------

/**
 * Helper for GPU tests: returns the test data used in these CPU tests.
 * GPU tests import this to run the same assertions on GPU output.
 */
export function getTestData() {
  return { calibrations, topRangeImage, frontRangeImage, allRangeImages }
}

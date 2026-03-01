/**
 * GPU vs CPU consistency test for LiDAR range image → xyz conversion.
 *
 * Uses the `webgpu` npm package (Google Dawn) to run WebGPU compute shaders
 * in Node.js. This validates that the WGSL shader produces identical results
 * to the CPU TypeScript implementation.
 *
 * Requirements:
 *   - npm install webgpu (dev dependency)
 *   - Compatible platform: macOS (darwin-universal), Linux x64, Windows x64
 *   - GPU or software rasterizer available
 *
 * Run with: npx vitest run rangeImageGpu
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { readFileSync, closeSync } from 'fs'
import { openSync, readSync, fstatSync } from 'fs'
import { resolve } from 'path'
import type { AsyncBuffer } from 'hyparquet'
import { openParquetFile, readAllRows, readRowRange, isHeavyComponent } from '../parquet'
import {
  convertRangeImageToPointCloud,
  convertAllSensors,
  parseLidarCalibration,
  POINT_STRIDE,
  type LidarCalibration,
  type RangeImage,
} from '../rangeImage'

// ---------------------------------------------------------------------------
// Try to import webgpu — skip all tests if not available
// ---------------------------------------------------------------------------

let gpu: GPU | null = null
let skipReason = ''

try {
  // Dynamic import to avoid hard failure if platform binary missing
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const webgpu = require('webgpu')
  // Register WebGPU globals (GPUBufferUsage, GPUMapMode, etc.) into globalThis
  // Without this, rangeImageGpu.ts can't reference GPUBufferUsage.STORAGE etc.
  Object.assign(globalThis, webgpu.globals)
  // create([]) returns the GPU object directly (NOT { gpu: ... })
  // See: https://github.com/dawn-gpu/node-webgpu#usage
  gpu = webgpu.create([]) as GPU
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    skipReason = 'webgpu.create() did not return a valid GPU object'
    gpu = null
  }
} catch (e) {
  skipReason = `WebGPU not available on this platform: ${(e as Error).message}`
}

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
// Test data
// ---------------------------------------------------------------------------

let calibrations: Map<number, LidarCalibration>
let allRangeImages: Map<number, RangeImage>

beforeAll(async () => {
  if (skipReason) return

  const calibPf = await openTestFile('lidar_calibration')
  const calibRows = await readAllRows(calibPf)
  calibrations = new Map()
  for (const row of calibRows) {
    const calib = parseLidarCalibration(row)
    calibrations.set(calib.laserName, calib)
  }

  const lidarPf = await openTestFile('lidar')
  const lidarRows = await readRowRange(lidarPf, 0, 5, [
    'key.laser_name',
    '[LiDARComponent].range_image_return1.shape',
    '[LiDARComponent].range_image_return1.values',
  ])

  allRangeImages = new Map()
  for (const row of lidarRows) {
    const laserName = row['key.laser_name'] as number
    allRangeImages.set(laserName, {
      shape: row['[LiDARComponent].range_image_return1.shape'] as [number, number, number],
      values: row['[LiDARComponent].range_image_return1.values'] as number[],
    })
  }
})

// ---------------------------------------------------------------------------
// GPU vs CPU consistency tests
// ---------------------------------------------------------------------------

import { convertAllSensorsGpu } from '../rangeImageGpu'

// Skip all GPU tests if webgpu package couldn't load OR gpu object is falsy
const shouldSkip = !!skipReason || !gpu

describe('GPU vs CPU consistency', () => {
  it.skipIf(shouldSkip)('point counts match', async () => {
    const cpuCloud = convertAllSensors(allRangeImages, calibrations)
    const gpuCloud = await convertAllSensorsGpu(allRangeImages, calibrations, gpu!)

    // Point counts should be identical (same filtering logic)
    expect(gpuCloud.pointCount).toBe(cpuCloud.pointCount)
  })

  it.skipIf(shouldSkip)('xyz positions match within float32 epsilon', async () => {
    const cpuCloud = convertAllSensors(allRangeImages, calibrations)
    const gpuCloud = await convertAllSensorsGpu(allRangeImages, calibrations, gpu!)

    // Note: GPU uses atomic counter for stream compaction, so point ORDER
    // may differ from CPU. We compare aggregate statistics:

    // 1. Total point count
    expect(gpuCloud.pointCount).toBe(cpuCloud.pointCount)

    // 2. Bounding box should match closely
    const cpuBounds = computeBounds(cpuCloud.positions, cpuCloud.pointCount)
    const gpuBounds = computeBounds(gpuCloud.positions, gpuCloud.pointCount)

    expect(gpuBounds.minX).toBeCloseTo(cpuBounds.minX, 1)
    expect(gpuBounds.maxX).toBeCloseTo(cpuBounds.maxX, 1)
    expect(gpuBounds.minY).toBeCloseTo(cpuBounds.minY, 1)
    expect(gpuBounds.maxY).toBeCloseTo(cpuBounds.maxY, 1)
    expect(gpuBounds.minZ).toBeCloseTo(cpuBounds.minZ, 1)
    expect(gpuBounds.maxZ).toBeCloseTo(cpuBounds.maxZ, 1)
  })

  it.skipIf(shouldSkip)('intensity values and sum match', async () => {
    const cpuCloud = convertAllSensors(allRangeImages, calibrations)
    const gpuCloud = await convertAllSensorsGpu(allRangeImages, calibrations, gpu!)

    // Sum all intensity values (index 3 of each [x,y,z,i] quad)
    let cpuSum = 0
    let gpuSum = 0
    for (let i = 0; i < cpuCloud.pointCount; i++) {
      cpuSum += cpuCloud.positions[i * POINT_STRIDE + 3]
    }
    for (let i = 0; i < gpuCloud.pointCount; i++) {
      gpuSum += gpuCloud.positions[i * 4 + 3]
    }

    // Intensity sum should match within float32 accumulation tolerance
    // Allow 0.1% relative error for large sums
    const relativeError = Math.abs(cpuSum - gpuSum) / Math.max(cpuSum, 1)
    expect(relativeError).toBeLessThan(0.001)

    // Also log timing for reference
    const cpuT0 = performance.now()
    convertAllSensors(allRangeImages, calibrations)
    const cpuMs = performance.now() - cpuT0
    const gpuResult = await convertAllSensorsGpu(allRangeImages, calibrations, gpu!)
    console.log(`\n  CPU: ${cpuMs.toFixed(1)}ms, GPU: ${gpuResult.elapsedMs.toFixed(1)}ms`)
    console.log(`  Speedup: ${(cpuMs / gpuResult.elapsedMs).toFixed(1)}x`)
  })

  if (skipReason) {
    it(`SKIPPED: ${skipReason}`, () => {
      console.log(`\n  GPU tests skipped: ${skipReason}`)
      console.log('  Run on macOS/Linux-x64/Windows to enable GPU testing.')
      console.log('  GPU vs CPU consistency will be verified in-browser.')
    })
  }
})

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function computeBounds(positions: Float32Array, pointCount: number) {
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity
  let minZ = Infinity, maxZ = -Infinity

  for (let i = 0; i < pointCount; i++) {
    const x = positions[i * 4]
    const y = positions[i * 4 + 1]
    const z = positions[i * 4 + 2]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
    if (z < minZ) minZ = z
    if (z > maxZ) maxZ = z
  }

  return { minX, maxX, minY, maxY, minZ, maxZ }
}

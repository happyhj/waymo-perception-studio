/**
 * WebGPU compute shader for LiDAR range image → xyz conversion.
 *
 * Primary path for browsers with WebGPU support (Chrome, Edge, Safari 17.4+).
 * Each GPU thread processes one pixel — embarrassingly parallel.
 *
 * Pipeline:
 *   1. Upload range image values + calibration data as storage buffers
 *   2. Dispatch compute shader (one invocation per pixel)
 *   3. Read back xyz + intensity as Float32Array
 *
 * This module is pure GPU logic — no DOM, no React.
 */

import type { LidarCalibration, RangeImage, PointCloud } from './rangeImage'
import { computeInclinations, computeAzimuths } from './rangeImage'

// ---------------------------------------------------------------------------
// WGSL Compute Shader
// ---------------------------------------------------------------------------

const WGSL_SHADER = /* wgsl */ `
// Per-pixel range image conversion: spherical → cartesian → vehicle frame

struct Params {
  height: u32,
  width: u32,
  channels: u32,
  _pad: u32,
  // 4×4 extrinsic matrix (row-major), stored as 4 vec4s
  extrinsic0: vec4<f32>,
  extrinsic1: vec4<f32>,
  extrinsic2: vec4<f32>,
  extrinsic3: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> rangeValues: array<f32>;
@group(0) @binding(2) var<storage, read> inclinations: array<f32>;
@group(0) @binding(3) var<storage, read> azimuths: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
// Atomic counter for compacted output index
@group(0) @binding(5) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixelIdx = gid.x;
  let totalPixels = params.height * params.width;
  if (pixelIdx >= totalPixels) { return; }

  let row = pixelIdx / params.width;
  let col = pixelIdx % params.width;
  let valueIdx = pixelIdx * params.channels;

  let range = rangeValues[valueIdx];
  if (range <= 0.0) { return; }

  let intensity = rangeValues[valueIdx + 1u];

  // Spherical → Cartesian (sensor frame)
  let inc = inclinations[row];
  let az = azimuths[col];
  let ci = cos(inc);
  let si = sin(inc);
  let ca = cos(az);
  let sa = sin(az);

  let x = range * ci * ca;
  let y = range * ci * sa;
  let z = range * si;

  // Extrinsic transform (sensor → vehicle frame)
  let e0 = params.extrinsic0;
  let e1 = params.extrinsic1;
  let e2 = params.extrinsic2;

  let vx = e0.x * x + e0.y * y + e0.z * z + e0.w;
  let vy = e1.x * x + e1.y * y + e1.z * z + e1.w;
  let vz = e2.x * x + e2.y * y + e2.z * z + e2.w;

  // Atomic increment to get output index (stream compaction)
  let outIdx = atomicAdd(&counter, 1u);
  let base = outIdx * 4u;
  output[base] = vx;
  output[base + 1u] = vy;
  output[base + 2u] = vz;
  output[base + 3u] = intensity;
}
`

// ---------------------------------------------------------------------------
// GPU Device Management
// ---------------------------------------------------------------------------

let cachedDevice: GPUDevice | null = null
let injectedGpu: GPU | null = null

/**
 * Inject a GPU instance for testing (e.g., from `webgpu` npm package in Node.js).
 * Call this before any conversion functions.
 */
export function setGPU(gpu: GPU): void {
  injectedGpu = gpu
  cachedDevice = null // reset cached device when GPU instance changes
}

/**
 * Get the GPU instance — injected (testing) or browser-native.
 */
function getGPU(): GPU {
  if (injectedGpu) return injectedGpu
  if (typeof navigator !== 'undefined' && 'gpu' in navigator) return navigator.gpu
  throw new Error('WebGPU: not available. Use setGPU() to inject a GPU instance for testing.')
}

/**
 * Check if WebGPU is available in the current browser.
 */
export function isWebGPUAvailable(): boolean {
  if (injectedGpu) return true
  return typeof navigator !== 'undefined' && 'gpu' in navigator
}

/**
 * Get (or create) the GPU device. Caches for reuse.
 */
async function getDevice(gpuOverride?: GPU): Promise<GPUDevice> {
  if (cachedDevice) return cachedDevice

  const gpu = gpuOverride || getGPU()
  const adapter = await gpu.requestAdapter()
  if (!adapter) throw new Error('WebGPU: No adapter found')

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: 256 * 1024 * 1024, // 256MB
      maxBufferSize: 256 * 1024 * 1024,
    },
  })

  device.lost.then(() => { cachedDevice = null })
  cachedDevice = device
  return device
}

// ---------------------------------------------------------------------------
// Single-sensor GPU conversion
// ---------------------------------------------------------------------------

/**
 * Convert a single sensor's range image to point cloud using WebGPU.
 */
async function convertSensorGpu(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  rangeImage: RangeImage,
  calibration: LidarCalibration,
): Promise<PointCloud> {
  const [height, width, channels] = rangeImage.shape
  const totalPixels = height * width
  const maxOutputFloats = totalPixels * 4 // worst case: all valid

  // Precompute angles on CPU (small arrays, not worth GPU)
  const inclinations = computeInclinations(height, calibration)
  const azimuths = computeAzimuths(width)

  // Pack params: height, width, channels, pad, extrinsic (4×vec4)
  const paramsData = new Float32Array(4 + 16)
  const paramsU32 = new Uint32Array(paramsData.buffer)
  paramsU32[0] = height
  paramsU32[1] = width
  paramsU32[2] = channels
  paramsU32[3] = 0 // padding
  for (let i = 0; i < 16; i++) {
    paramsData[4 + i] = calibration.extrinsic[i]
  }

  // Convert range values to Float32Array if needed
  const rangeF32 = rangeImage.values instanceof Float32Array
    ? rangeImage.values
    : new Float32Array(rangeImage.values)

  // Create GPU buffers
  const paramsBuffer = createBuffer(device, paramsData, GPUBufferUsage.STORAGE)
  const rangeBuffer = createBuffer(device, rangeF32, GPUBufferUsage.STORAGE)
  const incBuffer = createBuffer(device, inclinations, GPUBufferUsage.STORAGE)
  const azBuffer = createBuffer(device, azimuths, GPUBufferUsage.STORAGE)

  const outputBuffer = device.createBuffer({
    size: maxOutputFloats * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  })

  const counterBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
  // Initialize counter to 0
  device.queue.writeBuffer(counterBuffer, 0, new Uint32Array([0]))

  // Staging buffers for readback
  const outputStaging = device.createBuffer({
    size: maxOutputFloats * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })
  const counterStaging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  // Bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: rangeBuffer } },
      { binding: 2, resource: { buffer: incBuffer } },
      { binding: 3, resource: { buffer: azBuffer } },
      { binding: 4, resource: { buffer: outputBuffer } },
      { binding: 5, resource: { buffer: counterBuffer } },
    ],
  })

  // Dispatch
  const workgroupSize = 256
  const numWorkgroups = Math.ceil(totalPixels / workgroupSize)

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(numWorkgroups)
  pass.end()

  // Copy results to staging
  encoder.copyBufferToBuffer(counterBuffer, 0, counterStaging, 0, 4)
  encoder.copyBufferToBuffer(outputBuffer, 0, outputStaging, 0, maxOutputFloats * 4)

  device.queue.submit([encoder.finish()])

  // Read back counter first to know how many points
  await counterStaging.mapAsync(GPUMapMode.READ)
  const pointCount = new Uint32Array(counterStaging.getMappedRange())[0]
  counterStaging.unmap()

  // Read back output positions
  await outputStaging.mapAsync(GPUMapMode.READ)
  const positions = new Float32Array(outputStaging.getMappedRange().slice(0, pointCount * 4 * 4))
  outputStaging.unmap()

  // Cleanup GPU buffers
  paramsBuffer.destroy()
  rangeBuffer.destroy()
  incBuffer.destroy()
  azBuffer.destroy()
  outputBuffer.destroy()
  counterBuffer.destroy()
  outputStaging.destroy()
  counterStaging.destroy()

  return { positions, pointCount }
}

// ---------------------------------------------------------------------------
// Multi-sensor GPU conversion (public API)
// ---------------------------------------------------------------------------

/**
 * Convert all sensors' range images to a merged point cloud using WebGPU.
 *
 * Public API — matches convertAllSensors() signature from rangeImage.ts.
 *
 * @param gpuOverride — optional GPU instance (for Node.js testing with `webgpu` package).
 *   In the browser, omit this and the module uses `navigator.gpu` automatically.
 */
export async function convertAllSensorsGpu(
  rangeImages: Map<number, RangeImage>,
  calibrations: Map<number, LidarCalibration>,
  gpuOverride?: GPU,
): Promise<PointCloud & { elapsedMs: number }> {
  const t0 = performance.now()

  const device = await getDevice(gpuOverride)

  // Create pipeline (could be cached, but creation is fast)
  const shaderModule = device.createShaderModule({ code: WGSL_SHADER })
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: shaderModule, entryPoint: 'main' },
  })

  // Convert each sensor
  const clouds: PointCloud[] = []
  let totalPoints = 0

  for (const [laserName, rangeImage] of rangeImages) {
    const calib = calibrations.get(laserName)
    if (!calib) {
      console.warn(`[rangeImageGpu] No calibration for laser_name=${laserName}, skipping`)
      continue
    }
    const cloud = await convertSensorGpu(device, pipeline, rangeImage, calib)
    clouds.push(cloud)
    totalPoints += cloud.pointCount
  }

  // Merge into single Float32Array
  const merged = new Float32Array(totalPoints * 4)
  let offset = 0
  for (const cloud of clouds) {
    merged.set(cloud.positions, offset)
    offset += cloud.pointCount * 4
  }

  const elapsedMs = performance.now() - t0

  return { positions: merged, pointCount: totalPoints, elapsedMs }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createBuffer(
  device: GPUDevice,
  data: Float32Array | Uint32Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  })
  if (data instanceof Float32Array) {
    new Float32Array(buffer.getMappedRange()).set(data)
  } else {
    new Uint32Array(buffer.getMappedRange()).set(data)
  }
  buffer.unmap()
  return buffer
}

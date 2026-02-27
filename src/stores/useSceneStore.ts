/**
 * Scene store — Zustand-based central state for Waymo Perception Studio.
 *
 * Heavy work (Parquet I/O + BROTLI decompress + LiDAR conversion) runs in
 * a Data Worker — main thread stays free for 60fps rendering.
 *
 * Prefetch strategy: load ALL row groups sequentially (start → end).
 * Each row group decompression yields ~51 frames at once — only 4 reads
 * to cache the entire 199-frame segment.
 *
 * Usage in React:
 *   const pointCloud = useSceneStore(s => s.currentFrame?.pointCloud)
 *   const { loadDataset, nextFrame } = useSceneStore(s => s.actions)
 */

import { create } from 'zustand'
import type { ParquetRow } from '../utils/merge'
import { groupIndexBy } from '../utils/merge'
import {
  openParquetFile,
  readAllRows,
  readRowGroupRows,
  buildFrameIndex,
  buildHeavyFileFrameIndex,
  readFrameData,
  type WaymoParquetFile,
  type FrameRowIndex,
} from '../utils/parquet'
import {
  parseLidarCalibration,
  convertAllSensors,
  type LidarCalibration,
  type PointCloud,
  type RangeImage,
} from '../utils/rangeImage'
import type {
  DataWorkerRequest,
  DataWorkerResponse,
  DataWorkerRowGroupResult,
  FrameResult,
  SensorCloudResult,
} from '../workers/dataWorker'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type LoadStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface FrameData {
  timestamp: bigint
  pointCloud: PointCloud | null
  /** Per-sensor point clouds for toggle UI (keyed by laser_name: 1=TOP,2=FRONT,3=SIDE_LEFT,4=SIDE_RIGHT,5=REAR) */
  sensorClouds: Map<number, PointCloud>
  boxes: ParquetRow[]
  cameraImages: Map<number, ArrayBuffer>
  vehiclePose: number[] | null
}

interface SceneActions {
  loadDataset: (sources: Map<string, File | string>) => Promise<void>
  loadFrame: (index: number) => Promise<void>
  nextFrame: () => Promise<void>
  prevFrame: () => Promise<void>
  seekFrame: (index: number) => Promise<void>
  play: () => void
  pause: () => void
  togglePlayback: () => void
  setPlaybackSpeed: (speed: number) => void
  toggleSensor: (laserName: number) => void
  reset: () => void
}

export interface SceneState {
  // Loading
  status: LoadStatus
  error: string | null
  availableComponents: string[]
  loadProgress: number

  // Frame navigation
  totalFrames: number
  currentFrameIndex: number
  isPlaying: boolean
  playbackSpeed: number

  // Current frame data
  currentFrame: FrameData | null

  // Calibrations (loaded once)
  lidarCalibrations: Map<number, LidarCalibration>
  cameraCalibrations: ParquetRow[]

  // Performance
  lastFrameLoadMs: number
  lastConvertMs: number

  // Prefetch progress (for YouTube-style buffer bar)
  /** Sorted array of cached frame indices */
  cachedFrames: number[]
  /** Which sensors are visible (1=TOP,2=FRONT,3=SIDE_LEFT,4=SIDE_RIGHT,5=REAR) */
  visibleSensors: Set<number>
  // Actions
  actions: SceneActions
}

// ---------------------------------------------------------------------------
// Internal state (not exposed to React — no re-renders on mutation)
// ---------------------------------------------------------------------------

const internal = {
  parquetFiles: new Map<string, WaymoParquetFile>(),
  timestamps: [] as bigint[],
  /** Reverse lookup: timestamp → frame index */
  timestampToFrame: new Map<bigint, number>(),
  lidarBoxByFrame: new Map<unknown, ParquetRow[]>(),
  vehiclePoseByFrame: new Map<unknown, ParquetRow[]>(),
  /** No eviction needed — row-group loading caches all ~199 frames, which is the goal. */
  frameCache: new Map<number, FrameData>(),
  playIntervalId: null as ReturnType<typeof setInterval> | null,
  worker: null as Worker | null,
  workerReady: false,
  numRowGroups: 0,
  /** Track which row groups have been loaded or are in-flight */
  loadedRowGroups: new Set<number>(),
  /** Prevent duplicate prefetchAllRowGroups calls (React StrictMode) */
  prefetchStarted: false,
  /** Last per-frame conversion time (for performance tracking) */
  lastConvertMs: 0,
  /** Frame index for per-frame fallback (test env / no Worker) */
  lidarFrameIndex: null as FrameRowIndex | null,
  pendingRequests: new Map<number, {
    resolve: (result: DataWorkerRowGroupResult) => void
    reject: (err: Error) => void
  }>(),
  nextRequestId: 0,
}

function resetInternal() {
  internal.parquetFiles.clear()
  internal.timestamps = []
  internal.timestampToFrame.clear()
  internal.lidarBoxByFrame.clear()
  internal.vehiclePoseByFrame.clear()
  internal.frameCache.clear()
  internal.loadedRowGroups.clear()
  internal.prefetchStarted = false
  if (internal.playIntervalId !== null) {
    clearInterval(internal.playIntervalId)
    internal.playIntervalId = null
  }
  if (internal.worker) {
    internal.worker.terminate()
    internal.worker = null
    internal.workerReady = false
  }
  internal.numRowGroups = 0
  internal.pendingRequests.clear()
  internal.nextRequestId = 0
}

// ---------------------------------------------------------------------------
// Worker communication
// ---------------------------------------------------------------------------

function postToWorker(msg: DataWorkerRequest) {
  internal.worker?.postMessage(msg)
}

function requestRowGroup(
  rowGroupIndex: number,
): Promise<DataWorkerRowGroupResult> {
  return new Promise((resolve, reject) => {
    const requestId = internal.nextRequestId++
    internal.pendingRequests.set(requestId, { resolve, reject })
    postToWorker({
      type: 'loadRowGroup',
      requestId,
      rowGroupIndex,
    })
  })
}

function handleWorkerMessage(e: MessageEvent<DataWorkerResponse>) {
  const msg = e.data

  if (msg.type === 'rowGroupReady' || msg.type === 'error') {
    const rid = 'requestId' in msg ? msg.requestId : -1
    const pending = internal.pendingRequests.get(rid ?? -1)
    if (pending) {
      internal.pendingRequests.delete(rid!)
      if (msg.type === 'error') {
        pending.reject(new Error(msg.message))
      } else {
        pending.resolve(msg)
      }
    }
  }
}

/** Cache all frames from a row group result into internal.frameCache */
function cacheRowGroupFrames(
  result: DataWorkerRowGroupResult,
  set: (partial: Partial<SceneState>) => void,
) {
  for (const frame of result.frames) {
    const timestamp = BigInt(frame.timestamp)
    const frameIndex = internal.timestampToFrame.get(timestamp)
    if (frameIndex === undefined) continue
    if (internal.frameCache.has(frameIndex)) continue

    const boxes = internal.lidarBoxByFrame.get(timestamp) ?? []
    const poseRows = internal.vehiclePoseByFrame.get(timestamp)
    const vehiclePose = (poseRows?.[0]?.['[VehiclePoseComponent].world_from_vehicle.transform'] as number[]) ?? null

    const sensorClouds = new Map<number, PointCloud>()
    if (frame.sensorClouds) {
      for (const sc of frame.sensorClouds) {
        sensorClouds.set(sc.laserName, { positions: sc.positions, pointCount: sc.pointCount })
      }
    }

    const frameData: FrameData = {
      timestamp,
      pointCloud: {
        positions: frame.positions,
        pointCount: frame.pointCount,
      },
      sensorClouds,
      boxes,
      cameraImages: new Map(),
      vehiclePose,
    }

    internal.frameCache.set(frameIndex, frameData)
  }

  syncCachedFrames(set)
}

/** Update the cachedFrames state for the buffer bar UI */
function syncCachedFrames(set: (partial: Partial<SceneState>) => void) {
  const indices = [...internal.frameCache.keys()].sort((a, b) => a - b)
  set({ cachedFrames: indices })
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useSceneStore = create<SceneState>((set, get) => ({
  status: 'idle',
  error: null,
  availableComponents: [],
  loadProgress: 0,
  totalFrames: 0,
  currentFrameIndex: 0,
  isPlaying: false,
  playbackSpeed: 1,
  currentFrame: null,
  lidarCalibrations: new Map(),
  cameraCalibrations: [],
  lastFrameLoadMs: 0,
  lastConvertMs: 0,
  cachedFrames: [],
  visibleSensors: new Set([1, 2, 3, 4, 5]),

  actions: {
    loadDataset: async (sources) => {
      resetInternal()
      set({
        status: 'loading',
        availableComponents: [...sources.keys()],
        error: null,
        loadProgress: 0,
        cachedFrames: [],
      })

      try {
        const totalSteps = sources.size + 2
        let completed = 0

        // 1. Open all Parquet files (footer only — lightweight, main thread OK)
        for (const [component, source] of sources) {
          const pf = await openParquetFile(component, source)
          internal.parquetFiles.set(component, pf)
          completed++
          set({ loadProgress: completed / totalSteps })
        }

        // 2. Load startup data (small files: poses, calibrations, boxes)
        await loadStartupData(set)
        completed++
        set({ loadProgress: completed / totalSteps })

        // 3. Init Data Worker for heavy lidar I/O
        await initDataWorker(sources, get)
        completed++
        set({ loadProgress: completed / totalSteps })

        // 4. Load first frame
        const rgT0 = performance.now()
        if (internal.workerReady) {
          // Worker path: load entire first row group → cache ~51 frames at once
          await loadAndCacheRowGroup(0, set)
        } else {
          // Main-thread fallback: build lidar frame index, load single frame
          const lidarPf = internal.parquetFiles.get('lidar')
          if (lidarPf) {
            internal.lidarFrameIndex = await buildHeavyFileFrameIndex(lidarPf)
            await loadFrameMainThread(0, set, get)
          }
        }
        const rgMs = performance.now() - rgT0

        // Show first frame
        const firstFrame = internal.frameCache.get(0)
        if (firstFrame) {
          set({
            currentFrameIndex: 0,
            currentFrame: firstFrame,
            lastFrameLoadMs: rgMs,
            lastConvertMs: internal.lastConvertMs,
          })
        }

        set({ status: 'ready', loadProgress: 1 })

        // Auto-play after first segment loads
        get().actions.play()

        // 5. Prefetch remaining row groups in background (start → end)
        //    Only when Worker is available — in test env, load on demand per loadFrame.
        //    Guard against duplicate calls from React StrictMode re-renders.
        if (internal.workerReady && !internal.prefetchStarted) {
          internal.prefetchStarted = true
          prefetchAllRowGroups(set, get)
        }
      } catch (e) {
        set({
          status: 'error',
          error: e instanceof Error ? e.message : String(e),
        })
      }
    },

    loadFrame: async (frameIndex) => {
      if (frameIndex < 0 || frameIndex >= internal.timestamps.length) return

      // Cache hit — instant (the common case after prefetch completes)
      const cached = internal.frameCache.get(frameIndex)
      if (cached) {
        set({
          currentFrameIndex: frameIndex,
          currentFrame: cached,
          lastFrameLoadMs: 0,
          lastConvertMs: cached.pointCloud ? get().lastConvertMs : 0,
        })
        return
      }

      // Cache miss — frame not yet prefetched, ignore navigation.
      // Prefetch loads all row groups sequentially; the frame will become
      // available shortly. This avoids contention with the prefetch queue.
    },

    nextFrame: () => get().actions.loadFrame(get().currentFrameIndex + 1),
    prevFrame: () => get().actions.loadFrame(get().currentFrameIndex - 1),
    seekFrame: (index) => get().actions.loadFrame(index),

    play: () => {
      if (get().isPlaying) return
      set({ isPlaying: true })
      const intervalMs = 100 / get().playbackSpeed
      internal.playIntervalId = setInterval(async () => {
        const next = get().currentFrameIndex + 1
        if (next >= get().totalFrames) {
          get().actions.pause()
          return
        }
        await get().actions.loadFrame(next)
      }, intervalMs)
    },

    pause: () => {
      if (!get().isPlaying) return
      if (internal.playIntervalId !== null) {
        clearInterval(internal.playIntervalId)
        internal.playIntervalId = null
      }
      set({ isPlaying: false })
    },

    togglePlayback: () => {
      if (get().isPlaying) get().actions.pause()
      else get().actions.play()
    },

    setPlaybackSpeed: (speed) => {
      const wasPlaying = get().isPlaying
      if (wasPlaying) get().actions.pause()
      set({ playbackSpeed: speed })
      if (wasPlaying) get().actions.play()
    },

    toggleSensor: (laserName: number) => {
      const prev = get().visibleSensors
      const next = new Set(prev)
      if (next.has(laserName)) next.delete(laserName)
      else next.add(laserName)
      set({ visibleSensors: next })
    },

    reset: () => {
      get().actions.pause()
      resetInternal()
      set({
        status: 'idle',
        error: null,
        availableComponents: [],
        loadProgress: 0,
        totalFrames: 0,
        currentFrameIndex: 0,
        isPlaying: false,
        playbackSpeed: 1,
        currentFrame: null,
        lidarCalibrations: new Map(),
        cameraCalibrations: [],
        lastFrameLoadMs: 0,
        lastConvertMs: 0,
        cachedFrames: [],
        visibleSensors: new Set([1, 2, 3, 4, 5]),
      })
    },
  },
}))

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Find which row group index contains a given frame.
 * Uses the lidar parquet file's row group boundaries + frame index mapping.
 */
function findRowGroupForFrame(frameIndex: number): number {
  const lidarPf = internal.parquetFiles.get('lidar')
  if (!lidarPf) return -1

  const timestamp = internal.timestamps[frameIndex]
  if (timestamp === undefined) return -1

  // Each lidar row group has a rowStart..rowEnd range.
  // We need to find which RG contains rows for this timestamp.
  // Since frames are time-sorted and row groups are sequential,
  // we can estimate: frameIndex / framesPerRG ≈ rowGroupIndex.
  // But for safety, use the frame index's row mapping.
  const totalFrames = internal.timestamps.length
  const numRGs = lidarPf.rowGroups.length
  const framesPerRG = Math.ceil(totalFrames / numRGs)
  const estimated = Math.min(Math.floor(frameIndex / framesPerRG), numRGs - 1)
  return estimated
}

const LIDAR_COLUMNS = [
  'key.frame_timestamp_micros',
  'key.laser_name',
  '[LiDARComponent].range_image_return1.shape',
  '[LiDARComponent].range_image_return1.values',
]

/** Load entire row group via Worker and cache all its frames. */
async function loadAndCacheRowGroup(
  rgIndex: number,
  set: (partial: Partial<SceneState>) => void,
): Promise<void> {
  if (internal.loadedRowGroups.has(rgIndex)) return
  internal.loadedRowGroups.add(rgIndex) // Mark as in-flight to prevent duplicates

  try {
    const result = await requestRowGroup(rgIndex)
    cacheRowGroupFrames(result, set)
  } catch {
    // If loading failed, allow retry
    internal.loadedRowGroups.delete(rgIndex)
  }
}

/**
 * Main-thread fallback: load a SINGLE frame via per-frame row range.
 * Used in test env / File-based sources where Worker is not available.
 * Each call decompresses the full row group (wasteful), but only keeps
 * 5 rows in memory — avoids OOM that row-group-level caching would cause.
 */
async function loadFrameMainThread(
  frameIndex: number,
  set: (partial: Partial<SceneState>) => void,
  get: () => SceneState,
): Promise<FrameData | null> {
  const lidarPf = internal.parquetFiles.get('lidar')
  if (!lidarPf || !internal.lidarFrameIndex) return null

  const timestamp = internal.timestamps[frameIndex]
  if (timestamp === undefined) return null

  const lidarRows = await readFrameData(
    lidarPf,
    internal.lidarFrameIndex,
    timestamp,
    LIDAR_COLUMNS,
  )

  const rangeImages = new Map<number, RangeImage>()
  for (const row of lidarRows) {
    const laserName = row['key.laser_name'] as number
    rangeImages.set(laserName, {
      shape: row['[LiDARComponent].range_image_return1.shape'] as [number, number, number],
      values: row['[LiDARComponent].range_image_return1.values'] as number[],
    })
  }

  const ct0 = performance.now()
  const result = convertAllSensors(rangeImages, get().lidarCalibrations)
  internal.lastConvertMs = performance.now() - ct0

  const boxes = internal.lidarBoxByFrame.get(timestamp) ?? []
  const poseRows = internal.vehiclePoseByFrame.get(timestamp)
  const vehiclePose = (poseRows?.[0]?.['[VehiclePoseComponent].world_from_vehicle.transform'] as number[]) ?? null

  const frameData: FrameData = {
    timestamp,
    pointCloud: result.merged,
    sensorClouds: result.perSensor,
    boxes,
    cameraImages: new Map(),
    vehiclePose,
  }

  internal.frameCache.set(frameIndex, frameData)
  syncCachedFrames(set)
  return frameData
}

async function loadStartupData(set: (partial: Partial<SceneState>) => void) {
  // Vehicle pose → master frame list
  const posePf = internal.parquetFiles.get('vehicle_pose')
  if (posePf) {
    const rows = await readAllRows(posePf)
    const index = buildFrameIndex(rows)
    internal.timestamps = index.timestamps
    internal.timestampToFrame = index.frameByTimestamp
    internal.vehiclePoseByFrame = groupIndexBy(rows, 'key.frame_timestamp_micros')
    set({ totalFrames: index.timestamps.length })
  }

  // LiDAR calibration
  const lidarCalibPf = internal.parquetFiles.get('lidar_calibration')
  if (lidarCalibPf) {
    const rows = await readAllRows(lidarCalibPf)
    const calibMap = new Map<number, LidarCalibration>()
    for (const row of rows) {
      const calib = parseLidarCalibration(row)
      calibMap.set(calib.laserName, calib)
    }
    set({ lidarCalibrations: calibMap })
  }

  // Camera calibration
  const cameraCalibPf = internal.parquetFiles.get('camera_calibration')
  if (cameraCalibPf) {
    set({ cameraCalibrations: await readAllRows(cameraCalibPf) })
  }

  // LiDAR boxes
  const lidarBoxPf = internal.parquetFiles.get('lidar_box')
  if (lidarBoxPf) {
    const rows = await readAllRows(lidarBoxPf)
    internal.lidarBoxByFrame = groupIndexBy(rows, 'key.frame_timestamp_micros')
  }
}

// ---------------------------------------------------------------------------
// Data Worker init
// ---------------------------------------------------------------------------

async function initDataWorker(
  sources: Map<string, File | string>,
  get: () => SceneState,
) {
  const lidarSource = sources.get('lidar')
  if (!lidarSource || typeof lidarSource !== 'string') {
    return
  }

  return new Promise<void>((resolve, reject) => {
    const worker = new Worker(
      new URL('../workers/dataWorker.ts', import.meta.url),
      { type: 'module' },
    )

    worker.onmessage = (e: MessageEvent<DataWorkerResponse>) => {
      if (e.data.type === 'ready') {
        internal.workerReady = true
        internal.numRowGroups = e.data.numRowGroups
        worker.onmessage = handleWorkerMessage
        resolve()
      } else if (e.data.type === 'error') {
        reject(new Error(e.data.message))
      }
    }

    worker.onerror = (e) => reject(new Error(e.message))

    internal.worker = worker

    const calibEntries = [...get().lidarCalibrations.entries()]

    postToWorker({
      type: 'init',
      lidarUrl: lidarSource,
      calibrationEntries: calibEntries,
    })
  })
}

// ---------------------------------------------------------------------------
// Row-group-level prefetching — load ALL row groups start → end
// ---------------------------------------------------------------------------

/**
 * Sequentially load all row groups from the lidar file.
 * Each RG yields ~51 frames — after 4 RGs, all 199 frames are cached.
 * Sequential (not parallel) to avoid saturating bandwidth.
 */
/**
 * Sequentially load all row groups from the lidar file via Worker.
 * Each RG yields ~51 frames — after 4 RGs, all 199 frames are cached.
 * Sequential (not parallel) to avoid saturating bandwidth.
 */
async function prefetchAllRowGroups(
  set: (partial: Partial<SceneState>) => void,
  _get: () => SceneState,
) {
  for (let rg = 0; rg < internal.numRowGroups; rg++) {
    if (internal.loadedRowGroups.has(rg)) continue

    try {
      await loadAndCacheRowGroup(rg, set)
    } catch {
      // Non-critical: prefetch failure doesn't block user interaction
    }
  }
}

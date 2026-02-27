/**
 * Scene store — Zustand-based central state for Waymo Perception Studio.
 *
 * Heavy work (Parquet I/O + BROTLI decompress + LiDAR conversion) runs in
 * a Data Worker — main thread stays free for 60fps rendering.
 *
 * Usage in React:
 *   const pointCloud = useSceneStore(s => s.currentFrame?.pointCloud)
 *   const { loadDataset, nextFrame } = useSceneStore(s => s.actions)
 *
 * Usage outside React:
 *   useSceneStore.getState().actions.nextFrame()
 */

import { create } from 'zustand'
import type { ParquetRow } from '../utils/merge'
import { groupIndexBy } from '../utils/merge'
import {
  openParquetFile,
  readAllRows,
  buildFrameIndex,
  type WaymoParquetFile,
} from '../utils/parquet'
import {
  parseLidarCalibration,
  type LidarCalibration,
  type PointCloud,
} from '../utils/rangeImage'
import type {
  DataWorkerRequest,
  DataWorkerResponse,
  DataWorkerFrameResult,
} from '../workers/dataWorker'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type LoadStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface FrameData {
  timestamp: bigint
  pointCloud: PointCloud | null
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

  // Actions
  actions: SceneActions
}

// ---------------------------------------------------------------------------
// Internal state (not exposed to React — no re-renders on mutation)
// ---------------------------------------------------------------------------

const internal = {
  parquetFiles: new Map<string, WaymoParquetFile>(),
  timestamps: [] as bigint[],
  lidarBoxByFrame: new Map<unknown, ParquetRow[]>(),
  vehiclePoseByFrame: new Map<unknown, ParquetRow[]>(),
  frameCache: new Map<number, FrameData>(),
  prefetchInFlight: new Set<number>(),
  playIntervalId: null as ReturnType<typeof setInterval> | null,
  worker: null as Worker | null,
  workerReady: false,
  pendingRequests: new Map<number, {
    resolve: (result: DataWorkerFrameResult) => void
    reject: (err: Error) => void
  }>(),
  nextRequestId: 0,
  CACHE_SIZE: 10,
  PREFETCH_AHEAD: 3,
}

function resetInternal() {
  internal.parquetFiles.clear()
  internal.timestamps = []
  internal.lidarBoxByFrame.clear()
  internal.vehiclePoseByFrame.clear()
  internal.frameCache.clear()
  internal.prefetchInFlight.clear()
  if (internal.playIntervalId !== null) {
    clearInterval(internal.playIntervalId)
    internal.playIntervalId = null
  }
  if (internal.worker) {
    internal.worker.terminate()
    internal.worker = null
    internal.workerReady = false
  }
  internal.pendingRequests.clear()
  internal.nextRequestId = 0
}

// ---------------------------------------------------------------------------
// Worker communication
// ---------------------------------------------------------------------------

function postToWorker(msg: DataWorkerRequest) {
  internal.worker?.postMessage(msg)
}

function requestWorkerFrame(
  frameIndex: number,
  timestamp: bigint,
): Promise<DataWorkerFrameResult> {
  return new Promise((resolve, reject) => {
    const requestId = internal.nextRequestId++
    internal.pendingRequests.set(requestId, { resolve, reject })
    postToWorker({
      type: 'loadFrame',
      requestId,
      frameIndex,
      timestamp: timestamp.toString(),
    })
  })
}

function handleWorkerMessage(e: MessageEvent<DataWorkerResponse>) {
  const msg = e.data

  if (msg.type === 'frameReady' || msg.type === 'error') {
    const pending = internal.pendingRequests.get(msg.requestId ?? -1)
    if (pending) {
      internal.pendingRequests.delete(msg.requestId!)
      if (msg.type === 'error') {
        pending.reject(new Error(msg.message))
      } else {
        pending.resolve(msg)
      }
    }
  }
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

        // 4. Load first frame (via worker — non-blocking)
        await get().actions.loadFrame(0)

        set({ status: 'ready', loadProgress: 1 })
      } catch (e) {
        set({
          status: 'error',
          error: e instanceof Error ? e.message : String(e),
        })
      }
    },

    loadFrame: async (frameIndex) => {
      if (frameIndex < 0 || frameIndex >= internal.timestamps.length) return

      // Cache hit — instant
      const cached = internal.frameCache.get(frameIndex)
      if (cached) {
        set({
          currentFrameIndex: frameIndex,
          currentFrame: cached,
          lastFrameLoadMs: 0,
          lastConvertMs: cached.pointCloud ? get().lastConvertMs : 0,
        })
        // Still prefetch ahead from new position
        prefetchAhead(frameIndex, set)
        return
      }

      const timestamp = internal.timestamps[frameIndex]
      const t0 = performance.now()

      // Request from worker (non-blocking — main thread stays free)
      let pointCloud: PointCloud | null = null
      let convertMs = 0

      if (internal.workerReady) {
        const result = await requestWorkerFrame(frameIndex, timestamp)
        pointCloud = {
          positions: result.positions,
          pointCount: result.pointCount,
        }
        convertMs = result.convertMs
      }

      // Boxes + pose (already in memory from startup)
      const boxes = internal.lidarBoxByFrame.get(timestamp) ?? []
      const poseRows = internal.vehiclePoseByFrame.get(timestamp)
      const vehiclePose = (poseRows?.[0]?.['[VehiclePoseComponent].world_from_vehicle.transform'] as number[]) ?? null

      const frameData: FrameData = {
        timestamp,
        pointCloud,
        boxes,
        cameraImages: new Map(),
        vehiclePose,
      }

      // LRU cache
      evictIfNeeded()
      internal.frameCache.set(frameIndex, frameData)

      set({
        currentFrameIndex: frameIndex,
        currentFrame: frameData,
        lastFrameLoadMs: performance.now() - t0,
        lastConvertMs: convertMs,
      })

      syncCachedFrames(set)

      // Prefetch next N frames in background
      prefetchAhead(frameIndex, set)
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
      })
    },
  },
}))

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function evictIfNeeded() {
  if (internal.frameCache.size >= internal.CACHE_SIZE) {
    const oldest = internal.frameCache.keys().next().value!
    internal.frameCache.delete(oldest)
  }
}

async function loadStartupData(set: (partial: Partial<SceneState>) => void) {
  // Vehicle pose → master frame list
  const posePf = internal.parquetFiles.get('vehicle_pose')
  if (posePf) {
    const rows = await readAllRows(posePf)
    const index = buildFrameIndex(rows)
    internal.timestamps = index.timestamps
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
    // Worker only supports URL-based sources for now
    // File-based (drag & drop) will fall back to main-thread loading
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
        // Switch to persistent handler
        worker.onmessage = handleWorkerMessage
        resolve()
      } else if (e.data.type === 'error') {
        reject(new Error(e.data.message))
      }
    }

    worker.onerror = (e) => reject(new Error(e.message))

    internal.worker = worker

    // Serialize calibrations as entries (Map can't be postMessage'd)
    const calibEntries = [...get().lidarCalibrations.entries()]

    postToWorker({
      type: 'init',
      lidarUrl: lidarSource,
      calibrationEntries: calibEntries,
    })
  })
}

// ---------------------------------------------------------------------------
// Prefetching — load upcoming frames via worker in background
// ---------------------------------------------------------------------------

function prefetchAhead(
  currentIndex: number,
  set: (partial: Partial<SceneState>) => void,
) {
  if (!internal.workerReady) return

  const total = internal.timestamps.length
  for (let offset = 1; offset <= internal.PREFETCH_AHEAD; offset++) {
    const idx = currentIndex + offset
    if (idx >= total) break
    if (internal.frameCache.has(idx)) continue
    if (internal.prefetchInFlight.has(idx)) continue

    internal.prefetchInFlight.add(idx)
    prefetchFrame(idx, set).finally(() => {
      internal.prefetchInFlight.delete(idx)
    })
  }
}

async function prefetchFrame(
  frameIndex: number,
  set: (partial: Partial<SceneState>) => void,
) {
  if (internal.frameCache.has(frameIndex)) return

  const timestamp = internal.timestamps[frameIndex]
  if (timestamp === undefined) return

  try {
    const result = await requestWorkerFrame(frameIndex, timestamp)

    const pointCloud: PointCloud = {
      positions: result.positions,
      pointCount: result.pointCount,
    }

    // Boxes + pose (already in memory)
    const boxes = internal.lidarBoxByFrame.get(timestamp) ?? []
    const poseRows = internal.vehiclePoseByFrame.get(timestamp)
    const vehiclePose = (poseRows?.[0]?.['[VehiclePoseComponent].world_from_vehicle.transform'] as number[]) ?? null

    const frameData: FrameData = {
      timestamp,
      pointCloud,
      boxes,
      cameraImages: new Map(),
      vehiclePose,
    }

    evictIfNeeded()
    internal.frameCache.set(frameIndex, frameData)

    // Update UI buffer bar
    syncCachedFrames(set)
  } catch {
    // Prefetch failure is non-critical — silently ignore
  }
}

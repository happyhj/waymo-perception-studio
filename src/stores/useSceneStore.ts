/**
 * Scene store — Zustand-based central state for Waymo Perception Studio.
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
  buildHeavyFileFrameIndex,
  readFrameData,
  type WaymoParquetFile,
  type FrameRowIndex,
} from '../utils/parquet'
import {
  parseLidarCalibration,
  convertAllSensors,
  type LidarCalibration,
  type RangeImage,
  type PointCloud,
} from '../utils/rangeImage'

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

  // Actions
  actions: SceneActions
}

// ---------------------------------------------------------------------------
// Internal state (not exposed to React — no re-renders on mutation)
// ---------------------------------------------------------------------------

const internal = {
  parquetFiles: new Map<string, WaymoParquetFile>(),
  heavyFileIndices: new Map<string, FrameRowIndex>(),
  timestamps: [] as bigint[],
  lidarBoxByFrame: new Map<unknown, ParquetRow[]>(),
  vehiclePoseByFrame: new Map<unknown, ParquetRow[]>(),
  frameCache: new Map<number, FrameData>(),
  playIntervalId: null as ReturnType<typeof setInterval> | null,
  CACHE_SIZE: 10,
}

function resetInternal() {
  internal.parquetFiles.clear()
  internal.heavyFileIndices.clear()
  internal.timestamps = []
  internal.lidarBoxByFrame.clear()
  internal.vehiclePoseByFrame.clear()
  internal.frameCache.clear()
  if (internal.playIntervalId !== null) {
    clearInterval(internal.playIntervalId)
    internal.playIntervalId = null
  }
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

  actions: {
    loadDataset: async (sources) => {
      resetInternal()
      set({
        status: 'loading',
        availableComponents: [...sources.keys()],
        error: null,
        loadProgress: 0,
      })

      try {
        const totalSteps = sources.size + 2
        let completed = 0

        // 1. Open all Parquet files (footer only)
        for (const [component, source] of sources) {
          const pf = await openParquetFile(component, source)
          internal.parquetFiles.set(component, pf)
          completed++
          set({ loadProgress: completed / totalSteps })
        }

        // 2. Load startup data
        await loadStartupData(set)
        completed++
        set({ loadProgress: completed / totalSteps })

        // 3. Build heavy file indices
        await buildHeavyIndices()
        completed++
        set({ loadProgress: completed / totalSteps })

        // 4. Load first frame
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

      // Cache hit
      const cached = internal.frameCache.get(frameIndex)
      if (cached) {
        set({ currentFrameIndex: frameIndex, currentFrame: cached })
        return
      }

      const timestamp = internal.timestamps[frameIndex]
      const t0 = performance.now()

      // LiDAR → point cloud
      let pointCloud: PointCloud | null = null
      let convertMs = 0

      const lidarPf = internal.parquetFiles.get('lidar')
      const lidarIndex = internal.heavyFileIndices.get('lidar')
      if (lidarPf && lidarIndex) {
        const lidarRows = await readFrameData(lidarPf, lidarIndex, timestamp, [
          'key.laser_name',
          '[LiDARComponent].range_image_return1.shape',
          '[LiDARComponent].range_image_return1.values',
        ])

        const rangeImages = new Map<number, RangeImage>()
        for (const row of lidarRows) {
          const laserName = row['key.laser_name'] as number
          rangeImages.set(laserName, {
            shape: row['[LiDARComponent].range_image_return1.shape'] as [number, number, number],
            values: row['[LiDARComponent].range_image_return1.values'] as number[],
          })
        }

        const ct0 = performance.now()
        pointCloud = convertAllSensors(rangeImages, get().lidarCalibrations)
        convertMs = performance.now() - ct0
      }

      // Boxes + pose for this frame
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
      if (internal.frameCache.size >= internal.CACHE_SIZE) {
        const oldest = internal.frameCache.keys().next().value!
        internal.frameCache.delete(oldest)
      }
      internal.frameCache.set(frameIndex, frameData)

      set({
        currentFrameIndex: frameIndex,
        currentFrame: frameData,
        lastFrameLoadMs: performance.now() - t0,
        lastConvertMs: convertMs,
      })
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
      })
    },
  },
}))

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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

async function buildHeavyIndices() {
  for (const component of ['lidar', 'camera_image', 'lidar_camera_projection', 'lidar_pose']) {
    const pf = internal.parquetFiles.get(component)
    if (pf) {
      internal.heavyFileIndices.set(component, await buildHeavyFileFrameIndex(pf))
    }
  }
}

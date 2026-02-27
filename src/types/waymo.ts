/**
 * Waymo Open Dataset v2.0 type definitions
 * Reference: https://waymo.com/open/data/perception/
 */

/** LiDAR sensor names in Waymo dataset */
export const LidarName = {
  TOP: 1,
  FRONT: 2,
  SIDE_LEFT: 3,
  SIDE_RIGHT: 4,
  REAR: 5,
} as const
export type LidarName = (typeof LidarName)[keyof typeof LidarName]

/** Camera sensor names in Waymo dataset */
export const CameraName = {
  FRONT: 1,
  FRONT_LEFT: 2,
  FRONT_RIGHT: 3,
  SIDE_LEFT: 4,
  SIDE_RIGHT: 5,
} as const
export type CameraName = (typeof CameraName)[keyof typeof CameraName]

/** 3D bounding box label types */
export const BoxType = {
  TYPE_UNKNOWN: 0,
  TYPE_VEHICLE: 1,
  TYPE_PEDESTRIAN: 2,
  TYPE_SIGN: 3,
  TYPE_CYCLIST: 4,
} as const
export type BoxType = (typeof BoxType)[keyof typeof BoxType]

/** A single LiDAR point */
export interface LidarPoint {
  x: number
  y: number
  z: number
  intensity: number
  elongation: number
  lidarName: LidarName
}

/** 3D bounding box */
export interface Box3D {
  centerX: number
  centerY: number
  centerZ: number
  length: number
  width: number
  height: number
  heading: number
  type: BoxType
  id: string
}

/** Vehicle pose (ego vehicle transform) */
export interface VehiclePose {
  worldFromVehicle: number[] // 4x4 matrix, row-major
}

/** Camera calibration parameters */
export interface CameraCalibration {
  name: CameraName
  intrinsic: number[] // [fx, fy, cx, cy, k1, k2, p1, p2, k3]
  extrinsic: number[] // 4x4 matrix, row-major
  width: number
  height: number
}

/** LiDAR calibration parameters */
export interface LidarCalibration {
  name: LidarName
  extrinsic: number[] // 4x4 matrix, row-major
}

/** A single frame of data */
export interface Frame {
  frameIndex: number
  timestamp: number
  vehiclePose: VehiclePose
  lidarPoints: Float32Array // interleaved [x,y,z,intensity, x,y,z,intensity, ...]
  lidarPointCount: number
  boxes3D: Box3D[]
  cameraImages?: Map<CameraName, string> // camera name -> image data URL
}

/** Segment metadata */
export interface SegmentInfo {
  segmentId: string
  frameCount: number
  cameraCalibrations: CameraCalibration[]
  lidarCalibrations: LidarCalibration[]
}

/** Application state for playback */
export interface PlaybackState {
  currentFrame: number
  totalFrames: number
  isPlaying: boolean
  playbackSpeed: number // 1x, 2x, 0.5x, etc.
  fps: number // dataset fps (10Hz for Waymo)
}

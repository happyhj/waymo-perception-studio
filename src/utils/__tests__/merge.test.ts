/**
 * Unit tests for merge.ts — JS port of Waymo v2.merge()
 *
 * Tests mirror the official Python SDK behavior:
 * - Auto-detect key columns via `key.` prefix
 * - Inner join on common keys
 * - groupBy to prevent cartesian products
 * - indexBy / groupIndexBy convenience functions
 */

import { describe, it, expect } from 'vitest'
import { merge, indexBy, groupIndexBy, type ParquetRow } from '../merge'

// ---------------------------------------------------------------------------
// Test data — mimics Waymo v2.0 table structure
// ---------------------------------------------------------------------------

/** vehicle_pose: 1 row per frame */
const poseRows: ParquetRow[] = [
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 1000n, '[VehiclePoseComponent].world_from_vehicle.transform': [1, 0, 0, 0] },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 2000n, '[VehiclePoseComponent].world_from_vehicle.transform': [0, 1, 0, 0] },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 3000n, '[VehiclePoseComponent].world_from_vehicle.transform': [0, 0, 1, 0] },
]

/** camera_image: 5 rows per frame (one per camera) */
const cameraRows: ParquetRow[] = [
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 1000n, 'key.camera_name': 1, '[CameraImageComponent].image': 'jpeg_f1_cam1' },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 1000n, 'key.camera_name': 2, '[CameraImageComponent].image': 'jpeg_f1_cam2' },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 2000n, 'key.camera_name': 1, '[CameraImageComponent].image': 'jpeg_f2_cam1' },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 2000n, 'key.camera_name': 2, '[CameraImageComponent].image': 'jpeg_f2_cam2' },
]

/** lidar_box: variable rows per frame (one per object) */
const boxRows: ParquetRow[] = [
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 1000n, 'key.laser_object_id': 'obj_a', '[LiDARBoxComponent].type': 1, '[LiDARBoxComponent].box.center.x': 10.0 },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 1000n, 'key.laser_object_id': 'obj_b', '[LiDARBoxComponent].type': 2, '[LiDARBoxComponent].box.center.x': 20.0 },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 1000n, 'key.laser_object_id': 'obj_c', '[LiDARBoxComponent].type': 1, '[LiDARBoxComponent].box.center.x': 30.0 },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 2000n, 'key.laser_object_id': 'obj_a', '[LiDARBoxComponent].type': 1, '[LiDARBoxComponent].box.center.x': 11.0 },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 2000n, 'key.laser_object_id': 'obj_b', '[LiDARBoxComponent].type': 2, '[LiDARBoxComponent].box.center.x': 21.0 },
  { 'key.segment_context_name': 'seg1', 'key.frame_timestamp_micros': 3000n, 'key.laser_object_id': 'obj_a', '[LiDARBoxComponent].type': 1, '[LiDARBoxComponent].box.center.x': 12.0 },
]

/** lidar_calibration: per-sensor, no frame timestamp */
const calibRows: ParquetRow[] = [
  { 'key.segment_context_name': 'seg1', 'key.laser_name': 1, '[LiDARCalibrationComponent].extrinsic.transform': [1, 0, 0, 0] },
  { 'key.segment_context_name': 'seg1', 'key.laser_name': 2, '[LiDARCalibrationComponent].extrinsic.transform': [0, 1, 0, 0] },
]

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('merge()', () => {
  it('joins pose with boxes on common keys (segment + timestamp)', () => {
    // pose has 2 key cols: segment_context_name, frame_timestamp_micros
    // boxes has 3 key cols: segment_context_name, frame_timestamp_micros, laser_object_id
    // Common: segment_context_name, frame_timestamp_micros
    const result = merge(poseRows, boxRows)

    // frame 1000 has 3 boxes, 2000 has 2, 3000 has 1 = 6 total
    expect(result).toHaveLength(6)

    // Each result row has both pose and box columns
    const first = result[0]
    expect(first).toHaveProperty('[VehiclePoseComponent].world_from_vehicle.transform')
    expect(first).toHaveProperty('[LiDARBoxComponent].box.center.x')
    expect(first).toHaveProperty('key.laser_object_id')
  })

  it('joins pose with camera on common keys (segment + timestamp)', () => {
    // pose: segment + timestamp
    // camera: segment + timestamp + camera_name
    // Common: segment + timestamp
    const result = merge(poseRows, cameraRows)

    // frame 1000: 2 cameras, frame 2000: 2 cameras = 4
    // frame 3000 has no camera rows → not in result (inner join)
    expect(result).toHaveLength(4)

    // Each result has both pose transform and camera image
    expect(result[0]).toHaveProperty('[VehiclePoseComponent].world_from_vehicle.transform')
    expect(result[0]).toHaveProperty('[CameraImageComponent].image')
  })

  it('inner join excludes frames with no match', () => {
    const result = merge(poseRows, cameraRows)

    // frame 3000 exists in pose but not in camera → excluded
    const timestamps = result.map((r) => r['key.frame_timestamp_micros'])
    expect(timestamps).not.toContain(3000n)
  })

  it('rightGroup aggregates right side into arrays', () => {
    // Merge pose with boxes, grouping boxes per frame
    const result = merge(poseRows, boxRows, { rightGroup: true })

    // Should have one row per frame that has boxes
    expect(result).toHaveLength(3)

    // Frame 1000 should have 3 boxes aggregated
    const frame1 = result.find((r) => r['key.frame_timestamp_micros'] === 1000n)!
    expect(frame1['[LiDARBoxComponent].box.center.x']).toEqual([10.0, 20.0, 30.0])
    expect(frame1['key.laser_object_id']).toEqual(['obj_a', 'obj_b', 'obj_c'])
  })

  it('leftGroup aggregates left side into arrays', () => {
    // Merge cameras (grouped by frame) with pose
    const result = merge(cameraRows, poseRows, { leftGroup: true })

    // 2 frames have camera data
    expect(result).toHaveLength(2)

    // Frame 1000 has 2 cameras grouped
    const frame1 = result.find((r) => r['key.frame_timestamp_micros'] === 1000n)!
    expect(frame1['[CameraImageComponent].image']).toEqual(['jpeg_f1_cam1', 'jpeg_f1_cam2'])
  })

  it('handles empty arrays', () => {
    expect(merge([], boxRows)).toEqual([])
    expect(merge(poseRows, [])).toEqual([])
    expect(merge([], [])).toEqual([])
  })

  it('warns and returns empty when no common keys', () => {
    const noKeyRows = [{ 'some_col': 123 }]
    const result = merge(noKeyRows, poseRows)
    expect(result).toEqual([])
  })

  it('joins calibration with pose on segment_context_name only', () => {
    // calib has: segment_context_name, laser_name (no timestamp)
    // pose has: segment_context_name, frame_timestamp_micros (no laser_name)
    // Common: segment_context_name only
    const result = merge(poseRows, calibRows)

    // 3 poses × 2 calibs = 6 (cartesian on the common key)
    expect(result).toHaveLength(6)
  })

  it('custom key prefix works', () => {
    const left = [{ 'custom.id': 1, value: 'a' }]
    const right = [{ 'custom.id': 1, other: 'b' }]
    const result = merge(left, right, { keyPrefix: 'custom.' })
    expect(result).toHaveLength(1)
    expect(result[0]).toEqual({ 'custom.id': 1, value: 'a', other: 'b' })
  })
})

describe('indexBy()', () => {
  it('creates O(1) lookup map by column', () => {
    const map = indexBy(poseRows, 'key.frame_timestamp_micros')

    expect(map.size).toBe(3)
    expect(map.get(1000n)).toBe(poseRows[0])
    expect(map.get(2000n)).toBe(poseRows[1])
    expect(map.get(3000n)).toBe(poseRows[2])
  })

  it('last row wins for duplicate keys', () => {
    const map = indexBy(boxRows, 'key.frame_timestamp_micros')
    // frame 1000 has 3 rows — last one (obj_c) wins
    expect(map.get(1000n)?.['key.laser_object_id']).toBe('obj_c')
  })
})

describe('groupIndexBy()', () => {
  it('groups rows by column into arrays', () => {
    const map = groupIndexBy(boxRows, 'key.frame_timestamp_micros')

    expect(map.size).toBe(3)
    expect(map.get(1000n)).toHaveLength(3) // 3 objects
    expect(map.get(2000n)).toHaveLength(2) // 2 objects
    expect(map.get(3000n)).toHaveLength(1) // 1 object
  })

  it('preserves row references', () => {
    const map = groupIndexBy(boxRows, 'key.frame_timestamp_micros')
    expect(map.get(1000n)![0]).toBe(boxRows[0])
  })
})

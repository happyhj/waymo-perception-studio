/**
 * PointCloud renderer — single draw call for ~168K points.
 *
 * Supports per-sensor visibility toggle via store's visibleSensors set.
 * When all sensors visible, uses pre-merged buffer (zero copy).
 * When some hidden, merges only visible sensor clouds on the fly.
 *
 * Color mode: turbo colormap based on intensity.
 */

import { useRef, useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { useSceneStore } from '../../stores/useSceneStore'
import type { PointCloud as PointCloudType } from '../../utils/rangeImage'

// ---------------------------------------------------------------------------
// Intensity colormap — cool-tinted white ramp
//
// Dark blue-gray → neutral mid → bright white.
// Avoids warm hues so perception box colors (orange, blue, crimson, magenta)
// pop clearly on top. Perceptually uniform luminance progression.
// ---------------------------------------------------------------------------

const INTENSITY_STOPS: [number, number, number][] = [
  [0.08, 0.09, 0.16],  // 0.0 — near-black (dark navy)
  [0.16, 0.20, 0.32],  // ~0.2 — dark slate
  [0.30, 0.38, 0.52],  // ~0.4 — cool gray
  [0.52, 0.60, 0.72],  // ~0.6 — silver blue
  [0.78, 0.84, 0.90],  // ~0.8 — light gray
  [0.95, 0.97, 1.00],  // 1.0 — near-white
]

function intensityColor(t: number): [number, number, number] {
  const tc = Math.max(0, Math.min(1, t))
  const idx = tc * (INTENSITY_STOPS.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, INTENSITY_STOPS.length - 1)
  const f = idx - lo
  return [
    INTENSITY_STOPS[lo][0] + f * (INTENSITY_STOPS[hi][0] - INTENSITY_STOPS[lo][0]),
    INTENSITY_STOPS[lo][1] + f * (INTENSITY_STOPS[hi][1] - INTENSITY_STOPS[lo][1]),
    INTENSITY_STOPS[lo][2] + f * (INTENSITY_STOPS[hi][2] - INTENSITY_STOPS[lo][2]),
  ]
}

// Sensor color map for per-sensor coloring mode (cool-tone Waymo palette)
const SENSOR_COLORS: Record<number, [number, number, number]> = {
  1: [0.0, 0.91, 0.62],  // TOP — teal (#00E89D)
  2: [0.0, 0.79, 0.86],  // FRONT — cyan (#00C9DB)
  3: [0.30, 0.66, 1.0],  // SIDE_LEFT — sky blue (#4DA8FF)
  4: [0.48, 0.44, 1.0],  // SIDE_RIGHT — indigo (#7B6FFF)
  5: [0.71, 0.56, 1.0],  // REAR — lavender (#B490FF)
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** Maximum points we'll ever allocate buffers for (avoids realloc). */
const MAX_POINTS = 200_000

/** All 5 sensor IDs */
const ALL_SENSORS = new Set([1, 2, 3, 4, 5])

export default function PointCloud() {
  const currentFrame = useSceneStore((s) => s.currentFrame)
  const visibleSensors = useSceneStore((s) => s.visibleSensors)
  const pointOpacity = useSceneStore((s) => s.pointOpacity)
  const geometryRef = useRef<THREE.BufferGeometry>(null)

  // Check if all sensors visible (fast path — use pre-merged buffer)
  const allVisible = visibleSensors.size === ALL_SENSORS.size

  // Pre-allocate position & color buffers once
  const { posAttr, colorAttr } = useMemo(() => {
    const pos = new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3)
    const col = new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3)
    pos.setUsage(THREE.DynamicDrawUsage)
    col.setUsage(THREE.DynamicDrawUsage)
    return { posAttr: pos, colorAttr: col }
  }, [])

  // Update geometry when pointCloud or visibleSensors changes
  useEffect(() => {
    const geom = geometryRef.current
    if (!geom || !currentFrame) {
      if (geom) geom.setDrawRange(0, 0)
      return
    }

    const posArr = posAttr.array as Float32Array
    const colArr = colorAttr.array as Float32Array

    if (allVisible) {
      // Fast path: use pre-merged buffer, turbo colormap
      const pc = currentFrame.pointCloud
      if (!pc || pc.pointCount === 0) {
        geom.setDrawRange(0, 0)
        return
      }
      const { positions, pointCount } = pc
      const count = Math.min(pointCount, MAX_POINTS)

      for (let i = 0; i < count; i++) {
        const src = i * 4
        const dst = i * 3
        posArr[dst] = positions[src]
        posArr[dst + 1] = positions[src + 1]
        posArr[dst + 2] = positions[src + 2]
        const [r, g, b] = intensityColor(positions[src + 3])
        colArr[dst] = r
        colArr[dst + 1] = g
        colArr[dst + 2] = b
      }

      posAttr.needsUpdate = true
      colorAttr.needsUpdate = true
      geom.setDrawRange(0, count)
      geom.computeBoundingSphere()
    } else {
      // Per-sensor path: merge visible sensors with sensor coloring
      const sensorClouds = currentFrame.sensorClouds
      if (!sensorClouds || sensorClouds.size === 0) {
        geom.setDrawRange(0, 0)
        return
      }

      let total = 0
      for (const [laserName, cloud] of sensorClouds) {
        if (!visibleSensors.has(laserName)) continue
        const count = Math.min(cloud.pointCount, MAX_POINTS - total)
        const { positions } = cloud

        for (let i = 0; i < count; i++) {
          const src = i * 4
          const dst = (total + i) * 3
          posArr[dst] = positions[src]
          posArr[dst + 1] = positions[src + 1]
          posArr[dst + 2] = positions[src + 2]

          // Sensor coloring
          const color = SENSOR_COLORS[laserName] ?? [0.5, 0.5, 0.5]
          colArr[dst] = color[0]
          colArr[dst + 1] = color[1]
          colArr[dst + 2] = color[2]
        }
        total += count
      }

      posAttr.needsUpdate = true
      colorAttr.needsUpdate = true
      geom.setDrawRange(0, total)
      geom.computeBoundingSphere()
    }
  }, [currentFrame, visibleSensors, allVisible, posAttr, colorAttr])

  return (
    <points frustumCulled={false}>
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute attach="attributes-position" {...posAttr} />
        <bufferAttribute attach="attributes-color" {...colorAttr} />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        sizeAttenuation
        vertexColors
        transparent
        opacity={pointOpacity}
        depthWrite={false}
      />
    </points>
  )
}

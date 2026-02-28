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
// Turbo colormap (simplified 8-stop LUT, linearly interpolated)
// ---------------------------------------------------------------------------

const TURBO_STOPS: [number, number, number][] = [
  [0.19, 0.07, 0.23],  // 0.0 — dark purple
  [0.11, 0.32, 0.76],  // ~0.14
  [0.06, 0.57, 0.87],  // ~0.29
  [0.17, 0.80, 0.64],  // ~0.43
  [0.49, 0.93, 0.36],  // ~0.57
  [0.80, 0.89, 0.17],  // ~0.71
  [0.97, 0.64, 0.10],  // ~0.86
  [0.90, 0.18, 0.15],  // 1.0 — red
]

function turboColor(t: number): [number, number, number] {
  const tc = Math.max(0, Math.min(1, t))
  const idx = tc * (TURBO_STOPS.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.min(lo + 1, TURBO_STOPS.length - 1)
  const f = idx - lo
  return [
    TURBO_STOPS[lo][0] + f * (TURBO_STOPS[hi][0] - TURBO_STOPS[lo][0]),
    TURBO_STOPS[lo][1] + f * (TURBO_STOPS[hi][1] - TURBO_STOPS[lo][1]),
    TURBO_STOPS[lo][2] + f * (TURBO_STOPS[hi][2] - TURBO_STOPS[lo][2]),
  ]
}

// Sensor color map for per-sensor coloring mode
const SENSOR_COLORS: Record<number, [number, number, number]> = {
  1: [1.0, 0.3, 0.3],   // TOP — red
  2: [0.3, 1.0, 0.3],   // FRONT — green
  3: [0.3, 0.5, 1.0],   // SIDE_LEFT — blue
  4: [1.0, 0.8, 0.2],   // SIDE_RIGHT — yellow
  5: [0.8, 0.3, 1.0],   // REAR — purple
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
        const [r, g, b] = turboColor(positions[src + 3])
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
        opacity={0.85}
        depthWrite={false}
      />
    </points>
  )
}

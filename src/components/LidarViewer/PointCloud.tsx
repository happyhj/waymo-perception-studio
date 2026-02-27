/**
 * PointCloud renderer — single draw call for ~168K points.
 *
 * Uses InterleavedBufferAttribute to read [x,y,z] directly from the
 * store's Float32Array ([x,y,z,intensity] stride-4 layout) without copying.
 * Intensity is mapped to a turbo-like colormap for visual clarity.
 */

import { useRef, useEffect, useMemo } from 'react'
import * as THREE from 'three'
import { useSceneStore } from '../../stores/useSceneStore'

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

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** Maximum points we'll ever allocate buffers for (avoids realloc). */
const MAX_POINTS = 200_000

export default function PointCloud() {
  const pointCloud = useSceneStore((s) => s.currentFrame?.pointCloud)
  const geometryRef = useRef<THREE.BufferGeometry>(null)

  // Pre-allocate position & color buffers once
  const { posAttr, colorAttr } = useMemo(() => {
    const pos = new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3)
    const col = new THREE.Float32BufferAttribute(new Float32Array(MAX_POINTS * 3), 3)
    pos.setUsage(THREE.DynamicDrawUsage)
    col.setUsage(THREE.DynamicDrawUsage)
    return { posAttr: pos, colorAttr: col }
  }, [])

  // Update geometry when pointCloud changes
  useEffect(() => {
    const geom = geometryRef.current
    if (!geom) return

    if (!pointCloud || pointCloud.pointCount === 0) {
      geom.setDrawRange(0, 0)
      return
    }

    const { positions, pointCount } = pointCloud
    const count = Math.min(pointCount, MAX_POINTS)
    const posArr = posAttr.array as Float32Array
    const colArr = colorAttr.array as Float32Array

    // Extract xyz from stride-4 layout [x, y, z, intensity, x, y, z, intensity, ...]
    // and build color from intensity
    for (let i = 0; i < count; i++) {
      const src = i * 4
      const dst = i * 3
      posArr[dst] = positions[src]       // x
      posArr[dst + 1] = positions[src + 1] // y
      posArr[dst + 2] = positions[src + 2] // z

      // Intensity is in channel 3, typically 0–1 range (already normalized by conversion)
      const intensity = positions[src + 3]
      const [r, g, b] = turboColor(intensity)
      colArr[dst] = r
      colArr[dst + 1] = g
      colArr[dst + 2] = b
    }

    posAttr.needsUpdate = true
    colorAttr.needsUpdate = true
    geom.setDrawRange(0, count)
    geom.computeBoundingSphere()
  }, [pointCloud, posAttr, colorAttr])

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

/**
 * CameraFrustums — renders 5 Waymo camera frustums in the 3D scene.
 *
 * Each frustum is a wireframe pyramid showing the camera's field of view.
 * POV switching is triggered from the CameraPanel overlay buttons (not here).
 */

import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { useSceneStore } from '../../stores/useSceneStore'
import { parseCameraCalibrations, buildFrustumLines, type CameraCalib } from '../../utils/cameraCalibration'
import { CameraName } from '../../types/waymo'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FRUSTUM_NEAR = 0.3
const FRUSTUM_FAR = 6

const CAMERA_LABELS: Record<number, string> = {
  [CameraName.FRONT]: 'F',
  [CameraName.FRONT_LEFT]: 'FL',
  [CameraName.FRONT_RIGHT]: 'FR',
  [CameraName.SIDE_LEFT]: 'SL',
  [CameraName.SIDE_RIGHT]: 'SR',
}

const CAMERA_COLORS: Record<number, string> = {
  [CameraName.FRONT]: '#ffffff',
  [CameraName.FRONT_LEFT]: '#4ade80',
  [CameraName.FRONT_RIGHT]: '#60a5fa',
  [CameraName.SIDE_LEFT]: '#fb923c',
  [CameraName.SIDE_RIGHT]: '#c084fc',
}

// ---------------------------------------------------------------------------
// Single frustum
// ---------------------------------------------------------------------------

function CameraFrustum({
  calib,
  active,
}: {
  calib: CameraCalib
  active: boolean
}) {
  const groupRef = useRef<THREE.Group>(null)
  const color = CAMERA_COLORS[calib.cameraName] ?? '#888888'
  const label = CAMERA_LABELS[calib.cameraName] ?? '?'

  const linePositions = useMemo(
    () => buildFrustumLines(calib.hFov, calib.vFov, FRUSTUM_NEAR, FRUSTUM_FAR),
    [calib.hFov, calib.vFov],
  )

  return (
    <group
      ref={groupRef}
      position={calib.position}
      quaternion={calib.quaternion}
    >
      {/* Frustum wireframe lines */}
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            array={linePositions}
            count={linePositions.length / 3}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={color}
          transparent
          opacity={active ? 1.0 : 0.5}
          linewidth={1}
        />
      </lineSegments>

      {/* Camera position marker */}
      <mesh>
        <boxGeometry args={[0.15, 0.10, 0.10]} />
        <meshBasicMaterial color={color} transparent opacity={active ? 1.0 : 0.6} />
      </mesh>

      {/* Label */}
      <Html
        position={[0, -0.3, 0]}
        center
        style={{
          color,
          fontSize: '11px',
          fontWeight: 700,
          fontFamily: 'monospace',
          textShadow: '0 0 4px rgba(0,0,0,0.8)',
          pointerEvents: 'none',
          userSelect: 'none',
          whiteSpace: 'nowrap',
          opacity: active ? 1 : 0.7,
        }}
      >
        {label}
      </Html>
    </group>
  )
}

// ---------------------------------------------------------------------------
// All frustums
// ---------------------------------------------------------------------------

export default function CameraFrustums({
  activeCam,
}: {
  activeCam: number | null
}) {
  const cameraCalibrations = useSceneStore((s) => s.cameraCalibrations)

  const calibMap = useMemo(
    () => parseCameraCalibrations(cameraCalibrations),
    [cameraCalibrations],
  )

  if (calibMap.size === 0) return null

  return (
    <group>
      {[...calibMap.values()].map((calib) => (
        <CameraFrustum
          key={calib.cameraName}
          calib={calib}
          active={activeCam === calib.cameraName}
        />
      ))}
    </group>
  )
}

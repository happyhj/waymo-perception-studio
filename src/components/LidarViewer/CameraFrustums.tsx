/**
 * CameraFrustums — renders 5 Waymo camera frustums in the 3D scene.
 *
 * Each frustum is a wireframe pyramid showing the camera's field of view.
 * Frustums highlight when the corresponding camera image is hovered.
 */

import { useMemo } from 'react'
import * as THREE from 'three'
import { useSceneStore } from '../../stores/useSceneStore'
import { parseCameraCalibrations, buildFrustumLines, type CameraCalib } from '../../utils/cameraCalibration'
import { CameraName } from '../../types/waymo'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FRUSTUM_FAR = 6

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
  hovered,
}: {
  calib: CameraCalib
  active: boolean
  hovered: boolean
}) {
  const color = CAMERA_COLORS[calib.cameraName] ?? '#888888'

  const linePositions = useMemo(
    () => buildFrustumLines(calib.hFov, calib.vFov, FRUSTUM_FAR),
    [calib.hFov, calib.vFov],
  )

  // Highlight: hovered or active → full opacity, else dim
  const lineOpacity = hovered ? 1.0 : active ? 1.0 : 0.25
  const lineColor = hovered ? '#ffffff' : color

  return (
    <group
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
          color={lineColor}
          transparent
          opacity={lineOpacity}
          linewidth={1}
        />
      </lineSegments>
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
  const hoveredCam = useSceneStore((s) => s.hoveredCam)

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
          hovered={hoveredCam === calib.cameraName}
        />
      ))}
    </group>
  )
}

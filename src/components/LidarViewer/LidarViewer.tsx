/**
 * LidarViewer — R3F Canvas wrapper for 3D point cloud visualization.
 *
 * Renders the Waymo LiDAR point cloud with OrbitControls.
 * Camera starts from a bird's-eye-ish angle looking down at the vehicle.
 * Waymo vehicle frame: X = forward, Y = left, Z = up.
 *
 * Includes camera frustum visualization — click a frustum to switch
 * to that camera's POV. Press ESC or click the button to return.
 */

import { useEffect, useRef, useMemo } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei'
import * as THREE from 'three'
import PointCloud from './PointCloud'
import BoundingBoxes from './BoundingBoxes'
import CameraFrustums from './CameraFrustums'
import { useSceneStore } from '../../stores/useSceneStore'
import { parseCameraCalibrations, type CameraCalib } from '../../utils/cameraCalibration'

const SENSOR_INFO: { id: number; label: string; color: string }[] = [
  { id: 1, label: 'TOP', color: '#ff4d4d' },
  { id: 2, label: 'FRONT', color: '#4dff4d' },
  { id: 3, label: 'SIDE_L', color: '#4d80ff' },
  { id: 4, label: 'SIDE_R', color: '#ffcc33' },
  { id: 5, label: 'REAR', color: '#cc4dff' },
]

// ---------------------------------------------------------------------------
// POV Camera Controller — animates the camera to a Waymo camera's viewpoint
// ---------------------------------------------------------------------------

/** Lerp speed — higher = faster snap (0..1 per frame) */
const LERP_SPEED = 0.08

/**
 * Flip from optical camera convention (+Z forward, -Y up) to
 * Three.js camera convention (-Z forward, +Y up): 180° around X.
 */
const OPTICAL_TO_THREEJS_CAM = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  Math.PI,
)

/** Distance threshold to consider the return animation "done" */
const SNAP_THRESHOLD = 0.05

function PovController({
  targetCalib,
  orbitRef,
  returningRef,
}: {
  targetCalib: CameraCalib | null
  orbitRef: React.RefObject<any>
  /** Shared ref so parent can disable OrbitControls during return animation */
  returningRef: React.MutableRefObject<boolean>
}) {
  const { camera } = useThree()
  const savedState = useRef<{ pos: THREE.Vector3; fov: number; target: THREE.Vector3 } | null>(null)
  /** When non-null we're animating back to the saved orbital view */
  const returnTarget = useRef<{ pos: THREE.Vector3; fov: number; target: THREE.Vector3 } | null>(null)

  // Save orbital camera state when entering POV
  useEffect(() => {
    if (targetCalib && !savedState.current) {
      // Cancel any ongoing return animation
      returnTarget.current = null
      savedState.current = {
        pos: camera.position.clone(),
        fov: (camera as THREE.PerspectiveCamera).fov,
        target: orbitRef.current?.target?.clone() ?? new THREE.Vector3(),
      }
    }
  }, [targetCalib, camera, orbitRef])

  // Start return animation when leaving POV
  useEffect(() => {
    if (!targetCalib && savedState.current) {
      returnTarget.current = savedState.current
      returningRef.current = true
      savedState.current = null
    }
  }, [targetCalib, returningRef])

  // Animate: either toward POV target or back to orbital view
  useFrame(() => {
    const pc = camera as THREE.PerspectiveCamera

    // Keep OrbitControls disabled during POV & return animation
    if (orbitRef.current && (targetCalib || returnTarget.current)) {
      orbitRef.current.enabled = false
    }

    if (targetCalib) {
      // ---- Entering / holding POV ----
      camera.position.lerp(targetCalib.position, LERP_SPEED)

      const povQuat = targetCalib.quaternion.clone().multiply(OPTICAL_TO_THREEJS_CAM)
      camera.quaternion.slerp(povQuat, LERP_SPEED)

      const targetFov = THREE.MathUtils.radToDeg(targetCalib.vFov)
      pc.fov = THREE.MathUtils.lerp(pc.fov, targetFov, LERP_SPEED)
      pc.updateProjectionMatrix()
      return
    }

    if (returnTarget.current) {
      // ---- Animating back to orbital view ----
      const rt = returnTarget.current

      camera.position.lerp(rt.pos, LERP_SPEED)
      pc.fov = THREE.MathUtils.lerp(pc.fov, rt.fov, LERP_SPEED)
      pc.updateProjectionMatrix()

      // Compute an orientation that looks from the returning position toward the orbit target
      const lookAtMatrix = new THREE.Matrix4().lookAt(camera.position, rt.target, new THREE.Vector3(0, 0, 1))
      const targetQuat = new THREE.Quaternion().setFromRotationMatrix(lookAtMatrix)
      camera.quaternion.slerp(targetQuat, LERP_SPEED)

      // Check if close enough to snap and finish
      const dist = camera.position.distanceTo(rt.pos)
      if (dist < SNAP_THRESHOLD) {
        camera.position.copy(rt.pos)
        camera.quaternion.copy(targetQuat)
        pc.fov = rt.fov
        pc.updateProjectionMatrix()
        if (orbitRef.current) {
          orbitRef.current.target.copy(rt.target)
          orbitRef.current.update()
          orbitRef.current.enabled = true
        }
        returnTarget.current = null
        returningRef.current = false
      }
    }
  })

  return null
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function LidarViewer() {
  const visibleSensors = useSceneStore((s) => s.visibleSensors)
  const toggleSensor = useSceneStore((s) => s.actions.toggleSensor)
  const sensorClouds = useSceneStore((s) => s.currentFrame?.sensorClouds)
  const boxMode = useSceneStore((s) => s.boxMode)
  const cycleBoxMode = useSceneStore((s) => s.actions.cycleBoxMode)
  const trailLength = useSceneStore((s) => s.trailLength)
  const setTrailLength = useSceneStore((s) => s.actions.setTrailLength)
  const cameraCalibrations = useSceneStore((s) => s.cameraCalibrations)
  const activeCam = useSceneStore((s) => s.activeCam)
  const setActiveCam = useSceneStore((s) => s.actions.setActiveCam)
  const orbitRef = useRef<any>(null)
  const returningRef = useRef(false)

  // Parse calibrations once
  const calibMap = useMemo(
    () => parseCameraCalibrations(cameraCalibrations),
    [cameraCalibrations],
  )
  const activeCalib = activeCam !== null ? calibMap.get(activeCam) ?? null : null

  // ESC to exit POV
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && activeCam !== null) setActiveCam(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [activeCam, setActiveCam])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{
          position: [-30, -30, 40],
          fov: 60,
          near: 0.1,
          far: 500,
          up: [0, 0, 1],
        }}
        gl={{ antialias: false }}
        style={{ width: '100%', height: '100%' }}
        onCreated={({ gl }) => {
          gl.setClearColor('#1a1a2e')
        }}
      >
        <ambientLight intensity={0.3} />
        <directionalLight position={[50, -30, 80]} intensity={1.0} />
        <directionalLight position={[-30, 40, 20]} intensity={0.4} />
        <PointCloud />
        <BoundingBoxes />
        <CameraFrustums activeCam={activeCam} />

        {/* POV animation controller */}
        <PovController targetCalib={activeCalib} orbitRef={orbitRef} returningRef={returningRef} />

        {/* Ground grid (XY plane, Z=0) */}
        <gridHelper
          args={[200, 40, '#334155', '#1e293b']}
          rotation={[Math.PI / 2, 0, 0]}
        />

        {/* Vehicle origin marker */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial color="#e94560" />
        </mesh>

        <OrbitControls
          ref={orbitRef}
          makeDefault
          enableDamping
          dampingFactor={0.1}
          minDistance={5}
          maxDistance={200}
          /* enabled is controlled imperatively by PovController via orbitRef */
        />

        <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
          <GizmoViewport
            axisColors={['#e94560', '#22d3ee', '#a3e635']}
            labelColor="white"
          />
        </GizmoHelper>
      </Canvas>

      {/* Sensor toggle overlay */}
      <div style={{
        position: 'absolute',
        top: 8,
        left: 8,
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        pointerEvents: 'auto',
      }}>
        {SENSOR_INFO.map(({ id, label, color }) => {
          const active = visibleSensors.has(id)
          const cloud = sensorClouds?.get(id)
          const pts = cloud ? cloud.pointCount.toLocaleString() : '—'
          return (
            <button
              key={id}
              onClick={() => toggleSensor(id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '3px 8px',
                fontSize: '11px',
                fontFamily: 'monospace',
                border: 'none',
                borderRadius: '3px',
                cursor: 'pointer',
                backgroundColor: active ? 'rgba(22, 33, 62, 0.85)' : 'rgba(22, 33, 62, 0.4)',
                color: active ? '#e0e0e0' : '#666',
                opacity: active ? 1 : 0.5,
              }}
            >
              <span style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: active ? color : '#444',
                display: 'inline-block',
                flexShrink: 0,
              }} />
              {label}
              <span style={{ color: '#8892a0', marginLeft: 'auto', paddingLeft: 8 }}>{pts}</span>
            </button>
          )
        })}

        {/* Perception controls */}
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <button
            onClick={cycleBoxMode}
            style={{
              padding: '3px 8px',
              fontSize: '11px',
              fontFamily: 'monospace',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              backgroundColor: boxMode !== 'off' ? 'rgba(233, 69, 96, 0.7)' : 'rgba(22, 33, 62, 0.6)',
              color: boxMode !== 'off' ? '#fff' : '#8892a0',
            }}
          >
            {boxMode === 'off' ? 'DETECT OFF' : boxMode === 'box' ? 'DETECT BOX' : 'DETECT 3D'}
          </button>

          {boxMode !== 'off' && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '3px 8px',
              backgroundColor: 'rgba(22, 33, 62, 0.85)',
              borderRadius: '3px',
            }}>
              <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#8892a0', whiteSpace: 'nowrap' }}>
                TRAIL
              </span>
              <input
                type="range"
                min={0}
                max={100}
                value={trailLength}
                onChange={(e) => setTrailLength(Number(e.target.value))}
                style={{ width: 60, height: 2, accentColor: '#e94560' }}
              />
              <span style={{ fontSize: '10px', fontFamily: 'monospace', color: '#e0e0e0', minWidth: 16, textAlign: 'right' }}>
                {trailLength}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* POV mode indicator + exit button */}
      {activeCam !== null && (
        <div style={{
          position: 'absolute',
          top: 8,
          right: 8,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          pointerEvents: 'auto',
        }}>
          <span style={{
            fontSize: '12px',
            fontFamily: 'monospace',
            color: '#e0e0e0',
            backgroundColor: 'rgba(22, 33, 62, 0.85)',
            padding: '4px 10px',
            borderRadius: '3px',
          }}>
            CAM {['', 'FRONT', 'FL', 'FR', 'SL', 'SR'][activeCam] ?? activeCam}
          </span>
          <button
            onClick={() => setActiveCam(null)}
            style={{
              padding: '4px 10px',
              fontSize: '11px',
              fontFamily: 'monospace',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              backgroundColor: 'rgba(233, 69, 96, 0.8)',
              color: '#fff',
            }}
          >
            ESC
          </button>
        </div>
      )}
    </div>
  )
}

/**
 * LidarViewer — R3F Canvas wrapper for 3D point cloud visualization.
 *
 * Renders the Waymo LiDAR point cloud with OrbitControls.
 * Camera starts from a bird's-eye-ish angle looking down at the vehicle.
 * Waymo vehicle frame: X = forward, Y = left, Z = up.
 */

import { Canvas } from '@react-three/fiber'
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei'
import PointCloud from './PointCloud'
import BoundingBoxes from './BoundingBoxes'
import { useSceneStore } from '../../stores/useSceneStore'

const SENSOR_INFO: { id: number; label: string; color: string }[] = [
  { id: 1, label: 'TOP', color: '#ff4d4d' },
  { id: 2, label: 'FRONT', color: '#4dff4d' },
  { id: 3, label: 'SIDE_L', color: '#4d80ff' },
  { id: 4, label: 'SIDE_R', color: '#ffcc33' },
  { id: 5, label: 'REAR', color: '#cc4dff' },
]

export default function LidarViewer() {
  const visibleSensors = useSceneStore((s) => s.visibleSensors)
  const toggleSensor = useSceneStore((s) => s.actions.toggleSensor)
  const sensorClouds = useSceneStore((s) => s.currentFrame?.sensorClouds)
  const boxMode = useSceneStore((s) => s.boxMode)
  const cycleBoxMode = useSceneStore((s) => s.actions.cycleBoxMode)
  const trailLength = useSceneStore((s) => s.trailLength)
  const setTrailLength = useSceneStore((s) => s.actions.setTrailLength)

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{
          position: [-30, -30, 40],  // behind-left, elevated
          fov: 60,
          near: 0.1,
          far: 500,
          up: [0, 0, 1],  // Z-up (Waymo vehicle frame)
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
          makeDefault
          enableDamping
          dampingFactor={0.1}
          minDistance={5}
          maxDistance={200}
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

        {/* Perception: box mode toggle + trail slider */}
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
    </div>
  )
}

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

export default function LidarViewer() {
  return (
    <Canvas
      camera={{
        position: [-30, -30, 40],  // behind-left, elevated
        fov: 60,
        near: 0.1,
        far: 500,
        up: [0, 0, 1],  // Z-up (Waymo vehicle frame)
      }}
      gl={{ antialias: false }}  // points don't benefit from AA, save GPU
      style={{ width: '100%', height: '100%' }}
      onCreated={({ gl }) => {
        gl.setClearColor('#1a1a2e')
      }}
    >
      {/* Ambient light for potential future meshes */}
      <ambientLight intensity={0.5} />

      {/* Point cloud */}
      <PointCloud />

      {/* Ground grid (XY plane, Z=0) */}
      <gridHelper
        args={[200, 40, '#334155', '#1e293b']}
        rotation={[Math.PI / 2, 0, 0]}  // rotate from XZ to XY plane
      />

      {/* Vehicle origin marker */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshBasicMaterial color="#e94560" />
      </mesh>

      {/* Orbit controls */}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.1}
        minDistance={5}
        maxDistance={200}
      />

      {/* Orientation gizmo */}
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport
          axisColors={['#e94560', '#22d3ee', '#a3e635']}
          labelColor="white"
        />
      </GizmoHelper>
    </Canvas>
  )
}

import { useEffect, useCallback, useMemo } from 'react'
import { useSceneStore } from './stores/useSceneStore'
import LidarViewer from './components/LidarViewer/LidarViewer'
import CameraPanel from './components/CameraPanel/CameraPanel'


// ---------------------------------------------------------------------------
// Dev auto-load: known segment served by Vite plugin (serveWaymoData)
// ---------------------------------------------------------------------------

const DEV_SEGMENT = '10023947602400723454_1120_000_1140_000'
const DEV_COMPONENTS = [
  'vehicle_pose',
  'lidar_calibration',
  'camera_calibration',
  'lidar_box',
  'lidar',
  'camera_image',
]

function useDevAutoLoad() {
  const status = useSceneStore((s) => s.status)
  const loadDataset = useSceneStore((s) => s.actions.loadDataset)

  useEffect(() => {
    if (status !== 'idle') return
    if (!import.meta.env.DEV) return

    // Build URL map for each component
    const sources = new Map<string, string>()
    for (const comp of DEV_COMPONENTS) {
      sources.set(comp, `/waymo_data/${comp}/${DEV_SEGMENT}.parquet`)
    }
    loadDataset(sources as Map<string, File | string>)
  }, [status, loadDataset])
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  useDevAutoLoad()

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#1a1a2e',
      color: '#e0e0e0',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <SensorView />
      </main>

      {/* Timeline */}
      <footer style={{
        padding: '8px 16px',
        backgroundColor: '#16213e',
        borderTop: '1px solid #0f3460',
        flexShrink: 0,
      }}>
        <Timeline />
      </footer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

function Header() {
  const status = useSceneStore((s) => s.status)
  const totalFrames = useSceneStore((s) => s.totalFrames)
  const loadProgress = useSceneStore((s) => s.loadProgress)
  const cachedFrames = useSceneStore((s) => s.cachedFrames)
  const cameraLoadedCount = useSceneStore((s) => s.cameraLoadedCount)
  const cameraTotalCount = useSceneStore((s) => s.cameraTotalCount)

  let statusText: string
  if (status === 'idle') {
    statusText = 'No segment loaded'
  } else if (status === 'loading') {
    statusText = `Loading… ${Math.round(loadProgress * 100)}%`
  } else if (status === 'error') {
    statusText = 'Error'
  } else {
    // status === 'ready' — show prefetch progress
    const lidarDone = cachedFrames.length >= totalFrames
    const cameraDone = cameraTotalCount > 0 && cameraLoadedCount >= cameraTotalCount
    if (lidarDone && cameraDone) {
      statusText = `${totalFrames} frames`
    } else {
      const parts: string[] = []
      if (!lidarDone) parts.push(`LiDAR ${cachedFrames.length}/${totalFrames}`)
      if (!cameraDone && cameraTotalCount > 0) parts.push(`Camera ${cameraLoadedCount}/${cameraTotalCount}`)
      statusText = `Caching… ${parts.join(' · ')}`
    }
  }

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '8px 16px',
      backgroundColor: '#16213e',
      borderBottom: '1px solid #0f3460',
      flexShrink: 0,
    }}>
      <h1 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>
        Waymo Perception Studio
      </h1>
      <div style={{ fontSize: '12px', opacity: 0.6 }}>
        {statusText}
      </div>
    </header>
  )
}

// ---------------------------------------------------------------------------
// Main Views
// ---------------------------------------------------------------------------

function SensorView() {
  const status = useSceneStore((s) => s.status)

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* LiDAR 3D View — main area */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0 }}>
          {status === 'ready' ? (
            <LidarViewer />
          ) : status === 'loading' ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', backgroundColor: '#0f3460', opacity: 0.5 }}>
              Loading LiDAR data…
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', backgroundColor: '#0f3460', opacity: 0.5 }}>
              3D LiDAR View
            </div>
          )}
        </div>
      </div>

      {/* Camera Image Strip — bottom */}
      {status === 'ready' && <CameraPanel />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

function Timeline() {
  const status = useSceneStore((s) => s.status)
  const currentFrameIndex = useSceneStore((s) => s.currentFrameIndex)
  const totalFrames = useSceneStore((s) => s.totalFrames)
  const isPlaying = useSceneStore((s) => s.isPlaying)
  const cachedFrames = useSceneStore((s) => s.cachedFrames)
  const actions = useSceneStore((s) => s.actions)

  const disabled = status !== 'ready'
  const maxFrame = Math.max(totalFrames - 1, 0)

  // Clamp slider to the highest cached frame — prevent jumping to unloaded area
  const maxCached = cachedFrames.length > 0 ? cachedFrames[cachedFrames.length - 1] : 0

  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const target = parseInt(e.target.value, 10)
    if (target <= maxCached) {
      actions.seekFrame(target)
    }
  }, [actions, maxCached])

  // Compute buffer bar segments (continuous ranges of cached frames)
  const bufferSegments = useMemo(() => {
    if (totalFrames <= 1) return []
    const segments: { start: number; end: number }[] = []
    let segStart = -1
    for (let i = 0; i < cachedFrames.length; i++) {
      const f = cachedFrames[i]
      if (segStart === -1) {
        segStart = f
      }
      const next = cachedFrames[i + 1]
      if (next === undefined || next !== f + 1) {
        segments.push({ start: segStart, end: f })
        segStart = -1
      }
    }
    return segments
  }, [cachedFrames, totalFrames])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '13px' }}>
      {cachedFrames.length === 0 ? (
        <div style={{
          width: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '14px',
          color: '#8892a0',
        }}>
          ⏳
        </div>
      ) : (
        <button
          onClick={() => actions.togglePlayback()}
          disabled={disabled}
          style={{ background: 'none', border: 'none', color: disabled ? '#4a5568' : '#e0e0e0', cursor: disabled ? 'default' : 'pointer', fontSize: '16px', width: '24px' }}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
      )}

      {/* Custom slider with buffer bar */}
      <div style={{ flex: 1, position: 'relative', height: '20px', display: 'flex', alignItems: 'center' }}>
        {/* Track background */}
        <div style={{
          position: 'absolute',
          left: 0,
          right: 0,
          height: '4px',
          backgroundColor: '#1a2744',
          borderRadius: '2px',
          pointerEvents: 'none',
        }} />

        {/* Buffer segments — loaded frames */}
        {bufferSegments.map((seg, i) => {
          const left = (seg.start / maxFrame) * 100
          const width = ((seg.end - seg.start + 1) / maxFrame) * 100
          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: `${left}%`,
                width: `${width}%`,
                height: '4px',
                backgroundColor: 'rgba(233, 69, 96, 0.3)',
                borderRadius: '2px',
                pointerEvents: 'none',
              }}
            />
          )
        })}

        {/* Played progress (red bar) */}
        <div style={{
          position: 'absolute',
          left: 0,
          width: `${maxFrame > 0 ? (currentFrameIndex / maxFrame) * 100 : 0}%`,
          height: '4px',
          backgroundColor: '#e94560',
          borderRadius: '2px',
          pointerEvents: 'none',
        }} />

        {/* Invisible range input on top */}
        <input
          type="range"
          min={0}
          max={maxCached}
          value={currentFrameIndex}
          onChange={handleSliderChange}
          disabled={disabled}
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            width: '100%',
            height: '20px',
            opacity: 0,
            cursor: disabled ? 'default' : 'pointer',
            margin: 0,
          }}
        />
      </div>

      <span style={{ opacity: 0.6, fontSize: '12px', minWidth: '60px', textAlign: 'center' }}>
        {currentFrameIndex} / {maxFrame}
      </span>
    </div>
  )
}

export default App

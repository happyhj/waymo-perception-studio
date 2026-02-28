import { useEffect, useCallback, useMemo } from 'react'
import { useSceneStore } from './stores/useSceneStore'
import LidarViewer from './components/LidarViewer/LidarViewer'
import CameraPanel from './components/CameraPanel/CameraPanel'


// ---------------------------------------------------------------------------
// Segment discovery: fetch available segments from Vite API, auto-load if 1
// ---------------------------------------------------------------------------

function useSegmentDiscovery() {
  const status = useSceneStore((s) => s.status)
  const availableSegments = useSceneStore((s) => s.availableSegments)
  const actions = useSceneStore((s) => s.actions)

  useEffect(() => {
    if (!import.meta.env.DEV) return
    if (availableSegments.length > 0) return // already discovered

    fetch('/api/segments')
      .then((r) => r.json())
      .then(({ segments }: { segments: string[] }) => {
        if (segments.length === 0) return
        actions.setAvailableSegments(segments)

        // Auto-load if only 1 segment available
        if (segments.length === 1) {
          actions.selectSegment(segments[0])
        }
      })
      .catch(() => {})
  }, [availableSegments.length, actions])
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

function App() {
  useSegmentDiscovery()
  const status = useSceneStore((s) => s.status)
  const togglePlayback = useSceneStore((s) => s.actions.togglePlayback)

  // Global spacebar → play/pause toggle
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'Space' && status === 'ready') {
        // Don't trigger if user is typing in an input/textarea
        const tag = (e.target as HTMLElement)?.tagName
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return
        e.preventDefault()
        togglePlayback()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [status, togglePlayback])

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
  const availableSegments = useSceneStore((s) => s.availableSegments)
  const currentSegment = useSceneStore((s) => s.currentSegment)
  const actions = useSceneStore((s) => s.actions)

  let statusText: string
  if (status === 'idle') {
    statusText = availableSegments.length > 1 ? 'Select a segment' : 'No segment loaded'
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
      gap: '12px',
    }}>
      <h1 style={{ margin: 0, fontSize: '16px', fontWeight: 600, whiteSpace: 'nowrap' }}>
        Waymo Perception Studio
      </h1>

      {/* Segment selector — only shown when multiple segments available */}
      {availableSegments.length > 1 && (
        <select
          value={currentSegment ?? ''}
          onChange={(e) => {
            if (e.target.value) actions.selectSegment(e.target.value)
          }}
          disabled={status === 'loading'}
          style={{
            flex: '0 1 auto',
            minWidth: 0,
            maxWidth: '360px',
            padding: '4px 8px',
            fontSize: '12px',
            fontFamily: 'monospace',
            backgroundColor: '#0f3460',
            color: '#e0e0e0',
            border: '1px solid #1a4a8a',
            borderRadius: '4px',
            cursor: status === 'loading' ? 'not-allowed' : 'pointer',
            opacity: status === 'loading' ? 0.5 : 1,
          }}
        >
          <option value="">-- select segment --</option>
          {availableSegments.map((seg) => (
            <option key={seg} value={seg}>{seg}</option>
          ))}
        </select>
      )}

      <div style={{ fontSize: '12px', opacity: 0.6, whiteSpace: 'nowrap' }}>
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

import { useEffect, useCallback, useMemo } from 'react'
import { useSceneStore } from './stores/useSceneStore'
import LidarViewer from './components/LidarViewer/LidarViewer'
import CameraPanel from './components/CameraPanel/CameraPanel'
import { colors, fonts, radius, gradients } from './theme'


// ---------------------------------------------------------------------------
// Segment discovery: fetch available segments from Vite API, auto-load if 1
// ---------------------------------------------------------------------------

function useSegmentDiscovery() {
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
      backgroundColor: colors.bgBase,
      color: colors.textPrimary,
      fontFamily: fonts.sans,
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
        padding: '10px 20px',
        background: colors.bgSurface,
        borderTop: `1px solid ${colors.border}`,
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
      padding: '12px 24px',
      background: colors.bgSurface,
      borderBottom: `1px solid ${colors.border}`,
      flexShrink: 0,
      gap: '16px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <h1 style={{
          margin: 0,
          fontSize: '15px',
          fontWeight: 600,
          fontFamily: fonts.sans,
          letterSpacing: '-0.01em',
          color: colors.textPrimary,
        }}>
          Perception Studio <span style={{ fontWeight: 400, opacity: 0.5, fontSize: '12px' }}>for Waymo Open Dataset</span>
        </h1>
      </div>

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
            padding: '6px 12px',
            fontSize: '12px',
            fontFamily: fonts.mono,
            backgroundColor: colors.bgOverlay,
            color: colors.textPrimary,
            border: `1px solid ${colors.border}`,
            borderRadius: radius.md,
            cursor: status === 'loading' ? 'not-allowed' : 'pointer',
            opacity: status === 'loading' ? 0.5 : 1,
            outline: 'none',
            boxShadow: `0 0 0 0px ${colors.accentGlow}`,
            transition: 'box-shadow 0.2s, border-color 0.2s',
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = colors.accent
            e.currentTarget.style.boxShadow = `0 0 8px ${colors.accentGlow}`
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = colors.border
            e.currentTarget.style.boxShadow = 'none'
          }}
        >
          <option value="">-- select segment --</option>
          {availableSegments.map((seg) => (
            <option key={seg} value={seg}>{seg}</option>
          ))}
        </select>
      )}

      <div style={{
        fontSize: '12px',
        fontFamily: fonts.mono,
        color: colors.textSecondary,
        whiteSpace: 'nowrap',
      }}>
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
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              backgroundColor: colors.bgDeep,
              color: colors.textSecondary,
              fontFamily: fonts.sans,
              fontSize: '14px',
            }}>
              Loading LiDAR data…
            </div>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              backgroundColor: colors.bgDeep,
              color: colors.textDim,
              fontFamily: fonts.sans,
              fontSize: '14px',
            }}>
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
    <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '13px' }}>
      {cachedFrames.length === 0 ? (
        <div style={{
          width: '28px',
          height: '28px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '14px',
          color: colors.textDim,
        }}>
          ⏳
        </div>
      ) : (
        <button
          onClick={() => actions.togglePlayback()}
          disabled={disabled}
          style={{
            width: '28px',
            height: '28px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'none',
            border: 'none',
            color: disabled ? colors.textDim : colors.textPrimary,
            cursor: disabled ? 'default' : 'pointer',
            fontSize: '16px',
            borderRadius: radius.sm,
            transition: 'color 0.15s',
          }}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
      )}

      {/* Custom slider with buffer bar */}
      <div style={{ flex: 1, position: 'relative', height: '24px', display: 'flex', alignItems: 'center' }}>
        {/* Track background */}
        <div style={{
          position: 'absolute',
          left: 0,
          right: 0,
          height: '6px',
          backgroundColor: colors.bgOverlay,
          borderRadius: radius.pill,
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
                height: '6px',
                backgroundColor: colors.accentDim,
                borderRadius: radius.pill,
                pointerEvents: 'none',
              }}
            />
          )
        })}

        {/* Played progress (gradient bar) */}
        <div style={{
          position: 'absolute',
          left: 0,
          width: `${maxFrame > 0 ? (currentFrameIndex / maxFrame) * 100 : 0}%`,
          height: '6px',
          background: gradients.accent,
          borderRadius: radius.pill,
          pointerEvents: 'none',
          boxShadow: `0 0 8px ${colors.accentGlow}`,
        }} />

        {/* Playhead dot */}
        {maxFrame > 0 && (
          <div style={{
            position: 'absolute',
            left: `${(currentFrameIndex / maxFrame) * 100}%`,
            top: '50%',
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            backgroundColor: colors.accent,
            transform: 'translate(-50%, -50%)',
            boxShadow: `0 0 6px ${colors.accentDim}`,
            pointerEvents: 'none',
            transition: 'left 0.05s linear',
          }} />
        )}

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
            height: '24px',
            opacity: 0,
            cursor: disabled ? 'default' : 'pointer',
            margin: 0,
          }}
        />
      </div>

      <span style={{
        fontFamily: fonts.mono,
        fontSize: '11px',
        color: colors.textSecondary,
        minWidth: '64px',
        textAlign: 'right',
      }}>
        {currentFrameIndex} / {maxFrame}
      </span>
    </div>
  )
}

export default App

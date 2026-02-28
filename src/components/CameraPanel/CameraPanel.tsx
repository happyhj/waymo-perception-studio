/**
 * CameraPanel — displays all 5 Waymo camera images in a horizontal strip.
 *
 * Layout: SIDE_LEFT | FRONT_LEFT | FRONT | FRONT_RIGHT | SIDE_RIGHT
 * The FRONT camera is slightly larger (primary view).
 *
 * Each camera image has a POV button overlay that switches the 3D view
 * to that camera's perspective.
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { useSceneStore } from '../../stores/useSceneStore'
import { CameraName } from '../../types/waymo'

// Camera display order: surround view left-to-right
const CAMERA_ORDER: { id: number; label: string }[] = [
  { id: CameraName.SIDE_LEFT, label: 'SIDE LEFT' },
  { id: CameraName.FRONT_LEFT, label: 'FRONT LEFT' },
  { id: CameraName.FRONT, label: 'FRONT' },
  { id: CameraName.FRONT_RIGHT, label: 'FRONT RIGHT' },
  { id: CameraName.SIDE_RIGHT, label: 'SIDE RIGHT' },
]

/** Height of the camera strip in pixels */
const STRIP_HEIGHT = 160

const CAMERA_COLORS: Record<number, string> = {
  [CameraName.FRONT]: '#ffffff',
  [CameraName.FRONT_LEFT]: '#4ade80',
  [CameraName.FRONT_RIGHT]: '#60a5fa',
  [CameraName.SIDE_LEFT]: '#fb923c',
  [CameraName.SIDE_RIGHT]: '#c084fc',
}

export default function CameraPanel() {
  const cameraImages = useSceneStore((s) => s.currentFrame?.cameraImages)
  const activeCam = useSceneStore((s) => s.activeCam)
  const toggleActiveCam = useSceneStore((s) => s.actions.toggleActiveCam)

  return (
    <div style={{
      height: STRIP_HEIGHT,
      flexShrink: 0,
      display: 'flex',
      gap: '2px',
      padding: '4px',
      backgroundColor: '#0d1b2a',
      borderTop: '1px solid #1b2838',
      overflow: 'hidden',
    }}>
      {CAMERA_ORDER.map(({ id, label }) => (
        <CameraView
          key={id}
          cameraName={id}
          label={label}
          imageBuffer={cameraImages?.get(id) ?? null}
          active={activeCam === id}
          onTogglePov={toggleActiveCam}
        />
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Single camera view
// ---------------------------------------------------------------------------

interface CameraViewProps {
  cameraName: number
  label: string
  imageBuffer: ArrayBuffer | null
  active: boolean
  onTogglePov: (cameraName: number) => void
}

function CameraView({ cameraName, label, imageBuffer, active, onTogglePov }: CameraViewProps) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null)
  const prevUrlRef = useRef<string | null>(null)
  const [hovered, setHovered] = useState(false)

  useEffect(() => {
    // Revoke previous URL
    if (prevUrlRef.current) {
      URL.revokeObjectURL(prevUrlRef.current)
      prevUrlRef.current = null
    }

    if (!imageBuffer) {
      setObjectUrl(null)
      return
    }

    // Create blob URL from JPEG ArrayBuffer
    const blob = new Blob([imageBuffer], { type: 'image/jpeg' })
    const url = URL.createObjectURL(blob)
    prevUrlRef.current = url
    setObjectUrl(url)

    return () => {
      URL.revokeObjectURL(url)
      prevUrlRef.current = null
    }
  }, [imageBuffer])

  const handleClick = useCallback(() => {
    onTogglePov(cameraName)
  }, [onTogglePov, cameraName])

  // FRONT camera gets slightly more space
  const isFront = cameraName === CameraName.FRONT
  const flex = isFront ? 1.3 : 1
  const accentColor = CAMERA_COLORS[cameraName] ?? '#888'

  return (
    <div
      style={{
        flex,
        position: 'relative',
        backgroundColor: '#111827',
        borderRadius: '4px',
        overflow: 'hidden',
        minWidth: 0,
        cursor: 'pointer',
        border: active ? `2px solid ${accentColor}` : '2px solid transparent',
        transition: 'border-color 0.15s',
      }}
      onClick={handleClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {objectUrl ? (
        <img
          src={objectUrl}
          alt={label}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: 'block',
          }}
        />
      ) : (
        <div style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#4a5568',
          fontSize: '11px',
        }}>
          No image
        </div>
      )}

      {/* POV indicator / hover overlay */}
      {(hovered || active) && (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: active ? 'rgba(0, 0, 0, 0.25)' : 'rgba(0, 0, 0, 0.35)',
          transition: 'background-color 0.15s',
        }}>
          <span style={{
            fontSize: '11px',
            fontFamily: 'monospace',
            fontWeight: 700,
            color: active ? accentColor : '#fff',
            backgroundColor: 'rgba(0, 0, 0, 0.6)',
            padding: '3px 10px',
            borderRadius: '3px',
            letterSpacing: '0.5px',
          }}>
            {active ? 'EXIT POV' : 'POV'}
          </span>
        </div>
      )}

      {/* Label overlay */}
      <div style={{
        position: 'absolute',
        bottom: 4,
        left: 6,
        fontSize: '10px',
        fontFamily: 'monospace',
        color: 'rgba(255, 255, 255, 0.7)',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        padding: '1px 4px',
        borderRadius: '2px',
        pointerEvents: 'none',
      }}>
        {label}
      </div>

      {/* Active dot indicator */}
      {active && (
        <div style={{
          position: 'absolute',
          top: 6,
          right: 6,
          width: 8,
          height: 8,
          borderRadius: '50%',
          backgroundColor: accentColor,
          boxShadow: `0 0 6px ${accentColor}`,
        }} />
      )}
    </div>
  )
}

/**
 * CameraPanel — displays all 5 Waymo camera images in a horizontal strip.
 *
 * Layout: FRONT_LEFT | FRONT | FRONT_RIGHT | SIDE_LEFT | SIDE_RIGHT
 * The FRONT camera is slightly larger (primary view).
 *
 * Images come as JPEG ArrayBuffers from the camera worker.
 * We create object URLs for efficient rendering and revoke them on cleanup.
 */

import { useEffect, useRef, useState } from 'react'
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

export default function CameraPanel() {
  const cameraImages = useSceneStore((s) => s.currentFrame?.cameraImages)

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
}

function CameraView({ cameraName, label, imageBuffer }: CameraViewProps) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null)
  const prevUrlRef = useRef<string | null>(null)

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

  // FRONT camera gets slightly more space
  const isFront = cameraName === CameraName.FRONT
  const flex = isFront ? 1.3 : 1

  return (
    <div style={{
      flex,
      position: 'relative',
      backgroundColor: '#111827',
      borderRadius: '4px',
      overflow: 'hidden',
      minWidth: 0,
    }}>
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
    </div>
  )
}

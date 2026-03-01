/**
 * BoundingBoxes — renders perception objects from lidar_box data in two modes:
 *
 *   box   → semi-transparent solid boxes/cylinders + edge outlines (class-colored)
 *   model → low-poly 3D models: sedan, humanoid, cyclist, sign (class-colored)
 *
 * Plus trajectory trails showing each tracked object's past positions.
 *
 * Waymo coordinate frame: X=forward, Y=left, Z=up.
 */

import { useMemo, useCallback } from 'react'
import * as THREE from 'three'
import { useSceneStore, getObjectTrajectories, hasLaserAssociation } from '../../stores/useSceneStore'
import { BoxType, BOX_TYPE_COLORS } from '../../types/waymo'
import { VehicleModel, PedestrianModel, CyclistModel, SignModel } from './ObjectModels'
import type { ParquetRow } from '../../utils/merge'

// ---------------------------------------------------------------------------
// Parsed box data
// ---------------------------------------------------------------------------

interface ParsedBox {
  cx: number
  cy: number
  cz: number
  sx: number // length (X)
  sy: number // width (Y)
  sz: number // height (Z)
  heading: number
  type: number
  id: string
  /** Whether this box has a camera↔lidar association (pedestrian/cyclist only) */
  isAssociated: boolean
}

function parseBoxes(rows: ParquetRow[]): ParsedBox[] {
  const result: ParsedBox[] = []
  for (const row of rows) {
    const cx = row['[LiDARBoxComponent].box.center.x'] as number | undefined
    const cy = row['[LiDARBoxComponent].box.center.y'] as number | undefined
    const cz = row['[LiDARBoxComponent].box.center.z'] as number | undefined
    const sx = row['[LiDARBoxComponent].box.size.x'] as number | undefined
    const sy = row['[LiDARBoxComponent].box.size.y'] as number | undefined
    const sz = row['[LiDARBoxComponent].box.size.z'] as number | undefined
    const heading = row['[LiDARBoxComponent].box.heading'] as number | undefined
    const type = row['[LiDARBoxComponent].type'] as number | undefined
    const id = (row['key.laser_object_id'] as string) ?? ''

    if (cx == null || cy == null || cz == null) continue
    if (sx == null || sy == null || sz == null) continue

    result.push({
      cx, cy, cz,
      sx, sy, sz,
      heading: heading ?? 0,
      type: type ?? BoxType.TYPE_UNKNOWN,
      id,
      isAssociated: id ? hasLaserAssociation(id) : false,
    })
  }
  return result
}

// ---------------------------------------------------------------------------
// Shared geometries for "box" mode
// ---------------------------------------------------------------------------

const _unitBox = new THREE.BoxGeometry(1, 1, 1)
const _unitEdges = new THREE.EdgesGeometry(_unitBox)
const _unitCylinder = new THREE.CylinderGeometry(0.5, 0.5, 1, 12)
_unitCylinder.rotateX(Math.PI / 2) // Z-up

// ---------------------------------------------------------------------------
// "box" mode — semi-transparent solid + edge outline
// ---------------------------------------------------------------------------

/** Highlight colors for cross-modal hover */
const HIGHLIGHT_SELF = '#FFFF00'   // yellow — the box you're hovering
const HIGHLIGHT_LINKED = '#00FFAA' // green — linked counterpart

function BoxMesh({ box, highlighted, onHover }: {
  box: ParsedBox
  highlighted: 'self' | 'linked' | false
  onHover?: (id: string | null) => void
}) {
  const baseColor = BOX_TYPE_COLORS[box.type] ?? BOX_TYPE_COLORS[BoxType.TYPE_UNKNOWN]
  const color = highlighted === 'self' ? HIGHLIGHT_SELF : highlighted === 'linked' ? HIGHLIGHT_LINKED : baseColor
  const opacity = highlighted ? 0.5 : 0.25
  const isPedestrian = box.type === BoxType.TYPE_PEDESTRIAN
  const geometry = isPedestrian ? _unitCylinder : _unitBox

  const handlePointerEnter = useCallback((e: THREE.Event) => {
    if (onHover) {
      (e as unknown as { stopPropagation: () => void }).stopPropagation()
      onHover(box.id)
    }
  }, [onHover, box.id])

  const handlePointerLeave = useCallback(() => {
    if (onHover) onHover(null)
  }, [onHover])

  return (
    <group
      position={[box.cx, box.cy, box.cz]}
      rotation={[0, 0, box.heading]}
      onPointerEnter={box.isAssociated ? handlePointerEnter : undefined}
      onPointerLeave={box.isAssociated ? handlePointerLeave : undefined}
    >
      <mesh scale={[box.sx, box.sy, box.sz]} geometry={geometry}>
        <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
      </mesh>
      <lineSegments scale={[box.sx, box.sy, box.sz]} geometry={_unitEdges}>
        <lineBasicMaterial color={color} />
      </lineSegments>
    </group>
  )
}

// ---------------------------------------------------------------------------
// "model" mode — low-poly 3D models
// ---------------------------------------------------------------------------

const MODEL_OPACITY = 0.55

function ModelMesh({ box, highlighted, onHover }: {
  box: ParsedBox
  highlighted: 'self' | 'linked' | false
  onHover?: (id: string | null) => void
}) {
  const baseColor = BOX_TYPE_COLORS[box.type] ?? BOX_TYPE_COLORS[BoxType.TYPE_UNKNOWN]
  const color = highlighted === 'self' ? HIGHLIGHT_SELF : highlighted === 'linked' ? HIGHLIGHT_LINKED : baseColor
  const opacity = highlighted ? 0.8 : MODEL_OPACITY

  const handlePointerEnter = useCallback((e: THREE.Event) => {
    if (onHover) {
      (e as unknown as { stopPropagation: () => void }).stopPropagation()
      onHover(box.id)
    }
  }, [onHover, box.id])

  const handlePointerLeave = useCallback(() => {
    if (onHover) onHover(null)
  }, [onHover])

  let model: React.ReactNode
  switch (box.type) {
    case BoxType.TYPE_VEHICLE:
      model = <VehicleModel color={color} opacity={opacity} />
      break
    case BoxType.TYPE_PEDESTRIAN:
      model = <PedestrianModel color={color} opacity={opacity} />
      break
    case BoxType.TYPE_CYCLIST:
      model = <CyclistModel color={color} opacity={opacity} />
      break
    case BoxType.TYPE_SIGN:
      model = <SignModel color={color} opacity={opacity} />
      break
    default:
      model = <VehicleModel color={color} opacity={opacity} />
  }

  return (
    <group
      position={[box.cx, box.cy, box.cz]}
      rotation={[0, 0, box.heading]}
      scale={[box.sx, box.sy, box.sz]}
      onPointerEnter={box.isAssociated ? handlePointerEnter : undefined}
      onPointerLeave={box.isAssociated ? handlePointerLeave : undefined}
    >
      {model}
    </group>
  )
}

// ---------------------------------------------------------------------------
// Trajectory trail line
// ---------------------------------------------------------------------------

function TrajectoryTrail({ objectId, type, currentFrame, trailLength }: {
  objectId: string
  type: number
  currentFrame: number
  trailLength: number
}) {
  const color = BOX_TYPE_COLORS[type] ?? BOX_TYPE_COLORS[BoxType.TYPE_UNKNOWN]

  const geometry = useMemo(() => {
    if (trailLength <= 0) return null

    const trajectories = getObjectTrajectories()
    const trail = trajectories.get(objectId)
    if (!trail) return null

    let endIdx = trail.length
    for (let i = 0; i < trail.length; i++) {
      if (trail[i].frameIndex > currentFrame) {
        endIdx = i
        break
      }
    }
    const startIdx = Math.max(0, endIdx - trailLength)
    if (endIdx - startIdx < 2) return null

    const points: THREE.Vector3[] = []
    for (let i = startIdx; i < endIdx; i++) {
      const p = trail[i]
      points.push(new THREE.Vector3(p.x, p.y, p.z))
    }

    return new THREE.BufferGeometry().setFromPoints(points)
  }, [objectId, currentFrame, trailLength])

  if (!geometry) return null

  return (
    <line geometry={geometry}>
      {/* @ts-expect-error — R3F line material child */}
      <lineBasicMaterial color={color} transparent opacity={0.6} />
    </line>
  )
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export default function BoundingBoxes() {
  const boxMode = useSceneStore((s) => s.boxMode)
  const boxRows = useSceneStore((s) => s.currentFrame?.boxes)
  const currentFrameIndex = useSceneStore((s) => s.currentFrameIndex)
  const trailLength = useSceneStore((s) => s.trailLength)
  const hoveredBoxId = useSceneStore((s) => s.hoveredBoxId)
  const highlightedLaserBoxId = useSceneStore((s) => s.highlightedLaserBoxId)
  const setHoveredBox = useSceneStore((s) => s.actions.setHoveredBox)

  const handleHover = useCallback((id: string | null) => {
    setHoveredBox(id, id ? 'laser' : null)
  }, [setHoveredBox])

  const parsed = useMemo(() => {
    if (!boxRows || boxRows.length === 0) return []
    return parseBoxes(boxRows)
  }, [boxRows])

  if (boxMode === 'off' || parsed.length === 0) return null

  const Renderer = boxMode === 'model' ? ModelMesh : BoxMesh

  return (
    <>
      {parsed.map((box, i) => {
        // Determine highlight state: 'self' if directly hovered from 3D,
        // 'linked' if highlighted via 2D camera hover
        const highlighted: 'self' | 'linked' | false =
          hoveredBoxId === box.id ? 'self'
          : highlightedLaserBoxId === box.id ? 'linked'
          : false

        return (
          <Renderer
            key={i}
            box={box}
            highlighted={highlighted}
            onHover={box.isAssociated ? handleHover : undefined}
          />
        )
      })}
      {trailLength > 0 && parsed.map((box) =>
        box.id ? (
          <TrajectoryTrail
            key={`trail-${box.id}`}
            objectId={box.id}
            type={box.type}
            currentFrame={currentFrameIndex}
            trailLength={trailLength}
          />
        ) : null,
      )}
    </>
  )
}

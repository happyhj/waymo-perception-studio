# Performance Optimization Log

Tracking all performance work with measured before/after data.
See `.claude/CONVENTIONS.md` for process rules.

---

## OPT-001: Fix geometry memory leak on frame scrub

**Date:** 2026-03-01
**Status:** Implemented
**Files:** `src/components/LidarViewer/BoundingBoxes.tsx`, `src/components/LidarViewer/CameraFrustums.tsx`

### Problem

Profiling with chrome-devtools-mcp heap snapshots showed **+105,592 objects (+6.5 MB)** leaked after scrubbing 50 frames. Two sources identified:

1. **TrajectoryTrail** (`BoundingBoxes.tsx:236`): `useMemo` creates a new `BufferGeometry.setFromPoints()` every frame change (~20-30 tracked objects per frame). Old geometries were never disposed — their GPU-side vertex buffers accumulated indefinitely.

2. **CameraFrustums** (`CameraFrustums.tsx:75-87`): Pyramid edge meshes used conditional mount (`{highlighted && <lineSegments>}`), creating and destroying `bufferGeometry` + `lineBasicMaterial` on every hover toggle. Unmounting a R3F element disposes the Three.js object, but the mount/unmount churn creates allocation pressure and short-lived objects that stress GC.

### Alternatives considered

| Approach | Tradeoff |
|---|---|
| **A) useEffect cleanup + dispose()** | Simple, targeted. Requires a ref to track previous geometry. |
| B) Reuse single geometry, update buffer in-place | More complex, avoids all allocation. Overkill for trail lines (<50 vertices). |
| C) Object pool for geometries | High complexity, marginal benefit for this use case. |

| Approach | Tradeoff |
|---|---|
| **A) Toggle `visible` prop** | Zero allocation on hover. Invisible meshes still exist in scene graph but are skipped by renderer. |
| B) Dispose manually in useEffect | Doesn't prevent the mount/unmount churn itself. |

### Decision

- **TrajectoryTrail**: Option A — added `useRef` + `useEffect` that calls `geometry.dispose()` on the previous geometry when `useMemo` produces a new one, and on unmount.
- **CameraFrustums**: Option A — replaced `{highlighted && (<lineSegments>...)}` with `<lineSegments visible={highlighted}>`. Meshes are always mounted; visibility toggles without allocation.

### Measurements

Methodology: chrome-devtools-mcp `take_memory_snapshot` before and after programmatically scrubbing 50 frames (150ms between frames, boxes enabled with trail length 10).

| Metric | Before fix | After fix | Improvement |
|---|---|---|---|
| Object delta (50 scrubs) | +105,592 | +41,916 | **-60%** |
| Heap size delta (50 scrubs) | +6.5 MB | +3.2 MB | **-51%** |

Remaining +41,916 objects are expected growth from the prefetch cache filling with decoded frame data (point clouds, camera images, box rows) — working storage, not leaked geometry.

### Baseline performance (no regression)

Measured during the same profiling session, confirming no frame budget regression:

| Metric | Value | Within 16.6ms budget? |
|---|---|---|
| Idle frame time (p50) | 16.7 ms | Yes (60 fps) |
| Colormap loop (170K pts) | 11.3 ms | Yes (runs once per frame change) |
| computeBoundingSphere | 2.1 ms | Yes |
| Scrubbing frame time (p90) | 33.3 ms | Occasional double-vsync, acceptable |

---

## OPT-002: Camera thumbnail bitmap resize

**Date:** 2026-03-01
**Status:** Implemented
**Files:** `src/components/CameraPanel/CameraPanel.tsx`

### Problem

Camera images (1920x1280 front, 1920x886 side) are displayed in 160px-height thumbnail cards but were decoded as full-size bitmaps via `new Blob()` → `URL.createObjectURL()` → `new Image()`. Each full decode allocates a raster backing store:

| Camera | Full decode | Thumbnail (160px) | Reduction |
|---|---|---|---|
| FRONT / FRONT_LEFT / FRONT_RIGHT | 1920x1280 = 9.4 MB | 240x160 = 150 KB | 98% |
| SIDE_LEFT / SIDE_RIGHT | 1920x886 = 6.5 MB | 347x160 = 217 KB | 97% |
| **Total per frame (5 cameras)** | **41.1 MB** | **884 KB** | **97.8%** |

This raster memory lives outside the V8 JS heap (in GPU/compositor process memory), so it doesn't appear in heap snapshots but contributes to overall process memory pressure.

### Alternatives considered

| Approach | Tradeoff |
|---|---|
| **A) `createImageBitmap` with `resizeHeight`** | Browser decodes JPEG directly to thumbnail size. Zero intermediate full-res bitmap. Replaces Blob URL + `<img>` with canvas draw. |
| B) OffscreenCanvas in worker | Moves decode off main thread but adds complexity. `createImageBitmap` already decodes off-thread. |
| C) Server-side thumbnails | Requires a server. Breaks zero-install browser-only constraint. |
| D) CSS `image-rendering` / `content-visibility` | Browser still decodes full resolution; CSS only affects display. No memory savings. |

### Decision

Option A — replaced the entire `Blob` → `URL.createObjectURL` → `new Image()` → `<img>` pipeline with:
1. `createImageBitmap(blob, { resizeHeight: 160, resizeQuality: 'low' })` — decodes JPEG directly to thumbnail size
2. Draw to a `<canvas>` element via `ctx.drawImage(bmp, 0, 0)`
3. `bmp.close()` on the previous bitmap when a new frame arrives, and on unmount
4. Sequence counter (`seqRef`) to discard stale decodes during fast scrubbing

Original JPEG `ArrayBuffer` is kept intact in the cache (for potential future full-res POV use).

### Measurements

**Bitmap memory (raster, per-frame, 5 cameras):**

| Metric | Before fix | After fix | Improvement |
|---|---|---|---|
| Decoded bitmap size | 41.1 MB | 884 KB | **-97.8%** |
| Pixel count | 12.5M px | 0.26M px | **-97.9%** |

**JS heap delta (50 frame scrubs, heap snapshots):**

| Metric | Before fix | After fix |
|---|---|---|
| Object delta | +68,743 | +66,273 |
| Size delta | +3.8 MB | +3.7 MB |

JS heap delta is similar because the savings are in raster memory (decoded pixel backing stores), not V8 heap objects. The Blob and ImageBitmap JS wrapper objects are similar in size.

**Visual quality:** Confirmed via screenshot — thumbnail cards render correctly with `objectFit: 'cover'`, 2D bounding box SVG overlays remain functional.

---

## Rejected / Deferred

### computeBoundingSphere optimization
**Reason deferred:** Measured at 2.1 ms average, only fires once per frame change (not every rAF tick). Well within frame budget. Total useFrame dirty cost is ~13.4 ms including the colormap loop.

### Colormap loop vectorization
**Reason deferred:** 11.3 ms average for 170K points. Combined with computeBoundingSphere (2.1 ms), total dirty-frame cost is ~13.4 ms — under the 16.6 ms budget. Would only matter if point counts increase significantly.

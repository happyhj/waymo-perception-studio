# Technical Plan — Waymo Perception Studio

Portfolio project targeting Waymo Fullstack Engineer (Applications & Tools) role.
Timeline: 4–5 days.

## 1. Architecture Decisions

### Data Pipeline — Browser-Native Parquet (Zero Preprocessing)

**Decision**: Read Parquet files directly in the browser via `hyparquet` (pure JS). No Python, no server, no preprocessing.

**Why this works**: Parquet files have footer metadata with row group offsets. Browser reads footer (few KB) first, then fetches only the needed row group via `File.slice(offset, length)`. This enables random access into 328MB camera_image and 162MB lidar files without loading them fully into memory.

**Why this matters**: v1.0 TFRecord requires sequential reads via TensorFlow — that's why erksch needed a Python server. v2.0 Parquet was designed for selective/columnar access, and we exploit this to the maximum: zero-install, browser-only, file-in → visualization-out. This should be emphasized in interviews.

### Data Loading (Two Modes)

- **Local dev**: `VITE_WAYMO_DATA_PATH=./waymo_data` in `.env`. Vite serves data as static assets. Auto-loads on startup.
- **Deployed demo / visitors**: Folder drag & drop into browser. App scans `{component}/{segment_id}.parquet` structure, auto-detects segments/components. If multiple segments, show picker.
- 3DGS Lab tab always works immediately (bundled .ply).

### Licensing

- **Raw data**: Cannot redistribute. Users must download from Waymo (free, license agreement required).
- **Trained model weights (.ply)**: Distributable (non-commercial). Pre-built 3DGS .ply bundled with app.
- Result: 3DGS Lab = zero-download demo. Sensor View = user provides data.

## 2. Waymo v2.0 Data Structure

Files: `{component}/{segment_id}.parquet`
Key columns: `key.segment_context_name`, `key.frame_timestamp_micros`, `key.laser_name` (1-5), `key.camera_name` (1-5)

### Sample Segment: `10023947602400723454_1120_000_1140_000`
- SF downtown, daytime, sunny
- 199 frames (~20 sec at 10Hz)
- Avg 94 objects/frame, 115 unique tracked objects
- Types: 1=VEHICLE(36/frame), 2=PEDESTRIAN(33), 3=SIGN(23), 4=CYCLIST(1)

### LiDAR — CRITICAL: Range Image format, NOT xyz points

`lidar` component stores **range images**, not point clouds. Must convert to xyz in browser.

| LiDAR | laser_name | Range Image Shape | Pixels |
|-------|-----------|-------------------|--------|
| TOP | 1 | 64 × 2650 × 4 | 169,600 |
| FRONT | 2 | 200 × 600 × 4 | 120,000 |
| SIDE_LEFT | 3 | 200 × 600 × 4 | 120,000 |
| SIDE_RIGHT | 4 | 200 × 600 × 4 | 120,000 |
| REAR | 5 | 200 × 600 × 4 | 120,000 |

- 4 channels = [range, intensity, elongation, is_in_no_label_zone]
- Total ~649K range pixels/frame (valid points fewer — filter range > 0)
- Two returns per pulse: `range_image_return1` (primary) and `range_image_return2` (secondary reflection). MVP uses return1 only.

#### Range Image → XYZ Conversion Math

Source: Official Waymo SDK `lidar_utils.convert_range_image_to_point_cloud()` and GitHub issues #656, #51, #307, #863.

**Step 1: Compute inclination and azimuth per pixel**

- **Inclination** (vertical angle):
  - TOP LiDAR: **non-uniform** — `lidar_calibration` provides a `beam_inclination.values` array (64 exact angles, one per row).
  - Other 4 LiDARs: **uniform** — only `beam_inclination.min` and `beam_inclination.max` provided. Linear interpolation: `inclination = max - (row / height) * (max - min)` (row 0 = top = max angle).
- **Azimuth** (horizontal angle):
  - `azimuth = azimuth_offset + (col / width) * 2π`
  - Column 0 = rear direction (azimuth ≈ π), center column = forward (azimuth ≈ 0).

**Step 2: Spherical → Cartesian**

```
x = range × cos(inclination) × cos(azimuth)
y = range × cos(inclination) × sin(azimuth)
z = range × sin(inclination)
```

Skip pixels where `range <= 0` (invalid).

**Step 3: Extrinsic transform (sensor frame → vehicle frame)**

Apply 4×4 extrinsic matrix from `lidar_calibration`:
```
[x_v, y_v, z_v, 1]ᵀ = extrinsic × [x, y, z, 1]ᵀ
```

**Step 4: Per-point ego-motion correction (TOP only)**

`lidar_pose` provides a per-pixel vehicle pose for the TOP LiDAR to correct rolling shutter distortion. Other 4 LiDARs don't need this (their sweep is fast enough). For MVP, this step can be deferred — the visual difference is subtle.

#### Conversion Strategy: WebGPU Compute Shader + CPU Web Worker Fallback

The conversion is **embarrassingly parallel** — each pixel is independent (cos, sin, matrix mul). Two implementations:

- **WebGPU Compute Shader** (primary): 649K pixels as GPU threads. Expected ~1-2ms/frame. Available in Chrome, Edge, Safari 17.4+.
- **CPU Web Worker** (fallback): For Firefox and older browsers. Expected ~30-50ms/frame.

Both paths share the same math; only the execution environment differs. Benchmark comparison in README demonstrates the speedup.

```
src/utils/rangeImage.ts        ← Pure conversion math (shared, testable)
src/workers/lidarWorker.ts     ← CPU fallback (Web Worker)
src/utils/rangeImageGpu.ts     ← WebGPU compute shader
src/hooks/useLidarConverter.ts ← Auto-selects GPU or Worker
```

#### Gotchas from Waymo SDK Issues

- **#656**: beam_inclination.values can be null for uniform sensors — always check before using.
- **#307**: Raw data is already corrected to vehicle frame — don't apply additional azimuth corrections.
- **#863**: When merging DataFrames, laser_name must match between lidar and lidar_calibration.
- **#51**: range_image_top_pose is per-pixel, not per-frame — only TOP LiDAR has this.

#### Reference: erksch viewer (v1.0)

erksch doesn't do this conversion in the browser at all. Python server calls `frame_utils.convert_range_image_to_point_cloud()` (official Waymo util with TensorFlow), converts to xyz, then sends `[x, y, z, intensity, laser_id, label]` as Float32 binary over WebSocket. Our project does this **entirely in the browser** — no Python, no TensorFlow.

### Camera

| Camera | camera_name | Resolution |
|--------|-----------|------------|
| FRONT | 1 | 1920 × 1280 |
| FRONT_LEFT | 2 | 1920 × 1280 |
| FRONT_RIGHT | 3 | 1920 × 1280 |
| SIDE_LEFT | 4 | 1920 × 886 |
| SIDE_RIGHT | 5 | 1920 × 886 |

- `camera_image` stores JPEG binary in `[CameraImageComponent].image`
- `camera_segmentation` is **1Hz** (not 10Hz) — only 20 frames have segmentation

### File Sizes (1 segment = ~597MB total)

| Component | Size | Load Strategy |
|-----------|------|--------------|
| camera_image | 328MB | Lazy per-frame (row group) |
| lidar | 162MB | Lazy per-frame (row group, 4 RGs) |
| lidar_camera_projection | 74MB | Lazy per-frame |
| lidar_pose | 22MB | Lazy per-frame |
| camera_segmentation | 2.3MB | Load at startup |
| lidar_box | 976KB | Load at startup |
| lidar_camera_synced_box | 543KB | Load at startup |
| lidar_segmentation | 531KB | Load at startup |
| projected_lidar_box | 611KB | Load at startup |
| camera_box | 291KB | Load at startup |
| camera_hkp | 116KB | Load at startup |
| lidar_hkp | 29KB | Load at startup |
| vehicle_pose | 28KB | Load at startup |
| stats | 24KB | Load at startup |
| camera_to_lidar_box_association | 24KB | Load at startup |
| camera_calibration | 8.8KB | Load at startup |
| lidar_calibration | 4.7KB | Load at startup |

### Component Schemas (Key Columns)

**lidar_box** (18,633 rows = ~94/frame × 199 frames):
- `key.laser_object_id` — tracking ID, persistent across frames
- `[LiDARBoxComponent].box.center.{x,y,z}` — double
- `[LiDARBoxComponent].box.size.{x,y,z}` — double
- `[LiDARBoxComponent].box.heading` — double
- `[LiDARBoxComponent].type` — int8 (1=vehicle, 2=pedestrian, 3=sign, 4=cyclist)
- `[LiDARBoxComponent].speed.{x,y,z}` — double
- `[LiDARBoxComponent].acceleration.{x,y,z}` — double

**vehicle_pose** (199 rows):
- `[VehiclePoseComponent].world_from_vehicle.transform` — fixed_size_list<double>[16] (4×4 matrix)

**lidar_calibration** (5 rows):
- `[LiDARCalibrationComponent].extrinsic.transform` — fixed_size_list<double>[16]
- `[LiDARCalibrationComponent].beam_inclination.{min,max}` — double
- `[LiDARCalibrationComponent].beam_inclination.values` — list<double>

**camera_calibration** (5 rows):
- `[CameraCalibrationComponent].intrinsic.{f_u,f_v,c_u,c_v,k1,k2,p1,p2,k3}` — double
- `[CameraCalibrationComponent].extrinsic.transform` — fixed_size_list<double>[16]
- `[CameraCalibrationComponent].{width,height}` — int32

**camera_image** (995 rows = 5 cameras × 199 frames):
- `[CameraImageComponent].image` — binary (JPEG)
- `[CameraImageComponent].pose.transform` — fixed_size_list<double>[16]

**lidar** (995 rows = 5 LiDARs × 199 frames):
- `[LiDARComponent].range_image_return1.values` — list<float>
- `[LiDARComponent].range_image_return1.shape` — fixed_size_list<int32>[3]
- `[LiDARComponent].range_image_return2.{values,shape}` — second return

## 3. Reference Projects (Prior Art)

### erksch/waymo-open-dataset-viewer (2019)
- **GitHub**: https://github.com/erksch/waymo-open-dataset-viewer
- **Stack**: Webpack + TypeScript, Python WebSocket server (TensorFlow GPU)
- **What it does**: LiDAR point cloud (5 sensors), 3D bounding boxes, point color by label/intensity, per-LiDAR toggle, frame slider, OrbitControls
- **What it doesn't do**: No camera images, no map data, no segmentation, no play/pause animation (commented out)
- **Architecture**: Python server reads v1.0 TFRecord → parses via TensorFlow → streams binary frames over WebSocket → Three.js renders in browser
- **Key limitation**: v1.0 only, TensorFlow dependency, requires Python server
- **What we learn**: Basic Three.js point cloud rendering approach (BufferGeometry + Points), WebSocket frame streaming pattern, UI layout inspiration

### Foxglove Studio (industry standard)
- **GitHub (open-source fork)**: https://github.com/AD-EYE/foxglove-opensource (v1.87.0, MPL-2.0)
- **Stack**: React + Three.js, desktop app (Electron)
- **What it does**: Multi-panel layout (diagnostics, aerial view, camera views, 3D perspective), camera segmentation overlay, play/pause/speed control, dual 3D views, camera vs LiDAR object count comparison
- **What it doesn't do**: No Waymo v2.0 Parquet support (requires ROS/MCAP conversion), no 3DGS
- **Key insight**: Panel-based "studio" UI pattern — resizable panels, multiple views of same data. Our UI layout is inspired by this.
- **Closed source since v2.0**: Only open-source fork (v1.87.0) is available

### hailanyi/3D-Detection-Tracking-Viewer (522 stars)
- **GitHub**: https://github.com/hailanyi/3D-Detection-Tracking-Viewer
- **Stack**: Python + VTK/Vedo, desktop app
- **What it does**: GT vs Prediction dual box rendering (red/blue), tracking ID color mapping via matplotlib colormaps, 3D car OBJ model rendering, 2D camera projection, supports KITTI + Waymo (OpenPCDet format)
- **What it doesn't do**: No browser, requires Python + preprocessed npy/pkl files, no 3DGS
- **What we adopt**: (1) Tracking ID → rainbow colormap per `laser_object_id`, (2) BoxType-specific 3D meshes (car/pedestrian/cyclist), (3) Camera frustum visualization in 3D view
- **What we skip**: GT vs Prediction comparison (not portfolio-relevant for a viewer tool)

### Street Gaussians (ECCV 2024)
- **GitHub**: https://github.com/zju3dv/street_gaussians
- **Paper**: Street Gaussians for Modeling Dynamic Urban Scenes
- **What it does**: Static background (standard 3DGS) + dynamic foreground (tracked pose + 4D SH). Clean background rendering with vehicle removal. Novel view synthesis including BEV.
- **Performance**: PSNR ~28, 135 FPS, ~30 min training per segment
- **Why we use it**: Waymo has no top-down camera → need Novel View Synthesis for BEV → 3DGS is the approach. Street Gaussians is Waymo-compatible and fast to train.

### gsplat.js
- **GitHub**: https://github.com/dylanebert/gsplat.js
- **What it does**: Browser-native Gaussian Splat rendering. Loads .ply files, renders in WebGL.
- **Why we use it**: Renders Street Gaussians .ply output directly in browser. Separate canvas from R3F Three.js to avoid WebGL context conflicts.

## 4. Differentiation Matrix

| Feature | erksch (2019) | Foxglove (2024) | Ours (2026) |
|---------|--------------|----------------|-------------|
| Dataset | v1.0 TFRecord | ROS/MCAP | **v2.0 Parquet native** |
| Server | Python + TF | Desktop App | **None (browser)** |
| LiDAR | ✅ | ✅ | ✅ |
| Camera | ❌ | ✅ | ✅ |
| Segmentation | ❌ | ✅ | ✅ |
| Dual 3D View | ❌ | ✅ | ✅ |
| 3DGS BEV | ❌ | ❌ | **✅ Killer Feature** |

## 4. Visualization Features

- **Tracking ID color mapping**: `key.laser_object_id` persistent across frames. Rainbow colormap per ID.
- **3D vehicle OBJ models**: BoxType-specific meshes (car, pedestrian, cyclist) instead of wireframe boxes.
- **Camera frustum visualization**: Semi-transparent frustums in 3D view showing each camera's FOV.

## 5. UI Design

Dark theme (#1a1a2e). Two tabs: [Sensor View] [3DGS Lab 🧪]

### Sensor View
```
┌────────────────────────┬─────────────────────────┐
│ Camera Views (5)       │ Bird's Eye View         │
│ FL | F | FR            │ (3DGS or LiDAR BEV)     │
│ SL |   | SR            │                         │
├────────────────────────┴─────────────────────────┤
│ 3D LiDAR View (point cloud + bbox, OrbitControls)│
├──────────────────────────────────────────────────┤
│ ◀ ▶  ────●──────────── 00:05/00:20   1x         │
└──────────────────────────────────────────────────┘
```

### Data Loading Screen
- "Open Waymo Dataset Folder" button + folder drag & drop
- After load: component checklist (lidar ✅ camera ✅ segmentation ❌ etc.)

### 3DGS Lab
- Full viewport gsplat.js renderer with pre-built .ply
- Separate tab to avoid WebGL context conflicts

## 6. Implementation Phases

1. **MVP (2 days)**: Parquet loading + LiDAR point cloud (range image→xyz) + bounding boxes + timeline
2. **Camera (1 day)**: 5 camera panels + Camera-LiDAR sync + segmentation overlay
3. **Dual View (0.5 day)**: Aerial + Perspective split
4. **3DGS BEV (1 day)**: Street Gaussians training + .ply export + gsplat.js renderer
5. **Polish (0.5 day)**: README, deployment, demo GIF, LinkedIn post

## 7. 3DGS Strategy

### Approach: Street Gaussians (ECCV 2024)
- Static background (standard 3DGS) + Dynamic foreground (tracked pose + 4D SH)
- ~30 min training per segment, 135 FPS rendering
- Export .ply → bundle with app → orthographic BEV camera

### Distribution
- Trained .ply is distributable under Waymo license (non-commercial)
- Same segment as README's recommended download → direct raw-vs-reconstructed comparison
- 3DGS Lab works with zero data download

## 8. Performance Notes

- LiDAR range image → xyz: WebGPU Compute Shader (~1-2ms) with CPU Web Worker fallback (~30-50ms)
- Point cloud: BufferGeometry + Points
- Bounding boxes: InstancedMesh (avg 94/frame)
- Lazy frame loading: current ± N frames in memory, prefetch ahead
- camera_image/lidar: row-group random access, never full file
- Calibrations + boxes + poses: full load at startup (<2MB total)
- Perf-critical rendering: useFrame + imperative refs
- WebGPU availability: Chrome 113+, Edge 113+, Safari 17.4+. Firefox fallback to Web Worker.

### R3F vs Vanilla Three.js — 성능 동등성

R3F(@react-three/fiber)는 Three.js 위의 얇은 React 바인딩이며, 렌더 루프 자체는 동일한 Three.js `WebGLRenderer`가 실행한다. 성능 차이가 없는 이유:

1. **렌더 루프**: R3F의 `useFrame` 훅은 Three.js의 `requestAnimationFrame` 루프에 직접 콜백을 등록. 포인트 클라우드 업데이트 시 `bufferGeometry.attributes.position.needsUpdate = true`를 imperative하게 호출 — vanilla Three.js와 완전히 동일한 코드 경로.

2. **Draw call 최소화**: 168K 포인트를 개별 `<mesh>`로 만들면 React reconciliation 오버헤드 발생. 우리는 `BufferGeometry` + `<points>`로 **한 번의 draw call**에 전체 포인트 클라우드 렌더. 바운딩 박스도 `InstancedMesh`로 94개를 **1 draw call**.

3. **React 오버헤드 구간**: 컴포넌트 마운트/언마운트 시 Three.js 오브젝트 생성/삭제에만 발생. 이건 초기화 때 한 번이고 60fps 렌더 루프에는 영향 없음.

4. **R3F 선택 이유**: Waymo JD가 React/TypeScript를 명시. R3F는 React 생태계 숙련도를 보여주면서도 Three.js 성능을 그대로 유지. 선언적 씬 그래프 구성 (카메라 패널, 컨트롤 등)에서 개발 생산성도 높음.

## 9. Interview Narrative

"I analyzed Foxglove Studio and erksch's viewer, then built a Waymo v2.0-native visualization tool. Existing viewers require Python + TensorFlow servers or ROS conversion. Mine reads v2.0 Parquet natively in the browser — no server, no install. The 3DGS Bird's Eye View is unique — Waymo has no top-down camera, so I used Street Gaussians for Novel View Synthesis. I applied WebGL optimization experience from View360 (530+ stars, adopted by Amazon.com)."

## 10. Decision Log

Chronological record of technical decisions and the reasoning behind them.

### D1. Project Name → `waymo-perception-studio`
- **Alternatives considered**: waymo-viewer, waymo-3d-viewer, waymo-scene-studio, wod-viewer
- **Reasoning**: "Waymo" gives brand recognition on LinkedIn/GitHub. "Perception" matches the Waymo Perception team and dataset domain. "Studio" implies multi-panel tool (not just a simple viewer) — justified by our multi-view layout, tab system, and panel customization. Foxglove also calls itself "Studio."
- **Rejected**: `wod-viewer` (too obscure — only paper authors know "WOD"), `waymo-scene-studio` (less domain-specific)

### D2. Build Tool → Vite over Webpack/Next.js
- **Alternatives considered**: Webpack (erksch used it), Next.js, Vite
- **Reasoning**: (1) Near-instant dev server startup via native ES modules — critical for iterating on Three.js scenes. Webpack bundles everything upfront. (2) 2026 standard for React + TS SPA — CRA is deprecated, Next.js is overkill for pure client-side app (no SSR needed). (3) Web Worker support out of the box (`new Worker(new URL(..., import.meta.url))`). (4) `waymo_data/` static serving trivial via config. (5) erksch used Webpack in 2019 — Vite modernizes the stack.
- **Rejected**: Next.js (SSR/SSG unnecessary, adds complexity for what is a pure browser-side tool)

### D3. 3D Library → @react-three/fiber (R3F) over Vanilla Three.js
- **Reasoning**: Waymo JD emphasizes React/TypeScript. R3F demonstrates React ecosystem mastery. Declarative patterns for UI-heavy panels (cameras, controls). Performance-critical paths (200K point cloud) handled via `useFrame` + imperative refs — no performance gap vs vanilla.
- **Trade-off accepted**: Slightly larger bundle, but developer velocity and interview alignment win.

### D4. Data Pipeline → Browser-native Parquet, no Python preprocessing
- **Alternatives considered**: (A) Python script → JSON/Binary, (B) Browser Parquet parsing
- **Reasoning**: Option B makes "Parquet native" claim genuine. Zero setup friction vs erksch (Python + TF). Leverages Parquet row-group random access — read footer metadata, then slice into specific frames without loading full file. This is only possible because Waymo v2.0 chose Parquet over v1.0 TFRecord (sequential-only).
- **Key insight**: `File.slice(offset, length)` in browser = HTTP Range Request for local files. Footer metadata gives row group offsets. 328MB camera_image becomes manageable.

### D5. No GitHub Pages for data → Local-first with deployed demo for 3DGS
- **Reasoning**: Waymo license prohibits data redistribution. erksch also requires "download yourself." But we go further: deployed URL hosts the app + bundled 3DGS .ply (trained weights are distributable). Sensor View requires user data; 3DGS Lab works immediately.
- **User flow impact**: Interview → share URL → 3DGS BEV loads instantly → "wow" moment → motivated to try Sensor View.

### D6. Data loading → .env path (dev) + folder drag & drop (deployed)
- **Alternatives considered**: (A) Drag & drop only, (B) showDirectoryPicker() only, (C) .env + dev server
- **Reasoning**: Developers clone → .env path → auto-load (zero friction). Visitors → drag & drop `waymo_data/` folder. `showDirectoryPicker()` as bonus for Chrome/Edge. Multiple entry points, same parsing pipeline.
- **Rejected**: Individual file drag & drop (17 component folders with same filename — would overwrite each other).

### D7. 3DGS .ply is distributable → bundle with app
- **Evidence**: Waymo license explicitly allows "trained model architectures, including weights and biases, developed using the Dataset" for non-commercial purposes.
- **Impact**: Killer feature (3DGS BEV) becomes zero-download demo. Same segment as recommended gsutil download → raw vs reconstructed comparison possible.

### D8. LiDAR data is range images → need spherical-to-cartesian conversion
- **Discovery**: Parquet analysis revealed `lidar` component stores range images (64×2650×4 for TOP, 200×600×4 for others), not xyz point clouds.
- **Impact**: Must implement range image → xyz conversion in browser. Uses beam inclination angles from `lidar_calibration` + extrinsic matrix. CPU intensive → Web Worker.
- **Positive spin**: This is non-trivial engineering that demonstrates understanding of LiDAR data representation — good interview talking point.

### D9. Tracking ID → zero-cost tracking visualization
- **Discovery**: `lidar_box` has `key.laser_object_id` that persists across frames for the same physical object. 115 unique objects in sample segment.
- **Impact**: Assign color per object ID with rainbow colormap → automatic tracking visualization with no additional ML. Same color = same car across 20 seconds.

### D10. Camera segmentation is 1Hz, not 10Hz
- **Discovery**: `camera_segmentation` has 100 rows (5 cameras × 20 frames), not 995. Segmentation is sampled at 1Hz.
- **Impact**: Segmentation overlay only updates every 10 frames. UI should indicate when segmentation data is available vs interpolated/unavailable.

### D11. Parquet row-group structure enables lazy loading without preprocessing
- **Discovery**: lidar file has 4 row groups (~50 frames each). camera_image also has 4 row groups. Browser can read specific row groups without loading the entire file.
- **Impact**: No preprocessing step needed. User drags folder → app reads Parquet footer → fetches frame data on demand. This is the fundamental architectural advantage over v1.0 TFRecord approach.
- **Interview angle**: "I chose v2.0 Parquet specifically because its columnar format enables browser-native random access — something impossible with v1.0 TFRecord."

### D12. Lazy loading mechanism — File.slice() vs HTTP Range Requests
- **Two paths, same interface**: hyparquet's `AsyncBuffer` abstracts byte access. We implement two backends:
  - **Drag & drop (File API)**: `file.slice(start, end).arrayBuffer()` — reads bytes from local file handle, no network involved.
  - **Static serving (Vite dev / deployed)**: `fetch(url, { headers: { Range: 'bytes=start-end' } })` → server responds `206 Partial Content` with only the requested bytes. Vite dev server, nginx, S3, Cloudflare Pages all support Range Requests by default.
- **Loading flow**: (1) Read last 8 bytes → get footer length. (2) Read footer (few KB) → get all row group offsets/sizes. (3) On frame request → read only that row group's byte range → decode.
- **Result**: 328MB camera_image file, but each frame request reads ~1.6MB. No server-side logic. No preprocessing.
- **Why this works**: Parquet was designed for distributed storage (HDFS, S3) where Range Requests are the access primitive. We're just using the same mechanism in the browser.
- **AsyncBuffer interface**:
  ```typescript
  interface AsyncBuffer {
    byteLength: number
    slice(start: number, end: number): Promise<ArrayBuffer>
  }
  // File API backend
  const fileBuffer: AsyncBuffer = {
    byteLength: file.size,
    slice: (s, e) => file.slice(s, e).arrayBuffer()
  }
  // HTTP Range Request backend
  const urlBuffer: AsyncBuffer = {
    byteLength: totalSize,
    slice: (s, e) => fetch(url, {
      headers: { Range: `bytes=${s}-${e - 1}` }
    }).then(r => r.arrayBuffer())
  }
  ```

### D13. Component merge strategy — JS port of Waymo's v2.merge()
- **Source**: Official Waymo Python SDK `v2.dataframe_utils.merge()` ([GitHub](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/dataframe_utils.py))
- **What the official API does**: (1) Auto-detect join keys via `key.` prefix. (2) Find common keys between two tables (intersection). (3) Optional `groupby → agg(list)` to prevent cartesian product when row counts differ. (4) Pandas merge on common keys.
- **Our JS port** (`src/utils/merge.ts`): Same logic, but using `Map` lookup instead of Pandas merge. Faster in browser (no Pandas overhead).
- **Key insight from reference projects**:
  - erksch (v1.0): No join needed — TFRecord bundles everything per frame.
  - 3D-Detection-Tracking-Viewer: Pre-processes into per-frame `.npy` files — join happens offline.
  - Waymo official v2.0: `key.frame_timestamp_micros` is the universal join key. Components with different granularity (per-frame vs per-sensor vs per-object) joined via common key intersection.
- **Our approach**: Port v2.merge() to TypeScript. `vehicle_pose` (199 rows) provides master frame list. All other components join via `key.frame_timestamp_micros` + optional `key.camera_name` / `key.laser_name`.
- **Interview angle**: "I studied the official Waymo v2 Python SDK's merge strategy and ported it to TypeScript for browser-native use — same relational join pattern, zero Python dependency."

### D15. Real Data Observations — Range Image Conversion

실제 Waymo v2.0 데이터로 range image → xyz 변환을 구현하면서 발견한 사실들:

**Range image format**:
- Values는 `number[]` (not Float32Array). hyparquet가 Parquet의 float list를 JS Array로 디코딩함.
- 4 channels [range, intensity, elongation, nlz] 이 flat array로 인터리브됨: `[r0, i0, e0, n0, r1, i1, e1, n1, ...]`
- Shape은 `[height, width, channels]` = `[64, 2650, 4]` for TOP → 169,600 pixels × 4 values = 678,400 elements.

**Valid pixel ratio**:
- TOP lidar: 149,796 valid (range > 0) out of 169,600 total → **88.3%** valid.
- Invalid pixels have `range = -1` (not `0`). Filter condition `range > 0` correctly handles both.
- First ~32 pixels in row 0 are typically invalid (very top of FOV, sky region).

**Spatial distribution (vehicle frame)**:
- Range: 2.3m ~ 75m (this segment, SF downtown).
- X/Y: max distance ~75m from vehicle center, reasonable for urban lidar.
- **Z range: -20m ~ +30m** — much wider than the naïve expectation of "ground at -2m, buildings at +10m". SF downtown has steep hills, underground parking ramp exits visible to lidar, and tall buildings/overpasses. Test thresholds adjusted to `[-25, +40]`.

**Beam inclination**:
- TOP (`laser_name=1`): `beam_inclination.values` is a 64-element `number[]` — non-uniform spacing, denser near horizon.
- FRONT/SIDE/REAR (`laser_name=2-5`): `beam_inclination.values` is `undefined` (not present in Parquet row). Must use `min`/`max` for uniform linear interpolation. Height is 200 rows for these sensors.

**Extrinsic calibration**:
- All 5 sensors have 16-element `number[]` (4×4 row-major). TOP's extrinsic includes ~1.8m Z translation (roof mount height).
- Extrinsic transforms sensor-frame xyz → vehicle-frame xyz. Vehicle frame: X=forward, Y=left, Z=up.

**Performance (CPU, single thread, M2 MacBook Air)**:
- TOP (64×2650, 149K valid points): ~2.3ms per frame
- All 5 sensors merged: ~5ms per frame, ~168K total points
- Already fast enough for 10Hz (100ms budget). WebGPU will further improve in real browser with hardware GPU.
- Dawn software renderer (Node.js `webgpu` pkg): ~35ms — slower due to no hardware acceleration. Browser Metal/Vulkan path expected ~1-2ms.

### D14. Waymo Parquet uses BROTLI compression → hyparquet-compressors required
- **Discovery**: All Waymo v2.0 Parquet files use BROTLI compression codec. hyparquet core only includes Snappy; BROTLI requires `hyparquet-compressors` plugin.
- **Why BROTLI**: Standard in Google's big data stack (BigQuery, Cloud Storage). ~20-30% better compression than GZIP on structured data. Waymo chose it for storage efficiency across petabyte-scale datasets.
- **Why this is good for us**: BROTLI was originally designed by Google for web content delivery (`Content-Encoding: br`). Browser-native support exists for HTTP streams. The JS WASM decompressor in hyparquet-compressors is well-optimized. So Waymo's infrastructure choice accidentally aligns perfectly with browser-based access.
- **Impact**: Must pass `compressors` option to all `parquetReadObjects()` calls. Added to `parquet.ts` as default. ~3KB additional dependency.

### D18. Data Worker — Parquet I/O + 변환을 메인 스레드에서 완전 분리

- **문제**: 프레임 전환에 ~4.5초 소요. 원인은 BROTLI 해제 + Parquet 컬럼 디코딩이 메인 스레드에서 동기적으로 실행되어 UI 프레임 드랍 유발.
- **단순 프리페칭이 안 되는 이유**: 프리페칭은 동일 작업을 미리 실행할 뿐, BROTLI 해제 자체가 메인 스레드 CPU를 점유. 3프레임 프리페칭 시 블로킹이 3배로 증가.
- **해결**: `dataWorker.ts` — 전체 파이프라인(fetch → BROTLI 해제 → Parquet 디코딩 → range image → xyz 변환)을 Web Worker에서 실행. 메인 스레드는 최종 `Float32Array`만 `transfer`로 수신 (zero-copy).
- **아키텍처 — 모듈 책임 분리 유지**:
  ```
  dataWorker.ts (얇은 오케스트레이션, ~130줄)
    ├── import { readFrameData } from parquet.ts      ← Parquet I/O 책임 (변경 없음)
    ├── import { convertAllSensors } from rangeImage.ts ← 변환 책임 (변경 없음)
    └── postMessage(Float32Array, [buffer])             ← zero-copy transfer
  ```
  Worker는 기존 모듈을 import해서 호출만 함. 각 모듈의 단일 책임 원칙 유지. Vite의 `new Worker(new URL(...))` 문법이 import를 자동 번들링.
- **통신 패턴**: Promise-based request/response. `requestId`로 동시 다발 프리페치 요청 구분.
  ```
  Main Thread                          Data Worker
  ──────────                          ───────────
  init(lidarUrl, calibrations) ──→   openParquetFile + buildFrameIndex
                               ←──   { type: 'ready' }
  loadFrame(requestId, ts)     ──→   readFrameData → convertAllSensors
                               ←──   { type: 'frameReady', positions: Float32Array } (transfer)
  loadFrame(requestId+1, ts)   ──→   (프리페치 — 동시 처리)
  loadFrame(requestId+2, ts)   ──→
  ```
- **프리페칭**: 현재 프레임 로드 완료 후 다음 3프레임을 Worker에 요청. Worker가 별도 스레드에서 처리하므로 메인 스레드 블로킹 제로. 순차 탐색 시 캐시 히트로 즉시 전환.
- **YouTube-style buffer bar**: `cachedFrames: number[]` 상태로 캐시된 프레임 인덱스를 React에 노출. Timeline에서 연속 구간을 계산하여 반투명 바로 표시. 유저가 프리페치 진행 상황을 시각적으로 확인 가능.
- **성능 영향**:
  - 이전: 프레임 전환 시 메인 스레드 ~4.5초 블로킹 → UI 멈춤
  - 이후: 메인 스레드 블로킹 0ms (postMessage 수신 + 캐시 저장만). Worker에서 ~4.5초 처리되지만 UI는 60fps 유지.
  - 프리페치 적중 시: 프레임 전환 0ms (캐시에서 즉시 로드)
- **면접 포인트**: "162MB LiDAR Parquet의 BROTLI 해제가 메인 스레드를 4.5초 블로킹하는 문제를 발견하고, Data Worker + transfer 패턴으로 메인 스레드 블로킹을 제로로 만들었습니다. 기존 parquet.ts와 rangeImage.ts 모듈은 변경 없이 Worker에서 import하여 재사용 — 단일 책임 원칙을 유지하면서 실행 컨텍스트만 이동했습니다."

### D17. CPU 변환 성능 regression guard — `lastConvertMs < 50ms`
- **목적**: `convertAllSensors()` (range image → xyz) 알고리즘 변경 시 성능 퇴행을 로컬 테스트에서 자동 감지.
- **측정 기준**: M2 MacBook Air에서 5 센서 ~168K 포인트 기준 ~5ms. Threshold은 10배 마진인 50ms.
- **적용 위치**: `useSceneStore.test.ts`의 frame 로딩 테스트 (`nextFrame`, `seekFrame`, `first frame timing`)에 `expect(lastConvertMs).toBeLessThan(50)` assertion.
- **감지 불가능한 것**: Parquet I/O 성능 (8초 중 5ms만 변환이므로 I/O 노이즈에 묻힘). GPU 변환 (Dawn 소프트웨어 렌더러는 비현실적, 브라우저 프로파일링 필요).
- **보완**: `rangeImageBenchmark.test.ts`에서 순수 변환만 5회 반복 평균 측정 가능 (기본 vitest run에서 제외, 수동 실행).

### D16. State management → Zustand
- **Alternatives considered**: (A) Zustand, (B) `useSyncExternalStore` + TS class, (C) Context + `useReducer`
- **Reasoning**:
  - **(C) Context 탈락**: 상태 하나 바뀌면 트리 전체 리렌더. 168K 포인트 매 프레임 업데이트하는 이 프로젝트에서 치명적.
  - **(B) `useSyncExternalStore` 검토**: 의존성 제로, 면접 임팩트. 그러나 subscribe/emit/getSnapshot 보일러플레이트가 커짐. Zustand 내부도 `useSyncExternalStore` 기반이므로 성능 동일하면서 코드가 훨씬 심플.
  - **(A) Zustand 채택**: selector 기반 슬라이스 구독으로 불필요한 리렌더 없음. React 밖에서 `getState()` 접근 가능 (Worker 결과 반영 등). ~1KB. 보일러플레이트 최소. 나중에 `devtools`/`persist` middleware 필요하면 한 줄 추가로 끝.
- **구현**: `useSceneStore` — Zustand store에 state + actions 통합. 내부 데이터(Parquet files, frame indices, cache)는 모듈 스코프에 분리하여 React 리렌더에서 제외.

## 11. Next Actions

1. ✅ Project scaffolding (Vite + React + TS + R3F)
2. ✅ Waymo Dataset v2.0 download (sample segment)
3. ✅ Parquet schema analysis
4. ✅ Parquet loading infrastructure (hyparquet + merge + tests — 27 passing)
5. ✅ Range image → xyz pure math (rangeImage.ts + 14 tests passing against real Waymo data)
6. ✅ CPU Web Worker + WebGPU compute shader (lidarWorker.ts, rangeImageGpu.ts, GPU vs CPU 3 tests passing)
7. ⬜ Phase 1 MVP implementation
7. ⬜ Street Gaussians training
8. ⬜ Deploy + LinkedIn post + Amy DM

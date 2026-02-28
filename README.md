# Perception Studio

In-browser perception explorer for [Waymo Open Dataset](https://waymo.com/open/) v2.0.1.
No setup, no server — just drop Parquet files and explore.

**[Live Demo → heejaekim.github.io/waymo-perception-studio](https://heejaekim.github.io/waymo-perception-studio)**

## Features

- **LiDAR point cloud** — 5 sensors, ~168K pts/frame, turbo colormap, per-sensor toggle
- **3D bounding boxes** — wireframe or GLB models (car/pedestrian/cyclist), tracking-ID colors
- **Trajectory trails** — past N frames of object motion as fading polylines
- **5 camera views** — synchronized JPEG panels with POV switching
- **Camera frustums** — FOV visualization in 3D view with hover sync
- **Timeline** — scrubber, play/pause, speed control (0.5×–4×), buffer bar
- **Multi-segment** — dropdown selector with metadata (location, time, weather)
- **Drag & drop** — drop a folder of Parquet files, no server needed
- **Keyboard shortcuts** — `← →` frame, `J L` ±10, `Space` play/pause, `Shift+← →` segment, `?` help

## Download Data

You need [Waymo Open Dataset v2.0.1](https://waymo.com/open/) Parquet files. Access is free with a Google account.

### Prerequisites

```bash
# Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
```

### Quick start — download 1 segment (~500 MB)

```bash
BUCKET="gs://waymo_open_dataset_v_2_0_1/training"
SEGMENT="10203656353524179475_7625_000_7645_000"
COMPONENTS="vehicle_pose lidar_calibration camera_calibration lidar_box lidar camera_image"

for C in $COMPONENTS; do
  mkdir -p waymo_data/$C
  gsutil cp "$BUCKET/$C/$SEGMENT.parquet" "waymo_data/$C/"
done
```

### Download multiple segments

```bash
BUCKET="gs://waymo_open_dataset_v_2_0_1/training"
COMPONENTS="vehicle_pose lidar_calibration camera_calibration lidar_box lidar camera_image"
N=3

SEGMENTS=$(gsutil ls "$BUCKET/vehicle_pose/*.parquet" | head -$N | xargs -I{} basename {} .parquet)

for SEG in $SEGMENTS; do
  echo "Downloading $SEG"
  for C in $COMPONENTS; do
    mkdir -p waymo_data/$C
    gsutil -m cp "$BUCKET/$C/$SEG.parquet" "waymo_data/$C/"
  done
done
```

### Expected folder structure

```
waymo_data/
├── vehicle_pose/         ← ego vehicle world transform
├── lidar/                ← range images from 5 LiDAR sensors
├── lidar_box/            ← 3D bounding boxes with tracking IDs
├── lidar_calibration/    ← LiDAR extrinsic transforms
├── camera_image/         ← JPEGs from 5 cameras
└── camera_calibration/   ← camera intrinsic/extrinsic
```

Each segment is a **20-second driving clip** at **10 Hz** (~200 frames). A single segment is ~500 MB across 6 components.

Drag & drop the `waymo_data/` folder into the app. Multiple segments are auto-detected with a dropdown selector.

## For Developers

```bash
git clone https://github.com/heejaekim/waymo-perception-studio.git
cd waymo-perception-studio
npm install
npm run dev
```

Place `waymo_data/` in the project root — the dev server auto-discovers segments on startup.

```bash
npm run build   # type-check + production build
npm run lint    # ESLint
npm test        # Vitest
```

## Tech Stack

- **UI**: React 19 + Vite 7 + TypeScript 5.9
- **3D**: @react-three/fiber + drei
- **Data**: hyparquet + hyparquet-compressors (browser-native Parquet with BROTLI)
- **Workers**: 3 LiDAR + 2 camera workers for parallel row group decompression

## Browser Support

Chrome / Edge recommended (folder drag & drop via File System Access API). Firefox / Safari work with individual file drag & drop.

## License

MIT

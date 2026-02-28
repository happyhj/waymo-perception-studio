# Perception Studio for Waymo Open Dataset

A browser-based 3D visualization tool for [Waymo Open Dataset v2.0](https://waymo.com/open/data/perception/). No server required — explore LiDAR point clouds, camera feeds, 3D bounding boxes, and photorealistic 3DGS Bird's Eye View directly in the browser.

## Try the Live Demo

Visit **[heejaekim.github.io/waymo-perception-studio](https://heejaekim.github.io/waymo-perception-studio)** — the 3DGS Lab tab works instantly with no data download needed.

To explore the Sensor View, download a sample segment (see below) and drag & drop the `waymo_data/` folder into the app.

## Download Data

The viewer works with [Waymo Open Dataset v2.0](https://waymo.com/open/) Parquet files. Access is free — you just need a Google account and to accept the license agreement.

### Prerequisites

```bash
# Install Google Cloud CLI (if you don't have it)
# https://cloud.google.com/sdk/docs/install

# Sign in to your Google account
gcloud auth login
```

### Option A: Download a single segment (~500MB)

A single segment is a 20-second driving clip. This is the quickest way to try the viewer.

```bash
# Training (~798 segments) or validation (~202 segments)
BUCKET="gs://waymo_open_dataset_v_2_0_1/training"

# Pick any segment — this is one example
SEGMENT="10203656353524179475_7625_000_7645_000"

# These 6 components are what the viewer uses
COMPONENTS="vehicle_pose lidar_calibration camera_calibration lidar_box lidar camera_image stats"

# Download
for COMP in $COMPONENTS; do
  mkdir -p waymo_data/$COMP
  gsutil cp "$BUCKET/$COMP/$SEGMENT.parquet" "waymo_data/$COMP/"
done
```

### Option B: Download multiple segments

```bash
# Change to ".../validation" for the validation split
BUCKET="gs://waymo_open_dataset_v_2_0_1/training"
COMPONENTS="vehicle_pose lidar_calibration camera_calibration lidar_box lidar camera_image stats"

# How many segments to download
# Training: ~798 segments, Validation: ~202 segments (~1,000 total)
# Change N to download more, or remove "| head -$N" below to download all
N=5

# List available segments and download the first N
SEGMENTS=$(gsutil ls "$BUCKET/vehicle_pose/*.parquet" 2>/dev/null | head -$N | xargs -I{} basename {} .parquet)

for SEGMENT in $SEGMENTS; do
  echo "Downloading segment: $SEGMENT"
  for COMP in $COMPONENTS; do
    mkdir -p waymo_data/$COMP
    gsutil -m cp "$BUCKET/$COMP/$SEGMENT.parquet" "waymo_data/$COMP/" 2>/dev/null
  done
done
```

### Expected folder structure

After downloading, your folder should look like this. Drag & drop the entire `waymo_data/` folder into the viewer.

```
waymo_data/
├── vehicle_pose/
│   └── {segment_id}.parquet
├── lidar/
│   └── {segment_id}.parquet
├── lidar_box/
│   └── {segment_id}.parquet
├── lidar_calibration/
│   └── {segment_id}.parquet
├── camera_image/
│   └── {segment_id}.parquet
├── camera_calibration/
│   └── {segment_id}.parquet
└── stats/
    └── {segment_id}.parquet
```

> **Disk space:** Each segment ≈ 500MB across the 7 components. 5 segments ≈ 2.5GB, full training set ≈ 400GB.
>
> **Multiple segments:** When multiple segments are present, a dropdown appears to switch between them.
>
> **Test set:** The test split has no `lidar_box` labels — the viewer still works but the Perception panel (bounding boxes, trails) will be hidden.

## For Developers

```bash
git clone https://github.com/heejaekim/waymo-perception-studio.git
cd waymo-perception-studio
pnpm install

# Point the app to your data folder
echo "VITE_WAYMO_DATA_PATH=./waymo_data" > .env

# Start dev server — data auto-loads on startup
pnpm run dev
```

The app auto-detects segments and available components from the folder structure. If multiple segments are present, pick one from the list.

## Dataset Structure

Each segment is an independent **20-second driving clip** at **10Hz** (~200 frames).

| Component | Description | Use |
|-----------|-------------|-----|
| `lidar` | Range images from 5 LiDAR sensors (~200K points/frame) | **Point cloud rendering** |
| `lidar_box` | 3D bounding boxes with heading, speed, type, `laser_object_id` | **Box visualization + tracking** |
| `lidar_calibration` | Extrinsic transforms per LiDAR sensor | **Coordinate transforms** |
| `camera_image` | Images from 5 cameras (FRONT, FL, FR, SL, SR) | **Camera panels** |
| `camera_calibration` | Intrinsic/extrinsic per camera | **Frustum visualization** |
| `vehicle_pose` | Ego vehicle world transform (4x4 matrix) | **Global coordinates** |
| `lidar_pose` | Per-point ego-motion correction | Future: point cloud accuracy |
| `lidar_camera_projection` | LiDAR-to-camera point mapping | Future: cross-modal overlay |
| `camera_box` | 2D bounding boxes per camera | Future: image overlay |
| `projected_lidar_box` | 3D boxes projected onto camera images | Future: image overlay |
| `lidar_segmentation` | Semantic segmentation per LiDAR point | Sparse coverage, not used |
| `camera_segmentation` | Semantic segmentation masks (1Hz) | Sparse coverage, not used |
| `lidar_hkp` | Human keypoints from LiDAR | Future: pedestrian pose |
| `camera_hkp` | Human keypoints from camera | Future: pedestrian pose |
| `lidar_camera_synced_box` | Synced boxes across LiDAR & camera | Future |
| `camera_to_lidar_box_association` | Links camera boxes to LiDAR boxes | Future |
| `stats` | Time of day, location, weather, object counts | Future: metadata display |

## Tech Stack

- **Frontend**: React 19 + Vite 7 + TypeScript 5.9
- **3D**: @react-three/fiber + drei
- **3DGS**: gsplat.js
- **Data**: hyparquet + hyparquet-compressors (browser-native Parquet reading with BROTLI support)

## Browser Support

Chrome and Edge recommended (folder drag & drop and File System Access API). Firefox and Safari supported via individual file drag & drop.

## License

MIT

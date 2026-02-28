# Waymo Perception Studio

A browser-based 3D visualization tool for [Waymo Open Dataset v2.0](https://waymo.com/open/data/perception/). No server required — explore LiDAR point clouds, camera feeds, 3D bounding boxes, and photorealistic 3DGS Bird's Eye View directly in the browser.

## Try the Live Demo

Visit **[heejaekim.github.io/waymo-perception-studio](https://heejaekim.github.io/waymo-perception-studio)** — the 3DGS Lab tab works instantly with no data download needed.

To explore the Sensor View, download a sample segment (see below) and drag & drop the `waymo_data/` folder into the app.

## Download Sample Segments

You need access to the [Waymo Open Dataset](https://waymo.com/open/) (free, requires Google account and license agreement).

```bash
# Authenticate with Google Cloud
gcloud auth login

BUCKET="gs://waymo_open_dataset_v_2_0_1/training"

# Essential components (the 6 used by the viewer)
COMPONENTS="vehicle_pose lidar_calibration camera_calibration lidar_box lidar camera_image"

# Create local data directories
for COMP in $COMPONENTS; do mkdir -p waymo_data/$COMP; done

# Discover first 10 segments from the bucket and download
SEGMENTS=$(gsutil ls $BUCKET/vehicle_pose/ | head -10 | xargs -I{} basename {} .parquet)

for SEGMENT in $SEGMENTS; do
  echo "Downloading segment: $SEGMENT"
  for COMP in $COMPONENTS; do
    gsutil -m cp "$BUCKET/$COMP/$SEGMENT.parquet" "waymo_data/$COMP/" 2>/dev/null
  done
done
```

To download a specific segment or all 17 components, use the manual approach:

```bash
SEGMENT="10023947602400723454_1120_000_1140_000"
for COMP in \
  lidar lidar_box lidar_calibration lidar_pose lidar_segmentation lidar_hkp \
  lidar_camera_projection lidar_camera_synced_box \
  camera_image camera_box camera_calibration camera_segmentation camera_hkp \
  camera_to_lidar_box_association projected_lidar_box \
  vehicle_pose stats; do
  gsutil -m cp $BUCKET/$COMP/$SEGMENT.parquet waymo_data/$COMP/
done
```

This preserves Waymo's original folder structure:

```
waymo_data/
├── lidar/
│   ├── {segment_a}.parquet
│   └── {segment_b}.parquet
├── camera_image/
│   ├── {segment_a}.parquet
│   └── {segment_b}.parquet
├── vehicle_pose/
│   ├── {segment_a}.parquet
│   └── {segment_b}.parquet
└── ...
```

> **Disk space:** Each segment is ~500MB (6 essential components). 10 segments ≈ 5GB.
>
> **Multiple segments:** When multiple segments are present, a dropdown selector appears in the header to switch between them.

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
| `lidar` | Range images from 5 LiDAR sensors (~200K points/frame) | Point cloud rendering |
| `lidar_box` | 3D bounding boxes with heading, speed, type, `laser_object_id` | Box visualization + tracking |
| `lidar_calibration` | Extrinsic transforms per LiDAR sensor | Coordinate transforms |
| `lidar_pose` | Per-point ego-motion correction | Point cloud accuracy |
| `lidar_segmentation` | Semantic segmentation per LiDAR point | Point coloring |
| `lidar_hkp` | Human keypoints from LiDAR | Pedestrian pose |
| `lidar_camera_projection` | LiDAR-to-camera point mapping | Cross-modal overlay |
| `lidar_camera_synced_box` | Synced boxes across LiDAR & camera | Consistent annotations |
| `camera_image` | Images from 5 cameras (FRONT, FL, FR, SL, SR) | Camera panels |
| `camera_box` | 2D bounding boxes per camera | Image overlay |
| `camera_calibration` | Intrinsic/extrinsic per camera | Projection math |
| `camera_segmentation` | Semantic segmentation masks | Image overlay |
| `camera_hkp` | Human keypoints from camera | Pedestrian pose |
| `camera_to_lidar_box_association` | Links camera boxes to LiDAR boxes | Cross-modal matching |
| `projected_lidar_box` | 3D boxes projected onto camera images | Image overlay |
| `vehicle_pose` | Ego vehicle world transform (4x4 matrix) | Global coordinates |
| `stats` | Time of day, location, weather, object counts | Segment metadata |

## Tech Stack

- **Frontend**: React 19 + Vite 7 + TypeScript 5.9
- **3D**: @react-three/fiber + drei
- **3DGS**: gsplat.js
- **Data**: parquet-wasm / hyparquet (browser-native Parquet reading)

## Browser Support

Chrome and Edge recommended (folder drag & drop and File System Access API). Firefox and Safari supported via individual file drag & drop.

## License

MIT

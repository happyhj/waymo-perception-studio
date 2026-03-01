# Change to ".../validation" for the validation split
BUCKET="gs://waymo_open_dataset_v_2_0_1/training"
COMPONENTS="vehicle_pose lidar_calibration camera_calibration lidar_box camera_box camera_to_lidar_box_association lidar camera_image stats"

# How many segments to download
# Training: ~798 segments, Validation: ~202 segments (~1,000 total)
# Change N to download more, or remove "| head -$N" below to download all
N=15

# List available segments and download the first N
SEGMENTS=$(gsutil ls "$BUCKET/vehicle_pose/*.parquet" 2>/dev/null | head -$N | xargs -I{} basename {} .parquet)

for SEGMENT in $SEGMENTS; do
  echo "Downloading segment: $SEGMENT"
  for COMP in $COMPONENTS; do
    mkdir -p waymo_data/$COMP
    gsutil -m cp "$BUCKET/$COMP/$SEGMENT.parquet" "waymo_data/$COMP/" 2>/dev/null
  done
done

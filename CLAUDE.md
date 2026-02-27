# Waymo Perception Studio

Browser-based 3D visualization tool for Waymo Open Dataset v2.0.

## Quick Context
- **Detailed technical plan**: `docs/TECHNICAL_PLAN.md` (read for architecture decisions, data schemas, reasoning)
- Zero-install: browser-only, no server, no Python
- Parquet native: `hyparquet` for browser parsing with row-group random access
- LiDAR data is **range images** (not xyz) — must convert spherical→cartesian in Web Worker
- 3DGS .ply is bundled (Waymo license allows trained weight distribution)

## Tech Stack
React 19 + Vite 7 + TypeScript 5.9 + @react-three/fiber + drei + gsplat.js + hyparquet

## Structure
```
src/
├── components/    # Layout, CameraPanel, LidarViewer, BEVViewer, Timeline, Controls
├── hooks/         # useDataLoader, usePlayback, usePointCloud
├── workers/       # Web Worker for Parquet/range-image parsing
├── utils/         # coordinate transforms, calibration math
├── types/         # waymo.ts (v2.0 type definitions)
└── stores/        # App state
```

## Commands
```bash
npm run dev     # Start dev server
npm run build   # Type-check + build
npm run lint    # ESLint
```

## Current Phase
Phase 1 MVP — Parquet loading + LiDAR point cloud + bounding boxes + timeline

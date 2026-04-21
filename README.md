# lyte-isaac-capture

Headless Isaac Sim capture on ANYmal quadruped with simulated RGB+depth camera. Writes per-frame RGB, depth, derived point cloud, and metadata to disk.

Built as the hands-on artifact for experimenting with NVIDIA stack.

## Output layout

    out/<run_name>/
      rgb/0000.png, 0001.png, ...
      depth/0000.npz, 0001.npz, ...        # float32 meters, shape (H, W)
      pointcloud/0000.ply, 0001.ply, ...   # world-frame XYZ
      meta.json                             # per-frame intrinsics + extrinsics + timestamp

## Running

- Local (tests only): `pip install -e '.[dev]' && pytest`
- Remote (Isaac Sim on Brev): `$ISAACSIM_PYTHON scripts/capture_anymal.py --mount head --frames 60 --out out/head-run`

"""Isaac Sim-dependent orchestration. Imports isaacsim -- remote only."""
from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
from isaacsim import SimulationApp  # must be first

# Add src/ to path for pure-Python modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _intrinsics_from_camera(camera, width: int, height: int) -> dict:
    focal = camera.get_focal_length()
    h_aperture = camera.get_horizontal_aperture()
    v_aperture = camera.get_vertical_aperture()
    fx = width * focal / h_aperture
    fy = height * focal / v_aperture
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
        "width": int(width),
        "height": int(height),
    }


def run_capture(args: argparse.Namespace, mount_offset: tuple[float, float, float]) -> int:
    sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    # Imports that require the kit runtime must come AFTER SimulationApp().
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from PIL import Image
    from capture.meta import MetaFrame, write_meta_json
    from capture.geometry import depth_to_pointcloud, write_ply_ascii

    world = World(stage_units_in_meters=1.0)
    assets_root = get_assets_root_path()

    simple_room_usd = assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd"
    add_reference_to_stage(usd_path=simple_room_usd, prim_path="/World/simple_room")

    anymal_usd = assets_root + "/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd"
    add_reference_to_stage(usd_path=anymal_usd, prim_path="/World/anymal")

    cam_prim = "/World/anymal/base/capture_cam"
    camera = Camera(
        prim_path=cam_prim,
        frequency=int(args.fps),
        resolution=(args.width, args.height),
        translation=np.array(mount_offset, dtype=np.float32),
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
    )

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    camera.set_horizontal_aperture(2.0 * np.tan(np.deg2rad(args.fov_deg) / 2.0))

    # Physics warmup (no render — fast).
    for _ in range(60):
        world.step(render=False)

    out_root = Path(args.out)
    frames_meta: list[MetaFrame] = []
    dt = 1.0 / args.fps

    for i in range(args.frames):
        world.step(render=True)
        frame = camera.get_current_frame()
        rgba = frame["rgba"]
        depth = frame["distance_to_image_plane"]
        pos, quat_wxyz = camera.get_world_pose()

        w, x, y, z = quat_wxyz
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = np.asarray(pos, dtype=np.float32)

        intr = _intrinsics_from_camera(camera, args.width, args.height)

        rgb_rel = f"rgb/{i:04d}.png"
        depth_rel = f"depth/{i:04d}.npz"
        pc_rel = f"pointcloud/{i:04d}.ply"

        Image.fromarray(rgba[..., :3]).save(out_root / rgb_rel)
        np.savez_compressed(out_root / depth_rel, depth=depth.astype(np.float32))
        pts = depth_to_pointcloud(depth, intr, extrinsic)
        write_ply_ascii(out_root / pc_rel, pts)

        frames_meta.append(MetaFrame(
            frame=i,
            timestamp=i * dt,
            intrinsics=intr,
            extrinsic=extrinsic.tolist(),
            rgb_path=rgb_rel,
            depth_path=depth_rel,
            pointcloud_path=pc_rel,
        ))

    write_meta_json(
        out_root / "meta.json",
        frames_meta,
        run_name=out_root.name,
        robot="anymal_c",
        mount=args.mount,
    )
    Path("/tmp/capture_done.txt").write_text(
        f"wrote {args.frames} frames to {out_root}\n"
    )

    sim.close()
    return 0

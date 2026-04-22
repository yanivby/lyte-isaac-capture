"""Isaac Sim-dependent orchestration. Imports isaacsim -- remote only."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from isaacsim import SimulationApp  # must be first


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

    world = World(stage_units_in_meters=1.0)

    assets_root = get_assets_root_path()

    # Load Simple_Room environment for meaningful depth maps instead of bare ground plane.
    simple_room_usd = assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd"
    add_reference_to_stage(usd_path=simple_room_usd, prim_path="/World/simple_room")

    anymal_usd = assets_root + "/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd"
    add_reference_to_stage(usd_path=anymal_usd, prim_path="/World/anymal")

    # Camera parented to ANYmal's base link at the configured mount offset.
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

    for _ in range(30):
        world.step(render=False)

    # One render step to populate the camera frame buffers.
    world.step(render=True)
    frame = camera.get_current_frame()
    rgb_shape = frame["rgba"].shape if "rgba" in frame else None
    depth_shape = frame.get("distance_to_image_plane", np.zeros(0)).shape
    Path("/tmp/task8.txt").write_text(
        f"camera ok. rgb={rgb_shape} depth={depth_shape}\n"
    )

    sim.close()
    return 0

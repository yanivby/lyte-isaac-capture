"""Isaac Sim-dependent orchestration. Imports isaacsim -- remote only."""
from __future__ import annotations
from pathlib import Path
import argparse
from isaacsim import SimulationApp  # must be first


def run_capture(args: argparse.Namespace, mount_offset: tuple[float, float, float]) -> int:
    sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})

    # Imports that require the kit runtime must come AFTER SimulationApp().
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path

    world = World(stage_units_in_meters=1.0)

    assets_root = get_assets_root_path()

    # Load Simple_Room environment for meaningful depth maps instead of bare ground plane.
    simple_room_usd = assets_root + "/Isaac/Environments/Simple_Room/simple_room.usd"
    add_reference_to_stage(usd_path=simple_room_usd, prim_path="/World/simple_room")

    anymal_usd = assets_root + "/Isaac/Robots/ANYbotics/anymal_c/anymal_c.usd"
    add_reference_to_stage(usd_path=anymal_usd, prim_path="/World/anymal")

    world.reset()
    Path("/tmp/debug.txt").write_text("before steps\n")
    for _ in range(30):
        world.step(render=False)

    msg = f"[capture_isaac] ANYmal spawned at {anymal_usd}. Mount offset placeholder: {mount_offset}\n"
    Path("/tmp/debug.txt").write_text(msg)
    import sys
    print(msg, flush=True)
    sys.stdout.flush()
    sim.close()
    return 0

#!/usr/bin/env python
"""Headless Isaac Sim capture on ANYmal. Runs ONLY inside Isaac Sim's Python env.

Usage:
    $ISAACSIM_PYTHON scripts/capture_anymal.py \
        --mount head --frames 60 --out out/head-run
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


MOUNTS = {
    # (x, y, z) body-frame offsets in meters, +X forward, +Z up. Approximate for ANYmal.
    "head": (0.35, 0.0, 0.10),
    "back": (0.00, 0.0, 0.25),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mount", choices=list(MOUNTS), default="head")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fov-deg", type=float, default=60.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true", help="Skip Isaac imports, just validate args + dirs")
    return p.parse_args(argv)


def prepare_output(out_root: Path) -> None:
    for sub in ("rgb", "depth", "pointcloud"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepare_output(args.out)
    print(f"[capture] mount={args.mount} offset={MOUNTS[args.mount]} frames={args.frames} out={args.out}")
    if args.dry_run:
        print("[capture] dry-run OK")
        return 0
    # Isaac path follows in later tasks.
    from capture_isaac import run_capture  # noqa: E402  (imports isaacsim)
    return run_capture(args, MOUNTS[args.mount])


if __name__ == "__main__":
    sys.exit(main())

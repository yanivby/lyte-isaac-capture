# src/capture/meta.py
"""Per-frame metadata — schema + JSON I/O."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class MetaFrame:
    frame: int
    timestamp: float
    intrinsics: dict           # fx, fy, cx, cy, width, height
    extrinsic: list            # 4x4 world-from-camera, nested lists
    rgb_path: str
    depth_path: str
    pointcloud_path: str


def write_meta_json(path: Path, frames: list[MetaFrame], run_name: str, robot: str, mount: str) -> None:
    path = Path(path)
    payload = {
        "run_name": run_name,
        "robot": robot,
        "mount": mount,
        "frames": [asdict(f) for f in frames],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_meta_json(path: Path) -> list[MetaFrame]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [MetaFrame(**f) for f in data["frames"]]

# src/capture/geometry.py
"""Pure numpy geometry — depth maps to world-frame point clouds, PLY I/O."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: dict,
    extrinsic: np.ndarray,
) -> np.ndarray:
    """Back-project a depth map into world-frame XYZ points.

    depth:      (H, W) float, meters. NaN/0/inf pixels are dropped.
    intrinsics: {"fx","fy","cx","cy","width","height"}.
    extrinsic:  (4, 4) world-from-camera transform.
    Returns:    (N, 3) float32 world-frame points.
    """
    H, W = depth.shape
    if intrinsics["height"] != H or intrinsics["width"] != W:
        raise ValueError(
            f"depth shape {depth.shape} does not match intrinsics "
            f"({intrinsics['height']}x{intrinsics['width']})"
        )

    valid = np.isfinite(depth) & (depth > 0.0)
    vs, us = np.nonzero(valid)
    zs = depth[vs, us].astype(np.float32)

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    xs_cam = (us.astype(np.float32) - cx) * zs / fx
    ys_cam = (vs.astype(np.float32) - cy) * zs / fy
    pts_cam_h = np.stack([xs_cam, ys_cam, zs, np.ones_like(zs)], axis=1)  # (N, 4)

    pts_world_h = pts_cam_h @ extrinsic.T  # (N, 4)
    return pts_world_h[:, :3].astype(np.float32)


def write_ply_ascii(path: Path, points: np.ndarray) -> None:
    """Write (N, 3) float points as an ASCII PLY file."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    path = Path(path)
    n = points.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    lines = [f"{p[0]:.8g} {p[1]:.8g} {p[2]:.8g}" for p in points]
    path.write_text(header + "\n".join(lines) + ("\n" if n else ""), encoding="ascii")

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
    assert intrinsics["height"] == H and intrinsics["width"] == W

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

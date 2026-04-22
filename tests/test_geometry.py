# tests/test_geometry.py
import numpy as np
from capture.geometry import depth_to_pointcloud


def test_depth_to_pointcloud_identity_extrinsic_center_pixel():
    """Center pixel at depth 5.0 with identity extrinsic -> (0, 0, 5)."""
    H, W = 3, 3
    depth = np.full((H, W), np.nan, dtype=np.float32)
    depth[1, 1] = 5.0  # only center pixel is valid
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": W, "height": H}
    extrinsic = np.eye(4, dtype=np.float32)

    points = depth_to_pointcloud(depth, intrinsics, extrinsic)

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 5.0], atol=1e-5)


def test_depth_to_pointcloud_translated_extrinsic():
    """Translate camera by +10 in X; same pixel should now be at (10, 0, 5)."""
    depth = np.full((3, 3), np.nan, dtype=np.float32)
    depth[1, 1] = 5.0
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 3, "height": 3}
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[0, 3] = 10.0  # translate +X

    points = depth_to_pointcloud(depth, intrinsics, extrinsic)

    np.testing.assert_allclose(points[0], [10.0, 0.0, 5.0], atol=1e-5)


def test_depth_to_pointcloud_offset_pixel():
    """Pixel one unit right of center, fx=100, depth=100 -> X = 1 in camera frame."""
    depth = np.full((3, 3), np.nan, dtype=np.float32)
    depth[1, 2] = 100.0  # u=2, cx=1 -> du=1
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 3, "height": 3}
    extrinsic = np.eye(4, dtype=np.float32)

    points = depth_to_pointcloud(depth, intrinsics, extrinsic)

    np.testing.assert_allclose(points[0], [1.0, 0.0, 100.0], atol=1e-5)


def test_depth_to_pointcloud_filters_nan_and_zero():
    depth = np.zeros((2, 2), dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = 0.0
    depth[1, 0] = 1.5
    depth[1, 1] = np.inf
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "width": 2, "height": 2}
    extrinsic = np.eye(4, dtype=np.float32)

    points = depth_to_pointcloud(depth, intrinsics, extrinsic)

    assert points.shape[0] == 1  # only the 1.5-depth pixel survives

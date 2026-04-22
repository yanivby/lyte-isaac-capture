# tests/test_meta.py
import json
import numpy as np
from capture.meta import MetaFrame, write_meta_json, read_meta_json


def test_metaframe_roundtrip(tmp_path):
    frames = [
        MetaFrame(
            frame=0,
            timestamp=0.0,
            intrinsics={"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
            extrinsic=np.eye(4, dtype=np.float32).tolist(),
            rgb_path="rgb/0000.png",
            depth_path="depth/0000.npz",
            pointcloud_path="pointcloud/0000.ply",
        ),
        MetaFrame(
            frame=1,
            timestamp=1 / 30.0,
            intrinsics={"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
            extrinsic=(np.eye(4, dtype=np.float32) + 0.01).tolist(),
            rgb_path="rgb/0001.png",
            depth_path="depth/0001.npz",
            pointcloud_path="pointcloud/0001.ply",
        ),
    ]
    out = tmp_path / "meta.json"
    write_meta_json(out, frames, run_name="test", robot="anymal", mount="head")

    loaded = json.loads(out.read_text())
    assert loaded["run_name"] == "test"
    assert loaded["robot"] == "anymal"
    assert loaded["mount"] == "head"
    assert len(loaded["frames"]) == 2
    assert loaded["frames"][0]["frame"] == 0
    assert loaded["frames"][1]["timestamp"] > 0.0

    roundtrip = read_meta_json(out)
    assert len(roundtrip) == 2
    assert roundtrip[1].frame == 1


def test_write_meta_json_rejects_raw_ndarray(tmp_path):
    """MetaFrame with a raw ndarray extrinsic should raise TypeError from json.dumps."""
    import pytest as _pytest
    frame = MetaFrame(
        frame=0,
        timestamp=0.0,
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "width": 1, "height": 1},
        extrinsic=np.eye(4, dtype=np.float32),  # raw ndarray, not .tolist()
        rgb_path="rgb/0000.png",
        depth_path="depth/0000.npz",
        pointcloud_path="pointcloud/0000.ply",
    )
    out = tmp_path / "meta.json"
    with _pytest.raises(TypeError):
        write_meta_json(out, [frame], run_name="test", robot="anymal", mount="head")

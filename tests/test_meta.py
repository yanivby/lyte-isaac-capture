# tests/test_meta.py
import json
import numpy as np
from capture.meta import MetaFrame, write_meta_json, read_meta_json


def test_metaframe_serializes_numpy(tmp_path):
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

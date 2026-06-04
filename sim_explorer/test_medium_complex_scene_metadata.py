#!/usr/bin/env python3
"""Pure Python metadata tests for the Stage 4A-3.2 medium scene."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from scene_factory import build_medium_complex_scene


def _inside_rect_xy(x: float, y: float, rect: dict[str, list[float]], margin: float = 0.0) -> bool:
    return (
        float(rect["x"][0]) - margin <= float(x) <= float(rect["x"][1]) + margin
        and float(rect["y"][0]) - margin <= float(y) <= float(rect["y"][1]) + margin
    )


def test_metadata_schema() -> None:
    metadata = build_medium_complex_scene(seed=0, variant="three_rooms", spawn=False)
    required = {
        "floor",
        "walls",
        "doors",
        "openings",
        "obstacles",
        "rooms",
        "corridors",
        "camera_poses",
        "map_bounds",
        "expected_output_files",
        "leakage_checks",
    }
    missing = sorted(required.difference(metadata))
    assert not missing, missing

    assert metadata["floor"]["size"] == [12.0, 12.0]
    assert metadata["map_bounds"]["x"] == [-6.0, 6.0]
    assert metadata["map_bounds"]["y"] == [-6.0, 6.0]
    assert metadata["map_bounds"]["z"] == [0.0, 3.0]
    assert math.isclose(float(metadata["wall_height_m"]), 2.2)

    assert len(metadata["rooms"]) >= 3
    assert len(metadata["corridors"]) >= 1
    assert len(metadata["openings"]) >= 3
    assert len(metadata["doors"]) == len(metadata["openings"])
    assert len(metadata["obstacles"]) >= 10
    assert len(metadata["camera_poses"]) >= 5


def test_doors_obstacles_and_camera_poses_are_sane() -> None:
    metadata = build_medium_complex_scene(seed=0, variant="three_rooms", spawn=False)

    for door in metadata["openings"]:
        assert 0.9 <= float(door["width"]) <= 1.4
        assert len(door["connects"]) == 2
        rect = door["clear_rect"]
        assert rect["x"][1] > rect["x"][0]
        assert rect["y"][1] > rect["y"][0]

    for obstacle in metadata["obstacles"]:
        x, y, z = (float(v) for v in obstacle["position"])
        sx, sy, sz = (float(v) for v in obstacle["size"])
        assert sx > 0.0 and sy > 0.0 and sz > 0.0
        assert math.isclose(z, sz * 0.5, rel_tol=0.0, abs_tol=1.0e-6)
        for door in metadata["openings"]:
            assert not _inside_rect_xy(x, y, door["clear_rect"], margin=0.20), (obstacle["name"], door["name"])

    pose_rooms = {pose["room"] for pose in metadata["camera_poses"]}
    assert {"room_a", "corridor", "room_c"}.issubset(pose_rooms)
    for pose in metadata["camera_poses"]:
        assert len(pose["position"]) == 3
        assert "yaw_rad" in pose and "yaw_deg" in pose
        assert "note" in pose and pose["note"]


def test_expected_outputs_and_json_roundtrip() -> None:
    metadata = build_medium_complex_scene(seed=0, variant="three_rooms", spawn=False)
    expected = metadata["expected_output_files"]
    smoke = set(expected["smoke"])
    viz = set(expected["visualization"])

    for idx in range(5):
        assert f"depth_{idx:03d}.npy" in smoke
        assert f"rgb_{idx:03d}.png" in smoke
        assert f"pose_{idx:03d}.json" in smoke
        assert f"observed_state_step{idx}.npy" in smoke
    assert {"camera_info.json", "scene_metadata.json", "observed_summary.json"}.issubset(smoke)
    assert "observed_state_final.npy" in smoke
    assert {
        "scene_overview_rgb.png",
        "scene_overview_depth_color.png",
        "scene_layout_topdown.png",
        "camera_rgb_grid.png",
        "camera_depth_grid.png",
        "observed_topdown_compare.png",
        "free_occupied_voxels_3d_final.png",
        "slices_final.png",
    }.issubset(viz)

    encoded = json.dumps(metadata, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["scene_id"] == "medium_complex_three_rooms"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene_metadata.json"
        path.write_text(encoded + "\n", encoding="utf-8")
        reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["topology_summary"]["room_count"] >= 3
    assert reloaded["topology_summary"]["opening_count"] >= 3
    assert reloaded["topology_summary"]["obstacle_count"] >= 10


def test_leakage_and_no_training_flags() -> None:
    metadata = build_medium_complex_scene(seed=0, variant="three_rooms", spawn=False)
    checks = metadata["leakage_checks"]
    forbidden_true = [
        "prediction_used",
        "prediction_wrote_observed_map",
        "target_lr_used",
        "target_hr_used",
        "scene_ground_truth_used_for_exploration",
        "rl_or_ppo_training",
        "behavior_cloning_training",
        "imitation_learning_training",
        "sscnet_training",
    ]
    for key in forbidden_true:
        assert checks[key] is False, key


def main() -> None:
    test_metadata_schema()
    test_doors_obstacles_and_camera_poses_are_sane()
    test_expected_outputs_and_json_roundtrip()
    test_leakage_and_no_training_flags()
    print("medium complex scene metadata tests passed")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Validate Stage 4A-6.6 larger_complex_scene_v1 construction outputs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_OUTPUT_DIR = Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation")
SCENE_FACTORY = Path("/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py")

REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "code_hash_audit.json",
    "code_hash_audit.md",
    "scene_construction_config.json",
    "scene_construction_config.md",
    "larger_complex_scene_v1_metadata.json",
    "larger_complex_scene_v1_metadata.md",
    "scene_topology_graph.json",
    "scene_topology_graph.md",
    "room_inventory.csv",
    "room_inventory.json",
    "room_inventory.md",
    "corridor_inventory.csv",
    "corridor_inventory.json",
    "corridor_inventory.md",
    "opening_inventory.csv",
    "opening_inventory.json",
    "opening_inventory.md",
    "wall_inventory.csv",
    "wall_inventory.json",
    "obstacle_inventory.csv",
    "obstacle_inventory.json",
    "obstacle_inventory.md",
    "start_variants.json",
    "start_variants.md",
    "validation_pose_manifest.json",
    "validation_pose_manifest.md",
    "topology_connectivity_summary.json",
    "topology_connectivity_summary.md",
    "preliminary_complexity_metrics.json",
    "preliminary_complexity_metrics.md",
    "preliminary_complexity_target_checklist.json",
    "preliminary_complexity_target_checklist.md",
    "scene_load_validation.json",
    "scene_load_validation.md",
    "fixed_capture_validation.json",
    "fixed_capture_validation.md",
    "observed_state_validation_summary.json",
    "observed_state_validation_summary.md",
    "observed_state_transition_summary.json",
    "observed_state_transition_summary.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_formal_expert_sampling_report.json",
    "no_formal_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "stage4a66_larger_complex_scene_v1_summary.json",
    "stage4a66_larger_complex_scene_v1_summary.md",
    "audit_input_bundle_manifest.json",
    "audit_input_bundle_manifest.md",
    "future_stage4a66a_scene_complexity_audit_command_sketch.md",
    "do_not_start_expert_sampling_before_audit.md",
    "recommended_next_faithful_step.md",
    "long_term_rl_gdpo_note.md",
    "observed_state_final.npy",
    "observed_summary.json",
    "camera_info.json",
    "scene_metadata.json",
    "scene_layout_topdown.png",
    "scene_topology_graph.png",
    "rooms_corridors_openings_topdown.png",
    "obstacles_topdown.png",
    "start_variants_topdown.png",
    "validation_camera_poses_topdown.png",
    "camera_rgb_grid.png",
    "camera_depth_grid.png",
    "observed_topdown_final.png",
    "observed_ratio_fixed_views.png",
    "connectivity_graph.png",
    "complexity_target_checklist.png",
    "start_distance_matrix.png",
    "obstacle_density_topdown.png",
    "audit_gate_flowchart.png",
    "missing_fields_report.json",
]

FORBIDDEN_PATTERNS = [
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "episode_manifest*",
    "expert_dataset_manifest*",
    "*replay_buffer*",
    "*policy_checkpoint*",
    "global_prediction_layer.npz",
    "prediction*.npz",
    "frame003*",
    "action002*",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"


def assert_png(path: Path) -> None:
    assert_file(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
    image = Image.open(path)
    assert image.size[0] > 0 and image.size[1] > 0, f"empty PNG dimensions: {path}"


def test_required_files(output_dir: Path) -> dict[str, Any]:
    for name in REQUIRED_FILES:
        path = output_dir / name
        if name.endswith(".png"):
            assert_png(path)
        else:
            assert_file(path)
    return {"passed": True, "required_count": len(REQUIRED_FILES)}


def test_metadata_and_metrics(output_dir: Path) -> dict[str, Any]:
    metadata = load_json(output_dir / "larger_complex_scene_v1_metadata.json")
    metrics = load_json(output_dir / "preliminary_complexity_metrics.json")
    checklist = load_json(output_dir / "preliminary_complexity_target_checklist.json")

    assert metadata["scene_id"] == "larger_complex_scene_v1"
    assert metadata["scene_seed"] == 0
    assert metadata["map_bounds"]["x"] == [-12.0, 12.0]
    assert metadata["map_bounds"]["y"] == [-12.0, 12.0]
    assert metadata["map_bounds"]["z"] == [0.0, 3.0]
    assert metadata["expected_observed_state_shape"] == [240, 240, 30]
    assert len(metadata["rooms"]) >= 8
    assert len(metadata["corridors"]) >= 3
    assert len(metadata["openings"]) >= 12
    assert len(metadata["walls"]) > 13
    assert len(metadata["obstacles"]) >= 40
    assert len(metadata["start_variants"]) >= 8
    assert len(metadata["validation_camera_poses"]) >= 12

    assert metrics["room_count"] >= 8
    assert metrics["corridor_count"] >= 3
    assert metrics["opening_count"] >= 12
    assert metrics["branch_point_count"] >= 4
    assert metrics["cycle_rank"] >= 2
    assert metrics["narrow_passage_count"] >= 4
    assert metrics["minimum_doorway_width_m"] <= 0.9
    assert metrics["start_variant_count"] >= 8
    assert metrics["validation_pose_count"] >= 12
    assert metrics["formal_expert_sampling_ready"] is False
    assert metrics["scene_complexity_audit_passed"] is False
    assert metrics["connectivity_between_start_variants"]["all_start_variants_mutually_connected"] is True
    assert metrics["starts_inside_bounds"] is True
    assert metrics["starts_collision_free_by_metadata"] is True
    assert metrics["starts_not_duplicated"] is True

    for target, item in checklist.items():
        if target == "formal_expert_sampling_ready":
            assert item["passed"] is False
        else:
            assert item["passed"] is True, target

    return {
        "passed": True,
        "rooms": len(metadata["rooms"]),
        "corridors": len(metadata["corridors"]),
        "openings": len(metadata["openings"]),
        "obstacles": len(metadata["obstacles"]),
    }


def test_capture_and_observed(output_dir: Path) -> dict[str, Any]:
    capture = load_json(output_dir / "fixed_capture_validation.json")
    scene_load = load_json(output_dir / "scene_load_validation.json")
    observed_summary = load_json(output_dir / "observed_state_validation_summary.json")
    transition = load_json(output_dir / "observed_state_transition_summary.json")

    assert scene_load["scene_loaded"] is True
    assert int(scene_load["isaac_startup_count"]) == 1
    assert capture["fixed_validation_pose_count"] >= 12
    assert capture["rgb_nonblank_count"] >= 8
    assert capture["depth_positive_count"] >= 8
    assert capture["no_actions_executed"] is True
    assert capture["no_map_predict"] is True
    assert capture["no_sscnet_inference"] is True

    pose_count = int(capture["fixed_validation_pose_count"])
    for idx in range(pose_count):
        assert_png(output_dir / f"validation_rgb_{idx:03d}.png")
        assert_png(output_dir / f"validation_depth_{idx:03d}.png")
        assert_file(output_dir / f"validation_pose_{idx:03d}.json")
        depth = np.load(output_dir / f"validation_depth_{idx:03d}.npy")
        assert depth.ndim == 2, f"depth {idx} shape {depth.shape}"
        assert np.count_nonzero(np.isfinite(depth) & (depth > 0.0)) > 0, f"depth {idx} has no finite positive values"
        observed = np.load(output_dir / f"observed_state_step{idx:03d}.npy")
        assert tuple(observed.shape) == (240, 240, 30), f"observed step {idx} shape {observed.shape}"
        assert np.all(np.isin(observed, [-1, 0, 1])), f"invalid labels in observed step {idx}"

    observed_final = np.load(output_dir / "observed_state_final.npy")
    assert tuple(observed_final.shape) == (240, 240, 30)
    assert np.all(np.isin(observed_final, [-1, 0, 1]))
    assert observed_summary["shape"] == [240, 240, 30]
    assert observed_summary["shape_matches_expected"] is True
    assert observed_summary["invalid_label_count"] == 0
    assert observed_summary["measured_only"] is True
    assert observed_summary["prediction_used"] is False
    assert observed_summary["observed_count"] > 0
    assert transition["total_newly_observed"] > 0
    assert transition["invalid_label_count_final"] == 0

    return {
        "passed": True,
        "pose_count": pose_count,
        "observed_ratio": observed_summary["observed_ratio"],
    }


def test_safety_and_summary(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a66_larger_complex_scene_v1_summary.json")
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_expert = load_json(output_dir / "no_formal_expert_sampling_report.json")
    no_map_predict = load_json(output_dir / "no_map_predict_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
    missing = load_json(output_dir / "missing_fields_report.json")

    assert summary["larger_scene_constructed"] is True
    assert summary["isaac_headless_load_validated"] is True
    assert summary["isaac_startup_count"] == 1
    assert summary["fixed_capture_validated"] is True
    assert summary["measured_only_observed_state"] is True
    assert summary["preliminary_complexity_targets_met"] is True
    assert summary["scene_complexity_audit_passed"] is False
    assert summary["formal_expert_sampling_ready"] is False
    assert summary["formal_expert_sampling_blocked"] is True
    assert summary["rollout_run"] is False
    assert summary["map_predict_called"] is False
    assert summary["sscnet_inference_called"] is False
    assert summary["rl_gdpo_ppo_bc_il_run"] is False

    assert no_rollout["rollout_run"] is False
    assert no_rollout["selected_expert_action_executed"] is False
    assert no_expert["formal_expert_sampling_run"] is False
    assert no_expert["formal_expert_sampling_ready"] is False
    assert no_map_predict["map_predict_called"] is False
    assert no_map_predict["prediction_npz_created"] is False
    assert no_rl["rl_run"] is False
    assert no_rl["gdpo_run"] is False
    assert no_rl["policy_checkpoint_created"] is False
    assert missing["all_required_plots_generated"] is True

    return {"passed": True}


def test_hashes_and_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    hashes = load_json(output_dir / "code_hash_audit.json")
    assert hashes["scene_factory_py"]["sha256_after_stage4a66_implementation"] == sha256_file(SCENE_FACTORY)
    assert hashes["checkpoint_audit_only"]["loaded"] is False
    assert hashes["map_predict_code_loaded"] is False
    assert hashes["sscnet_checkpoint_loaded"] is False

    hits = []
    for pattern in FORBIDDEN_PATTERNS:
        hits.extend(glob.glob(str(output_dir / pattern)))
        hits.extend(glob.glob(str(output_dir / "**" / pattern), recursive=True))
    hits = sorted(set(hits))
    assert not hits, f"forbidden outputs found: {hits}"
    return {"passed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6.6 outputs.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    assert output_dir.is_dir(), f"output_dir missing: {output_dir}"

    results = {
        "required": test_required_files(output_dir),
        "metadata_metrics": test_metadata_and_metrics(output_dir),
        "capture_observed": test_capture_and_observed(output_dir),
        "safety_summary": test_safety_and_summary(output_dir),
        "hashes_forbidden": test_hashes_and_forbidden_outputs(output_dir),
    }
    print(json.dumps({"all_passed": True, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

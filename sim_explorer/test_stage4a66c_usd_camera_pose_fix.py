#!/usr/bin/env python3
"""Validate Stage 4A-6.6c camera-pose-fix outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REQUIRED_REPORTS = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_defaultprim_validation_manifest.json",
    "loaded_defaultprim_validation_manifest.md",
    "previous_camera_issue_report.json",
    "previous_camera_issue_report.md",
    "usd_interior_region_analysis.json",
    "usd_interior_region_analysis.md",
    "candidate_pose_generation_report.json",
    "candidate_pose_generation_report.md",
    "interior_pose_scoring_table.csv",
    "interior_pose_scoring_table.json",
    "interior_pose_scoring_table.md",
    "selected_validation_pose_manifest.json",
    "selected_validation_pose_manifest.md",
    "selected_inspection_pose_manifest.json",
    "selected_inspection_pose_manifest.md",
    "start_variants_interior.json",
    "start_variants_interior.md",
    "start_variant_interior_report.json",
    "start_variant_interior_report.md",
    "scene_load_validation_camera_fix.json",
    "scene_load_validation_camera_fix.md",
    "fixed_capture_validation_camera_fix.json",
    "fixed_capture_validation_camera_fix.md",
    "visual_inspection_capture_validation_camera_fix.json",
    "visual_inspection_capture_validation_camera_fix.md",
    "observed_state_validation_summary_camera_fix.json",
    "observed_state_validation_summary_camera_fix.md",
    "observed_summary.json",
    "interior_visual_quality_report.json",
    "interior_visual_quality_report.md",
    "exterior_pose_rejection_report.json",
    "exterior_pose_rejection_report.md",
    "manual_review_gate.json",
    "manual_review_gate.md",
    "human_visual_review_checklist.json",
    "human_visual_review_checklist.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_expert_sampling_report.json",
    "no_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "stage4a66c_usd_camera_pose_fix_summary.json",
    "stage4a66c_usd_camera_pose_fix_summary.md",
    "recommended_next_faithful_step.md",
    "observed_state_final.npy",
    "camera_info.json",
    "scene_metadata.json",
    "corrected_rgb_validation_grid.png",
    "corrected_depth_validation_grid.png",
    "corrected_rgb_inspection_grid.png",
    "corrected_depth_inspection_grid.png",
    "corrected_observed_topdown_final.png",
    "corrected_start_variants_topdown.png",
    "corrected_validation_camera_poses_topdown.png",
    "corrected_inspection_camera_poses_topdown.png",
    "corrected_interior_region_topdown.png",
    "corrected_closeup_contact_sheet.png",
    "corrected_warning_regions.png",
    "visual_inspection_index.html",
]
FORBIDDEN_EXACT = {
    "transitions.jsonl",
    "rollout_index.html",
    "rollout_topdown_path.png",
    "expert_dataset_manifest.jsonl",
    "expert_dataset_manifest.json",
    "selected_action_execution_report.json",
    "selected_action_report.json",
    "action_execution_report.json",
    "global_prediction_layer.npz",
}
FORBIDDEN_TOKENS = (
    "replay_buffer",
    "policy_checkpoint",
    "checkpoint_epoch",
    "ppo_checkpoint",
    "gdpo_checkpoint",
    "behavior_cloning_checkpoint",
    "prediction_npz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"


def assert_png(path: Path, *, nonblank: bool = False) -> None:
    assert_file(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
    image = Image.open(path).convert("RGB")
    assert image.size[0] > 0 and image.size[1] > 0, f"bad PNG size: {path}"
    if nonblank:
        arr = np.asarray(image)
        assert int(arr.max()) > 2 and float(arr.std()) >= 1.0, f"blank PNG: {path}"


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_REPORTS:
        assert_file(output_dir / name)
    assert (output_dir / "usd_scene_flythrough.mp4").is_file() or (output_dir / "usd_scene_flythrough_frames").is_dir()
    return {"passed": True, "required_count": len(REQUIRED_REPORTS)}


def test_context_previous_and_usd(output_dir: Path, stage4a66c_defaultprim_dir: Path, source_usd: Path, fixed_usd: Path) -> dict[str, Any]:
    loaded = load_json(output_dir / "loaded_context_manifest.json")
    previous = load_json(output_dir / "loaded_defaultprim_validation_manifest.json")
    summary = load_json(output_dir / "stage4a66c_usd_camera_pose_fix_summary.json")
    assert loaded["current_task"] == "camera_validation_inspection_start_pose_fix_only"
    assert previous["fixed_usd_exists"] is True
    assert previous["fixed_usd_defaultPrim_valid"] is True
    assert previous["fixed_usd_defaultPrim"] in {"/World", "World"}
    assert previous["dependency_closure_complete"] is True
    assert previous["remote_official_refs_remaining"] == 0
    assert previous["omniverse_refs_remaining"] == 0
    assert previous["unresolved_local_deps_remaining"] == 0
    assert previous["previous_isaac_load_succeeded"] is True
    assert previous["previous_validation_rgb_count"] >= 20
    assert previous["previous_inspection_rgb_count"] >= 36
    assert previous["previous_observed_state_final_exists"] is True
    assert (stage4a66c_defaultprim_dir / "observed_state_final.npy").is_file()
    assert source_usd.is_file()
    assert fixed_usd.is_file()
    assert summary["source_usd_modified"] is False
    assert summary["fixed_usd_modified"] is False
    assert sha256_file(source_usd) == summary["source_sha256_before"] == summary["source_sha256_after"]
    assert sha256_file(fixed_usd) == summary["fixed_sha256_before"] == summary["fixed_sha256_after"]
    return {"passed": True}


def test_pose_manifests_and_scores(output_dir: Path) -> dict[str, Any]:
    starts = load_json(output_dir / "start_variants_interior.json")
    validation = load_json(output_dir / "selected_validation_pose_manifest.json")
    inspection = load_json(output_dir / "selected_inspection_pose_manifest.json")
    scoring = load_json(output_dir / "interior_pose_scoring_table.json")
    rejection = load_json(output_dir / "exterior_pose_rejection_report.json")
    assert len(starts) >= 8
    assert validation["pose_count"] >= 20
    assert inspection["pose_count"] >= 36
    assert len(scoring) >= validation["pose_count"] + inspection["pose_count"]
    assert rejection["previous_exterior_or_suspect_poses_rejected"] > 0
    for start in starts:
        assert start["pending_human_approval"] is True
        x, y, z = start["position"]
        assert -10.1 <= x <= 10.1 and 0.0 <= y <= 36.1 and 1.0 <= z <= 1.4
    for prefix, manifest in (("validation", validation), ("inspection", inspection)):
        for pose in manifest["poses"]:
            x, y, z = pose["position"]
            assert -10.1 <= x <= 10.1 and 0.0 <= y <= 36.1 and 1.0 <= z <= 1.4
            assert_file(output_dir / f"{prefix}_pose_score_{int(pose['index']):03d}.json")
    return {"passed": True, "starts": len(starts), "validation": validation["pose_count"], "inspection": inspection["pose_count"]}


def test_capture_files(output_dir: Path) -> dict[str, Any]:
    validation = load_json(output_dir / "selected_validation_pose_manifest.json")
    inspection = load_json(output_dir / "selected_inspection_pose_manifest.json")
    for prefix, count in (("validation", validation["pose_count"]), ("inspection", inspection["pose_count"])):
        for idx in range(count):
            rgb = output_dir / f"{prefix}_rgb_{idx:03d}.png"
            depth = output_dir / f"{prefix}_depth_{idx:03d}.npy"
            depth_color = output_dir / f"{prefix}_depth_color_{idx:03d}.png"
            pose_json = output_dir / f"{prefix}_pose_{idx:03d}.json"
            score_json = output_dir / f"{prefix}_pose_score_{idx:03d}.json"
            assert_png(rgb, nonblank=True)
            assert_png(depth_color)
            assert_file(depth)
            assert_file(pose_json)
            assert_file(score_json)
            arr = np.load(depth)
            positive = arr[np.isfinite(arr) & (arr > 0.0)]
            assert positive.size > 0, f"no finite positive depth: {depth}"
            score = load_json(score_json)
            assert "outside_likelihood" in score
            assert "accepted" in score
    assert_png(output_dir / "corrected_rgb_validation_grid.png", nonblank=True)
    assert_png(output_dir / "corrected_rgb_inspection_grid.png", nonblank=True)
    return {"passed": True}


def test_observed_state(output_dir: Path) -> dict[str, Any]:
    observed = np.load(output_dir / "observed_state_final.npy")
    labels = set(int(v) for v in np.unique(observed))
    assert {-1, 0, 1}.issubset(labels), f"observed_state missing labels: {labels}"
    assert labels.issubset({-1, 0, 1}), f"invalid observed_state labels: {labels}"
    summary = load_json(output_dir / "observed_state_validation_summary_camera_fix.json")
    assert summary["invalid_label_count"] == 0
    assert summary["labels_present"]["UNKNOWN"] is True
    assert summary["labels_present"]["FREE"] is True
    assert summary["labels_present"]["OCCUPIED"] is True
    assert summary["measured_only"] is True
    assert summary["prediction_used"] is False
    assert summary["map_predict_called"] is False
    assert_png(output_dir / "corrected_observed_topdown_final.png")
    return {"passed": True, "shape": list(observed.shape), "observed_ratio": summary["observed_ratio"]}


def test_gates_and_negative_scope(output_dir: Path) -> dict[str, Any]:
    manual = load_json(output_dir / "manual_review_gate.json")
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_expert = load_json(output_dir / "no_expert_sampling_report.json")
    no_map = load_json(output_dir / "no_map_predict_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
    summary = load_json(output_dir / "stage4a66c_usd_camera_pose_fix_summary.json")
    assert manual["human_visual_inspection_done"] is False
    assert manual["user_needs_to_review_visuals"] is True
    assert manual["formal_expert_sampling_ready"] is False
    assert manual["full_expert_dataset_ready"] is False
    assert manual["stage4a66d_executed"] is False
    assert manual["stage4a67_executed"] is False
    assert no_rollout["rollout_run"] is False
    assert no_rollout["selected_action_executed"] is False
    assert no_expert["formal_expert_sampling_run"] is False
    assert no_expert["expert_dataset_generated"] is False
    assert no_map["map_predict_called"] is False
    assert no_map["sscnet_inference_called"] is False
    assert no_map["prediction_npz_created"] is False
    assert no_rl["rl_run"] is False
    assert no_rl["gdpo_run"] is False
    assert no_rl["ppo_run"] is False
    assert no_rl["behavior_cloning_run"] is False
    assert no_rl["imitation_learning_run"] is False
    assert no_rl["policy_checkpoint_created"] is False
    assert no_rl["checkpoint_modified"] is False
    assert summary["stage4a66d_executed"] is False
    assert summary["stage4a67_executed"] is False
    bad = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if path.name in FORBIDDEN_EXACT:
            bad.append(str(path.relative_to(output_dir)))
        if path.suffix.lower() == ".npz":
            bad.append(str(path.relative_to(output_dir)))
        if any(token in lower for token in FORBIDDEN_TOKENS):
            bad.append(str(path.relative_to(output_dir)))
    assert not bad, f"forbidden artifacts found: {bad}"
    return {"passed": True}


def test_scene_factory_old_scene_disabled() -> dict[str, Any]:
    import scene_factory

    try:
        scene_factory.build_larger_complex_scene_v1(seed=0, spawn=False)
    except RuntimeError:
        return {"passed": True, "larger_complex_scene_v1_disabled": True}
    raise AssertionError("larger_complex_scene_v1 should remain disabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6.6c camera pose fix outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage4a66c_defaultprim_dir", required=True)
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--fixed_usd", required=True)
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_formal_expert_sampling", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_stage4a66d_not_executed", action="store_true")
    parser.add_argument("--expect_stage4a67_not_executed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (
        args.expect_no_rollout
        and args.expect_no_formal_expert_sampling
        and args.expect_no_map_predict
        and args.expect_no_rl_gdpo
        and args.expect_stage4a66d_not_executed
        and args.expect_stage4a67_not_executed
    ):
        raise ValueError("All expectation flags are required.")
    output_dir = Path(args.output_dir).resolve()
    stage4a66c_defaultprim_dir = Path(args.stage4a66c_defaultprim_dir).resolve()
    source_usd = Path(args.source_usd).resolve()
    fixed_usd = Path(args.fixed_usd).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "context_previous_and_usd": test_context_previous_and_usd(output_dir, stage4a66c_defaultprim_dir, source_usd, fixed_usd),
        "pose_manifests_and_scores": test_pose_manifests_and_scores(output_dir),
        "capture_files": test_capture_files(output_dir),
        "observed_state": test_observed_state(output_dir),
        "gates_and_negative_scope": test_gates_and_negative_scope(output_dir),
        "scene_factory_old_scene_disabled": test_scene_factory_old_scene_disabled(),
    }
    print(json.dumps({"all_passed": True, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

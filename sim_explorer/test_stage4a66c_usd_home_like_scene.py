#!/usr/bin/env python3
"""Validate Stage 4A-6.6c USD-import home_like_scene_v1 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


OLD_LARGER_OUTPUT_DIRS = [
    Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66_larger_complex_scene_v1_validation"),
    Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66a_scene_complexity_audit"),
    Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66b_gui_visual_inspection"),
]

REQUIRED_REPORTS = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "old_scene_rejection_status.json",
    "old_scene_rejection_status.md",
    "old_scene_cleanup_verification.json",
    "old_scene_cleanup_verification.md",
    "usd_staging_report.json",
    "usd_staging_report.md",
    "usd_dependency_report.json",
    "usd_dependency_report.md",
    "usd_hash_manifest.csv",
    "usd_hash_manifest.json",
    "usd_hash_manifest.md",
    "usd_path_patch_report.json",
    "usd_path_patch_report.md",
    "usd_prim_inventory.csv",
    "usd_prim_inventory.json",
    "usd_prim_inventory.md",
    "usd_material_inventory.csv",
    "usd_material_inventory.json",
    "usd_material_inventory.md",
    "usd_texture_inventory.csv",
    "usd_texture_inventory.json",
    "usd_texture_inventory.md",
    "usd_semantic_guess_inventory.csv",
    "usd_semantic_guess_inventory.json",
    "usd_semantic_guess_inventory.md",
    "usd_scene_bounds_report.json",
    "usd_scene_bounds_report.md",
    "usd_quality_warning_report.json",
    "usd_quality_warning_report.md",
    "scene_factory_registration_report.json",
    "scene_factory_registration_report.md",
    "start_variants.json",
    "start_variants.md",
    "start_variant_proposal_report.json",
    "start_variant_proposal_report.md",
    "validation_pose_manifest.json",
    "validation_pose_manifest.md",
    "inspection_pose_manifest.json",
    "inspection_pose_manifest.md",
    "scene_load_validation.json",
    "scene_load_validation.md",
    "fixed_capture_validation.json",
    "fixed_capture_validation.md",
    "visual_inspection_capture_validation.json",
    "visual_inspection_capture_validation.md",
    "observed_state_validation_summary.json",
    "observed_state_validation_summary.md",
    "observed_state_transition_summary.json",
    "observed_state_transition_summary.md",
    "observed_summary.json",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_formal_expert_sampling_report.json",
    "no_formal_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "human_visual_review_checklist.md",
    "human_visual_review_checklist.json",
    "manual_review_gate.json",
    "manual_review_gate.md",
    "future_stage4a66d_usd_scene_audit_command_sketch.md",
    "stage4a66c_usd_home_like_scene_summary.json",
    "stage4a66c_usd_home_like_scene_summary.md",
    "recommended_next_faithful_step.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_STATE_FILES = [
    "observed_state_final.npy",
    "camera_info.json",
    "scene_metadata.json",
]

BLOCKED_REQUIRED_STATE_FILES = [
    "camera_info.json",
    "scene_metadata.json",
]

REQUIRED_IMAGES = [
    "usd_scene_layout_topdown.png",
    "usd_scene_bounds_topdown.png",
    "usd_semantic_guess_topdown.png",
    "usd_prim_category_topdown.png",
    "usd_material_color_topdown.png",
    "start_variants_topdown.png",
    "validation_camera_poses_topdown.png",
    "inspection_camera_poses_topdown.png",
    "rgb_validation_grid.png",
    "depth_validation_grid.png",
    "rgb_inspection_grid.png",
    "depth_inspection_grid.png",
    "observed_topdown_final.png",
    "living_area_closeup.png",
    "kitchen_or_counter_closeup.png",
    "bedroom_or_private_room_closeup.png",
    "bathroom_or_small_room_closeup.png",
    "hallway_or_junction_closeup.png",
    "clutter_or_storage_closeup.png",
]

BLOCKED_REQUIRED_IMAGES = [
    "usd_scene_layout_topdown.png",
    "usd_scene_bounds_topdown.png",
    "usd_semantic_guess_topdown.png",
    "usd_prim_category_topdown.png",
    "usd_material_color_topdown.png",
    "start_variants_topdown.png",
    "validation_camera_poses_topdown.png",
    "inspection_camera_poses_topdown.png",
    "living_area_closeup.png",
    "kitchen_or_counter_closeup.png",
    "bedroom_or_private_room_closeup.png",
    "bathroom_or_small_room_closeup.png",
    "hallway_or_junction_closeup.png",
    "clutter_or_storage_closeup.png",
]

BLOCKED_REQUIRED_REPORTS = [
    "isaac_headless_blocker_report.json",
    "isaac_headless_blocker_report.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
]

FORBIDDEN_EXACT = {
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "rollout_index.html",
    "expert_dataset_manifest.jsonl",
    "expert_dataset_manifest.json",
    "selected_action_execution_report.json",
    "selected_action_report.json",
    "action_execution_report.json",
}

FORBIDDEN_TOKENS = (
    "replay_buffer",
    "policy_checkpoint",
    "global_prediction_layer.npz",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"


def assert_png(path: Path, *, require_nonblank: bool = False) -> None:
    assert_file(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
    image = Image.open(path).convert("RGB")
    assert image.size[0] > 0 and image.size[1] > 0, f"invalid PNG size: {path}"
    if require_nonblank:
        arr = np.asarray(image)
        assert int(arr.max()) > 2 and float(arr.std()) >= 1.0, f"blank or nearly uniform RGB: {path}"


def test_inputs_and_old_scene(source_usd: Path, staged_usd: Path) -> dict[str, Any]:
    assert_file(source_usd)
    assert_file(staged_usd)
    for path in OLD_LARGER_OUTPUT_DIRS:
        assert not path.exists(), f"old larger scene output dir must remain absent: {path}"
    import scene_factory

    try:
        scene_factory.build_larger_complex_scene_v1(seed=0, spawn=False)
        raise AssertionError("larger_complex_scene_v1 builder did not raise")
    except RuntimeError:
        pass
    medium = scene_factory.build_medium_complex_scene(seed=0, spawn=False)
    assert medium["variant"] == "three_rooms"
    meta = scene_factory.build_home_like_scene_v1(seed=0, spawn=False)
    assert meta["staged_usd_path"] == str(staged_usd)
    assert meta["procedural_scene_generated"] is False
    return {"passed": True, "source_sha256": sha256_file(source_usd), "staged_sha256": sha256_file(staged_usd)}


def test_required_artifacts(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_REPORTS + REQUIRED_STATE_FILES:
        assert_file(output_dir / name)
    for name in REQUIRED_IMAGES:
        assert_png(output_dir / name)
    assert_file(output_dir / "visual_inspection_index.html")
    mp4 = output_dir / "usd_scene_flythrough.mp4"
    frames = sorted((output_dir / "usd_scene_flythrough_frames").glob("frame_*.png"))
    if mp4.is_file():
        assert mp4.stat().st_size > 0, "empty flythrough mp4"
    else:
        assert frames, "missing MP4 and fallback flythrough frames"
        for frame in frames[:3]:
            assert_png(frame, require_nonblank=True)
        assert_file(output_dir / "video_generation_skipped.md")
    return {"passed": True, "reports": len(REQUIRED_REPORTS), "images": len(REQUIRED_IMAGES), "flythrough_frames": len(frames)}


def test_blocked_required_artifacts(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_REPORTS + BLOCKED_REQUIRED_REPORTS + BLOCKED_REQUIRED_STATE_FILES:
        assert_file(output_dir / name)
    for name in BLOCKED_REQUIRED_IMAGES:
        assert_png(output_dir / name)
    assert_file(output_dir / "visual_inspection_index.html")
    absent = [
        "observed_state_final.npy",
        "rgb_validation_grid.png",
        "depth_validation_grid.png",
        "rgb_inspection_grid.png",
        "depth_inspection_grid.png",
        "observed_topdown_final.png",
        "usd_scene_flythrough.mp4",
    ]
    for name in absent:
        assert not (output_dir / name).exists(), f"blocked run should not create {name}"
    assert not list(output_dir.glob("validation_rgb_*.png")), "blocked run unexpectedly created validation RGB"
    assert not list(output_dir.glob("validation_depth_*.npy")), "blocked run unexpectedly created validation depth"
    assert not list(output_dir.glob("inspection_rgb_*.png")), "blocked run unexpectedly created inspection RGB"
    assert not list(output_dir.glob("inspection_depth_*.npy")), "blocked run unexpectedly created inspection depth"
    return {
        "passed": True,
        "reports": len(REQUIRED_REPORTS) + len(BLOCKED_REQUIRED_REPORTS),
        "images": len(BLOCKED_REQUIRED_IMAGES),
        "blocked_outputs_absent": len(absent),
    }


def test_usd_reports(output_dir: Path, source_usd: Path, staged_usd: Path) -> dict[str, Any]:
    staging = load_json(output_dir / "usd_staging_report.json")
    dep = load_json(output_dir / "usd_dependency_report.json")
    prims = load_json(output_dir / "usd_prim_inventory.json")
    materials = load_json(output_dir / "usd_material_inventory.json")
    reg = load_json(output_dir / "scene_factory_registration_report.json")
    assert staging["source_usd"] == str(source_usd)
    assert staging["staged_usd"] == str(staged_usd)
    assert staging["source_and_staged_hash_match"] is True
    assert staging["source_sha256"] == sha256_file(source_usd)
    assert staging["staged_sha256"] == sha256_file(staged_usd)
    assert isinstance(dep["missing_dependencies"], list)
    assert len(prims) > 0, "USD prim inventory is empty"
    assert isinstance(materials, list)
    assert reg["loads_staged_usd"] is True
    assert reg["larger_complex_scene_v1_disabled"] is True
    assert reg["medium_three_rooms_preserved"] is True
    return {"passed": True, "prim_count": len(prims), "material_count": len(materials), "dependencies_complete": dep["dependencies_complete"]}


def test_capture_outputs(output_dir: Path, min_validation: int = 20, min_inspection: int = 36) -> dict[str, Any]:
    validation_rgb = sorted(output_dir.glob("validation_rgb_*.png"))
    validation_depth = sorted(output_dir.glob("validation_depth_*.npy"))
    inspection_rgb = sorted(output_dir.glob("inspection_rgb_*.png"))
    inspection_depth = sorted(output_dir.glob("inspection_depth_*.npy"))
    assert len(validation_rgb) >= min_validation, f"validation RGB count {len(validation_rgb)} < {min_validation}"
    assert len(validation_depth) >= min_validation, f"validation depth count {len(validation_depth)} < {min_validation}"
    assert len(inspection_rgb) >= min_inspection, f"inspection RGB count {len(inspection_rgb)} < {min_inspection}"
    assert len(inspection_depth) >= min_inspection, f"inspection depth count {len(inspection_depth)} < {min_inspection}"
    nonblank = 0
    depth_positive = 0
    for idx in range(min_validation):
        assert_png(output_dir / f"validation_rgb_{idx:03d}.png", require_nonblank=True)
        assert_png(output_dir / f"validation_depth_color_{idx:03d}.png")
        assert_file(output_dir / f"validation_pose_{idx:03d}.json")
        depth = np.load(output_dir / f"validation_depth_{idx:03d}.npy")
        positive = np.isfinite(depth) & (depth > 0.0)
        assert depth.ndim == 2
        assert int(np.count_nonzero(positive)) > 0, f"no positive depth for validation {idx}"
        nonblank += 1
        depth_positive += 1
    for idx in range(min_inspection):
        assert_png(output_dir / f"inspection_rgb_{idx:03d}.png", require_nonblank=True)
        assert_png(output_dir / f"inspection_depth_color_{idx:03d}.png")
        assert_file(output_dir / f"inspection_pose_{idx:03d}.json")
        depth = np.load(output_dir / f"inspection_depth_{idx:03d}.npy")
        assert depth.ndim == 2
        assert int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0))) > 0, f"no positive depth for inspection {idx}"
    fixed = load_json(output_dir / "fixed_capture_validation.json")
    visual = load_json(output_dir / "visual_inspection_capture_validation.json")
    assert fixed["nonblank_rgb_count"] >= min_validation
    assert fixed["finite_positive_depth_count"] >= min_validation
    assert visual["nonblank_rgb_count"] >= min_inspection
    assert visual["finite_positive_depth_count"] >= min_inspection
    return {"passed": True, "validation_rgb": len(validation_rgb), "inspection_rgb": len(inspection_rgb), "nonblank_checked": nonblank, "depth_checked": depth_positive}


def test_observed_state(output_dir: Path) -> dict[str, Any]:
    observed = np.load(output_dir / "observed_state_final.npy")
    assert observed.ndim == 3
    labels = set(int(v) for v in np.unique(observed))
    assert {-1, 0, 1}.issubset(labels), f"missing observed labels: {labels}"
    assert labels.issubset({-1, 0, 1}), f"invalid labels present: {labels}"
    summary = load_json(output_dir / "observed_state_validation_summary.json")
    assert summary["invalid_label_count"] == 0
    assert summary["labels_present"]["UNKNOWN"] is True
    assert summary["labels_present"]["FREE"] is True
    assert summary["labels_present"]["OCCUPIED"] is True
    assert 0.02 < float(summary["observed_ratio"]) < 0.60
    assert summary["measured_only"] is True
    assert summary["prediction_used"] is False
    assert summary["map_predict_called"] is False
    return {"passed": True, "shape": list(observed.shape), "observed_ratio": summary["observed_ratio"]}


def test_blocked_isaac_load(output_dir: Path, source_usd: Path, staged_usd: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a66c_usd_home_like_scene_summary.json")
    blocker = load_json(output_dir / "isaac_headless_blocker_report.json")
    scene_load = load_json(output_dir / "scene_load_validation.json")
    fixed = load_json(output_dir / "fixed_capture_validation.json")
    visual = load_json(output_dir / "visual_inspection_capture_validation.json")
    observed = load_json(output_dir / "observed_state_validation_summary.json")
    observed_transition = load_json(output_dir / "observed_state_transition_summary.json")
    assert summary["blocked"] is True
    assert summary["source_usd"] == str(source_usd)
    assert summary["staged_usd"] == str(staged_usd)
    assert summary["source_sha256"] == sha256_file(source_usd)
    assert summary["staged_sha256"] == sha256_file(staged_usd)
    assert summary["isaac_headless_startup_count"] == 1
    assert summary["isaac_headless_loaded_usd"] is False
    assert summary["validation_rgb_depth_valid"] is False
    assert summary["inspection_rgb_depth_valid"] is False
    assert summary["observed_state_valid"] is False
    assert summary["dependencies_complete"] is False
    assert int(summary["missing_dependency_count"]) > 0
    assert int(summary["unresolved_path_count"]) > 0
    assert int(summary["total_prim_count"]) > 0
    assert "sofa_couch" in summary["semantic_category_counts"]
    assert "bed" in summary["semantic_category_counts"]
    assert "bathroom" in summary["semantic_category_counts"]
    assert blocker["blocked"] is True
    assert blocker["isaac_headless_startup_attempted_once"] is True
    assert blocker["isaac_headless_startup_count"] == 1
    assert "out of memory" in blocker["isaac_error_tail"].lower()
    assert blocker["procedural_fallback_used"] is False
    assert blocker["cuboid_fallback_used"] is False
    assert blocker["asset_download_script_run"] is False
    assert scene_load["blocked"] is True
    assert scene_load["scene_loaded"] is False
    assert "out of memory" in scene_load["failure_reason"].lower()
    assert fixed["blocked"] is True
    assert fixed["nonblank_rgb_count"] == 0
    assert fixed["finite_positive_depth_count"] == 0
    assert visual["blocked"] is True
    assert visual["nonblank_rgb_count"] == 0
    assert visual["finite_positive_depth_count"] == 0
    assert observed["blocked"] is True
    assert observed["observed_state_final_created"] is False
    assert observed["measured_only"] is True
    assert observed["prediction_used"] is False
    assert observed["map_predict_called"] is False
    assert observed_transition["blocked"] is True
    assert observed_transition["transition_records"] == []
    return {
        "passed": True,
        "failure_reason": scene_load["failure_reason"],
        "missing_dependency_count": summary["missing_dependency_count"],
        "total_prim_count": summary["total_prim_count"],
    }


def test_gates_and_negative_scope(output_dir: Path) -> dict[str, Any]:
    checklist = load_json(output_dir / "human_visual_review_checklist.json")
    gate = load_json(output_dir / "manual_review_gate.json")
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_expert = load_json(output_dir / "no_formal_expert_sampling_report.json")
    no_map = load_json(output_dir / "no_map_predict_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
    summary = load_json(output_dir / "stage4a66c_usd_home_like_scene_summary.json")
    assert checklist["human_visual_inspection_done"] is False
    assert checklist["user_needs_to_review_visuals"] is True
    assert len(checklist["items"]) >= 14
    assert gate["human_visual_inspection_done"] is False
    assert gate["user_needs_to_review_visuals"] is True
    assert gate["visual_approval_required_before_6_7"] is True
    assert gate["formal_expert_sampling_ready"] is False
    assert gate["full_expert_dataset_ready"] is False
    assert gate["stage4a67_executed"] is False
    assert no_rollout["rollout_run"] is False
    assert no_expert["formal_expert_sampling_run"] is False
    assert no_expert["expert_dataset_generated"] is False
    assert no_map["map_predict_called"] is False
    assert no_map["sscnet_inference_called"] is False
    assert no_rl["rl_run"] is False
    assert no_rl["gdpo_run"] is False
    assert no_rl["policy_checkpoint_created"] is False
    assert summary["formal_expert_sampling_ready"] is False
    assert summary["full_expert_dataset_ready"] is False
    assert summary["stage4a67_executed"] is False
    return {"passed": True}


def test_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    hits = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(output_dir))
        name = path.name
        if name in FORBIDDEN_EXACT:
            hits.append(rel)
        if name.endswith(".npz") and ("prediction" in name or "global_prediction_layer" in rel):
            hits.append(rel)
        if any(token in rel for token in FORBIDDEN_TOKENS):
            hits.append(rel)
    assert not hits, f"forbidden outputs present: {hits}"
    assert not (output_dir / "downloaded_assets").exists(), "asset download directory must not exist"
    return {"passed": True, "forbidden_hits": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6.6c USD import outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--staged_usd", required=True)
    parser.add_argument("--expected_scene_variant", default="home_like_scene_v1")
    parser.add_argument("--expect_old_larger_scene_disabled", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_formal_expert_sampling", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--allow_blocked_isaac_load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.expected_scene_variant == "home_like_scene_v1"
    assert args.expect_old_larger_scene_disabled
    assert args.expect_no_rollout and args.expect_no_formal_expert_sampling and args.expect_no_map_predict and args.expect_no_rl_gdpo
    output_dir = Path(args.output_dir).resolve()
    source_usd = Path(args.source_usd).resolve()
    staged_usd = Path(args.staged_usd).resolve()
    summary = load_json(output_dir / "stage4a66c_usd_home_like_scene_summary.json")
    if summary.get("blocked") is True and args.allow_blocked_isaac_load:
        results = {
            "inputs_and_old_scene": test_inputs_and_old_scene(source_usd, staged_usd),
            "blocked_required_artifacts": test_blocked_required_artifacts(output_dir),
            "usd_reports": test_usd_reports(output_dir, source_usd, staged_usd),
            "blocked_isaac_load": test_blocked_isaac_load(output_dir, source_usd, staged_usd),
            "gates_and_negative_scope": test_gates_and_negative_scope(output_dir),
            "forbidden_outputs": test_forbidden_outputs(output_dir),
        }
        payload = {"all_passed": True, "blocked_isaac_load_validated": True, "results": results}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    assert summary.get("blocked") is not True, "output is blocked; rerun with --allow_blocked_isaac_load to validate blocker evidence"
    results = {
        "inputs_and_old_scene": test_inputs_and_old_scene(source_usd, staged_usd),
        "required_artifacts": test_required_artifacts(output_dir),
        "usd_reports": test_usd_reports(output_dir, source_usd, staged_usd),
        "capture_outputs": test_capture_outputs(output_dir),
        "observed_state": test_observed_state(output_dir),
        "gates_and_negative_scope": test_gates_and_negative_scope(output_dir),
        "forbidden_outputs": test_forbidden_outputs(output_dir),
    }
    payload = {"all_passed": True, "results": results}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

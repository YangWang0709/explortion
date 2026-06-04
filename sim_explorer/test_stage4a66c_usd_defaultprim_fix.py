#!/usr/bin/env python3
"""Validate Stage 4A-6.6c USD defaultPrim-fix outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import fix_stage4a66c_usd_defaultprim as fix_stage


REQUIRED_REPORTS = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "previous_download_dependency_summary.json",
    "previous_download_dependency_summary.md",
    "localized_usd_root_inspection.json",
    "localized_usd_root_inspection.md",
    "top_level_prim_inventory.csv",
    "top_level_prim_inventory.json",
    "defaultprim_candidate_report.json",
    "defaultprim_candidate_report.md",
    "defaultprim_fix_plan.json",
    "defaultprim_fix_plan.md",
    "defaultprim_patch_report.json",
    "defaultprim_patch_report.md",
    "wrapper_usd_report.json",
    "wrapper_usd_report.md",
    "fixed_usd_validation_report.json",
    "fixed_usd_validation_report.md",
    "fixed_usd_hash_manifest.csv",
    "fixed_usd_hash_manifest.json",
    "fixed_usd_hash_manifest.md",
    "unresolved_after_defaultprim_fix.json",
    "unresolved_after_defaultprim_fix.md",
    "scene_factory_defaultprim_registration_report.json",
    "scene_factory_defaultprim_registration_report.md",
    "isaac_validation_gate_report.json",
    "isaac_validation_gate_report.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_expert_sampling_report.json",
    "no_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "manual_review_gate.json",
    "manual_review_gate.md",
    "human_visual_review_checklist.md",
    "human_visual_review_checklist.json",
    "stage4a66c_usd_defaultprim_fix_summary.json",
    "stage4a66c_usd_defaultprim_fix_summary.md",
    "recommended_next_faithful_step.md",
]

SUCCESS_REPORTS = [
    "scene_load_validation_defaultprim_retry.json",
    "scene_load_validation_defaultprim_retry.md",
    "fixed_capture_validation_defaultprim_retry.json",
    "fixed_capture_validation_defaultprim_retry.md",
    "visual_inspection_capture_validation_defaultprim_retry.json",
    "visual_inspection_capture_validation_defaultprim_retry.md",
    "observed_state_validation_summary_defaultprim_retry.json",
    "observed_state_validation_summary_defaultprim_retry.md",
    "observed_state_final.npy",
    "visual_inspection_index.html",
    "rgb_validation_grid.png",
    "depth_validation_grid.png",
    "rgb_inspection_grid.png",
    "depth_inspection_grid.png",
    "observed_topdown_final.png",
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


def assert_png(path: Path, *, require_nonblank: bool = False) -> None:
    assert_file(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
    image = Image.open(path).convert("RGB")
    assert image.size[0] > 0 and image.size[1] > 0, f"bad PNG size: {path}"
    if require_nonblank:
        arr = np.asarray(image)
        assert int(arr.max()) > 2 and float(arr.std()) >= 1.0, f"blank image: {path}"


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_REPORTS:
        assert_file(output_dir / name)
    return {"passed": True, "required_report_count": len(REQUIRED_REPORTS)}


def test_context_and_previous(output_dir: Path, previous_download_output_dir: Path) -> dict[str, Any]:
    loaded = load_json(output_dir / "loaded_context_manifest.json")
    previous = load_json(output_dir / "previous_download_dependency_summary.json")
    assert loaded["dependencies_complete_confirmed"] is True
    assert loaded["remote_official_refs_remaining_confirmed"] is True
    assert loaded["omniverse_refs_remaining_confirmed"] is True
    assert loaded["unresolved_local_deps_remaining_confirmed"] is True
    assert loaded["previous_retry_failed_only_missing_defaultPrim"] is True
    assert loaded["previous_no_rgb_depth"] is True
    assert loaded["previous_no_observed_state_final"] is True
    assert loaded["previous_no_visual_html"] is True
    assert loaded["previous_no_mp4"] is True
    assert loaded["stage4a66d_blocked"] is True
    assert loaded["stage4a67_blocked"] is True
    assert previous["initial_requested_urls"] == 67
    assert previous["dependency_file_count"] == 278
    assert previous["remote_official_refs_remaining"] == 0
    assert previous["omniverse_refs_remaining"] == 0
    assert previous["unresolved_local_deps_remaining"] == 0
    for name in fix_stage.PREVIOUS_REQUIRED_FILES:
        assert_file(previous_download_output_dir / name)
    return {"passed": True, "previous_dependency_file_count": previous["dependency_file_count"]}


def test_defaultprim_fix(output_dir: Path, source_usd: Path, original_staged_usd: Path, fixed_root: Path) -> dict[str, Any]:
    inspection = load_json(output_dir / "localized_usd_root_inspection.json")
    candidate = load_json(output_dir / "defaultprim_candidate_report.json")
    patch = load_json(output_dir / "defaultprim_patch_report.json")
    wrapper = load_json(output_dir / "wrapper_usd_report.json")
    validation = load_json(output_dir / "fixed_usd_validation_report.json")
    unresolved = load_json(output_dir / "unresolved_after_defaultprim_fix.json")
    hashes = load_json(output_dir / "fixed_usd_hash_manifest.json")

    assert inspection["can_open_layer"] is True
    assert inspection["can_open_stage"] is True
    assert inspection["existing_layer_defaultPrim"] == ""
    assert len(load_json(output_dir / "top_level_prim_inventory.json")) >= 1
    assert candidate["blocked"] is False
    assert candidate["chosen_defaultPrim"], "no chosen defaultPrim"
    assert patch["patch_ok"] is True
    assert wrapper["wrapper_used"] in {False, True}
    target = Path(patch["target_usd"])
    assert target.is_file(), f"fixed target missing: {target}"
    pxr = fix_stage.run_pxr_child("validate_fixed", {"path": str(target)}, timeout=240)
    assert pxr["ok"] is True, pxr
    assert pxr["stage_defaultPrim_valid"] is True
    assert pxr["defaultPrim_is_concrete"] is True
    assert pxr["defaultPrim_subtree_prim_count"] > 0
    assert validation["offline_validation_passed"] is True
    assert validation["defaultPrim_valid"] is True
    assert validation["remote_official_refs_remaining"] == 0
    assert validation["omniverse_refs_remaining"] == 0
    assert validation["unresolved_local_deps_remaining"] == 0
    assert unresolved["remote_official_ref_count"] == 0
    assert unresolved["omniverse_ref_count"] == 0
    assert unresolved["unresolved_local_dependency_count"] == 0
    assert validation["source_usd_hash_unchanged"] is True
    assert validation["original_staged_usd_hash_unchanged"] is True
    assert sha256_file(source_usd) == validation["source_sha256_before"]
    assert sha256_file(original_staged_usd) == validation["original_staged_sha256_before"]
    assert Path(hashes["fixed_root"]) == fixed_root.resolve()
    assert hashes["file_count"] >= 1
    return {"passed": True, "target_usd": str(target), "defaultPrim": validation["defaultPrim_path"]}


def test_scene_factory_and_gates(output_dir: Path) -> dict[str, Any]:
    scene = load_json(output_dir / "scene_factory_defaultprim_registration_report.json")
    gate = load_json(output_dir / "isaac_validation_gate_report.json")
    manual = load_json(output_dir / "manual_review_gate.json")
    summary = load_json(output_dir / "stage4a66c_usd_defaultprim_fix_summary.json")
    assert scene["larger_complex_scene_v1_disabled"] is True
    assert scene["home_like_scene_v1_points_to_fixed_usd"] is True
    assert scene["procedural_fallback"] is False
    assert gate["procedural_fallback"] is False
    assert manual["human_visual_inspection_done"] is False
    assert manual["formal_expert_sampling_ready"] is False
    assert manual["full_expert_dataset_ready"] is False
    assert manual["stage4a66d_executed"] is False
    assert manual["stage4a67_executed"] is False
    assert summary["stage4a66d_executed"] is False
    assert summary["stage4a67_executed"] is False
    return {"passed": True, "gate_passed": gate["validation_gate_passed"], "retry_result": gate["retry_result"]}


def test_retry_or_blocked(output_dir: Path, allow_root_blocked: bool, allow_isaac_failed: bool) -> dict[str, Any]:
    candidate = load_json(output_dir / "defaultprim_candidate_report.json")
    gate = load_json(output_dir / "isaac_validation_gate_report.json")
    summary = load_json(output_dir / "stage4a66c_usd_defaultprim_fix_summary.json")

    if candidate.get("blocked"):
        assert allow_root_blocked, "root selection blocked but --allow_blocked_if_no_safe_defaultprim was not set"
        assert_file(output_dir / "defaultprim_root_selection_blocker.json")
        assert gate["retry_attempted"] is False
        return {"passed": True, "mode": "root_selection_blocked"}

    if gate["retry_attempted"]:
        assert gate["retry_attempt_count"] == 1
        assert gate["exactly_one_attempt"] is True
        assert_file(output_dir / "scene_load_validation_defaultprim_retry.json") if gate["retry_result"] == "succeeded" else None
    else:
        assert gate["validation_gate_passed"] is False
        assert_file(output_dir / "defaultprim_root_selection_blocker.json")
        return {"passed": True, "mode": "offline_gate_blocked"}

    if gate["retry_result"] == "succeeded":
        for name in SUCCESS_REPORTS:
            assert_file(output_dir / name)
        assert_png(output_dir / "rgb_validation_grid.png", require_nonblank=True)
        assert_png(output_dir / "rgb_inspection_grid.png", require_nonblank=True)
        observed = np.load(output_dir / "observed_state_final.npy")
        labels = set(int(v) for v in np.unique(observed))
        assert {-1, 0, 1}.issubset(labels), f"observed_state missing labels: {labels}"
        assert labels.issubset({-1, 0, 1}), f"invalid observed_state labels: {labels}"
        manual = load_json(output_dir / "manual_review_gate.json")
        assert manual["human_visual_inspection_done"] is False
        assert manual["formal_expert_sampling_ready"] is False
        assert manual["full_expert_dataset_ready"] is False
        assert summary["completed"] is True
        return {"passed": True, "mode": "isaac_success"}

    assert allow_isaac_failed, "Isaac retry failed but --allow_blocked_if_isaac_retry_fails was not set"
    assert gate["retry_result"] == "failed"
    assert_file(output_dir / "isaac_defaultprim_retry_blocker.json")
    assert_file(output_dir / "isaac_defaultprim_retry_blocker.md")
    assert_file(output_dir / "kit_error_excerpt.txt")
    assert summary["blocked"] is True
    for name in SUCCESS_REPORTS:
        assert not (output_dir / name).exists(), f"success artifact should not be fabricated after failed retry: {name}"
    return {"passed": True, "mode": "isaac_failed"}


def test_negative_scope(output_dir: Path) -> dict[str, Any]:
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_expert = load_json(output_dir / "no_expert_sampling_report.json")
    no_map = load_json(output_dir / "no_map_predict_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
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

    bad = []
    for path in output_dir.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_EXACT:
            bad.append(path.name)
        lower = path.name.lower()
        if any(token in lower for token in FORBIDDEN_TOKENS):
            bad.append(path.name)
        if path.suffix.lower() == ".npz":
            bad.append(path.name)
    assert not bad, f"forbidden artifacts found: {bad}"
    return {"passed": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6.6c USD defaultPrim-fix outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--original_staged_usd", required=True)
    parser.add_argument("--fixed_root", required=True)
    parser.add_argument("--previous_download_output_dir", required=True)
    parser.add_argument("--allow_blocked_if_no_safe_defaultprim", action="store_true")
    parser.add_argument("--allow_blocked_if_isaac_retry_fails", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_formal_expert_sampling", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.expect_no_rollout and args.expect_no_formal_expert_sampling and args.expect_no_map_predict and args.expect_no_rl_gdpo):
        raise ValueError("All negative-scope expectation flags are required.")
    output_dir = Path(args.output_dir).resolve()
    source_usd = Path(args.source_usd).resolve()
    original_staged_usd = Path(args.original_staged_usd).resolve()
    fixed_root = Path(args.fixed_root).resolve()
    previous_download_output_dir = Path(args.previous_download_output_dir).resolve()

    results = {
        "required_outputs": test_required_outputs(output_dir),
        "context_and_previous": test_context_and_previous(output_dir, previous_download_output_dir),
        "defaultprim_fix": test_defaultprim_fix(output_dir, source_usd, original_staged_usd, fixed_root),
        "scene_factory_and_gates": test_scene_factory_and_gates(output_dir),
        "retry_or_blocked": test_retry_or_blocked(
            output_dir,
            allow_root_blocked=bool(args.allow_blocked_if_no_safe_defaultprim),
            allow_isaac_failed=bool(args.allow_blocked_if_isaac_retry_fails),
        ),
        "negative_scope": test_negative_scope(output_dir),
    }
    print(json.dumps({"all_passed": True, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

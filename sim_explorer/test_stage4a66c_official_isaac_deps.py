#!/usr/bin/env python3
"""Validate Stage 4A-6.6c official Isaac dependency download outputs."""

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
    "previous_dependency_fix_summary.json",
    "previous_dependency_fix_summary.md",
    "official_dependency_url_list.csv",
    "official_dependency_url_list.json",
    "official_dependency_download_manifest.csv",
    "official_dependency_download_manifest.json",
    "official_dependency_download_manifest.md",
    "official_dependency_download_errors.csv",
    "official_dependency_download_errors.json",
    "recursive_dependency_report.json",
    "recursive_dependency_report.md",
    "downloaded_file_hash_manifest.csv",
    "downloaded_file_hash_manifest.json",
    "downloaded_file_hash_manifest.md",
    "localized_package_manifest.json",
    "localized_package_manifest.md",
    "localized_patch_report.json",
    "localized_patch_report.md",
    "unresolved_after_patch.json",
    "unresolved_after_patch.md",
    "isaac_retry_gate_report.json",
    "isaac_retry_gate_report.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_expert_sampling_report.json",
    "no_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "stage4a66c_usd_download_official_isaac_deps_summary.json",
    "stage4a66c_usd_download_official_isaac_deps_summary.md",
    "recommended_next_faithful_step.md",
]

BLOCKED_REPORTS = [
    "official_dependency_download_blocker.json",
    "official_dependency_download_blocker.md",
    "dependency_package_request_still_needed.md",
]

RETRY_SUCCESS_REPORTS = [
    "scene_load_validation_retry.json",
    "scene_load_validation_retry.md",
    "fixed_capture_validation_retry.json",
    "fixed_capture_validation_retry.md",
    "visual_inspection_capture_validation_retry.json",
    "visual_inspection_capture_validation_retry.md",
    "observed_state_validation_summary_retry.json",
    "observed_state_validation_summary_retry.md",
    "observed_state_final.npy",
    "visual_inspection_index.html",
    "rgb_validation_grid.png",
    "depth_validation_grid.png",
    "rgb_inspection_grid.png",
    "depth_inspection_grid.png",
    "observed_topdown_final.png",
    "manual_review_gate.json",
    "manual_review_gate.md",
    "human_visual_review_checklist.md",
    "human_visual_review_checklist.json",
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
    "global_prediction_layer.npz",
}

FORBIDDEN_TOKENS = (
    "replay_buffer",
    "policy_checkpoint",
    "checkpoint_epoch",
    "ppo_checkpoint",
    "gdpo_checkpoint",
    "behavior_cloning_checkpoint",
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
    assert image.size[0] > 0 and image.size[1] > 0, f"bad image size: {path}"
    if require_nonblank:
        arr = np.asarray(image)
        assert int(arr.max()) > 2 and float(arr.std()) >= 1.0, f"blank image: {path}"


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_REPORTS:
        assert_file(output_dir / name)
    return {"passed": True, "required_report_count": len(REQUIRED_REPORTS)}


def test_context_and_previous(output_dir: Path, previous_output_dir: Path) -> dict[str, Any]:
    loaded = load_json(output_dir / "loaded_context_manifest.json")
    previous = load_json(output_dir / "previous_dependency_fix_summary.json")
    assert loaded["dependency_package_request_loaded"] is True
    assert loaded["previous_dependency_blocked"] is True
    assert loaded["previous_scene_validation_success"] is False
    assert loaded["previous_no_rgb_depth"] is True
    assert loaded["previous_no_observed_state_final"] is True
    assert loaded["previous_no_mp4"] is True
    assert loaded["stage4a66d_allowed"] is False
    assert loaded["stage4a67_allowed"] is False
    assert previous["previous_blocker_loaded"] is True
    assert previous["dependency_package_request_loaded"] is True
    assert previous["previous_dependencies_complete"] is False
    assert previous["previous_no_isaac_retry"] is True
    assert_file(previous_output_dir / "dependency_package_request.md")
    return {
        "passed": True,
        "previous_unique": previous["previous_remote_unique_dependencies"],
        "previous_occurrences": previous["previous_remote_reference_occurrences"],
    }


def test_url_list(output_dir: Path, previous_output_dir: Path) -> dict[str, Any]:
    urls = load_json(output_dir / "official_dependency_url_list.json")
    previous = load_json(previous_output_dir / "dependency_localization_summary.json")
    expected = int(previous.get("remote_unique_dependencies") or 67)
    assert isinstance(urls, list)
    assert len(urls) == expected, f"expected {expected} initial URLs, got {len(urls)}"
    unique = {row["url"] for row in urls}
    assert len(unique) == len(urls), "initial URL list is not unique"
    assert all("omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/" in row["url"] for row in urls)
    return {"passed": True, "initial_unique_urls": len(urls)}


def test_download_and_patch_reports(output_dir: Path) -> dict[str, Any]:
    downloads = load_json(output_dir / "official_dependency_download_manifest.json")
    errors = load_json(output_dir / "official_dependency_download_errors.json")
    recursive = load_json(output_dir / "recursive_dependency_report.json")
    hashes = load_json(output_dir / "downloaded_file_hash_manifest.json")
    package = load_json(output_dir / "localized_package_manifest.json")
    patch = load_json(output_dir / "localized_patch_report.json")
    unresolved = load_json(output_dir / "unresolved_after_patch.json")
    gate = load_json(output_dir / "isaac_retry_gate_report.json")
    assert isinstance(downloads, list) and downloads, "download manifest is empty"
    assert isinstance(errors, list)
    assert recursive["initial_requested_url_count"] >= 1
    assert hashes["file_count"] == len(hashes["files"])
    assert package["localized_usd_exists"] is True
    assert_file(Path(package["localized_usd"]))
    assert package["dependency_link"]["exists"] is True
    assert patch["source_usd_modified"] is False
    assert patch["original_staged_usd_modified"] is False
    assert "remote_official_ref_count" in unresolved
    assert "unresolved_local_dependency_count" in unresolved
    assert gate["procedural_fallback_used"] is False
    assert gate["cuboid_fallback_used"] is False
    return {
        "passed": True,
        "processed_urls": len(downloads),
        "errors": len(errors),
        "package_file_count": package["package_file_count"],
        "remote_remaining": unresolved["remote_official_ref_count"],
        "unresolved_remaining": unresolved["unresolved_local_dependency_count"],
    }


def test_hashes(output_dir: Path, source_usd: Path, staged_usd: Path, previous_output_dir: Path) -> dict[str, Any]:
    previous = load_json(previous_output_dir / "dependency_localization_summary.json")
    summary = load_json(output_dir / "stage4a66c_usd_download_official_isaac_deps_summary.json")
    source_hash = sha256_file(source_usd)
    staged_hash = sha256_file(staged_usd)
    assert source_hash == previous["source_sha256"], "source USD hash changed"
    assert staged_hash == previous["staged_sha256_after"], "original staged USD hash changed"
    assert summary["source_usd_hash_unchanged"] is True
    assert summary["staged_usd_hash_unchanged"] is True
    return {"passed": True, "source_sha256": source_hash, "staged_sha256": staged_hash}


def test_retry_or_blocked(output_dir: Path, allow_blocked: bool) -> dict[str, Any]:
    gate = load_json(output_dir / "isaac_retry_gate_report.json")
    summary = load_json(output_dir / "stage4a66c_usd_download_official_isaac_deps_summary.json")
    if gate["retry_result"] == "succeeded":
        assert gate["retry_allowed"] is True
        assert gate["retry_attempted"] is True
        assert gate["retry_attempt_count"] == 1
        assert gate["exactly_one_retry"] is True
        for name in RETRY_SUCCESS_REPORTS:
            assert_file(output_dir / name)
        assert_png(output_dir / "rgb_validation_grid.png", require_nonblank=True)
        assert_png(output_dir / "rgb_inspection_grid.png", require_nonblank=True)
        observed = np.load(output_dir / "observed_state_final.npy")
        labels = set(int(v) for v in np.unique(observed))
        assert {-1, 0, 1}.issubset(labels)
        assert labels.issubset({-1, 0, 1})
        manual = load_json(output_dir / "manual_review_gate.json")
        assert manual["human_visual_inspection_done"] is False
        assert manual["formal_expert_sampling_ready"] is False
        assert manual["stage4a66d_executed"] is False
        assert manual["stage4a67_executed"] is False
        return {"passed": True, "retry_succeeded": True}

    assert allow_blocked, "blocked or failed retry mode requires --allow_blocked_if_download_or_unresolved_dependencies"
    for name in BLOCKED_REPORTS:
        assert_file(output_dir / name)
    blocker = load_json(output_dir / "official_dependency_download_blocker.json")
    assert blocker["blocked"] is True
    assert summary["blocked"] is True
    if gate["retry_allowed"] is False:
        assert gate["retry_attempted"] is False
        assert gate["isaac_headless_startup_count"] == 0
    else:
        assert gate["retry_attempted"] is True
        assert gate["retry_attempt_count"] == 1
    if gate["retry_result"] != "succeeded":
        for name in RETRY_SUCCESS_REPORTS:
            assert not (output_dir / name).exists(), f"retry success artifact should not be fabricated: {name}"
    return {"passed": True, "blocked_or_failed_retry": True, "reason": gate.get("reason"), "retry_result": gate["retry_result"]}


def test_negative_scope(output_dir: Path) -> dict[str, Any]:
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_expert = load_json(output_dir / "no_expert_sampling_report.json")
    no_map = load_json(output_dir / "no_map_predict_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
    summary = load_json(output_dir / "stage4a66c_usd_download_official_isaac_deps_summary.json")
    assert no_rollout["rollout_run"] is False
    assert no_rollout["selected_action_executed"] is False
    assert no_expert["expert_sampling_run"] is False
    assert no_expert["expert_dataset_generated"] is False
    assert no_map["map_predict_called"] is False
    assert no_map["sscnet_inference_called"] is False
    assert no_map["prediction_npz_created"] is False
    assert no_rl["rl_run"] is False
    assert no_rl["gdpo_run"] is False
    assert no_rl["ppo_run"] is False
    assert no_rl["behavior_cloning_run"] is False
    assert no_rl["imitation_learning_run"] is False
    assert no_rl["checkpoint_modified"] is False
    assert summary["procedural_fallback"] is False
    assert summary["larger_complex_scene_restored"] is False
    assert summary["rollout"] is False
    assert summary["selected_action"] is False
    assert summary["expert_sampling"] is False
    assert summary["map_predict"] is False
    assert summary["sscnet_inference"] is False
    assert summary["prediction_npz"] is False
    assert summary["training_rl_gdpo_ppo_bc_il"] is False
    assert summary["checkpoint_modified"] is False
    assert summary["stage4a66d_executed"] is False
    assert summary["stage4a67_executed"] is False
    return {"passed": True}


def test_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    hits = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        name = path.name
        if name in FORBIDDEN_EXACT:
            hits.append(rel)
        if name.endswith(".npz") and ("prediction" in rel or "global_prediction_layer" in rel):
            hits.append(rel)
        if any(token in rel.lower() for token in FORBIDDEN_TOKENS):
            hits.append(rel)
    assert not hits, f"forbidden outputs present: {hits}"
    return {"passed": True, "forbidden_hits": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6.6c official Isaac dependency download outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--staged_usd", required=True)
    parser.add_argument("--dependency_request", required=True)
    parser.add_argument("--previous_output_dir", required=True)
    parser.add_argument("--localized_root", required=True)
    parser.add_argument("--allow_blocked_if_download_or_unresolved_dependencies", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_formal_expert_sampling", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.expect_no_rollout
    assert args.expect_no_formal_expert_sampling
    assert args.expect_no_map_predict
    assert args.expect_no_rl_gdpo
    output_dir = Path(args.output_dir).resolve()
    source_usd = Path(args.source_usd).resolve()
    staged_usd = Path(args.staged_usd).resolve()
    previous_output_dir = Path(args.previous_output_dir).resolve()
    assert_file(Path(args.dependency_request).resolve())
    assert Path(args.localized_root).exists(), "localized root missing"
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "context_and_previous": test_context_and_previous(output_dir, previous_output_dir),
        "url_list": test_url_list(output_dir, previous_output_dir),
        "download_and_patch_reports": test_download_and_patch_reports(output_dir),
        "hashes": test_hashes(output_dir, source_usd, staged_usd, previous_output_dir),
        "retry_or_blocked": test_retry_or_blocked(output_dir, args.allow_blocked_if_download_or_unresolved_dependencies),
        "negative_scope": test_negative_scope(output_dir),
        "forbidden_outputs": test_forbidden_outputs(output_dir),
    }
    print(json.dumps({"all_passed": True, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

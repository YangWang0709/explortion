#!/usr/bin/env python3
"""Validate Stage 4A-6.6c corrected-env USD dependency search outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_OUTPUTS = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "previous_dependency_fix_summary.json",
    "previous_dependency_fix_summary.md",
    "env_isaacsim_probe_report.json",
    "env_isaacsim_probe_report.md",
    "env_isaacsim_asset_roots.json",
    "env_isaacsim_asset_roots.md",
    "env_isaacsim_asset_search_results.csv",
    "env_isaacsim_asset_search_results.json",
    "env_isaacsim_asset_search_results.md",
    "localized_package_manifest.json",
    "localized_package_manifest.md",
    "localized_hash_manifest.csv",
    "localized_hash_manifest.json",
    "localized_hash_manifest.md",
    "env_isaacsim_path_patch_report.json",
    "env_isaacsim_path_patch_report.md",
    "unresolved_after_env_isaacsim_search.json",
    "unresolved_after_env_isaacsim_search.md",
    "isaac_retry_gate_report.json",
    "isaac_retry_gate_report.md",
    "env_isaacsim_dependency_blocker.json",
    "env_isaacsim_dependency_blocker.md",
    "dependency_package_request_updated.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_expert_sampling_report.json",
    "no_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "stage4a66c_usd_env_isaacsim_dependency_fix_summary.json",
    "stage4a66c_usd_env_isaacsim_dependency_fix_summary.md",
    "recommended_next_faithful_step.md",
]

RETRY_OUTPUTS = [
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
    "usd_scene_flythrough.mp4",
    "rgb_validation_grid.png",
    "depth_validation_grid.png",
    "rgb_inspection_grid.png",
    "depth_inspection_grid.png",
    "observed_topdown_final.png",
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
    "ppo",
    "gdpo_checkpoint",
    "behavior_cloning",
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


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    for name in REQUIRED_OUTPUTS:
        assert_file(output_dir / name)
    return {"passed": True, "required_output_count": len(REQUIRED_OUTPUTS)}


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
    assert int(previous["previous_remote_unique_dependencies"]) == 67
    assert int(previous["previous_remote_reference_occurrences"]) == 99
    assert previous["previous_no_isaac_retry"] is True
    assert_file(previous_output_dir / "dependency_package_request.md")
    assert_file(previous_output_dir / "dependency_localization_summary.json")
    return {
        "passed": True,
        "previous_unique": previous["previous_remote_unique_dependencies"],
        "previous_occurrences": previous["previous_remote_reference_occurrences"],
    }


def test_probe(output_dir: Path, expect_env: str) -> dict[str, Any]:
    probe = load_json(output_dir / "env_isaacsim_probe_report.json")
    assert probe["effective_conda_env_name"] == expect_env
    assert probe["user_corrected_conda_env_name"] == "env_isaaclab"
    assert probe["python"]["sys_executable"].endswith(f"/envs/{expect_env}/bin/python")
    assert probe["imports"]["isaacsim"]["ok"] is True
    assert "conda_prefix" in probe
    return {
        "passed": True,
        "conda_env": probe["effective_conda_env_name"],
        "python": probe["python"]["sys_executable"],
        "isaacsim_import": probe["imports"]["isaacsim"]["ok"],
        "omni_import": probe["imports"]["omni"]["ok"],
        "pxr_import": probe["imports"]["pxr"]["ok"],
        "pxr_import_with_omni_usd_libs": probe["pxr_import_with_omni_usd_libs"].get("ok"),
    }


def test_search_and_unresolved(output_dir: Path) -> dict[str, Any]:
    results = load_json(output_dir / "env_isaacsim_asset_search_results.json")
    unresolved = load_json(output_dir / "unresolved_after_env_isaacsim_search.json")
    retry_gate = load_json(output_dir / "isaac_retry_gate_report.json")
    assert isinstance(results, list) and len(results) == 67
    assert unresolved["unresolved_unique_remote_dependency_count"] == len(
        [row for row in results if row["still_missing"]]
    )
    assert unresolved["stage4a66d_blocked"] is True
    assert unresolved["stage4a67_blocked"] is True
    assert retry_gate["retry_executed"] is False
    if unresolved["unresolved_unique_remote_dependency_count"] > 0:
        assert retry_gate["retry_allowed"] is False
        assert retry_gate["isaac_headless_startup_count"] == 0
    return {
        "passed": True,
        "dependency_rows": len(results),
        "unresolved_unique": unresolved["unresolved_unique_remote_dependency_count"],
        "retry_allowed": retry_gate["retry_allowed"],
    }


def test_hashes(output_dir: Path, source_usd: Path, staged_usd: Path, previous_output_dir: Path) -> dict[str, Any]:
    previous = load_json(previous_output_dir / "dependency_localization_summary.json")
    hash_manifest = load_json(output_dir / "localized_hash_manifest.json")
    summary = load_json(output_dir / "stage4a66c_usd_env_isaacsim_dependency_fix_summary.json")
    source_hash = sha256_file(source_usd)
    staged_hash = sha256_file(staged_usd)
    assert source_hash == previous["source_sha256"], "source USD hash changed"
    assert staged_hash == previous["staged_sha256_after"], "staged USD hash changed"
    assert hash_manifest["source_usd_hash_unchanged"] is True
    assert hash_manifest["staged_usd_hash_unchanged"] is True
    assert summary["source_usd_hash_unchanged"] is True
    assert summary["staged_usd_hash_unchanged"] is True
    return {"passed": True, "source_sha256": source_hash, "staged_sha256": staged_hash}


def test_blocked_mode(output_dir: Path, allow_blocked: bool) -> dict[str, Any]:
    unresolved = load_json(output_dir / "unresolved_after_env_isaacsim_search.json")
    blocker = load_json(output_dir / "env_isaacsim_dependency_blocker.json")
    retry_gate = load_json(output_dir / "isaac_retry_gate_report.json")
    if unresolved["unresolved_unique_remote_dependency_count"] > 0:
        assert allow_blocked, "blocked unresolved mode requires --allow_blocked_if_unresolved_dependencies"
        assert blocker["blocked"] is True
        assert blocker["user_confirmed_conda_env"] == "env_isaaclab"
        assert blocker["current_usd_self_contained"] is False
        assert blocker["need_complete_dependency_package_or_exact_url_download_permission"] is True
        assert blocker["stage4a66d_blocked"] is True
        assert blocker["stage4a67_blocked"] is True
        assert retry_gate["retry_allowed"] is False
        for name in RETRY_OUTPUTS:
            assert not (output_dir / name).exists(), f"retry artifact should not exist in blocked mode: {name}"
        assert not list(output_dir.glob("validation_rgb_*.png")), "blocked mode created validation RGB"
        assert not list(output_dir.glob("inspection_rgb_*.png")), "blocked mode created inspection RGB"
        return {"passed": True, "blocked_mode": True}
    assert retry_gate["retry_allowed"] is True
    assert retry_gate["exactly_one_retry"] is True
    assert (output_dir / "observed_state_final.npy").is_file()
    return {"passed": True, "blocked_mode": False}


def test_negative_scope(output_dir: Path) -> dict[str, Any]:
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_expert = load_json(output_dir / "no_expert_sampling_report.json")
    no_map = load_json(output_dir / "no_map_predict_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
    summary = load_json(output_dir / "stage4a66c_usd_env_isaacsim_dependency_fix_summary.json")
    assert no_rollout["rollout_run"] is False
    assert no_expert["formal_expert_sampling_run"] is False
    assert no_expert["expert_dataset_generated"] is False
    assert no_map["map_predict_called"] is False
    assert no_map["sscnet_inference_called"] is False
    assert no_rl["rl_run"] is False
    assert no_rl["gdpo_run"] is False
    assert no_rl["checkpoint_modified"] is False
    assert summary["no_rollout"] is True
    assert summary["no_formal_expert_sampling"] is True
    assert summary["no_map_predict"] is True
    assert summary["no_rl_gdpo"] is True
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
        if name.endswith(".npz") and ("prediction" in name or "global_prediction_layer" in rel):
            hits.append(rel)
        if any(token in rel.lower() for token in FORBIDDEN_TOKENS):
            hits.append(rel)
    assert not hits, f"forbidden outputs present: {hits}"
    return {"passed": True, "forbidden_hits": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate corrected-env Stage 4A-6.6c dependency fix outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--staged_usd", required=True)
    parser.add_argument("--previous_output_dir", required=True)
    parser.add_argument("--expect_env_isaacsim", action="store_true")
    parser.add_argument("--expect_env_isaaclab", action="store_true")
    parser.add_argument("--allow_blocked_if_unresolved_dependencies", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_formal_expert_sampling", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    source_usd = Path(args.source_usd).resolve()
    staged_usd = Path(args.staged_usd).resolve()
    previous_output_dir = Path(args.previous_output_dir).resolve()
    assert args.expect_env_isaaclab or args.expect_env_isaacsim, "must pass an expected env flag"
    expect_env = "env_isaaclab" if args.expect_env_isaaclab else "env_isaacsim"
    assert args.expect_no_rollout
    assert args.expect_no_formal_expert_sampling
    assert args.expect_no_map_predict
    assert args.expect_no_rl_gdpo
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "context_and_previous": test_context_and_previous(output_dir, previous_output_dir),
        "probe": test_probe(output_dir, expect_env),
        "search_and_unresolved": test_search_and_unresolved(output_dir),
        "hashes": test_hashes(output_dir, source_usd, staged_usd, previous_output_dir),
        "blocked_mode": test_blocked_mode(output_dir, args.allow_blocked_if_unresolved_dependencies),
        "negative_scope": test_negative_scope(output_dir),
        "forbidden_outputs": test_forbidden_outputs(output_dir),
    }
    print(json.dumps({"all_passed": True, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

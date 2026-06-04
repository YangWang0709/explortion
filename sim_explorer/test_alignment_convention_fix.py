#!/usr/bin/env python3
"""Validate Stage 4A-6.3 alignment convention audit/eval outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FORBIDDEN_NPZ_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "optimizer",
    "ppo",
    "policy",
    "rl",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(path: str | Path, label: str | None = None) -> Path:
    target = Path(path)
    if not target.exists():
        raise AssertionError(f"missing {label or target}: {target}")
    if target.is_file() and target.stat().st_size <= 0:
        raise AssertionError(f"empty {label or target}: {target}")
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_false(summary: dict[str, Any], keys: list[str], context: str) -> None:
    for key in keys:
        if bool(summary.get(key, False)):
            raise AssertionError(f"{context} reports forbidden true flag: {key}")


def assert_npz_clean(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        lower = {name.lower() for name in data.files}
        bad = sorted(lower & FORBIDDEN_NPZ_KEYS)
        if bad:
            raise AssertionError(f"{path} contains forbidden fields: {bad}")


def validate_axis_audit(axis_dir: Path) -> dict[str, Any]:
    require(axis_dir / "axis_convention_audit.json")
    require(axis_dir / "axis_convention_audit.md")
    require(axis_dir / "position_flatten_examples.csv")
    require(axis_dir / "axis_diagram_text.txt")
    audit = load_json(axis_dir / "axis_convention_audit.json")
    if "code_consistent_v1" not in audit.get("conventions", {}):
        raise AssertionError("code_consistent_v1 missing from axis audit")
    if audit.get("recommended_code_consistent_convention") != "code_consistent_v1":
        raise AssertionError("axis audit did not recommend code_consistent_v1 as code-consistent")
    assert_false(
        audit,
        ["planning_or_training_used", "prediction_writeback", "future_observations_used_for_planning"],
        "axis audit",
    )
    return audit


def validate_convention_eval(eval_dir: Path) -> dict[str, Any]:
    required = [
        "convention_metrics.csv",
        "convention_metrics.json",
        "convention_summary.md",
        "recommendation_alignment_fix.md",
        "synthetic_blob_projection.csv",
        "synthetic_blob_projection.png",
        "synthetic_alignment_test.md",
        "reliability_compare_conventions.png",
        "brier_compare_conventions.png",
        "later_measured_fraction_compare.png",
    ]
    for rel in required:
        require(eval_dir / rel)
    summary = load_json(eval_dir / "convention_metrics.json")
    conventions = set(summary.get("conventions", []))
    if "current_default_v0" not in conventions or "code_consistent_v1" not in conventions:
        raise AssertionError(f"required conventions missing: {conventions}")
    if "post-hoc" not in str(summary.get("future_observations_usage", "")):
        raise AssertionError("future observations must be marked post-hoc/evaluation-only")
    assert_false(
        summary,
        [
            "planning_or_training_used",
            "prediction_writeback",
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_used_for_a_star",
            "prediction_blocks_rays",
            "rl_or_ppo_training",
            "optimizer_step",
            "behavior_cloning_training",
            "imitation_learning_training",
            "sscnet_training",
            "target_or_ground_truth_used_for_scoring",
        ],
        "convention eval",
    )
    for path in eval_dir.glob("global_prediction_layer_step*_*.npz"):
        assert_npz_clean(path)
        with np.load(path, allow_pickle=False) as data:
            if not bool(np.asarray(data["strict_no_observed_write"]).item()):
                raise AssertionError(f"{path} does not report strict_no_observed_write")
            valid = data["global_prediction_valid"]
            pred = data["global_pred_class"]
            if valid.shape != pred.shape:
                raise AssertionError(f"{path} valid/pred shape mismatch")
    return summary


def validate_fixed_single(single_dir: Path) -> dict[str, Any] | None:
    if not single_dir.exists():
        return None
    require(single_dir / "prediction_alignment_summary.json")
    require(single_dir / "global_prediction_layer.npz")
    summary = load_json(single_dir / "prediction_alignment_summary.json")
    if summary.get("alignment_convention") != "code_consistent_v1":
        raise AssertionError("fixed single-frame smoke did not use code_consistent_v1")
    if not bool(summary.get("strict_no_observed_write", False)):
        raise AssertionError("fixed single-frame smoke reports observed_state modification")
    observed_path = Path(summary["observed_state_source"])
    if observed_path.exists():
        before = str(summary.get("observed_state_sha256_before", ""))
        after = str(summary.get("observed_state_sha256_after", ""))
        actual = sha256_file(observed_path)
        if before != after or actual != before:
            raise AssertionError("observed_state hash changed in fixed single smoke")
    with np.load(single_dir / "global_prediction_layer.npz", allow_pickle=False) as data:
        if tuple(data["global_prediction_valid"].shape) != tuple(summary["observed_state_shape"]):
            raise AssertionError("fixed prediction shape does not match observed_state shape")
    assert_npz_clean(single_dir / "global_prediction_layer.npz")
    return summary


def validate_fixed_expert(expert_dir: Path) -> dict[str, Any] | None:
    if not expert_dir.exists():
        return None
    require(expert_dir / "expert_step_decision.json")
    require(expert_dir / "expert_step_decision.npz")
    summary = load_json(expert_dir / "expert_step_decision.json")
    diagnostics = summary.get("diagnostics", {})
    if not bool(diagnostics.get("observed_state_hash_unchanged", False)):
        raise AssertionError("fixed one-step expert reports observed_state hash changed")
    for key in ("prediction_used_for_traversability", "prediction_used_for_collision", "prediction_used_for_a_star"):
        if bool(diagnostics.get(key, False)):
            raise AssertionError(f"fixed one-step expert reports forbidden prediction use: {key}")
    assert_npz_clean(expert_dir / "expert_step_decision.npz")
    return summary


def validate_fixed_rollout(rollout_dir: Path) -> dict[str, Any] | None:
    if not rollout_dir.exists():
        return None
    require(rollout_dir / "episode_summary.json")
    require(rollout_dir / "transitions.jsonl")
    summary = load_json(rollout_dir / "episode_summary.json")
    if summary.get("alignment_convention") != "code_consistent_v1":
        raise AssertionError("fixed rollout did not use code_consistent_v1")
    assert_false(
        summary,
        [
            "checkpoint_modified",
            "prediction_writeback",
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_used_for_a_star",
            "prediction_blocks_rays",
            "prediction_used_for_candidate_reachability",
            "prediction_used_for_collision_checking",
            "prediction_used_for_a_star_traversability",
            "rl_optimizer_training_run",
            "rl_optimizer_bc_il_training_run",
            "rl_or_ppo_training",
            "optimizer_step",
            "behavior_cloning_training",
            "imitation_learning_training",
            "sscnet_training",
        ],
        "fixed rollout",
    )
    for prediction_dir in sorted(rollout_dir.glob("prediction_step*")):
        if prediction_dir.is_dir():
            require(prediction_dir / "prediction_alignment_summary.json")
            step_summary = load_json(prediction_dir / "prediction_alignment_summary.json")
            if step_summary.get("alignment_convention") != "code_consistent_v1":
                raise AssertionError(f"{prediction_dir} has wrong alignment convention")
            if not bool(step_summary.get("strict_no_observed_write", False)):
                raise AssertionError(f"{prediction_dir} reports observed write")
            require(prediction_dir / "global_prediction_layer.npz")
            assert_npz_clean(prediction_dir / "global_prediction_layer.npz")
    return summary


def run_tests(args: argparse.Namespace) -> None:
    axis_dir = Path(args.axis_audit_dir).resolve()
    eval_dir = Path(args.convention_eval_dir).resolve()
    validate_axis_audit(axis_dir)
    eval_summary = validate_convention_eval(eval_dir)
    single = validate_fixed_single(Path(args.fixed_single_dir).resolve())
    expert = validate_fixed_expert(Path(args.fixed_expert_dir).resolve())
    rollout = validate_fixed_rollout(Path(args.fixed_rollout_dir).resolve())

    print("Stage 4A-6.3 alignment convention validation passed.")
    print(f"axis_audit_dir: {axis_dir}")
    print(f"convention_eval_dir: {eval_dir}")
    print(f"recommended_convention: {eval_summary.get('recommended_convention')}")
    print(f"fixed_single_checked: {single is not None}")
    print(f"fixed_expert_checked: {expert is not None}")
    print(f"fixed_rollout_checked: {rollout is not None}")
    print("prediction_writeback: false")
    print("prediction_used_for_traversability: false")
    print("prediction_used_for_collision: false")
    print("prediction_used_for_a_star: false")
    print("prediction_blocks_rays: false")
    print("future_observations_used_for_planning: false")
    print("target_or_ground_truth_leakage: false")
    print("rl_optimizer_bc_il_training_run: false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis_audit_dir", type=Path, required=True)
    parser.add_argument("--convention_eval_dir", type=Path, required=True)
    parser.add_argument("--fixed_single_dir", type=Path, required=True)
    parser.add_argument("--fixed_expert_dir", type=Path, required=True)
    parser.add_argument("--fixed_rollout_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_tests(parse_args())


if __name__ == "__main__":
    main()

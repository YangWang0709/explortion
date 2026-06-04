#!/usr/bin/env python3
"""Validate Stage 4A-6.5h offline tree utility outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from offline_tree_utility_prototype import (
    ROLLOUT_LIKE_PATTERNS,
    run_synthetic_tests,
)


REQUIRED_OUTPUTS = [
    "synthetic_tree_tests.json",
    "synthetic_tree_tests.md",
    "tree_formula_reference.md",
    "loaded_candidate_fields.json",
    "missing_fields_report.json",
    "one_step_star_results.csv",
    "recorded_episode_chain_results.csv",
    "shallow_pseudo_tree_results.csv",
    "subsequent_best_decisions.csv",
    "tree_utility_comparison_summary.json",
    "tree_utility_comparison_summary.md",
    "recommended_next_faithful_step.md",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=path,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        return f"error: {exc}"
    if proc.returncode != 0:
        return f"not_git_or_error: {proc.stderr.strip()}"
    return proc.stdout.strip()


def run_tests(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")

    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.exists(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")

    synthetic = load_json(output_dir / "synthetic_tree_tests.json")
    _assert(synthetic.get("all_passed") is True, "saved synthetic tests did not pass")
    rerun_synthetic = run_synthetic_tests()
    _assert(rerun_synthetic.get("all_passed") is True, "rerun synthetic tests did not pass")

    summary = load_json(output_dir / "tree_utility_comparison_summary.json")
    safety = summary.get("safety", {})
    for key in (
        "isaac_startup",
        "rollout",
        "new_expert_step",
        "map_predict_rerun",
        "sscnet_inference",
        "training_rl_bc_il",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_writeback",
        "target_ground_truth_scoring",
        "external_source_modified_or_built",
    ):
        _assert(not bool(safety.get(key, False)), f"safety flag is true: {key}")

    rollout_like = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        rollout_like.extend(sorted(output_dir.glob(pattern)))
    _assert(not rollout_like, f"rollout-like files found in output dir: {rollout_like}")

    one_step_csv = (output_dir / "one_step_star_results.csv").read_text(encoding="utf-8")
    decisions_csv = (output_dir / "subsequent_best_decisions.csv").read_text(encoding="utf-8")
    _assert("one_step_star" in one_step_csv, "one_step_star_results.csv missing rows")
    _assert("shallow_pseudo_tree_topk" in decisions_csv, "subsequent_best_decisions.csv missing pseudo-tree decisions")

    external_repo = Path("/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning")
    ssc_repo = Path("/home/ubuntu22/sc_explorer_ws/ssc_exploration")
    result = {
        "output_dir": str(output_dir),
        "required_outputs_exist": True,
        "synthetic_tests_pass": True,
        "safety_flags_false": True,
        "rollout_like_files_in_output_dir": [],
        "external_repo_git_status_short": git_status(external_repo),
        "ssc_exploration_git_status_short": git_status(ssc_repo),
        "note": "Git status may include historical workspace changes; this test only reports it.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tests(Path(args.output_dir))

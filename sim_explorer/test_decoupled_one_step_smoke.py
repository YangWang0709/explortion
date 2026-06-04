#!/usr/bin/env python3
"""Stage 4A-6.5c one-step decoupled_sc smoke test.

Runs the existing expert scorer on one saved observed map and prediction layer.
No Isaac, rollout, map_predict, or training code is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from run_sim_expert_step import run_expert_step
from select_decoupled_one_step_case import (
    DEFAULT_COUNTERFACTUAL_DIR,
    DEFAULT_EMPTY_BASELINE_EPISODE,
    DEFAULT_FIXED_RAW_EPISODE,
    DEFAULT_GATING_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANK_SENSITIVITY_DIR,
    select_case,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_key(candidate: Any) -> str:
    grid = getattr(candidate, "grid_position")
    return f"grid:{int(grid[0])},{int(grid[1])},{int(grid[2])}"


def candidate_summary(candidate: Any | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "id": int(getattr(candidate, "id")),
        "key": candidate_key(candidate),
        "grid_position": [int(v) for v in getattr(candidate, "grid_position")],
        "world_position": [float(v) for v in getattr(candidate, "world_position")],
        "yaw": float(getattr(candidate, "yaw")),
        "score": float(getattr(candidate, "final_score")),
        "gain_exp": float(getattr(candidate, "gain_exp")),
        "gain_sc": float(getattr(candidate, "gain_sc")),
        "raw_gain_sc": float(getattr(candidate, "raw_gain_sc")),
        "effective_gain_sc": float(getattr(candidate, "effective_gain_sc")),
        "weighted_gain_sc": float(getattr(candidate, "weighted_gain_sc")),
        "base_exp_utility": float(getattr(candidate, "base_exp_utility", getattr(candidate, "utility_exp", 0.0))),
        "final_score_decoupled_sc": float(getattr(candidate, "final_score_decoupled_sc", 0.0)),
        "path_cost": float(getattr(candidate, "path_cost")),
        "astar_path_length_m": float(getattr(candidate, "astar_path_length_m")),
        "astar_num_expanded": int(getattr(candidate, "astar_num_expanded")),
        "visible_count": int(getattr(candidate, "visible_count")),
        "predicted_unmeasured_visible_count": int(getattr(candidate, "predicted_unmeasured_visible_count")),
        "sc_selected_voxel_count": int(getattr(candidate, "sc_selected_voxel_count")),
    }


def top_keys(result: dict[str, Any], k: int) -> list[str]:
    return [candidate_key(candidate) for candidate in result["top_candidates"][:k]]


def overlap(a: list[str], b: list[str]) -> dict[str, Any]:
    set_a = set(a)
    set_b = set(b)
    both = sorted(set_a & set_b)
    union = set_a | set_b
    return {
        "count": int(len(both)),
        "jaccard": float(len(both) / len(union)) if union else 1.0,
        "keys": both,
    }


def find_candidate(result: dict[str, Any], key: str) -> Any | None:
    for candidate in result.get("all_candidates", []):
        if candidate_key(candidate) == key:
            return candidate
    return None


def make_args(case: dict[str, Any], output_dir: Path, *, score_gain_mode: str, sc_gain_weight: float) -> Namespace:
    runtime = dict(case["runtime_args"])
    return Namespace(
        observed_state=runtime["observed_state"],
        observed_summary=runtime.get("observed_summary"),
        episode_summary=runtime["episode_summary"],
        camera_info=runtime["camera_info"],
        pose_json=runtime["pose_json"],
        output_dir=str(output_dir),
        num_candidates=int(runtime["num_candidates"]),
        top_n=int(runtime["top_n"]),
        gain_mode=str(runtime["gain_mode"]),
        prediction_mode="sim_npz",
        prediction_npz=runtime["prediction_npz"],
        tau=float(runtime["tau"]),
        sc_gain_formula=str(runtime["sc_gain_formula"]),
        sc_occ_threshold=float(runtime["sc_occ_threshold"]),
        sc_conf_threshold=float(runtime["sc_conf_threshold"]),
        sc_count_mode=str(runtime["sc_count_mode"]),
        calibration_table=runtime.get("calibration_table"),
        alignment_convention=str(runtime["alignment_convention"]),
        sc_gain_weight=float(sc_gain_weight),
        sc_gain_cap=runtime.get("sc_gain_cap"),
        score_gain_mode=str(score_gain_mode),
        path_cost_mode=str(runtime["path_cost_mode"]),
        candidate_sampling_mode=str(runtime["candidate_sampling_mode"]),
        snap_start_to_traversable=bool(runtime["snap_start_to_traversable"]),
        max_snap_radius_cells=int(runtime["max_snap_radius_cells"]),
        seed=int(runtime["seed"]),
        max_range_voxels=int(runtime["max_range_voxels"]),
        num_yaw=int(runtime["num_yaw"]),
        num_pitch=int(runtime["num_pitch"]),
        fov_yaw_deg=float(runtime["fov_yaw_deg"]),
        fov_pitch_deg=float(runtime["fov_pitch_deg"]),
        save_viz=False,
    )


def _write_blocked_report(output_dir: Path, case: dict[str, Any]) -> None:
    comparison = {
        "stage": "Stage 4A-6.5c decoupled_sc one-step smoke",
        "status": "blocked",
        "blocked": True,
        "main_blocker": case.get("blocker"),
        "selected_case": case,
    }
    save_json(output_dir / "one_step_comparison.json", comparison)
    md = [
        "# Stage 4A-6.5c One-Step Comparison",
        "",
        "## Stage 4A-6.5c Result",
        "- Completed: selected_case.json was written with blocked status",
        "- Blocked: one-step runtime was not run",
        f"- Main blocker: {case.get('blocker')}",
        "",
    ]
    (output_dir / "one_step_comparison.md").write_text("\n".join(md), encoding="utf-8")


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _md_for_comparison(comparison: dict[str, Any]) -> str:
    case = comparison["selected_case"]
    baseline = comparison["baseline"]["best_candidate"]
    decoupled = comparison["decoupled_sc"]["best_candidate"]
    offline = case["offline_expected"]
    safety = comparison["safety"]
    interpretation = comparison["interpretation"]
    return "\n".join(
        [
            "# Stage 4A-6.5c One-Step Comparison",
            "",
            "## Stage 4A-6.5c Result",
            "- Completed: selected_case.json, one_step_comparison.json, one_step_comparison.md",
            f"- Blocked: {comparison['blocked']}",
            f"- Main blocker: {comparison['main_blocker']}",
            "",
            "## Selected Case",
            f"- config: {case['config']}",
            f"- step: {case['step']}",
            f"- formula: {case['formula']}",
            f"- lambda: {case['lambda']}",
            f"- observed_state: {case['observed_state']}",
            f"- prediction: {case['prediction_npz']}",
            f"- offline expected changed action: {offline['changed_action']}",
            "",
            "## One-Step Baseline",
            f"- best candidate: {baseline['key']}",
            f"- best position: {baseline['world_position']}",
            f"- score: {_format_value(baseline['score'])}",
            f"- gain_exp: {_format_value(baseline['gain_exp'])}",
            f"- raw_gain_sc: {_format_value(baseline['raw_gain_sc'])}",
            f"- effective_gain_sc: {_format_value(baseline['effective_gain_sc'])}",
            f"- path_cost: {_format_value(baseline['path_cost'])}",
            "",
            "## One-Step Decoupled SC",
            f"- best candidate: {decoupled['key']}",
            f"- best position: {decoupled['world_position']}",
            f"- score: {_format_value(decoupled['score'])}",
            f"- gain_exp: {_format_value(decoupled['gain_exp'])}",
            f"- raw_gain_sc: {_format_value(decoupled['raw_gain_sc'])}",
            f"- effective_gain_sc: {_format_value(decoupled['effective_gain_sc'])}",
            f"- path_cost: {_format_value(decoupled['path_cost'])}",
            f"- changed vs baseline: {comparison['decoupled_sc']['changed_vs_baseline']}",
            f"- matches offline expected: {comparison['decoupled_sc']['matches_offline_expected']}",
            "",
            "## Top-K Overlap",
            f"- baseline vs decoupled top-5 overlap: {comparison['overlap']['baseline_vs_decoupled']['top5']['count']}/5",
            f"- baseline vs decoupled top-16 overlap: {comparison['overlap']['baseline_vs_decoupled']['top16']['count']}/16",
            f"- decoupled vs offline top-5 overlap: {comparison['overlap']['decoupled_vs_offline']['top5']['count']}/5",
            f"- decoupled vs offline top-16 overlap: {comparison['overlap']['decoupled_vs_offline']['top16']['count']}/16",
            "",
            "## Interpretation",
            f"- did decoupled scoring overcome path_cost dominance: {interpretation['did_decoupled_scoring_overcome_path_cost_dominance']}",
            f"- is this formula plausible: {interpretation['is_formula_plausible']}",
            f"- caution: {interpretation['caution']}",
            "",
            "## Safety",
            f"- rollout: {safety['rollout']}",
            f"- Isaac startup: {safety['isaac_startup']}",
            f"- map_predict rerun: {safety['map_predict_rerun']}",
            f"- RL/IL/training: {safety['rl_il_training']}",
            f"- checkpoint modified: {safety['checkpoint_modified']}",
            f"- observed_state modified: {safety['observed_state_modified']}",
            f"- prediction writeback: {safety['prediction_writeback']}",
            f"- leakage: {safety['leakage']}",
            "",
            "## Next Recommended Small Task",
            f"- {comparison['next_recommended_small_task']}",
            "",
        ]
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case = select_case(
        counterfactual_dir=Path(args.counterfactual_dir),
        rank_sensitivity_dir=Path(args.rank_sensitivity_dir),
        fixed_raw_episode=Path(args.fixed_raw_episode),
        empty_baseline_episode=Path(args.empty_baseline_episode),
        gating_root=Path(args.gating_root),
        output_dir=output_dir,
    )
    if case.get("status") != "selected":
        _write_blocked_report(output_dir, case)
        print(f"BLOCKED: {case.get('blocker')}")
        return {"status": "blocked", "case": case}

    observed_state_path = Path(case["observed_state"])
    observed_hash_before = sha256_file(observed_state_path)
    baseline_args = make_args(
        case,
        output_dir / "baseline_runtime",
        score_gain_mode=str(case["gating"]["score_gain_mode"]),
        sc_gain_weight=float(case["gating"]["sc_gain_weight"]),
    )
    decoupled_args = make_args(
        case,
        output_dir / "decoupled_sc_lambda0p5",
        score_gain_mode="decoupled_sc",
        sc_gain_weight=float(case["lambda"]),
    )

    baseline_result = run_expert_step(baseline_args)
    decoupled_result = run_expert_step(decoupled_args)
    observed_hash_after = sha256_file(observed_state_path)

    baseline_best = candidate_summary(baseline_result["best_candidate"])
    decoupled_best = candidate_summary(decoupled_result["best_candidate"])
    assert baseline_best is not None
    assert decoupled_best is not None

    offline_expected_key = str(case["offline_expected"]["top_candidate_key"])
    offline_own_key = str(case["offline_expected"]["own_selected_candidate_key"])
    runtime_expected_candidate = candidate_summary(find_candidate(decoupled_result, offline_expected_key))
    runtime_own_candidate = candidate_summary(find_candidate(decoupled_result, offline_own_key))

    baseline_top5 = top_keys(baseline_result, 5)
    baseline_top16 = top_keys(baseline_result, 16)
    decoupled_top5 = top_keys(decoupled_result, 5)
    decoupled_top16 = top_keys(decoupled_result, 16)
    offline_top16 = [str(row["candidate_key"]) for row in case.get("offline_decoupled_top_candidates", [])[:16]]
    offline_top5 = offline_top16[:5]

    changed_vs_baseline = decoupled_best["key"] != baseline_best["key"]
    matches_offline_expected = decoupled_best["key"] == offline_expected_key
    baseline_matches_offline_own = baseline_best["key"] == offline_own_key
    observed_state_modified = observed_hash_before != observed_hash_after
    safety = {
        "rollout": False,
        "isaac_startup": False,
        "map_predict_rerun": False,
        "rl_il_training": False,
        "checkpoint_modified": False,
        "observed_state_modified": observed_state_modified,
        "prediction_writeback": False,
        "leakage": (
            "none detected: prediction read-only; not used for A*/traversability/collision/ray blocking; "
            "no target_lr/target_hr/ground_truth scoring"
        ),
        "baseline_observed_hash_unchanged": bool(
            baseline_result["diagnostics"].get("observed_state_hash_unchanged", False)
        ),
        "decoupled_observed_hash_unchanged": bool(
            decoupled_result["diagnostics"].get("observed_state_hash_unchanged", False)
        ),
    }

    if changed_vs_baseline and matches_offline_expected:
        plausibility = "yes: one-step runtime reproduced the offline decoupled top-1 change"
        next_task = "If changed and plausible: spatial visualization for this one step."
        caution = "This is still one saved step only; no rollout or coverage claim."
    elif changed_vs_baseline:
        plausibility = "partial: runtime changed action but did not match the offline top-16 expectation"
        next_task = "If not changed or mismatch: candidate logging/reproducibility fix."
        caution = "Offline 6.5b ranked saved top-16 candidates; runtime may expose candidates outside that saved set."
    else:
        plausibility = "weak: runtime did not change the selected action"
        next_task = "If not changed or mismatch: candidate logging/reproducibility fix."
        caution = "The offline change may not transfer through full candidate generation and scoring."

    comparison = {
        "stage": "Stage 4A-6.5c decoupled_sc one-step smoke",
        "status": "completed",
        "blocked": False,
        "main_blocker": "None",
        "selected_case": case,
        "baseline": {
            "output_dir": str(Path(baseline_args.output_dir).resolve()),
            "score_gain_mode": baseline_args.score_gain_mode,
            "best_candidate": baseline_best,
            "matches_offline_own_selected": baseline_matches_offline_own,
            "top5_keys": baseline_top5,
            "top16_keys": baseline_top16,
        },
        "decoupled_sc": {
            "output_dir": str(Path(decoupled_args.output_dir).resolve()),
            "score_gain_mode": "decoupled_sc",
            "sc_gain_weight": float(case["lambda"]),
            "best_candidate": decoupled_best,
            "changed_vs_baseline": changed_vs_baseline,
            "matches_offline_expected": matches_offline_expected,
            "runtime_candidate_for_offline_expected": runtime_expected_candidate,
            "runtime_candidate_for_offline_own_selected": runtime_own_candidate,
            "top5_keys": decoupled_top5,
            "top16_keys": decoupled_top16,
        },
        "offline_expected": case["offline_expected"],
        "overlap": {
            "baseline_vs_decoupled": {
                "top5": overlap(baseline_top5, decoupled_top5),
                "top16": overlap(baseline_top16, decoupled_top16),
            },
            "decoupled_vs_offline": {
                "top5": overlap(decoupled_top5, offline_top5),
                "top16": overlap(decoupled_top16, offline_top16),
            },
        },
        "observed_state_sha256_before": observed_hash_before,
        "observed_state_sha256_after": observed_hash_after,
        "safety": safety,
        "interpretation": {
            "did_decoupled_scoring_overcome_path_cost_dominance": bool(changed_vs_baseline),
            "is_formula_plausible": plausibility,
            "caution": caution,
        },
        "next_recommended_small_task": next_task,
    }
    save_json(output_dir / "one_step_comparison.json", comparison)
    (output_dir / "one_step_comparison.md").write_text(_md_for_comparison(comparison), encoding="utf-8")

    assert not observed_state_modified, "observed_state was modified"
    assert safety["baseline_observed_hash_unchanged"], "baseline run reported observed_state hash change"
    assert safety["decoupled_observed_hash_unchanged"], "decoupled run reported observed_state hash change"
    assert baseline_result["diagnostics"]["prediction_used_for_traversability"] is False
    assert decoupled_result["diagnostics"]["prediction_used_for_traversability"] is False
    assert baseline_result["diagnostics"]["prediction_used_for_astar"] is False
    assert decoupled_result["diagnostics"]["prediction_used_for_astar"] is False
    assert baseline_result["diagnostics"]["target_lr_used"] is False
    assert decoupled_result["diagnostics"]["target_hr_used"] is False
    assert baseline_result["diagnostics"]["rl_or_training_used"] is False
    assert decoupled_result["diagnostics"]["rl_or_training_used"] is False

    print("Stage 4A-6.5c decoupled one-step smoke complete.")
    print(f"selected_case: {output_dir / 'selected_case.json'}")
    print(f"one_step_comparison: {output_dir / 'one_step_comparison.json'}")
    print(f"baseline_best: {baseline_best['key']} score={baseline_best['score']:.6f}")
    print(f"decoupled_best: {decoupled_best['key']} score={decoupled_best['score']:.6f}")
    print(f"changed_vs_baseline: {changed_vs_baseline}")
    print(f"matches_offline_expected: {matches_offline_expected}")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 4A-6.5c decoupled_sc one-step smoke.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--counterfactual_dir", type=Path, default=DEFAULT_COUNTERFACTUAL_DIR)
    parser.add_argument("--rank_sensitivity_dir", type=Path, default=DEFAULT_RANK_SENSITIVITY_DIR)
    parser.add_argument("--fixed_raw_episode", type=Path, default=DEFAULT_FIXED_RAW_EPISODE)
    parser.add_argument("--empty_baseline_episode", type=Path, default=DEFAULT_EMPTY_BASELINE_EPISODE)
    parser.add_argument("--gating_root", type=Path, default=DEFAULT_GATING_ROOT)
    return parser.parse_args()


def main() -> None:
    run_smoke(parse_args())


if __name__ == "__main__":
    main()

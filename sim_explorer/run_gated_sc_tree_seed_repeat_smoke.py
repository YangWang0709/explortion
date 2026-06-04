#!/usr/bin/env python3
"""Stage 4A-6.5t alternate-tree-seed gated SC tree two-frame smoke.

This wrapper repeats the Stage 4A-6.5s two-frame gated smoke with the same
scene, start pose, confidence-weighted primary formula, and cap25 shadow
formula, but changes only the mini-RRT/tree sampling seed to 1. It then writes
an explicit repeat/stability summary against the Stage 4A-6.5s seed-0
reference.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_PROFILE_NAME = "source_like_crop_min_length_0p25"
DEFAULT_REFERENCE_STAGE4A65S_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke"
)
PROHIBITED_OUTPUT_PATTERNS = [
    "frame003*",
    "transitions.jsonl",
    "step_*.npz",
    "step_topdown_*.png",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def profile_name_for_seed(seed: int) -> str:
    return BASE_PROFILE_NAME if int(seed) == 0 else f"{BASE_PROFILE_NAME}_seed{int(seed)}"


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def distance(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        av = [float(v) for v in a]
        bv = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if len(av) != len(bv):
        return None
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv))))


def decision(summary: dict[str, Any], frame_index: int, label: str) -> dict[str, Any]:
    return summary["frames"][f"frame{frame_index:03d}"][label]


def compare_decisions(repeat: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    selected_delta = distance(repeat.get("selected_child_world"), reference.get("selected_child_world"))
    best_delta = distance(repeat.get("best_descendant_world"), reference.get("best_descendant_world"))
    return {
        "selected_child_same_id": repeat.get("selected_child_id") == reference.get("selected_child_id"),
        "selected_child_same_grid": same_grid(repeat.get("selected_child_grid"), reference.get("selected_child_grid")),
        "selected_child_world_delta_m": selected_delta,
        "selected_child_spatially_close_m0p75": selected_delta is not None and selected_delta <= 0.75,
        "best_descendant_same_id": repeat.get("best_descendant_id") == reference.get("best_descendant_id"),
        "best_descendant_same_grid": same_grid(repeat.get("best_descendant_grid"), reference.get("best_descendant_grid")),
        "best_descendant_world_delta_m": best_delta,
        "best_descendant_spatially_close_m1p0": best_delta is not None and best_delta <= 1.0,
    }


def measured_change(sc: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    selected_delta = distance(sc.get("selected_child_world"), measured.get("selected_child_world"))
    best_delta = distance(sc.get("best_descendant_world"), measured.get("best_descendant_world"))
    selected_changed = not same_grid(sc.get("selected_child_grid"), measured.get("selected_child_grid"))
    best_changed = not same_grid(sc.get("best_descendant_grid"), measured.get("best_descendant_grid"))
    return {
        "selected_child_changed": selected_changed,
        "selected_child_world_delta_m": selected_delta,
        "selected_child_spatially_meaningful_m0p25": bool(
            selected_changed and selected_delta is not None and selected_delta >= 0.25
        ),
        "best_descendant_changed": best_changed,
        "best_descendant_world_delta_m": best_delta,
        "best_descendant_spatially_meaningful_m0p25": bool(
            best_changed and best_delta is not None and best_delta >= 0.25
        ),
    }


def frame_report(base_summary: dict[str, Any], frame_index: int) -> dict[str, Any]:
    key = f"frame{frame_index:03d}"
    frame = base_summary["frames"][key]
    return {
        "measured_only": frame["measured_only"],
        "confidence_weighted": frame["confidence_weighted"],
        "cap25_shadow": frame["cap25_shadow"],
        "prediction_stats": frame["prediction_stats"],
        "map_predict_timing": frame.get("map_predict_timing", {}),
        "required_gain_fields_present": {
            label: all(
                name in frame[label]
                for name in (
                    "accumulated_gain_exp",
                    "accumulated_raw_gain_sc",
                    "accumulated_gain_occ",
                    "accumulated_gain_conf",
                    "accumulated_effective_gain_sc",
                    "accumulated_gain_hybrid_effective",
                    "gain_stats",
                )
            )
            for label in ("measured_only", "confidence_weighted", "cap25_shadow")
        },
    }


def scan_prohibited_outputs(output_dir: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        matches = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
        if matches:
            found[pattern] = matches
    return found


def run_primary(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    script = Path(__file__).with_name("run_gated_sc_tree_two_frame_smoke.py")
    command = [
        sys.executable,
        str(script),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--scene_seed",
        str(args.scene_seed),
        "--primary_sc_gain_formula",
        "confidence_weighted",
        "--shadow_sc_gain_formula",
        "cap25",
        "--stage_tag",
        "stage4a65t",
    ]
    log_path = output_dir / "stage4a65t_gated_sc_tree_seed_repeat.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:])
        raise RuntimeError(f"Stage 4A-6.5t primary repeat failed with code {completed.returncode}\n{tail}")
    return {
        "command": command,
        "log_path": str(log_path),
        "summary_path": str(output_dir / "gated_sc_tree_two_frame_summary.json"),
        "summary": load_json(output_dir / "gated_sc_tree_two_frame_summary.json"),
    }


def make_summary(args: argparse.Namespace, output_dir: Path, primary: dict[str, Any]) -> dict[str, Any]:
    repeat = primary["summary"]
    reference_dir = Path(args.reference_stage4a65s_dir).resolve()
    reference = load_json(reference_dir / "gated_sc_tree_two_frame_summary.json")

    f1_measured = decision(repeat, 1, "measured_only")
    f1_conf = decision(repeat, 1, "confidence_weighted")
    f1_cap25 = decision(repeat, 1, "cap25_shadow")
    f2_measured = decision(repeat, 2, "measured_only")
    f2_conf = decision(repeat, 2, "confidence_weighted")
    f2_cap25 = decision(repeat, 2, "cap25_shadow")

    ref_f1_conf = decision(reference, 1, "confidence_weighted")
    ref_f1_cap25 = decision(reference, 1, "cap25_shadow")
    ref_f2_conf = decision(reference, 2, "confidence_weighted")
    ref_f2_cap25 = decision(reference, 2, "cap25_shadow")

    f1_conf_vs_measured = measured_change(f1_conf, f1_measured)
    f1_cap25_vs_measured = measured_change(f1_cap25, f1_measured)
    f2_conf_vs_measured = measured_change(f2_conf, f2_measured)
    f2_cap25_vs_conf = compare_decisions(f2_cap25, f2_conf)
    f2_vs_ref = compare_decisions(f2_conf, ref_f2_conf)

    frame2_exact_ref_branch = bool(
        f2_conf.get("selected_child_id") == "n0127" and f2_conf.get("best_descendant_id") == "n0162"
    )
    frame2_spatially_close_ref_branch = bool(
        f2_vs_ref["selected_child_spatially_close_m0p75"]
        and f2_vs_ref["best_descendant_spatially_close_m1p0"]
    )
    frame2_spatially_meaningful = bool(
        f2_conf_vs_measured["selected_child_spatially_meaningful_m0p25"]
        or f2_conf_vs_measured["best_descendant_spatially_meaningful_m0p25"]
    )
    frame2_returned_to_measured = same_grid(
        f2_conf.get("selected_child_grid"),
        f2_measured.get("selected_child_grid"),
    )
    enough_for_next_gated_smoke = bool(
        repeat["safety"]["frames_captured"] == 2
        and repeat["safety"]["selected_action_execution_count"] == 1
        and (frame2_exact_ref_branch or frame2_spatially_close_ref_branch or frame2_spatially_meaningful)
        and not frame2_returned_to_measured
    )

    answers = {
        "seed1_repeat_completed_exactly_2_frames": int(repeat["safety"]["frames_captured"]) == 2,
        "frame1_confidence_weighted_still_close_to_measured_only": bool(
            same_grid(f1_conf.get("selected_child_grid"), f1_measured.get("selected_child_grid"))
            or safe_float(f1_conf_vs_measured["selected_child_world_delta_m"], 999.0) <= 0.25
        ),
        "frame1_cap25_shadow_still_more_aggressive": bool(
            f1_cap25_vs_measured["selected_child_changed"] or f1_cap25_vs_measured["best_descendant_changed"]
        ),
        "executed_action_from_confidence_weighted": bool(
            repeat["safety"]["selected_action_formula"] == "confidence_weighted"
            and int(repeat["safety"]["selected_action_execution_count"]) == 1
        ),
        "frame2_confidence_weighted_changed_measured_selected_child": bool(
            f2_conf_vs_measured["selected_child_changed"]
        ),
        "frame2_confidence_weighted_selected_n0127_or_spatially_close_to_reference_branch": bool(
            frame2_exact_ref_branch or frame2_spatially_close_ref_branch
        ),
        "frame2_cap25_shadow_consistent_with_confidence_weighted": bool(
            f2_cap25_vs_conf["selected_child_same_grid"] and f2_cap25_vs_conf["best_descendant_same_grid"]
        ),
        "if_seed1_did_not_select_n0127_still_spatially_meaningful_sc_branch": bool(
            frame2_exact_ref_branch or frame2_spatially_meaningful
        ),
        "if_seed1_returned_to_measured_need_seed_robustness_diagnosis": bool(frame2_returned_to_measured),
        "prediction_completely_read_only": bool(repeat["answers"]["prediction_read_only"]),
        "prediction_not_used_for_traversability_collision_ray_blocking": bool(
            not repeat["answers"]["prediction_used_for_traversability_collision_ray_blocking"]
        ),
        "enough_for_same_seed_other_start_or_at_most_3frame_gated_smoke": enough_for_next_gated_smoke,
        "still_not_ready_for_rollout": True,
    }
    if frame2_returned_to_measured:
        recommended_next = "seed robustness diagnosis before any longer smoke"
    elif enough_for_next_gated_smoke:
        recommended_next = "same-seed/other-start repeated smoke or at most a 3-frame gated smoke; still not rollout"
    else:
        recommended_next = "inspect seed-1 branch ranking before extending beyond two frames"

    return {
        "stage": "Stage 4A-6.5t alternate-tree-seed gated SC tree two-frame smoke",
        "output_dir": str(output_dir),
        "reference_stage4a65s_dir": str(reference_dir),
        "profile_name": profile_name_for_seed(int(args.seed)),
        "tree_seed": int(args.seed),
        "scene_seed": int(args.scene_seed),
        "scene": "medium_three_rooms",
        "primary_sc_gain_formula": "confidence_weighted",
        "shadow_sc_gain_formula": "cap25",
        "primary_run": {
            "command": primary["command"],
            "log_path": primary["log_path"],
            "summary_path": primary["summary_path"],
        },
        "answers": answers,
        "frame_reports": {
            "frame001": frame_report(repeat, 1),
            "frame002": frame_report(repeat, 2),
        },
        "reference_comparison": {
            "frame001_confidence_weighted_vs_stage4a65s": compare_decisions(f1_conf, ref_f1_conf),
            "frame001_cap25_vs_stage4a65s": compare_decisions(f1_cap25, ref_f1_cap25),
            "frame002_confidence_weighted_vs_stage4a65s": f2_vs_ref,
            "frame002_cap25_vs_stage4a65s": compare_decisions(f2_cap25, ref_f2_cap25),
            "frame002_confidence_weighted_vs_measured": f2_conf_vs_measured,
        },
        "move_once": repeat["move_once"],
        "safety": {
            **repeat["safety"],
            "isaac_startup_count": 1,
            "tree_seed_only_change_from_stage4a65s": True,
            "stage4a65s_reference_tree_seed": 0,
            "stage4a65t_repeat_tree_seed": int(args.seed),
            "prohibited_output_matches": scan_prohibited_outputs(output_dir),
        },
        "prediction_safety_checklist": repeat["prediction_safety_checklist"],
        "source_protection_checklist": repeat["source_protection_checklist"],
        "recommended_next_faithful_step": recommended_next,
        "still_not_next": [
            "rollout",
            "open-ended online loop",
            "third-frame capture in this smoke",
            "second action execution in this smoke",
            "RL/PPO/BC/IL training",
            "prediction writeback",
            "prediction traversability/collision/ray blocking",
            "target or ground-truth scoring",
            "coverage-improvement claim",
        ],
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    a = summary["answers"]
    f1 = summary["frame_reports"]["frame001"]
    f2 = summary["frame_reports"]["frame002"]
    lines = [
        "# Stage 4A-6.5t Alternate Tree Seed Repeat",
        "",
        f"1. Seed=1 repeat completed exactly 2 frames? `{a['seed1_repeat_completed_exactly_2_frames']}`.",
        f"2. Frame 1 confidence-weighted still close to measured-only? `{a['frame1_confidence_weighted_still_close_to_measured_only']}`.",
        f"3. Frame 1 cap25 shadow still more aggressive? `{a['frame1_cap25_shadow_still_more_aggressive']}`.",
        f"4. Executed action came from confidence-weighted? `{a['executed_action_from_confidence_weighted']}`.",
        f"5. Frame 2 confidence-weighted changed measured-only selected child? `{a['frame2_confidence_weighted_changed_measured_selected_child']}`.",
        f"6. Frame 2 confidence-weighted selected n0127 or spatially close reference branch? `{a['frame2_confidence_weighted_selected_n0127_or_spatially_close_to_reference_branch']}`.",
        f"7. Frame 2 cap25 shadow consistent with confidence-weighted? `{a['frame2_cap25_shadow_consistent_with_confidence_weighted']}`.",
        f"8. If seed=1 did not select n0127, branch still spatially meaningful? `{a['if_seed1_did_not_select_n0127_still_spatially_meaningful_sc_branch']}`.",
        f"9. If seed=1 returned to measured-only, need seed robustness diagnosis? `{a['if_seed1_returned_to_measured_need_seed_robustness_diagnosis']}`.",
        f"10. Prediction completely read-only? `{a['prediction_completely_read_only']}`.",
        f"11. Prediction excluded from traversability / collision / ray blocking? `{a['prediction_not_used_for_traversability_collision_ray_blocking']}`.",
        f"12. Enough for same-seed/other-start repeat or at most 3-frame gated smoke? `{a['enough_for_same_seed_other_start_or_at_most_3frame_gated_smoke']}`.",
        f"13. Still not ready for rollout? `{a['still_not_ready_for_rollout']}`.",
        "",
        "## Frame 1",
        f"- measured selected: `{f1['measured_only']['selected_child_id']}` grid `{f1['measured_only']['selected_child_grid']}`.",
        f"- confidence-weighted selected: `{f1['confidence_weighted']['selected_child_id']}` grid `{f1['confidence_weighted']['selected_child_grid']}`.",
        f"- cap25 shadow selected: `{f1['cap25_shadow']['selected_child_id']}` grid `{f1['cap25_shadow']['selected_child_grid']}`.",
        "",
        "## Frame 2",
        f"- measured selected: `{f2['measured_only']['selected_child_id']}` grid `{f2['measured_only']['selected_child_grid']}`.",
        f"- confidence-weighted selected: `{f2['confidence_weighted']['selected_child_id']}` grid `{f2['confidence_weighted']['selected_child_grid']}`.",
        f"- confidence-weighted best: `{f2['confidence_weighted']['best_descendant_id']}` grid `{f2['confidence_weighted']['best_descendant_grid']}`.",
        f"- cap25 shadow selected: `{f2['cap25_shadow']['selected_child_id']}` grid `{f2['cap25_shadow']['selected_child_grid']}`.",
        f"- cap25 shadow best: `{f2['cap25_shadow']['best_descendant_id']}` grid `{f2['cap25_shadow']['best_descendant_grid']}`.",
        "",
        f"Recommended next faithful step: {summary['recommended_next_faithful_step']}.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.seed) != 1:
        raise ValueError("Stage 4A-6.5t is specifically the alternate tree seed=1 repeat")
    output_dir = Path(args.output_dir).resolve()
    primary = run_primary(args, output_dir)
    summary = make_summary(args, output_dir, primary)
    save_json(output_dir / "stage4a65t_alternate_seed_summary.json", summary)
    write_summary_md(output_dir / "stage4a65t_alternate_seed_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference_stage4a65s_dir", default=DEFAULT_REFERENCE_STAGE4A65S_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scene_seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

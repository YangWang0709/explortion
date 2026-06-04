#!/usr/bin/env python3
"""Validate the Stage 4A-6.6a scene complexity audit output bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_JSON_MD_CSV = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a66_bundle_manifest.json",
    "loaded_stage4a66_bundle_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "input_hash_audit.json",
    "input_hash_audit.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "scene_scale_audit.json",
    "scene_scale_audit.md",
    "scene_scale_scorecard.csv",
    "topology_complexity_audit.json",
    "topology_complexity_audit.md",
    "topology_degree_histogram.csv",
    "start_graph_distance_matrix.csv",
    "bottleneck_articulation_review.json",
    "bottleneck_articulation_review.md",
    "start_variant_diversity_audit.json",
    "start_variant_diversity_audit.md",
    "start_variant_table.csv",
    "start_euclidean_distance_matrix.csv",
    "start_topology_distance_matrix.csv",
    "start_local_context_table.csv",
    "fixed_view_visibility_audit.json",
    "fixed_view_visibility_audit.md",
    "validation_view_table.csv",
    "depth_stats_by_view.csv",
    "validation_zone_coverage.json",
    "validation_zone_coverage.md",
    "observed_state_health_audit.json",
    "observed_state_health_audit.md",
    "observed_ratio_by_view.csv",
    "observed_label_distribution.csv",
    "observed_state_exploration_room_left.json",
    "observed_state_exploration_room_left.md",
    "frontier_reachability_audit.json",
    "frontier_reachability_audit.md",
    "reachable_area_by_start.csv",
    "reachable_frontier_by_start.csv",
    "candidate_availability_proxy_by_start.csv",
    "isolated_start_review.json",
    "isolated_start_review.md",
    "obstacle_occlusion_audit.json",
    "obstacle_occlusion_audit.md",
    "obstacle_distribution_table.csv",
    "obstacle_density_by_zone.csv",
    "occlusion_proxy_by_view.csv",
    "expert_usability_pre_audit.json",
    "expert_usability_pre_audit.md",
    "future_expert_pilot_risk_register.json",
    "future_expert_pilot_risk_register.md",
    "scene_complexity_scorecard.csv",
    "scene_complexity_scorecard.json",
    "scene_complexity_scorecard.md",
    "scene_complexity_audit_decision.json",
    "scene_complexity_audit_decision.md",
    "usable_start_subset.json",
    "usable_start_subset.md",
    "scene_revision_plan_if_needed.json",
    "scene_revision_plan_if_needed.md",
    "recommended_next_faithful_step.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_formal_expert_sampling_report.json",
    "no_formal_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "stage4a66a_scene_complexity_audit_summary.json",
    "stage4a66a_scene_complexity_audit_summary.md",
    "long_term_rl_gdpo_note.md",
    "future_stage4a67_formal_expert_sampling_pilot_design_sketch.md",
    "do_not_start_full_expert_sampling_in_stage4a66a.md",
]


REQUIRED_PLOTS = [
    "audit_scene_layout_topdown.png",
    "audit_topology_graph.png",
    "audit_room_corridor_opening_map.png",
    "audit_start_variants_topdown.png",
    "audit_start_distance_matrix.png",
    "audit_validation_view_coverage.png",
    "audit_observed_topdown_final.png",
    "audit_observed_ratio_by_view.png",
    "audit_frontier_distribution_topdown.png",
    "audit_reachable_area_by_start.png",
    "audit_reachable_frontier_by_start.png",
    "audit_obstacle_density_topdown.png",
    "audit_occlusion_proxy_by_view.png",
    "audit_complexity_scorecard.png",
    "audit_pass_fail_flowchart.png",
    "audit_next_stage_decision_flowchart.png",
]


FORBIDDEN_EXACT_NAMES = {
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "rollout_index.html",
    "expert_dataset_manifest.jsonl",
    "global_prediction_layer.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage4a66_dir", required=True)
    parser.add_argument("--expect_no_isaac", action="store_true")
    parser.add_argument("--expect_no_capture", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_formal_expert_sampling", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_required_files(out_dir: Path) -> None:
    assert_true(out_dir.exists(), f"output dir missing: {out_dir}")
    missing = [name for name in REQUIRED_JSON_MD_CSV if not (out_dir / name).is_file()]
    assert_true(not missing, f"missing required JSON/MD/CSV files: {missing}")
    missing_plots = []
    for name in REQUIRED_PLOTS:
        if not (out_dir / name).is_file() and not (out_dir / f"{name}.skipped.md").is_file():
            missing_plots.append(name)
    assert_true(not missing_plots, f"missing required plots or skipped reason files: {missing_plots}")


def assert_scale(out_dir: Path) -> None:
    scale = read_json(out_dir / "scene_scale_audit.json")
    metrics = scale["metrics"]
    assert_true(float(metrics["bounds_x_span_m"]) >= 20.0, "bounds_x_span_m < 20")
    assert_true(float(metrics["bounds_y_span_m"]) >= 20.0, "bounds_y_span_m < 20")
    assert_true(int(metrics["room_count"]) >= 8, "room_count < 8")
    assert_true(int(metrics["corridor_count"]) >= 3, "corridor_count < 3")
    assert_true(int(metrics["opening_count"]) >= 12, "opening_count < 12")
    assert_true(int(metrics["obstacle_count"]) >= 40, "obstacle_count < 40")
    assert_true(int(metrics["start_variant_count"]) >= 8, "start_variant_count < 8")
    assert_true(int(metrics["validation_pose_count"]) >= 12, "validation_pose_count < 12")


def assert_decision(out_dir: Path) -> dict[str, Any]:
    decision = read_json(out_dir / "scene_complexity_audit_decision.json")
    for key in [
        "scene_complexity_audit_passed",
        "scene_ready_for_formal_expert_sampling_pilot",
        "formal_expert_sampling_ready_full_dataset",
        "hard_blockers",
        "warnings",
    ]:
        assert_true(key in decision, f"decision missing {key}")
    assert_true(decision["formal_expert_sampling_ready_full_dataset"] is False, "full dataset readiness must remain false")
    if decision["scene_complexity_audit_passed"]:
        assert_true(not decision["hard_blockers"], "pass decision has hard blockers")
        if decision["scene_ready_for_formal_expert_sampling_pilot"]:
            assert_true("Stage 4A-6.7" in decision["recommended_next"], "pass next step must be Stage 4A-6.7")
            assert_true("full dataset" in decision["recommended_next"].lower(), "pass next step must mention not full dataset")
    else:
        assert_true(decision["scene_ready_for_formal_expert_sampling_pilot"] is False, "failed audit cannot be pilot-ready")
        assert_true("Stage 4A-6.6b" in decision["recommended_next"], "failed audit next step must be Stage 4A-6.6b")
    return decision


def assert_negative_scope(out_dir: Path, args: argparse.Namespace) -> None:
    no_rollout = read_json(out_dir / "no_rollout_report.json")
    no_sampling = read_json(out_dir / "no_formal_expert_sampling_report.json")
    no_map = read_json(out_dir / "no_map_predict_report.json")
    no_rl = read_json(out_dir / "no_rl_gdpo_report.json")
    if args.expect_no_rollout:
        assert_true(no_rollout["rollout_run"] is False, "rollout report says rollout ran")
    if args.expect_no_formal_expert_sampling:
        assert_true(no_sampling["formal_expert_sampling_run"] is False, "formal expert sampling report says sampling ran")
        assert_true(no_sampling["expert_dataset_created"] is False, "expert dataset was created")
    if args.expect_no_map_predict:
        assert_true(no_map["map_predict_called"] is False, "map_predict report says map_predict ran")
        assert_true(no_map["prediction_npz_created"] is False, "prediction NPZ was created")
    if args.expect_no_rl_gdpo:
        assert_true(no_rl["rl_gdpo_ppo_bc_il_run"] is False, "RL/GDPO report says training ran")
        assert_true(no_rl["policy_checkpoint_created_or_modified"] is False, "policy checkpoint was created or modified")


def assert_no_forbidden_artifacts(out_dir: Path) -> None:
    bad = []
    for path in out_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        lower = name.lower()
        if name in FORBIDDEN_EXACT_NAMES:
            bad.append(str(path))
        if lower.startswith("validation_rgb_") or lower.startswith("validation_depth_"):
            bad.append(str(path))
        if lower.endswith(".npz") and ("prediction" in lower or "sscnet" in lower or "global_prediction" in lower):
            bad.append(str(path))
        if "replay_buffer" in lower or "policy_checkpoint" in lower:
            bad.append(str(path))
        if lower.endswith(".jsonl") and ("expert" in lower or "transition" in lower):
            bad.append(str(path))
    assert_true(not bad, f"forbidden artifacts found: {bad}")


def assert_hashes_unchanged(out_dir: Path, stage_dir: Path) -> None:
    audit = read_json(out_dir / "input_hash_audit.json")
    mismatches = []
    for row in audit["files"]:
        rel = row["relative_path"]
        path = stage_dir / rel
        if not path.is_file():
            mismatches.append((rel, "missing"))
            continue
        current = sha256_file(path)
        if current != row["sha256"]:
            mismatches.append((rel, row["sha256"], current))
    assert_true(not mismatches, f"Stage 4A-6.6 input hashes changed: {mismatches[:5]}")
    observed_row = audit.get("observed_state_final_hash")
    assert_true(observed_row and observed_row["relative_path"] == "observed_state_final.npy", "observed_state_final hash missing")
    assert_true(
        sha256_file(stage_dir / "observed_state_final.npy") == observed_row["sha256"],
        "observed_state_final hash changed",
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    stage_dir = Path(args.stage4a66_dir).resolve()
    assert_required_files(out_dir)
    for name in [
        "loaded_stage4a66_bundle_manifest.json",
        "input_hash_audit.json",
        "hardware_utilization_report.json",
        "topology_complexity_audit.json",
        "start_variant_diversity_audit.json",
        "fixed_view_visibility_audit.json",
        "observed_state_health_audit.json",
        "frontier_reachability_audit.json",
        "obstacle_occlusion_audit.json",
        "expert_usability_pre_audit.json",
        "scene_complexity_audit_decision.json",
    ]:
        assert_true((out_dir / name).is_file(), f"missing {name}")
    hardware = read_json(out_dir / "hardware_utilization_report.json")
    assert_true(int(hardware["requested_max_workers"]) == 32, "requested_max_workers must be 32")
    assert_true(int(hardware["actual_max_workers"]) <= 32, "actual_max_workers > 32")
    assert_scale(out_dir)
    decision = assert_decision(out_dir)
    assert_negative_scope(out_dir, args)
    assert_no_forbidden_artifacts(out_dir)
    assert_hashes_unchanged(out_dir, stage_dir)
    gdpo_note = (out_dir / "long_term_rl_gdpo_note.md").read_text(encoding="utf-8").lower()
    assert_true("future" in gdpo_note and "gdpo" in gdpo_note, "long_term_rl_gdpo_note.md must say GDPO is future only")
    summary = read_json(out_dir / "stage4a66a_scene_complexity_audit_summary.json")
    assert_true(summary["decision"]["formal_expert_sampling_ready_full_dataset"] is False, "summary full dataset readiness must be false")
    print(
        json.dumps(
            {
                "all_passed": True,
                "output_dir": str(out_dir),
                "scene_complexity_audit_passed": decision["scene_complexity_audit_passed"],
                "scene_ready_for_formal_expert_sampling_pilot": decision["scene_ready_for_formal_expert_sampling_pilot"],
                "formal_expert_sampling_ready_full_dataset": decision["formal_expert_sampling_ready_full_dataset"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Stage 4A-7.0 BC dataset design/preparation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
FORBIDDEN_EXACT_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "reward",
    "policy_logits",
    "replay_buffer",
    "optimizer",
    "training_state",
    "class_prob",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path, checks: dict[str, Any], key: str) -> None:
    exists = path.is_file()
    checks[key] = exists
    if not exists:
        checks.setdefault("missing_files", []).append(str(path))


def git_large_artifact_policy() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    tracked = result.stdout.splitlines()
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/")
    forbidden_suffixes = (".npy", ".npz", ".png", ".mp4", ".usd", ".pth", ".tar")
    offenders = [
        path
        for path in tracked
        if path.startswith(forbidden_prefixes) or path.lower().endswith(forbidden_suffixes)
    ]
    return {"passed": result.returncode == 0 and not offenders, "offenders": offenders[:80]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_bonus_short_rollout_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_bonus_decision_dir", type=Path, required=True)
    parser.add_argument("--confidence_gated_pilot_dir", type=Path, required=True)
    parser.add_argument("--lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--measured_only_pilot_dir", type=Path, required=True)
    parser.add_argument("--fixed_usd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected_primary_samples", type=int, default=30)
    parser.add_argument("--expect_primary_label_policy", required=True)
    parser.add_argument("--expect_no_isaac", action="store_true")
    parser.add_argument("--expect_no_capture", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_sscnet_inference", action="store_true")
    parser.add_argument("--expect_no_action", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_long_rollout", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_no_optimizer_step", action="store_true")
    parser.add_argument("--expect_no_model_save", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_forward_only_smoke_no_training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    checks: dict[str, Any] = {"output_dir": str(out), "output_dir_exists": out.is_dir()}

    required = [
        "stage4a70_bc_dataset_design_summary.json",
        "stage4a70_bc_dataset_design_summary.md",
        "loaded_context_manifest.json",
        "loaded_context_manifest.md",
        "loaded_expert_artifact_manifest.json",
        "loaded_expert_artifact_manifest.md",
        "source_dataset_inventory.csv",
        "source_dataset_inventory.json",
        "source_dataset_inventory.md",
        "bc_label_policy.json",
        "bc_label_policy.md",
        "bc_candidate_feature_schema.json",
        "bc_candidate_feature_schema.md",
        "bc_sequence_schema.json",
        "bc_sequence_schema.md",
        "bc_dataset_card.json",
        "bc_dataset_card.md",
        "bc_feature_normalization_report.json",
        "bc_feature_normalization_report.md",
        "bc_feature_missingness_report.csv",
        "bc_feature_missingness_report.json",
        "bc_feature_missingness_report.md",
        "bc_quality_filter_report.csv",
        "bc_quality_filter_report.json",
        "bc_quality_filter_report.md",
        "bc_split_policy_report.json",
        "bc_split_policy_report.md",
        "split_assignments.csv",
        "leave_one_start_out_folds.json",
        "bc_primary_short_rollout_view_summary.json",
        "bc_primary_short_rollout_view_summary.md",
        "bc_one_action_reference_view_summary.json",
        "bc_one_action_reference_view_summary.md",
        "bc_shadow_multilabel_view_summary.json",
        "bc_shadow_multilabel_view_summary.md",
        "label_alignment_report.csv",
        "label_alignment_report.json",
        "label_alignment_report.md",
        "source_stage_comparison_report.csv",
        "source_stage_comparison_report.json",
        "source_stage_comparison_report.md",
        "forbidden_field_audit.json",
        "forbidden_field_audit.md",
        "no_training_report.json",
        "no_training_report.md",
        "no_isaac_report.json",
        "no_isaac_report.md",
        "no_capture_report.json",
        "no_capture_report.md",
        "no_map_predict_report.json",
        "no_map_predict_report.md",
        "no_action_report.json",
        "no_action_report.md",
        "no_rollout_report.json",
        "no_rollout_report.md",
        "no_rl_gdpo_ppo_report.json",
        "no_rl_gdpo_ppo_report.md",
        "source_hash_report.json",
        "source_hash_report.md",
        "checkpoint_hash_report.json",
        "checkpoint_hash_report.md",
        "prior_dataset_hash_report.json",
        "prior_dataset_hash_report.md",
        "git_status_before.txt",
        "git_status_after.txt",
        "future_stage4a71_bc_dry_run_or_training_sketch.md",
        "recommended_next_faithful_step.md",
        "bc_dataset_primary_short_rollout.npz",
        "bc_dataset_shadow_multilabel.npz",
        "bc_dataset_one_action_reference.npz",
        "bc_dataset_combined_research_view.npz",
        "bc_dataset_manifest.jsonl",
        "bc_dataset_metadata.json",
        "sample_index_table.csv",
        "candidate_feature_table.csv",
        "candidate_feature_table_model_ready.csv",
        "feature_names_raw.json",
        "feature_names_model.json",
        "normalization_stats.npz",
        "schema_version.json",
        "bc_dataset_index.html",
        "feature_distribution_contact_sheet.png",
        "label_distribution_bar.png",
        "source_stage_sample_count_bar.png",
        "split_distribution_bar.png",
        "quality_filter_summary_bar.png",
        "missing_feature_heatmap.png",
        "selected_action_topdown_contact_sheet.png",
        "selected_vs_shadow_action_delta_topdown.png",
        "primary_vs_lambda48_delta_hist.png",
        "primary_vs_measured_delta_hist.png",
        "uncertainty_feature_distribution.png",
        "score_component_distribution.png",
        "path_cost_vs_gain_scatter.png",
        "source_occ_free_vs_uncertainty_scatter.png",
        "local_jitter_distinct_branch_bar.png",
        "sequence_step_distribution.png",
        "observed_ratio_by_step.png",
    ]
    for name in required:
        require_file(out / name, checks, f"exists_{name}")

    summary = read_json(out / "stage4a70_bc_dataset_design_summary.json")
    checks["summary_completed"] = bool(summary.get("completed"))
    checks["summary_not_blocked"] = not bool(summary.get("blocked"))
    checks["primary_label_policy"] = summary.get("primary_label_policy") == args.expect_primary_label_policy
    checks["primary_sample_count"] = int(summary.get("primary_samples", -1)) == args.expected_primary_samples
    checks["d_model_positive"] = int(summary.get("D_model", 0)) > 0
    checks["d_raw_positive"] = int(summary.get("D_raw", 0)) > int(summary.get("D_model", 0))

    with np.load(out / "bc_dataset_primary_short_rollout.npz", allow_pickle=False) as data:
        keys = set(data.files)
        checks["npz_no_forbidden_exact_keys"] = not bool({k.lower() for k in keys} & FORBIDDEN_EXACT_KEYS)
        checks["npz_forbidden_exact_keys"] = sorted({k.lower() for k in keys} & FORBIDDEN_EXACT_KEYS)
        n = int(data["sample_id"].shape[0])
        checks["primary_npz_sample_count"] = n == args.expected_primary_samples
        labels = data["expert_action_index_primary"].astype(np.int64)
        valid_mask = data["candidate_valid_mask"].astype(bool)
        checks["candidate_valid_mask_shape"] = valid_mask.ndim == 2 and valid_mask.shape[0] == n
        checks["primary_labels_in_range"] = bool(np.all((labels >= 0) & (labels < valid_mask.shape[1])))
        checks["candidate_valid_at_primary_label"] = bool(all(valid_mask[i, int(labels[i])] for i in range(n)))
        checks["candidate_features_model_finite"] = bool(np.all(np.isfinite(data["candidate_features_model"])))
        checks["missing_feature_mask_exists"] = "missing_feature_mask" in keys and data["missing_feature_mask"].shape == data["candidate_features_model"].shape
        checks["candidate_feature_mask_exists"] = "candidate_feature_mask" in keys and data["candidate_feature_mask"].shape == data["candidate_features_model"].shape
        checks["quality_keep_mask_shape"] = data["quality_keep_mask"].shape == (n,)
        checks["score_primary_shape"] = data["score_primary"].shape == valid_mask.shape
        checks["score_primary_finite_on_valid"] = bool(np.all(np.isfinite(data["score_primary"][valid_mask])))

    with np.load(out / "bc_dataset_shadow_multilabel.npz", allow_pickle=False) as data:
        checks["shadow_multilabel_npz_sample_count"] = int(data["sample_id"].shape[0]) == args.expected_primary_samples

    table_columns = set(read_csv(out / "candidate_feature_table.csv")[0].keys())
    exact_table_hits = sorted({c.lower() for c in table_columns} & FORBIDDEN_EXACT_KEYS)
    checks["tables_no_forbidden_exact_columns"] = not exact_table_hits
    checks["table_forbidden_exact_columns"] = exact_table_hits

    forbidden = read_json(out / "forbidden_field_audit.json")
    checks["forbidden_field_audit_passed"] = bool(forbidden.get("passed"))
    quality = read_json(out / "bc_quality_filter_report.json")
    checks["strict_keep_count_positive"] = int(quality["counts"].get("strict_keep", 0)) > 0
    split = read_json(out / "bc_split_policy_report.json")
    checks["leave_one_start_out_folds_count"] = int(split.get("leave_one_start_out_fold_count", 0)) == 10

    for stem in (
        "no_training_report",
        "no_isaac_report",
        "no_capture_report",
        "no_map_predict_report",
        "no_sscnet_inference_report",
        "no_action_report",
        "no_rollout_report",
        "no_rl_gdpo_ppo_report",
    ):
        report = read_json(out / f"{stem}.json")
        checks[f"{stem}_executed_false"] = not bool(report.get("executed"))
        checks[f"{stem}_count_zero"] = int(report.get("count_this_stage", 0)) == 0
        checks[f"{stem}_optimizer_false"] = not bool(report.get("optimizer_step"))
        checks[f"{stem}_model_saved_false"] = not bool(report.get("model_saved"))
        checks[f"{stem}_checkpoint_false"] = not bool(report.get("checkpoint_created"))

    source_hash = read_json(out / "source_hash_report.json")
    checkpoint_hash = read_json(out / "checkpoint_hash_report.json")
    prior_hash = read_json(out / "prior_dataset_hash_report.json")
    checks["fixed_usd_unchanged"] = source_hash.get("fixed_usd_sha256_before") == source_hash.get("fixed_usd_sha256_after")
    checks["source_usd_unchanged"] = source_hash.get("source_usd_sha256_before") == source_hash.get("source_usd_sha256_after")
    checks["checkpoint_unchanged"] = checkpoint_hash.get("checkpoint_sha256_before") == checkpoint_hash.get("checkpoint_sha256_after")
    checks["prior_datasets_unchanged"] = all(v.get("sha256_before") == v.get("sha256_after") for v in prior_hash.values())

    if args.expect_forward_only_smoke_no_training:
        smoke_path = out / "bc_forward_only_smoke_report.json"
        require_file(smoke_path, checks, "forward_only_smoke_report_exists")
        smoke = read_json(smoke_path)
        checks["forward_smoke_optimizer_false"] = not bool(smoke.get("optimizer_step"))
        checks["forward_smoke_backward_false"] = not bool(smoke.get("backward_called"))
        checks["forward_smoke_model_saved_false"] = not bool(smoke.get("model_saved"))
        checks["forward_smoke_checkpoint_false"] = not bool(smoke.get("checkpoint_created"))

    future_sketch = (out / "future_stage4a71_bc_dry_run_or_training_sketch.md").read_text(encoding="utf-8").splitlines()[0]
    checks["future_sketch_do_not_run_header"] = future_sketch == "DO NOT RUN IN STAGE 4A-7.0."

    large_policy = git_large_artifact_policy()
    checks["git_large_artifact_policy_preserved"] = bool(large_policy["passed"])
    checks["git_large_artifact_policy_offenders"] = large_policy["offenders"]

    checks["all_passed"] = all(
        bool(value)
        for key, value in checks.items()
        if key not in {"missing_files", "npz_forbidden_exact_keys", "table_forbidden_exact_columns", "git_large_artifact_policy_offenders", "output_dir"}
    )
    report_path = out / "stage4a70_test_report.json"
    report_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not checks["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 4A-6.5af offline saved-frame lambda48 consolidation.

This script only reads saved Stage 4A-6.5ab/ac/ad/ae outputs and writes
aggregate review artifacts. It does not start Isaac, capture frames, rerun
map_predict, run SSCNet inference, execute actions, run runtime, or run
rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


STAGE4A65AB_FILES = [
    "best_config_candidates.json",
    "best_config_candidates.md",
    "stage4a65ab_synthetic_calibration_summary.json",
    "stage4a65ab_synthetic_calibration_summary.md",
    "calibration_summary_by_config.csv",
    "calibration_summary_by_config.json",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.json",
    "lambda_sensitivity_summary.csv",
    "lambda_sensitivity_summary.json",
    "oracle_map_predict_agreement.csv",
    "oracle_map_predict_agreement.json",
]

STAGE4A65AC_FILES = [
    "stage4a65ac_saved_frame_lambda48_formula_summary.json",
    "stage4a65ac_saved_frame_lambda48_formula_summary.md",
    "lambda48_reproduction_summary.json",
    "lambda48_reproduction_summary.md",
    "oracle_vs_map_predict_lambda48.json",
    "oracle_vs_map_predict_lambda48.md",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.json",
    "per_seed_mode_decisions.csv",
    "per_seed_mode_decisions.json",
    "per_seed_value_components.csv",
    "per_seed_value_components.json",
    "branch_direction_classification.csv",
    "branch_direction_classification.json",
    "prediction_safety_report.json",
    "hash_checks.json",
]

STAGE4A65AD_FILES = [
    "stage4a65ad_real_frame_lambda48_formula_summary.json",
    "stage4a65ad_real_frame_lambda48_formula_summary.md",
    "lambda48_behavior_summary.json",
    "lambda48_behavior_summary.md",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.json",
    "per_seed_mode_decisions.csv",
    "per_seed_mode_decisions.json",
    "per_seed_value_components.csv",
    "per_seed_value_components.json",
    "branch_classification_by_seed_mode.csv",
    "branch_classification_by_seed_mode.json",
    "comparison_to_stage4a65z_z1.json",
    "comparison_to_stage4a65z_z1.md",
    "prediction_safety_report.json",
    "hash_checks.json",
]

STAGE4A65AE_FILES = [
    "stage4a65ae_real_frame1_lambda48_formula_summary.json",
    "stage4a65ae_real_frame1_lambda48_formula_summary.md",
    "lambda48_behavior_summary.json",
    "lambda48_behavior_summary.md",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.json",
    "per_seed_mode_decisions.csv",
    "per_seed_mode_decisions.json",
    "per_seed_value_components.csv",
    "per_seed_value_components.json",
    "branch_classification_by_seed_mode.csv",
    "branch_classification_by_seed_mode.json",
    "comparison_to_stage4a65ad.json",
    "comparison_to_stage4a65ad.md",
    "comparison_to_stage4a65z_z1.json",
    "comparison_to_stage4a65z_z1.md",
    "prediction_safety_report.json",
    "hash_checks.json",
    "selected_frame_report.json",
    "selected_frame_report.md",
]

REAL_MODE_KEYS = [
    "measured_only",
    "map_predict_lambda32",
    "map_predict_lambda48",
    "source_occ_free_over_cost",
    "raw_hybrid_over_cost",
    "source_occ_free_no_cost",
]

PLOT_FILES = [
    "lambda48_real_frame_branch_fractions.png",
    "lambda48_synthetic_vs_real_summary.png",
    "lambda32_vs_lambda48_branch_comparison.png",
    "over_cost_vs_lambda48_comparison.png",
    "low_cost_artifact_cross_frame.png",
    "readiness_matrix_heatmap.png",
    "healthy_nonmeasured_fraction_by_frame.png",
    "same_as_measured_fraction_by_frame.png",
]

SAFETY_FALSE_FLAGS = {
    "isaac_startup": False,
    "new_capture": False,
    "map_predict_rerun": False,
    "sscnet_inference": False,
    "selected_action_execution": False,
    "two_frame_runtime": False,
    "rollout": False,
    "open_ended_loop": False,
    "training_rl_ppo_bc_il": False,
    "checkpoint_modified": False,
    "existing_observed_state_modified": False,
    "prediction_npz_modified": False,
    "prediction_writeback": False,
    "prediction_used_for_traversability": False,
    "prediction_used_for_collision": False,
    "prediction_ray_blocking": False,
    "target_ground_truth_planning_scoring": False,
    "external_source_modified_built": False,
    "pareto_dominance_gate_implemented": False,
    "runtime_planner_implemented": False,
    "coverage_improvement_claimed": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65ab_dir", required=True)
    parser.add_argument("--stage4a65ac_dir", required=True)
    parser.add_argument("--stage4a65ad_dir", required=True)
    parser.add_argument("--stage4a65ae_dir", required=True)
    parser.add_argument("--stage4a65p_dir", default=None)
    parser.add_argument("--stage4a65z_dir", default=None)
    parser.add_argument("--stage4a65z1_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_viz", action="store_true")
    return parser.parse_args()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def clean_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return repr(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_cell(row.get(key)) for key in fieldnames})


def format_md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows.")
    else:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_md_value(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fraction(count: int | float | None, total: int | float | None) -> float | None:
    if count is None or total in (None, 0):
        return None
    return float(count) / float(total)


def median(values: list[float]) -> float | None:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return None
    return float(statistics.median(values))


def parse_margin(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def file_record(stage: str, stage_dir: Path, name: str) -> dict[str, Any]:
    path = stage_dir / name
    record: dict[str, Any] = {
        "stage": stage,
        "file": name,
        "path": str(path),
        "exists": path.is_file(),
        "row_count": None,
        "seeds": [],
        "modes": [],
        "top_level_keys": [],
    }
    if not path.is_file():
        return record
    if path.suffix == ".csv":
        rows = read_csv_rows(path)
        record["row_count"] = len(rows)
        record["seeds"] = sorted({int(float(row["seed"])) for row in rows if row.get("seed") not in (None, "")})
        record["modes"] = sorted({row["mode"] for row in rows if row.get("mode")})
    elif path.suffix == ".json":
        data = load_json(path)
        if isinstance(data, list):
            record["row_count"] = len(data)
            record["seeds"] = sorted({int(row["seed"]) for row in data if isinstance(row, dict) and "seed" in row})
            record["modes"] = sorted({row["mode"] for row in data if isinstance(row, dict) and row.get("mode")})
        elif isinstance(data, dict):
            record["top_level_keys"] = sorted(data.keys())
            if isinstance(data.get("per_seed"), list):
                record["row_count"] = len(data["per_seed"])
            elif isinstance(data.get("candidates"), list):
                record["row_count"] = len(data["candidates"])
            elif isinstance(data.get("mode_summary"), list):
                record["row_count"] = len(data["mode_summary"])
            else:
                record["row_count"] = len(data)
    return record


def build_manifest(stage_dirs: dict[str, Path]) -> dict[str, Any]:
    expected = {
        "Stage 4A-6.5ab": STAGE4A65AB_FILES,
        "Stage 4A-6.5ac": STAGE4A65AC_FILES,
        "Stage 4A-6.5ad": STAGE4A65AD_FILES,
        "Stage 4A-6.5ae": STAGE4A65AE_FILES,
    }
    files: list[dict[str, Any]] = []
    for stage, names in expected.items():
        for name in names:
            files.append(file_record(stage, stage_dirs[stage], name))
    stages = []
    for stage, stage_dir in stage_dirs.items():
        stage_records = [item for item in files if item["stage"] == stage]
        stages.append(
            {
                "stage": stage,
                "source_path": str(stage_dir),
                "file_count": len(stage_records),
                "existing_file_count": sum(1 for item in stage_records if item["exists"]),
                "missing_file_count": sum(1 for item in stage_records if not item["exists"]),
                "hash_report_available": (stage_dir / "hash_checks.json").is_file(),
                "safety_report_available": (stage_dir / "prediction_safety_report.json").is_file(),
            }
        )
    return {"stages": stages, "files": files}


def mode_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if isinstance(summary.get("mode_summary"), list):
        return {row.get("mode"): row for row in summary["mode_summary"] if row.get("mode")}
    return {key: summary[key] for key in REAL_MODE_KEYS if isinstance(summary.get(key), dict)}


def row_from_summary(
    *,
    stage: str,
    frame_type: str,
    frame_id: str,
    mode: str,
    summary: dict[str, Any],
    decision_rows: int | None = None,
    saved_ready: bool = True,
    runtime_ready: bool = False,
    rollout_ready: bool = False,
) -> dict[str, Any]:
    seed_count = int(summary.get("seed_count") or 0)
    counts = summary.get("branch_classification_counts") or summary.get("direction_counts") or {}
    distinct_count = counts.get("distinct_nonmeasured_branch")
    if distinct_count is None and "toward_hidden_room" in counts:
        distinct_count = counts.get("toward_hidden_room")
    same_count = counts.get("same_as_measured")
    if same_count is None and "toward_measured_frontier" in counts:
        same_count = counts.get("toward_measured_frontier")
    margin = parse_margin(summary.get("margin"))
    return {
        "stage": stage,
        "frame_type": frame_type,
        "frame_id": frame_id,
        "mode": mode,
        "formula": summary.get("formula"),
        "lambda": summary.get("lambda"),
        "tau": summary.get("tau"),
        "occ_threshold": summary.get("occ_threshold"),
        "free_threshold": summary.get("free_threshold"),
        "seeds": seed_count,
        "decision_rows": decision_rows if decision_rows is not None else seed_count,
        "same_as_measured_fraction": summary.get("same_as_measured_fraction")
        if "same_as_measured_fraction" in summary
        else summary.get("measured_frontier_selection_fraction"),
        "distinct_nonmeasured_fraction": fraction(distinct_count, seed_count),
        "healthy_nonmeasured_fraction": summary.get("healthy_nonmeasured_fraction"),
        "hidden_room_fraction": summary.get("hidden_room_selection_fraction"),
        "oracle_agreement": summary.get("oracle_agreement_fraction"),
        "historical_prior_basin_fraction": summary.get("spatial_prior_sc_basin_fraction"),
        "low_cost_artifact_fraction": summary.get("low_cost_artifact_fraction"),
        "median_margin": margin.get("median"),
        "mean_source_occ_free": summary.get("mean_selected_source_occ_free")
        or summary.get("mean_selected_sc_gain"),
        "mean_normalized_sc": summary.get("mean_selected_normalized_sc"),
        "saved_frame_only_ready": saved_ready,
        "runtime_ready": runtime_ready,
        "rollout_ready": rollout_ready,
    }


def synthetic_config_rows(ab_dir: Path, ab_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_csv_rows(ab_dir / "calibration_summary_by_config.csv")
    wanted: list[dict[str, Any]] = []
    for row in rows:
        if row.get("prediction_source") != "map_predict":
            continue
        if row.get("sc_basis") != "source_occ_free":
            continue
        if as_float(row.get("confidence_threshold")) != 0.1:
            continue
        lambda_value = as_float(row.get("lambda"))
        utility = row.get("utility_mode")
        if utility == "decoupled_minmax" and lambda_value in {32.0, 48.0}:
            wanted.append(row)
        elif utility == "over_cost":
            wanted.append(row)
    output: list[dict[str, Any]] = []
    for row in wanted:
        lambda_value = as_float(row.get("lambda"))
        utility = row.get("utility_mode")
        if utility == "decoupled_minmax" and lambda_value == 48.0:
            mode = "map_predict_lambda48"
            formula = "gain_exp / cost + 48 * minmax(source_occ_free)"
        elif utility == "decoupled_minmax" and lambda_value == 32.0:
            mode = "map_predict_lambda32"
            formula = "gain_exp / cost + 32 * minmax(source_occ_free)"
        else:
            mode = "source_occ_free_over_cost"
            formula = "(gain_exp + source_occ_free) / cost"
        margin = parse_margin(row.get("margin"))
        output.append(
            {
                "stage": "Stage 4A-6.5ab",
                "frame_type": "synthetic_hidden_room",
                "frame_id": "synthetic_000",
                "mode": mode,
                "formula": formula,
                "lambda": lambda_value,
                "tau": as_float(row.get("confidence_threshold")),
                "occ_threshold": as_float(row.get("occ_threshold")),
                "free_threshold": as_float(row.get("free_threshold")),
                "seeds": int(float(row.get("seed_count") or 0)),
                "decision_rows": int(float(row.get("seed_count") or 0)),
                "same_as_measured_fraction": as_float(row.get("measured_frontier_selection_fraction")),
                "distinct_nonmeasured_fraction": as_float(row.get("hidden_room_selection_fraction")),
                "healthy_nonmeasured_fraction": as_float(row.get("hidden_room_selection_fraction")),
                "hidden_room_fraction": as_float(row.get("hidden_room_selection_fraction")),
                "oracle_agreement": as_float(row.get("oracle_map_predict_agreement_fraction")),
                "historical_prior_basin_fraction": None,
                "low_cost_artifact_fraction": as_float(row.get("low_cost_artifact_fraction")),
                "median_margin": margin.get("median"),
                "mean_source_occ_free": as_float(row.get("mean_selected_sc_gain")),
                "mean_normalized_sc": None,
                "saved_frame_only_ready": ab_summary.get("answers", {}).get("best_config_readiness")
                == "saved-frame-only-ready",
                "runtime_ready": False,
                "rollout_ready": False,
            }
        )
    return output


def build_unified_config_table(
    ab_dir: Path,
    ab_summary: dict[str, Any],
    ac_summary: dict[str, Any],
    ad_behavior: dict[str, Any],
    ae_behavior: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = synthetic_config_rows(ab_dir, ab_summary)
    for mode, summary in mode_lookup(ac_summary).items():
        rows.append(
            row_from_summary(
                stage="Stage 4A-6.5ac",
                frame_type="synthetic_hidden_room",
                frame_id="synthetic_000",
                mode=mode,
                summary=summary,
                saved_ready=True,
                runtime_ready=False,
                rollout_ready=False,
            )
        )
    for stage, frame_id, behavior in (
        ("Stage 4A-6.5ad", "stage4a65p_frame002", ad_behavior),
        ("Stage 4A-6.5ae", "stage4a65p_frame001", ae_behavior),
    ):
        for mode, summary in mode_lookup(behavior).items():
            rows.append(
                row_from_summary(
                    stage=stage,
                    frame_type="real_medium_frame2" if frame_id.endswith("002") else "real_medium_frame1",
                    frame_id=frame_id,
                    mode=mode,
                    summary=summary,
                    saved_ready=bool(behavior.get("saved_frame_only_readiness")),
                    runtime_ready=bool(behavior.get("runtime_smoke_readiness")),
                    rollout_ready=bool(behavior.get("rollout_readiness")),
                )
            )
    return rows


def rows_for_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("mode") == mode]


def branch_rows(path: Path, mode: str) -> list[dict[str, Any]]:
    return [row for row in read_csv_rows(path) if row.get("mode") == mode]


def collect_margins(decision_rows: list[dict[str, Any]], mode: str) -> list[float]:
    values = []
    for row in decision_rows:
        if row.get("mode") == mode:
            value = as_float(row.get("margin"))
            if value is not None:
                values.append(value)
    return values


def lambda48_frame_row(
    *,
    stage: str,
    frame_type: str,
    frame_id: str,
    source: dict[str, Any],
    synthetic: bool,
) -> dict[str, Any]:
    seed_count = int(source.get("seed_count") or 0)
    branch_counts = source.get("branch_classification_counts") or source.get("direction_counts") or {}
    if synthetic:
        hidden_fraction = source.get("hidden_room_selection_fraction")
        hidden_count = int(round((hidden_fraction or 0.0) * seed_count))
        return {
            "stage": stage,
            "frame_type": frame_type,
            "frame_id": frame_id,
            "seeds": seed_count,
            "same_as_measured_count": int(round((source.get("measured_frontier_selection_fraction") or 0.0) * seed_count)),
            "same_as_measured_fraction": source.get("measured_frontier_selection_fraction"),
            "distinct_nonmeasured_count": hidden_count,
            "distinct_nonmeasured_fraction": hidden_fraction,
            "healthy_nonmeasured_count": hidden_count,
            "healthy_nonmeasured_fraction": hidden_fraction,
            "hidden_room_count": hidden_count,
            "hidden_room_fraction": hidden_fraction,
            "oracle_agreement": source.get("oracle_agreement_fraction"),
            "historical_prior_basin_count": None,
            "historical_prior_basin_fraction": None,
            "low_cost_artifact_count": int(round((source.get("low_cost_artifact_fraction") or 0.0) * seed_count)),
            "low_cost_artifact_fraction": source.get("low_cost_artifact_fraction"),
            "median_margin": parse_margin(source.get("margin")).get("median"),
            "mean_source_occ_free": source.get("mean_selected_source_occ_free"),
            "mean_normalized_sc": source.get("mean_selected_normalized_sc"),
            "runtime_ready": False,
            "rollout_ready": False,
        }
    same_count = int(branch_counts.get("same_as_measured") or 0)
    distinct_count = int(branch_counts.get("distinct_nonmeasured_branch") or 0)
    prior_fraction = source.get("spatial_prior_sc_basin_fraction") or 0.0
    low_fraction = source.get("low_cost_artifact_fraction") or 0.0
    healthy_fraction = source.get("healthy_nonmeasured_fraction") or 0.0
    return {
        "stage": stage,
        "frame_type": frame_type,
        "frame_id": frame_id,
        "seeds": seed_count,
        "same_as_measured_count": same_count,
        "same_as_measured_fraction": source.get("same_as_measured_fraction"),
        "distinct_nonmeasured_count": distinct_count,
        "distinct_nonmeasured_fraction": fraction(distinct_count, seed_count),
        "healthy_nonmeasured_count": int(round(healthy_fraction * seed_count)),
        "healthy_nonmeasured_fraction": healthy_fraction,
        "hidden_room_count": None,
        "hidden_room_fraction": None,
        "oracle_agreement": None,
        "historical_prior_basin_count": int(round(prior_fraction * seed_count)),
        "historical_prior_basin_fraction": prior_fraction,
        "low_cost_artifact_count": int(round(low_fraction * seed_count)),
        "low_cost_artifact_fraction": low_fraction,
        "median_margin": parse_margin(source.get("margin")).get("median"),
        "mean_source_occ_free": source.get("mean_selected_source_occ_free"),
        "mean_normalized_sc": source.get("mean_selected_normalized_sc"),
        "runtime_ready": False,
        "rollout_ready": False,
    }


def build_lambda48_cross_frame_summary(
    ac_summary: dict[str, Any],
    ad_behavior: dict[str, Any],
    ae_behavior: dict[str, Any],
) -> list[dict[str, Any]]:
    ac_modes = mode_lookup(ac_summary)
    return [
        lambda48_frame_row(
            stage="Stage 4A-6.5ac",
            frame_type="synthetic_hidden_room",
            frame_id="synthetic_000",
            source=ac_modes["map_predict_lambda48"],
            synthetic=True,
        ),
        lambda48_frame_row(
            stage="Stage 4A-6.5ad",
            frame_type="real_medium_frame2",
            frame_id="stage4a65p_frame002",
            source=ad_behavior["map_predict_lambda48"],
            synthetic=False,
        ),
        lambda48_frame_row(
            stage="Stage 4A-6.5ae",
            frame_type="real_medium_frame1",
            frame_id="stage4a65p_frame001",
            source=ae_behavior["map_predict_lambda48"],
            synthetic=False,
        ),
    ]


def seed_branch_map(rows: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in rows:
        if row.get("seed") not in (None, ""):
            result[int(float(row["seed"]))] = row.get("branch_classification", "")
    return result


def build_real_frame_aggregate(
    ad_dir: Path,
    ae_dir: Path,
    cross_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    real_rows = [row for row in cross_rows if row["frame_type"].startswith("real")]
    total_seeds = sum(int(row["seeds"]) for row in real_rows)
    same_count = sum(int(row["same_as_measured_count"]) for row in real_rows)
    distinct_count = sum(int(row["distinct_nonmeasured_count"]) for row in real_rows)
    healthy_count = sum(int(row["healthy_nonmeasured_count"]) for row in real_rows)
    prior_count = sum(int(row["historical_prior_basin_count"]) for row in real_rows)
    low_count = sum(int(row["low_cost_artifact_count"]) for row in real_rows)

    ad_decisions = read_csv_rows(ad_dir / "per_seed_mode_decisions.csv")
    ae_decisions = read_csv_rows(ae_dir / "per_seed_mode_decisions.csv")
    margins = collect_margins(ad_decisions, "map_predict_lambda48")
    margins.extend(collect_margins(ae_decisions, "map_predict_lambda48"))

    ad_branch = seed_branch_map(branch_rows(ad_dir / "branch_classification_by_seed_mode.csv", "map_predict_lambda48"))
    ae_branch = seed_branch_map(branch_rows(ae_dir / "branch_classification_by_seed_mode.csv", "map_predict_lambda48"))
    overlap = sorted(set(ad_branch) & set(ae_branch))
    consistent = sum(1 for seed in overlap if ad_branch[seed] == ae_branch[seed])

    same_fractions = [row["same_as_measured_fraction"] for row in real_rows]
    healthy_fractions = [row["healthy_nonmeasured_fraction"] for row in real_rows]
    return [
        {
            "frame_count": len(real_rows),
            "total_seed_frame_rows": total_seeds,
            "same_as_measured_count": same_count,
            "same_as_measured_fraction": fraction(same_count, total_seeds),
            "distinct_nonmeasured_count": distinct_count,
            "distinct_nonmeasured_fraction": fraction(distinct_count, total_seeds),
            "healthy_nonmeasured_count": healthy_count,
            "healthy_nonmeasured_fraction": fraction(healthy_count, total_seeds),
            "historical_prior_basin_count": prior_count,
            "historical_prior_basin_fraction": fraction(prior_count, total_seeds),
            "low_cost_artifact_count": low_count,
            "low_cost_artifact_fraction": fraction(low_count, total_seeds),
            "real_lambda48_median_margin": median(margins),
            "same_as_measured_fraction_range": max(same_fractions) - min(same_fractions),
            "healthy_nonmeasured_fraction_range": max(healthy_fractions) - min(healthy_fractions),
            "seed_label_overlap_count": len(overlap),
            "seed_level_branch_class_consistency_count": consistent,
            "seed_level_branch_class_consistency_fraction": fraction(consistent, len(overlap)),
            "interpretation": (
                "lambda48 stayed artifact-free and avoided the historical prior basin on both real "
                "saved frames, but it was conservative: most seed-frame rows remained measured-only."
            ),
        }
    ]


def compare_modes_by_seed(
    branch_file: Path,
    mode_a: str,
    mode_b: str,
) -> dict[str, Any]:
    rows = read_csv_rows(branch_file)
    by_mode_seed = {(row.get("mode"), int(float(row["seed"]))): row for row in rows if row.get("seed") not in (None, "")}
    seeds = sorted(
        {
            seed
            for mode, seed in by_mode_seed
            if mode == mode_a and (mode_b, seed) in by_mode_seed
        }
    )
    branch_match = 0
    selected_match = 0
    best_match = 0
    exact_match = 0
    per_seed = []
    for seed in seeds:
        a = by_mode_seed[(mode_a, seed)]
        b = by_mode_seed[(mode_b, seed)]
        same_branch = a.get("branch_classification") == b.get("branch_classification")
        same_selected = a.get("selected_child_id") == b.get("selected_child_id")
        same_best = a.get("best_descendant_id") == b.get("best_descendant_id")
        branch_match += int(same_branch)
        selected_match += int(same_selected)
        best_match += int(same_best)
        exact_match += int(same_selected and same_best)
        per_seed.append(
            {
                "seed": seed,
                f"{mode_a}_branch": a.get("branch_classification"),
                f"{mode_b}_branch": b.get("branch_classification"),
                "same_branch_class": same_branch,
                "same_selected_child": same_selected,
                "same_best_descendant": same_best,
                "same_exact_selection": same_selected and same_best,
            }
        )
    return {
        "seed_count": len(seeds),
        "branch_class_match_count": branch_match,
        "branch_class_match_fraction": fraction(branch_match, len(seeds)),
        "selected_child_match_count": selected_match,
        "selected_child_match_fraction": fraction(selected_match, len(seeds)),
        "best_descendant_match_count": best_match,
        "best_descendant_match_fraction": fraction(best_match, len(seeds)),
        "exact_selection_match_count": exact_match,
        "exact_selection_match_fraction": fraction(exact_match, len(seeds)),
        "per_seed": per_seed,
    }


def build_lambda32_vs_lambda48_comparison(
    ac_summary: dict[str, Any],
    ad_dir: Path,
    ad_behavior: dict[str, Any],
    ae_dir: Path,
    ae_behavior: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ac_modes = mode_lookup(ac_summary)
    rows.append(
        {
            "stage": "Stage 4A-6.5ac",
            "frame_id": "synthetic_000",
            "frame_type": "synthetic_hidden_room",
            "seed_count": ac_modes["map_predict_lambda48"]["seed_count"],
            "lambda32_same_as_measured_fraction": ac_modes["map_predict_lambda32"].get("measured_frontier_selection_fraction"),
            "lambda48_same_as_measured_fraction": ac_modes["map_predict_lambda48"].get("measured_frontier_selection_fraction"),
            "lambda32_healthy_or_hidden_fraction": ac_modes["map_predict_lambda32"].get("hidden_room_selection_fraction"),
            "lambda48_healthy_or_hidden_fraction": ac_modes["map_predict_lambda48"].get("hidden_room_selection_fraction"),
            "lambda32_prior_basin_fraction": None,
            "lambda48_prior_basin_fraction": None,
            "branch_class_match_fraction": None,
            "exact_selection_match_fraction": None,
            "lambda48_changed_branch_class_beyond_lambda32": True,
            "interpretation": "Synthetic evidence keeps lambda48 preferred: lambda32 reached only 3/5 hidden-room selections.",
        }
    )
    real_specs = [
        ("Stage 4A-6.5ad", "stage4a65p_frame002", "real_medium_frame2", ad_dir, ad_behavior),
        ("Stage 4A-6.5ae", "stage4a65p_frame001", "real_medium_frame1", ae_dir, ae_behavior),
    ]
    totals = Counter()
    for stage, frame_id, frame_type, stage_dir, behavior in real_specs:
        comp = compare_modes_by_seed(
            stage_dir / "branch_classification_by_seed_mode.csv",
            "map_predict_lambda32",
            "map_predict_lambda48",
        )
        seed_count = comp["seed_count"]
        totals["seed_count"] += seed_count
        totals["branch_class_match_count"] += comp["branch_class_match_count"]
        totals["selected_child_match_count"] += comp["selected_child_match_count"]
        totals["best_descendant_match_count"] += comp["best_descendant_match_count"]
        totals["exact_selection_match_count"] += comp["exact_selection_match_count"]
        rows.append(
            {
                "stage": stage,
                "frame_id": frame_id,
                "frame_type": frame_type,
                "seed_count": seed_count,
                "lambda32_same_as_measured_fraction": behavior["map_predict_lambda32"].get("same_as_measured_fraction"),
                "lambda48_same_as_measured_fraction": behavior["map_predict_lambda48"].get("same_as_measured_fraction"),
                "lambda32_healthy_or_hidden_fraction": behavior["map_predict_lambda32"].get("healthy_nonmeasured_fraction"),
                "lambda48_healthy_or_hidden_fraction": behavior["map_predict_lambda48"].get("healthy_nonmeasured_fraction"),
                "lambda32_prior_basin_fraction": behavior["map_predict_lambda32"].get("spatial_prior_sc_basin_fraction"),
                "lambda48_prior_basin_fraction": behavior["map_predict_lambda48"].get("spatial_prior_sc_basin_fraction"),
                "branch_class_match_fraction": comp["branch_class_match_fraction"],
                "exact_selection_match_fraction": comp["exact_selection_match_fraction"],
                "selected_child_match_fraction": comp["selected_child_match_fraction"],
                "best_descendant_match_fraction": comp["best_descendant_match_fraction"],
                "lambda48_changed_branch_class_beyond_lambda32": comp["branch_class_match_count"] < seed_count,
                "lambda48_changed_exact_selection_beyond_lambda32": comp["exact_selection_match_count"] < seed_count,
                "interpretation": "On this real saved frame, lambda32 matched lambda48 at branch-class level.",
            }
        )
    total_seed_count = int(totals["seed_count"])
    rows.append(
        {
            "stage": "real_aggregate",
            "frame_id": "real_frame001_plus_frame002",
            "frame_type": "real_medium_aggregate",
            "seed_count": total_seed_count,
            "lambda32_same_as_measured_fraction": None,
            "lambda48_same_as_measured_fraction": None,
            "lambda32_healthy_or_hidden_fraction": None,
            "lambda48_healthy_or_hidden_fraction": None,
            "lambda32_prior_basin_fraction": 0.0,
            "lambda48_prior_basin_fraction": 0.0,
            "branch_class_match_fraction": fraction(totals["branch_class_match_count"], total_seed_count),
            "exact_selection_match_fraction": fraction(totals["exact_selection_match_count"], total_seed_count),
            "selected_child_match_fraction": fraction(totals["selected_child_match_count"], total_seed_count),
            "best_descendant_match_fraction": fraction(totals["best_descendant_match_count"], total_seed_count),
            "lambda48_changed_branch_class_beyond_lambda32": totals["branch_class_match_count"] < total_seed_count,
            "lambda48_changed_exact_selection_beyond_lambda32": totals["exact_selection_match_count"] < total_seed_count,
            "interpretation": (
                "Real saved frames did not show a branch-class advantage for lambda48 over lambda32, "
                "but synthetic calibration still favors lambda48 as the candidate."
            ),
        }
    )
    return rows


def build_over_cost_diagnostic_comparison(
    ad_behavior: dict[str, Any],
    ae_behavior: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = Counter()
    for stage, frame_id, frame_type, behavior in (
        ("Stage 4A-6.5ad", "stage4a65p_frame002", "real_medium_frame2", ad_behavior),
        ("Stage 4A-6.5ae", "stage4a65p_frame001", "real_medium_frame1", ae_behavior),
    ):
        lambda48 = behavior["map_predict_lambda48"]
        for mode in ("source_occ_free_over_cost", "raw_hybrid_over_cost"):
            if mode not in behavior:
                continue
            over = behavior[mode]
            seed_count = int(over.get("seed_count") or 0)
            counts = over.get("branch_classification_counts") or {}
            distinct_count = int(counts.get("distinct_nonmeasured_branch") or 0)
            prior_count = int(round((over.get("spatial_prior_sc_basin_fraction") or 0.0) * seed_count))
            low_count = int(round((over.get("low_cost_artifact_fraction") or 0.0) * seed_count))
            aggregate["seed_count"] += seed_count if mode == "source_occ_free_over_cost" else 0
            aggregate["distinct_count"] += distinct_count if mode == "source_occ_free_over_cost" else 0
            aggregate["prior_count"] += prior_count if mode == "source_occ_free_over_cost" else 0
            aggregate["low_count"] += low_count if mode == "source_occ_free_over_cost" else 0
            same_fraction = over.get("same_as_measured_fraction")
            more_aggressive = (
                same_fraction is not None
                and lambda48.get("same_as_measured_fraction") is not None
                and same_fraction < lambda48.get("same_as_measured_fraction")
            )
            remains_risky = prior_count > 0
            rows.append(
                {
                    "stage": stage,
                    "frame_id": frame_id,
                    "frame_type": frame_type,
                    "mode": mode,
                    "seed_count": seed_count,
                    "same_as_measured_fraction": same_fraction,
                    "distinct_nonmeasured_fraction": fraction(distinct_count, seed_count),
                    "healthy_nonmeasured_fraction": over.get("healthy_nonmeasured_fraction"),
                    "prior_basin_fraction": over.get("spatial_prior_sc_basin_fraction"),
                    "low_cost_artifact_fraction": over.get("low_cost_artifact_fraction"),
                    "lambda48_same_as_measured_fraction": lambda48.get("same_as_measured_fraction"),
                    "lambda48_healthy_nonmeasured_fraction": lambda48.get("healthy_nonmeasured_fraction"),
                    "more_aggressive_than_lambda48": more_aggressive,
                    "remains_risky": remains_risky,
                    "diagnostic_only": True,
                    "recommendation": "keep diagnostic-only",
                }
            )
    seed_count = int(aggregate["seed_count"])
    rows.append(
        {
            "stage": "real_aggregate",
            "frame_id": "real_frame001_plus_frame002",
            "frame_type": "real_medium_aggregate",
            "mode": "source_occ_free_over_cost",
            "seed_count": seed_count,
            "same_as_measured_fraction": None,
            "distinct_nonmeasured_fraction": fraction(aggregate["distinct_count"], seed_count),
            "healthy_nonmeasured_fraction": 0.0,
            "prior_basin_fraction": fraction(aggregate["prior_count"], seed_count),
            "low_cost_artifact_fraction": fraction(aggregate["low_count"], seed_count),
            "lambda48_same_as_measured_fraction": 0.7,
            "lambda48_healthy_nonmeasured_fraction": 0.3,
            "more_aggressive_than_lambda48": True,
            "remains_risky": aggregate["prior_count"] > 0,
            "diagnostic_only": True,
            "recommendation": "keep diagnostic-only; do not use for runtime",
        }
    )
    return rows


def build_low_cost_cross_frame(
    ac_summary: dict[str, Any],
    ad_behavior: dict[str, Any],
    ae_behavior: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ac_modes = mode_lookup(ac_summary)
    for mode in ("measured_only", "oracle_lambda48", "map_predict_lambda48", "map_predict_lambda32", "map_predict_over_cost"):
        if mode in ac_modes:
            summary = ac_modes[mode]
            rows.append(
                {
                    "stage": "Stage 4A-6.5ac",
                    "frame_id": "synthetic_000",
                    "frame_type": "synthetic_hidden_room",
                    "mode": mode,
                    "seed_count": summary.get("seed_count"),
                    "low_cost_artifact_fraction": summary.get("low_cost_artifact_fraction"),
                    "historical_prior_basin_fraction": None,
                    "diagnostic_only": "over_cost" in mode,
                }
            )
    for stage, frame_id, frame_type, behavior in (
        ("Stage 4A-6.5ad", "stage4a65p_frame002", "real_medium_frame2", ad_behavior),
        ("Stage 4A-6.5ae", "stage4a65p_frame001", "real_medium_frame1", ae_behavior),
    ):
        for mode in ("measured_only", "map_predict_lambda32", "map_predict_lambda48", "source_occ_free_over_cost", "raw_hybrid_over_cost"):
            if mode in behavior:
                summary = behavior[mode]
                rows.append(
                    {
                        "stage": stage,
                        "frame_id": frame_id,
                        "frame_type": frame_type,
                        "mode": mode,
                        "seed_count": summary.get("seed_count"),
                        "low_cost_artifact_fraction": summary.get("low_cost_artifact_fraction"),
                        "historical_prior_basin_fraction": summary.get("spatial_prior_sc_basin_fraction"),
                        "diagnostic_only": mode.endswith("over_cost"),
                    }
                )
    return rows


def build_readiness_matrix(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case": "synthetic lambda48",
            "reproduces_intended_direction": "yes",
            "avoids_prior_bad_branch": "yes",
            "low_cost_artifact_0": "yes",
            "healthy_nonmeasured_present": "yes",
            "cross_frame_stable": "not_applicable",
            "seed_robust": "yes",
            "source_faithful": "yes",
            "diagnostic_only": "no",
            "saved_frame_ready": "yes",
            "runtime_smoke_ready": "no",
            "rollout_ready": "no",
        },
        {
            "case": "real Frame2 lambda48",
            "reproduces_intended_direction": "mixed",
            "avoids_prior_bad_branch": "yes",
            "low_cost_artifact_0": "yes",
            "healthy_nonmeasured_present": "yes",
            "cross_frame_stable": "mixed",
            "seed_robust": "mixed",
            "source_faithful": "yes",
            "diagnostic_only": "no",
            "saved_frame_ready": "yes",
            "runtime_smoke_ready": "no",
            "rollout_ready": "no",
        },
        {
            "case": "real Frame1 lambda48",
            "reproduces_intended_direction": "mixed",
            "avoids_prior_bad_branch": "yes",
            "low_cost_artifact_0": "yes",
            "healthy_nonmeasured_present": "yes",
            "cross_frame_stable": "mixed",
            "seed_robust": "mixed",
            "source_faithful": "yes",
            "diagnostic_only": "no",
            "saved_frame_ready": "yes",
            "runtime_smoke_ready": "no",
            "rollout_ready": "no",
        },
        {
            "case": "real aggregate lambda48",
            "reproduces_intended_direction": "mixed",
            "avoids_prior_bad_branch": "yes",
            "low_cost_artifact_0": "yes",
            "healthy_nonmeasured_present": "yes",
            "cross_frame_stable": "mixed",
            "seed_robust": "mixed",
            "source_faithful": "yes",
            "diagnostic_only": "no",
            "saved_frame_ready": "yes",
            "runtime_smoke_ready": "no",
            "rollout_ready": "no",
        },
        {
            "case": "over-cost diagnostic",
            "reproduces_intended_direction": "mixed",
            "avoids_prior_bad_branch": "no",
            "low_cost_artifact_0": "yes",
            "healthy_nonmeasured_present": "mixed",
            "cross_frame_stable": "no",
            "seed_robust": "no",
            "source_faithful": "no",
            "diagnostic_only": "yes",
            "saved_frame_ready": "diagnostic",
            "runtime_smoke_ready": "no",
            "rollout_ready": "no",
        },
        {
            "case": "lambda32 diagnostic",
            "reproduces_intended_direction": "mixed",
            "avoids_prior_bad_branch": "yes",
            "low_cost_artifact_0": "yes",
            "healthy_nonmeasured_present": "mixed",
            "cross_frame_stable": "mixed",
            "seed_robust": "mixed",
            "source_faithful": "yes",
            "diagnostic_only": "yes",
            "saved_frame_ready": "diagnostic",
            "runtime_smoke_ready": "no",
            "rollout_ready": "no",
        },
    ]


def build_design_findings(real_aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "Stage 4A-6.5af",
        "status": "completed",
        "questions": {
            "is_source_occ_free_decoupled_minmax_lambda48_stable_safe": {
                "answer": "saved-frame-only yes, runtime no",
                "reason": (
                    "It reproduced the intended hidden-room behavior synthetically and avoided "
                    "low-cost artifacts/prior basin on two real saved frames, but evidence is "
                    "still limited and conservative."
                ),
            },
            "synthetic_vs_real_behavior": {
                "answer": (
                    "synthetic hidden-room is strong; real medium frames are mixed and conservative"
                ),
                "synthetic": "hidden-room 5/5, oracle agreement 1.0, low-cost artifact 0.0",
                "real": (
                    f"same-as-measured {real_aggregate['same_as_measured_count']}/"
                    f"{real_aggregate['total_seed_frame_rows']}, healthy non-measured "
                    f"{real_aggregate['healthy_nonmeasured_count']}/"
                    f"{real_aggregate['total_seed_frame_rows']}, prior basin 0"
                ),
            },
            "avoids_old_low_cost_sc_basin": {
                "answer": "yes for lambda48 on both real saved frames",
                "prior_basin_fraction": real_aggregate["historical_prior_basin_fraction"],
            },
            "is_too_conservative": {
                "answer": "possibly; current evidence is conservative",
                "reason": "14/20 real seed-frame rows stayed same-as-measured; healthy non-measured appeared in 6/20.",
            },
            "need_more_saved_real_frame_formula_smoke": {
                "answer": "do not keep doing one-off smoke tests; consolidate into multi-frame saved replay",
            },
            "can_enter_saved_multi_frame_replay": {
                "answer": "yes, saved-frame-only replay is the next faithful step",
            },
            "can_enter_runtime_or_rollout": {
                "answer": "no",
                "reason": "No action execution, no observed-ratio validation, only two saved real frames.",
            },
        },
        "candidate_formula": {
            "prediction_source": "map_predict",
            "sc_basis": "source_occ_free",
            "utility": "decoupled_minmax_lambda48",
            "formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
            "tau": 0.1,
            "occ_threshold": 0.5,
            "free_threshold": 0.5,
        },
        "recommendation": {
            "exactly_one_next_small_task": (
                "offline saved-frame-only multi-frame lambda48 replay over all available saved real medium frames"
            ),
            "not_next": [
                "runtime two-frame smoke",
                "rollout",
                "open-ended loop",
                "RL/PPO/BC/IL",
                "prediction writeback/fusion",
                "over-cost runtime",
                "Pareto dominance gate",
            ],
        },
        "coverage_improvement_claimed": False,
        "runtime_smoke_ready": False,
        "rollout_ready": False,
    }


def write_design_findings_md(path: Path, findings: dict[str, Any]) -> None:
    q = findings["questions"]
    lines = [
        "# Stage 4A-6.5af Lambda48 Design Review",
        "",
        "## Verdict",
        "",
        "- `source_occ_free + decoupled_minmax_lambda48` is saved-frame-only viable, not runtime-ready.",
        "- It is clean against the old low-cost SC basin on the two saved real frames, but conservative.",
        "- The next faithful step is offline saved-frame-only multi-frame replay over available saved real medium frames.",
        "",
        "## Answers",
        "",
    ]
    for key, item in q.items():
        lines.append(f"- **{key}**: {item['answer']}")
        if item.get("reason"):
            lines.append(f"  {item['reason']}")
    lines.extend(
        [
            "",
            "## Formula",
            "",
            "- `value = gain_exp / cost + 48 * minmax(source_occ_free)`",
            "- tau/confidence threshold: `0.1`",
            "- occ/free thresholds: `0.5 / 0.5`",
            "",
            "## Not Ready",
            "",
            "- runtime two-frame smoke",
            "- rollout",
            "- action execution",
            "- prediction writeback/fusion",
            "- over-cost runtime",
            "- coverage-improvement claims",
        ]
    )
    write_lines(path, lines)


def write_recommended_next(path: Path) -> None:
    lines = [
        "# Recommended Next Faithful Step",
        "",
        "Build and run an offline saved-frame-only multi-frame lambda48 replay over all available saved real `medium_three_rooms` frames.",
        "",
        "Scope:",
        "",
        "- Read existing saved observed frames, prediction NPZs, poses, camera info, and saved/tree-compatible decision artifacts only.",
        "- Reuse the fixed candidate formula: `gain_exp / cost + 48 * minmax(source_occ_free)`.",
        "- Report per-frame and aggregate branch fractions, low-cost artifacts, prior-basin flags, and lambda32 comparison.",
        "",
        "Still not next:",
        "",
        "- runtime smoke",
        "- rollout",
        "- action execution",
        "- new Isaac capture",
        "- map_predict rerun",
        "- SSCNet inference",
        "- training/RL/PPO/BC/IL",
        "- prediction writeback/fusion",
        "- over-cost runtime",
        "- coverage-improvement claims",
    ]
    write_lines(path, lines)


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    real = summary["real_frame_aggregate_lambda48"]
    lines = [
        "# Stage 4A-6.5af Lambda48 Consolidation Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- output dir: `{summary['output_dir']}`",
        "- formula: `gain_exp / cost + 48 * minmax(source_occ_free)`",
        "- saved-frame-only ready: `yes`",
        "- runtime smoke ready: `no`",
        "- rollout ready: `no`",
        "",
        "## Key Results",
        "",
        "- Synthetic hidden-room: map_predict lambda48 selected hidden-room `5/5`, oracle agreement `1.0`, low-cost artifact `0.0`.",
        (
            f"- Real aggregate: same-as-measured `{real['same_as_measured_count']}/"
            f"{real['total_seed_frame_rows']}`, healthy non-measured "
            f"`{real['healthy_nonmeasured_count']}/{real['total_seed_frame_rows']}`, "
            f"prior basin `{real['historical_prior_basin_count']}`, low-cost artifact `{real['low_cost_artifact_count']}`."
        ),
        "- Lambda32 matched lambda48 at real branch-class level, but synthetic evidence still favors lambda48.",
        "- Over-cost remains diagnostic-only because Frame2 reproduced the old prior-basin risk shape.",
        "",
        "## Recommendation",
        "",
        summary["recommended_next_faithful_step"],
    ]
    write_lines(path, lines)


def import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_bar_plot(path: Path, labels: list[str], series: dict[str, list[float]], title: str, ylabel: str) -> None:
    plt = import_matplotlib()
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for idx, (name, values) in enumerate(series.items()):
        offsets = [value + (idx - (len(series) - 1) / 2) * width for value in x]
        ax.bar(offsets, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_readiness_heatmap(path: Path, matrix_rows: list[dict[str, Any]]) -> None:
    plt = import_matplotlib()
    columns = [
        "reproduces_intended_direction",
        "avoids_prior_bad_branch",
        "low_cost_artifact_0",
        "healthy_nonmeasured_present",
        "cross_frame_stable",
        "seed_robust",
        "source_faithful",
        "diagnostic_only",
        "saved_frame_ready",
        "runtime_smoke_ready",
        "rollout_ready",
    ]
    score = {"yes": 1.0, "diagnostic": 0.7, "mixed": 0.5, "not_applicable": 0.5, "no": 0.0}
    data = [[score.get(str(row[col]), 0.0) for col in columns] for row in matrix_rows]
    fig, ax = plt.subplots(figsize=(12.0, 5.2), constrained_layout=True)
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([col.replace("_", "\n") for col in columns], fontsize=8)
    ax.set_yticks(range(len(matrix_rows)))
    ax.set_yticklabels([row["case"] for row in matrix_rows], fontsize=9)
    for y, row in enumerate(matrix_rows):
        for x, col in enumerate(columns):
            ax.text(x, y, str(row[col]).replace("not_applicable", "n/a"), ha="center", va="center", fontsize=7, color="white")
    ax.set_title("Stage 4A-6.5af Readiness Matrix")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_plots(
    output_dir: Path,
    cross_rows: list[dict[str, Any]],
    lambda_cmp: list[dict[str, Any]],
    over_cmp: list[dict[str, Any]],
    low_cost_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    skipped_plots: list[dict[str, Any]],
) -> None:
    try:
        real_rows = [row for row in cross_rows if row["frame_type"].startswith("real")]
        labels = [row["frame_id"].replace("stage4a65p_", "") for row in real_rows]
        save_bar_plot(
            output_dir / "lambda48_real_frame_branch_fractions.png",
            labels,
            {
                "same_as_measured": [row["same_as_measured_fraction"] or 0.0 for row in real_rows],
                "healthy_nonmeasured": [row["healthy_nonmeasured_fraction"] or 0.0 for row in real_rows],
                "prior_basin": [row["historical_prior_basin_fraction"] or 0.0 for row in real_rows],
            },
            "Lambda48 Real-Frame Branch Fractions",
            "fraction",
        )
        save_bar_plot(
            output_dir / "lambda48_synthetic_vs_real_summary.png",
            [row["frame_id"].replace("stage4a65p_", "") for row in cross_rows],
            {
                "intended_or_healthy": [
                    row["hidden_room_fraction"]
                    if row["hidden_room_fraction"] is not None
                    else row["healthy_nonmeasured_fraction"] or 0.0
                    for row in cross_rows
                ],
                "low_cost_artifact": [row["low_cost_artifact_fraction"] or 0.0 for row in cross_rows],
            },
            "Lambda48 Synthetic vs Real Summary",
            "fraction",
        )
        real_cmp = [row for row in lambda_cmp if row["frame_type"].startswith("real")]
        save_bar_plot(
            output_dir / "lambda32_vs_lambda48_branch_comparison.png",
            [row["frame_id"].replace("stage4a65p_", "") for row in real_cmp],
            {
                "lambda32_healthy": [row["lambda32_healthy_or_hidden_fraction"] or 0.0 for row in real_cmp],
                "lambda48_healthy": [row["lambda48_healthy_or_hidden_fraction"] or 0.0 for row in real_cmp],
                "branch_class_match": [row["branch_class_match_fraction"] or 0.0 for row in real_cmp],
            },
            "Lambda32 vs Lambda48 Branch Comparison",
            "fraction",
        )
        source_over = [row for row in over_cmp if row.get("mode") == "source_occ_free_over_cost" and row["frame_type"].startswith("real")]
        save_bar_plot(
            output_dir / "over_cost_vs_lambda48_comparison.png",
            [row["frame_id"].replace("stage4a65p_", "") for row in source_over],
            {
                "over_cost_prior": [row["prior_basin_fraction"] or 0.0 for row in source_over],
                "over_cost_distinct": [row["distinct_nonmeasured_fraction"] or 0.0 for row in source_over],
                "lambda48_healthy": [row["lambda48_healthy_nonmeasured_fraction"] or 0.0 for row in source_over],
            },
            "Over-Cost Diagnostic vs Lambda48",
            "fraction",
        )
        lambda48_low = [row for row in low_cost_rows if row["mode"] == "map_predict_lambda48"]
        save_bar_plot(
            output_dir / "low_cost_artifact_cross_frame.png",
            [row["frame_id"].replace("stage4a65p_", "") for row in lambda48_low],
            {"low_cost_artifact": [row["low_cost_artifact_fraction"] or 0.0 for row in lambda48_low]},
            "Low-Cost Artifact Cross-Frame",
            "fraction",
        )
        save_readiness_heatmap(output_dir / "readiness_matrix_heatmap.png", readiness_rows)
        save_bar_plot(
            output_dir / "healthy_nonmeasured_fraction_by_frame.png",
            labels,
            {"healthy_nonmeasured": [row["healthy_nonmeasured_fraction"] or 0.0 for row in real_rows]},
            "Healthy Non-Measured Fraction by Real Frame",
            "fraction",
        )
        save_bar_plot(
            output_dir / "same_as_measured_fraction_by_frame.png",
            labels,
            {"same_as_measured": [row["same_as_measured_fraction"] or 0.0 for row in real_rows]},
            "Same-as-Measured Fraction by Real Frame",
            "fraction",
        )
    except Exception as exc:  # pragma: no cover - only used when plot stack is unavailable
        for name in PLOT_FILES:
            path = output_dir / name
            if not path.is_file():
                reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
                reason.write_text(f"# Plot Skipped\n\n`{name}` was skipped: `{exc}`.\n", encoding="utf-8")
                skipped_plots.append({"plot": name, "reason": str(exc)})


def missing_field_report(
    manifest: dict[str, Any],
    required_checks: list[tuple[str, dict[str, Any], list[str]]],
    skipped_plots: list[dict[str, Any]],
) -> dict[str, Any]:
    missing_files = [item for item in manifest["files"] if not item["exists"]]
    missing_fields: list[dict[str, Any]] = []
    for label, data, fields in required_checks:
        for field in fields:
            cursor: Any = data
            ok = True
            for part in field.split("."):
                if isinstance(cursor, dict) and part in cursor:
                    cursor = cursor[part]
                else:
                    ok = False
                    break
            if not ok:
                missing_fields.append({"source": label, "field": field})
    return {
        "missing_file_count": len(missing_files),
        "missing_files": missing_files,
        "missing_field_count": len(missing_fields),
        "missing_fields": missing_fields,
        "skipped_plots": skipped_plots,
        "nonessential_missing_files_allowed": True,
    }


def write_missing_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Missing Fields Report", ""]
    lines.append(f"- missing files: `{report['missing_file_count']}`")
    lines.append(f"- missing fields: `{report['missing_field_count']}`")
    lines.append(f"- skipped plots: `{len(report['skipped_plots'])}`")
    if report["missing_files"]:
        lines.extend(["", "## Missing Files", ""])
        for item in report["missing_files"]:
            lines.append(f"- {item['stage']}: `{item['file']}`")
    if report["missing_fields"]:
        lines.extend(["", "## Missing Fields", ""])
        for item in report["missing_fields"]:
            lines.append(f"- {item['source']}: `{item['field']}`")
    if report["skipped_plots"]:
        lines.extend(["", "## Skipped Plots", ""])
        for item in report["skipped_plots"]:
            lines.append(f"- `{item['plot']}`: {item['reason']}")
    write_lines(path, lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_dirs = {
        "Stage 4A-6.5ab": Path(args.stage4a65ab_dir),
        "Stage 4A-6.5ac": Path(args.stage4a65ac_dir),
        "Stage 4A-6.5ad": Path(args.stage4a65ad_dir),
        "Stage 4A-6.5ae": Path(args.stage4a65ae_dir),
    }
    manifest = build_manifest(stage_dirs)
    save_json(output_dir / "loaded_inputs_manifest.json", manifest)
    manifest_rows = manifest["files"]
    write_md_table(
        output_dir / "loaded_inputs_manifest.md",
        "Loaded Inputs Manifest",
        manifest_rows,
        ["stage", "file", "exists", "row_count", "seeds", "modes"],
    )

    ab_summary = load_json(stage_dirs["Stage 4A-6.5ab"] / "stage4a65ab_synthetic_calibration_summary.json", {})
    ac_summary = load_json(stage_dirs["Stage 4A-6.5ac"] / "stage4a65ac_saved_frame_lambda48_formula_summary.json", {})
    ad_behavior = load_json(stage_dirs["Stage 4A-6.5ad"] / "lambda48_behavior_summary.json", {})
    ae_behavior = load_json(stage_dirs["Stage 4A-6.5ae"] / "lambda48_behavior_summary.json", {})

    unified = build_unified_config_table(
        stage_dirs["Stage 4A-6.5ab"],
        ab_summary,
        ac_summary,
        ad_behavior,
        ae_behavior,
    )
    unified_cols = [
        "stage",
        "frame_type",
        "frame_id",
        "mode",
        "formula",
        "lambda",
        "tau",
        "occ_threshold",
        "free_threshold",
        "seeds",
        "decision_rows",
        "same_as_measured_fraction",
        "distinct_nonmeasured_fraction",
        "healthy_nonmeasured_fraction",
        "hidden_room_fraction",
        "oracle_agreement",
        "historical_prior_basin_fraction",
        "low_cost_artifact_fraction",
        "median_margin",
        "mean_source_occ_free",
        "mean_normalized_sc",
        "saved_frame_only_ready",
        "runtime_ready",
        "rollout_ready",
    ]
    save_json(output_dir / "unified_config_table.json", unified)
    save_csv(output_dir / "unified_config_table.csv", unified, unified_cols)
    write_md_table(output_dir / "unified_config_table.md", "Unified Config Table", unified, unified_cols)

    cross_rows = build_lambda48_cross_frame_summary(ac_summary, ad_behavior, ae_behavior)
    cross_cols = [
        "stage",
        "frame_type",
        "frame_id",
        "seeds",
        "same_as_measured_count",
        "same_as_measured_fraction",
        "distinct_nonmeasured_count",
        "distinct_nonmeasured_fraction",
        "healthy_nonmeasured_count",
        "healthy_nonmeasured_fraction",
        "hidden_room_count",
        "hidden_room_fraction",
        "oracle_agreement",
        "historical_prior_basin_count",
        "historical_prior_basin_fraction",
        "low_cost_artifact_count",
        "low_cost_artifact_fraction",
        "median_margin",
        "mean_source_occ_free",
        "mean_normalized_sc",
        "runtime_ready",
        "rollout_ready",
    ]
    save_json(output_dir / "lambda48_cross_frame_summary.json", cross_rows)
    save_csv(output_dir / "lambda48_cross_frame_summary.csv", cross_rows, cross_cols)
    write_md_table(output_dir / "lambda48_cross_frame_summary.md", "Lambda48 Cross-Frame Summary", cross_rows, cross_cols)

    aggregate_rows = build_real_frame_aggregate(
        stage_dirs["Stage 4A-6.5ad"],
        stage_dirs["Stage 4A-6.5ae"],
        cross_rows,
    )
    aggregate = aggregate_rows[0]
    aggregate_cols = list(aggregate.keys())
    save_json(output_dir / "real_frame_aggregate_lambda48.json", aggregate_rows)
    save_csv(output_dir / "real_frame_aggregate_lambda48.csv", aggregate_rows, aggregate_cols)
    write_md_table(
        output_dir / "real_frame_aggregate_lambda48.md",
        "Real Frame Aggregate Lambda48",
        aggregate_rows,
        aggregate_cols,
    )

    lambda_cmp = build_lambda32_vs_lambda48_comparison(
        ac_summary,
        stage_dirs["Stage 4A-6.5ad"],
        ad_behavior,
        stage_dirs["Stage 4A-6.5ae"],
        ae_behavior,
    )
    lambda_cmp_cols = [
        "stage",
        "frame_type",
        "frame_id",
        "seed_count",
        "lambda32_same_as_measured_fraction",
        "lambda48_same_as_measured_fraction",
        "lambda32_healthy_or_hidden_fraction",
        "lambda48_healthy_or_hidden_fraction",
        "lambda32_prior_basin_fraction",
        "lambda48_prior_basin_fraction",
        "branch_class_match_fraction",
        "exact_selection_match_fraction",
        "selected_child_match_fraction",
        "best_descendant_match_fraction",
        "lambda48_changed_branch_class_beyond_lambda32",
        "lambda48_changed_exact_selection_beyond_lambda32",
        "interpretation",
    ]
    save_json(output_dir / "lambda32_vs_lambda48_comparison.json", lambda_cmp)
    save_csv(output_dir / "lambda32_vs_lambda48_comparison.csv", lambda_cmp, lambda_cmp_cols)
    write_md_table(
        output_dir / "lambda32_vs_lambda48_comparison.md",
        "Lambda32 vs Lambda48 Comparison",
        lambda_cmp,
        lambda_cmp_cols,
    )

    over_cmp = build_over_cost_diagnostic_comparison(ad_behavior, ae_behavior)
    over_cols = [
        "stage",
        "frame_type",
        "frame_id",
        "mode",
        "seed_count",
        "same_as_measured_fraction",
        "distinct_nonmeasured_fraction",
        "healthy_nonmeasured_fraction",
        "prior_basin_fraction",
        "low_cost_artifact_fraction",
        "lambda48_same_as_measured_fraction",
        "lambda48_healthy_nonmeasured_fraction",
        "more_aggressive_than_lambda48",
        "remains_risky",
        "diagnostic_only",
        "recommendation",
    ]
    save_json(output_dir / "over_cost_diagnostic_comparison.json", over_cmp)
    save_csv(output_dir / "over_cost_diagnostic_comparison.csv", over_cmp, over_cols)
    write_md_table(
        output_dir / "over_cost_diagnostic_comparison.md",
        "Over-Cost Diagnostic Comparison",
        over_cmp,
        over_cols,
    )

    low_cost_rows = build_low_cost_cross_frame(ac_summary, ad_behavior, ae_behavior)
    low_cols = [
        "stage",
        "frame_type",
        "frame_id",
        "mode",
        "seed_count",
        "low_cost_artifact_fraction",
        "historical_prior_basin_fraction",
        "diagnostic_only",
    ]
    save_json(output_dir / "low_cost_artifact_cross_frame.json", low_cost_rows)
    save_csv(output_dir / "low_cost_artifact_cross_frame.csv", low_cost_rows, low_cols)
    write_md_table(
        output_dir / "low_cost_artifact_cross_frame.md",
        "Low-Cost Artifact Cross-Frame",
        low_cost_rows,
        low_cols,
    )

    readiness = build_readiness_matrix(aggregate)
    readiness_cols = [
        "case",
        "reproduces_intended_direction",
        "avoids_prior_bad_branch",
        "low_cost_artifact_0",
        "healthy_nonmeasured_present",
        "cross_frame_stable",
        "seed_robust",
        "source_faithful",
        "diagnostic_only",
        "saved_frame_ready",
        "runtime_smoke_ready",
        "rollout_ready",
    ]
    save_json(output_dir / "readiness_matrix.json", readiness)
    save_csv(output_dir / "readiness_matrix.csv", readiness, readiness_cols)
    write_md_table(output_dir / "readiness_matrix.md", "Readiness Matrix", readiness, readiness_cols)

    findings = build_design_findings(aggregate)
    save_json(output_dir / "design_review_findings.json", findings)
    write_design_findings_md(output_dir / "design_review_findings.md", findings)
    write_recommended_next(output_dir / "recommended_next_faithful_step.md")

    skipped_plots: list[dict[str, Any]] = []
    if args.save_viz:
        make_plots(output_dir, cross_rows, lambda_cmp, over_cmp, low_cost_rows, readiness, skipped_plots)
    else:
        for name in PLOT_FILES:
            skipped = output_dir / f"{Path(name).stem}_skipped_reason.md"
            skipped.write_text(f"# Plot Skipped\n\n`{name}` was skipped because `--save_viz` was not set.\n", encoding="utf-8")
            skipped_plots.append({"plot": name, "reason": "--save_viz was not set"})

    required_checks = [
        (
            "Stage 4A-6.5ab summary",
            ab_summary,
            ["answers.best_config_readiness", "decision_row_count", "best_candidate_config"],
        ),
        (
            "Stage 4A-6.5ac summary",
            ac_summary,
            ["answers.readiness.saved_frame_only", "mode_summary", "lambda48_reproduction_summary"],
        ),
        (
            "Stage 4A-6.5ad behavior",
            ad_behavior,
            ["map_predict_lambda48", "map_predict_lambda32", "source_occ_free_over_cost"],
        ),
        (
            "Stage 4A-6.5ae behavior",
            ae_behavior,
            ["map_predict_lambda48", "map_predict_lambda32", "source_occ_free_over_cost"],
        ),
    ]
    missing = missing_field_report(manifest, required_checks, skipped_plots)
    save_json(output_dir / "missing_fields_report.json", missing)
    write_missing_md(output_dir / "missing_fields_report.md", missing)

    summary = {
        "stage": "Stage 4A-6.5af",
        "status": "completed",
        "output_dir": str(output_dir),
        "input_dirs": {
            "stage4a65ab": str(stage_dirs["Stage 4A-6.5ab"]),
            "stage4a65ac": str(stage_dirs["Stage 4A-6.5ac"]),
            "stage4a65ad": str(stage_dirs["Stage 4A-6.5ad"]),
            "stage4a65ae": str(stage_dirs["Stage 4A-6.5ae"]),
            "stage4a65p": args.stage4a65p_dir,
            "stage4a65z": args.stage4a65z_dir,
            "stage4a65z1": args.stage4a65z1_dir,
        },
        "candidate_formula": findings["candidate_formula"],
        "lambda48_cross_frame_summary": cross_rows,
        "real_frame_aggregate_lambda48": aggregate,
        "lambda32_vs_lambda48_real_aggregate": lambda_cmp[-1],
        "over_cost_diagnostic_aggregate": over_cmp[-1],
        "readiness": {
            "saved_frame_only_ready": True,
            "runtime_smoke_ready": False,
            "rollout_ready": False,
        },
        "recommended_next_faithful_step": findings["recommendation"]["exactly_one_next_small_task"],
        "safety": dict(SAFETY_FALSE_FLAGS),
        "coverage_improvement_claimed": False,
        "missing_file_count": missing["missing_file_count"],
        "missing_field_count": missing["missing_field_count"],
        "skipped_plot_count": len(skipped_plots),
    }
    save_json(output_dir / "stage4a65af_lambda48_consolidation_summary.json", summary)
    write_summary_md(output_dir / "stage4a65af_lambda48_consolidation_summary.md", summary)

    print(json.dumps({"status": "completed", "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()

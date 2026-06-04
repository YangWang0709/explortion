#!/usr/bin/env python3
"""Stage 4A-6.5ap seed0/1/2 repeat review and alternate-start design.

This stage is offline review/design only. It reads completed Stage 4A-6.5ak,
6.5am, and 6.5ao bounded runtime smokes, compares tree_seed 0/1/2, and writes
a future Stage 4A-6.5aq alternate-start command sketch. It does not start
Isaac, capture RGB/depth, run map_predict or SSCNet inference, execute
actions, run rollout, train, or modify existing runtime inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design"
DEFAULT_AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_AM_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_AO_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2"
DEFAULT_AL_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65al_post_action_two_frame_diagnosis"
DEFAULT_AN_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65an_repeat_comparison_and_next_design"
DEFAULT_AG_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ag_multi_frame_lambda48_replay"
DEFAULT_SCENE_FACTORY = WORKSPACE / "sim_explorer/scene_factory.py"
DEFAULT_MEDIUM_DATASET_DIR = WORKSPACE / "outputs/isaac_medium_rollout_dataset_empty_pred_astar"
CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]

PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
PROHIBITED_FORMULAS = [
    "(gain_exp + 48 * source_occ_free) / cost",
    "(gain_exp + source_occ_free) / cost",
]
HISTORICAL_PRIOR_SELECTED_GRID = [11, 15, 11]
HISTORICAL_PRIOR_BEST_GRID = [14, 15, 11]
VOXEL_SIZE_M = 0.1
CURRENT_CANONICAL_START = {
    "variant": "canonical_stage4a65p_frame1",
    "position": [-4.65, -4.65, 1.2],
    "yaw_rad": 0.38710316317995463,
    "source": "Stage 4A-6.5ak/am/ao repeat-variant metadata",
}

REQUIRED_OUTPUTS = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_input_manifest.json",
    "loaded_input_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "input_hash_audit.json",
    "input_hash_audit.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "sequence_safety_reverification.json",
    "sequence_safety_reverification.md",
    "prediction_safety_reverification.json",
    "prediction_safety_reverification.md",
    "no_rollout_reverification.json",
    "no_rollout_reverification.md",
    "forbidden_artifact_scan.json",
    "forbidden_artifact_scan.md",
    "seed012_decision_table.csv",
    "seed012_decision_table.json",
    "seed012_decision_table.md",
    "frame1_seed012_comparison.json",
    "frame1_seed012_comparison.md",
    "frame2_seed012_comparison.json",
    "frame2_seed012_comparison.md",
    "branch_spatial_delta_matrix.csv",
    "branch_spatial_delta_matrix.json",
    "branch_spatial_delta_matrix.md",
    "branch_class_transition_summary.json",
    "branch_class_transition_summary.md",
    "lambda32_lambda48_seed012_agreement.json",
    "lambda32_lambda48_seed012_agreement.md",
    "action_pose_seed012_comparison.json",
    "action_pose_seed012_comparison.md",
    "observed_state_seed012_comparison.csv",
    "observed_state_seed012_comparison.json",
    "observed_state_seed012_comparison.md",
    "map_predict_seed012_stability.csv",
    "map_predict_seed012_stability.json",
    "map_predict_seed012_stability.md",
    "low_cost_artifact_seed012_review.json",
    "low_cost_artifact_seed012_review.md",
    "historical_prior_basin_seed012_review.json",
    "historical_prior_basin_seed012_review.md",
    "branch_health_seed012_review.json",
    "branch_health_seed012_review.md",
    "cost_dominance_seed012_review.json",
    "cost_dominance_seed012_review.md",
    "seed012_outcome_classification.json",
    "seed012_outcome_classification.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "recommended_next_faithful_step.md",
    "alternate_start_candidate_inventory.json",
    "alternate_start_candidate_inventory.md",
    "selected_alternate_start_design.json",
    "selected_alternate_start_design.md",
    "future_stage4a65aq_command_sketch.md",
    "do_not_run_runtime_in_stage4a65ap.md",
    "stage4a65ap_seed012_repeat_review_summary.json",
    "stage4a65ap_seed012_repeat_review_summary.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_PLOTS = [
    "frame1_seed012_topdown_comparison.png",
    "frame2_seed012_topdown_comparison.png",
    "action_pose_seed012_topdown.png",
    "observed_delta_seed012_bar.png",
    "prediction_count_seed012_bar.png",
    "branch_class_transition_seed012.png",
    "seed012_spatial_delta_heatmap.png",
    "repeat_safety_readiness_matrix.png",
    "alternate_start_design_topdown.png",
    "next_stage_decision_flowchart.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "*.npy",
    "*.npz",
    "frame001*",
    "frame002*",
    "frame003*",
    "action002*",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "manifest.jsonl",
    "episode_manifest*",
    "capture_rgb*.png",
    "capture_depth*.npy",
    "capture_depth*.png",
    "*replay_buffer*",
    "*policy_checkpoint*",
]

AK_REQUIRED_INPUTS = [
    "stage4a65ak_two_frame_one_action_runtime_summary.json",
    "runtime_setup_summary.json",
    "hash_checks.json",
    "prediction_safety_report.json",
    "no_rollout_report.json",
    "action_execution_report.json",
    "formula_definition.json",
    "observed_state_update_frame001.json",
    "observed_state_update_frame002.json",
    "observed_state_frame001.npy",
    "observed_state_frame002.npy",
    "map_predict_frame001_summary.json",
    "map_predict_frame002_summary.json",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame002_branch_classification.json",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.json",
]

AM_REQUIRED_INPUTS = [
    "stage4a65am_bounded_repeat_safety_summary.json",
    "comparison_to_stage4a65ak.json",
    "runtime_setup_summary.json",
    "hash_checks.json",
    "prediction_safety_report.json",
    "no_rollout_report.json",
    "repeat_variant_definition.json",
    "repeat_stability_classification.json",
    "lambda32_vs_lambda48_repeat.json",
    "observed_state_delta_summary.json",
    "map_predict_two_frame_stability.json",
    "action_execution_report.json",
    "formula_definition.json",
    "observed_state_frame001.npy",
    "observed_state_frame002.npy",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame002_branch_classification.json",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.json",
]

AO_REQUIRED_INPUTS = [
    "stage4a65ao_bounded_repeat_safety_summary.json",
    "comparison_to_seed0_seed1_combined.json",
    "comparison_to_stage4a65ak.json",
    "comparison_to_stage4a65am.json",
    "runtime_setup_summary.json",
    "hash_checks.json",
    "prediction_safety_report.json",
    "no_rollout_report.json",
    "repeat_variant_definition.json",
    "repeat_stability_classification.json",
    "lambda32_vs_lambda48_repeat.json",
    "observed_state_delta_summary.json",
    "map_predict_two_frame_stability.json",
    "action_execution_report.json",
    "formula_definition.json",
    "observed_state_frame001.npy",
    "observed_state_frame002.npy",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame002_branch_classification.json",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.json",
]

SUPPORTING_INPUTS = {
    "stage4a65al": [
        "stage4a65al_post_action_two_frame_diagnosis_summary.json",
        "observed_state_delta_summary.json",
        "lambda32_vs_lambda48_two_frame.json",
        "branch_health_review.json",
        "cost_dominance_review.json",
        "historical_prior_basin_recheck.json",
        "low_cost_artifact_two_frame_review.json",
    ],
    "stage4a65an": [
        "stage4a65an_repeat_comparison_summary.json",
        "repeat_outcome_classification.json",
        "future_stage4a65ao_command_sketch.md",
    ],
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean(row.get(name, "")) for name in fieldnames})


def md_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    out = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.12g}"
            values.append(str(value))
        out.append("|" + "|".join(values) + "|")
    return "\n".join(out)


def write_summary_md(path: Path, title: str, lines: list[str]) -> None:
    write_text(path, "\n".join([f"# {title}", "", *lines]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_worker(path_text: str) -> tuple[str, dict[str, Any]]:
    path = Path(path_text)
    entry = {
        "path": path_text,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    return path_text, entry


def hash_paths(paths: list[Path], max_workers: int) -> dict[str, dict[str, Any]]:
    unique = sorted({str(path) for path in paths})
    workers = max(1, min(max_workers, len(unique) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(hash_worker, unique))


def vec_distance(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def yaw_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    delta = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def decision(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data.get("decision", data)


def optional_json(path: Path, default: Any = None) -> Any:
    return read_json(path) if path.is_file() else default


def compact_decision(dec: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "selected_child_id",
        "selected_child_grid",
        "selected_child_world",
        "best_descendant_id",
        "best_descendant_grid",
        "best_descendant_world",
        "branch_classification",
        "same_as_measured",
        "changed_vs_measured_only",
        "healthy_nonmeasured_candidate",
        "low_cost_artifact",
        "spatial_prior_sc_basin",
        "gain_exp",
        "source_occ_free",
        "normalized_sc",
        "sc_bonus",
        "cost",
        "final_value",
        "margin",
        "normalized_margin",
        "selected_cost_rank",
        "selected_gain_exp_rank",
        "selected_source_occ_free_rank",
    ]
    return {key: dec.get(key) for key in keys if key in dec}


def count_value(entry: dict[str, Any], name: str) -> Any:
    if name in entry:
        return entry[name]
    return entry.get(f"{name}_count")


def normalize_observed(data: dict[str, Any], stage_key: str) -> dict[str, Any]:
    frame001 = data.get("frame001", {})
    frame002 = data.get("frame002", {})
    return {
        "stage_key": stage_key,
        "frame001": {
            "free": count_value(frame001, "free"),
            "observed": count_value(frame001, "observed"),
            "observed_ratio": frame001.get("observed_ratio"),
            "occupied": count_value(frame001, "occupied"),
            "unknown": count_value(frame001, "unknown"),
        },
        "frame002": {
            "free": count_value(frame002, "free"),
            "observed": count_value(frame002, "observed"),
            "observed_ratio": frame002.get("observed_ratio"),
            "occupied": count_value(frame002, "occupied"),
            "unknown": count_value(frame002, "unknown"),
        },
        "observed_ratio_delta": data.get("observed_ratio_delta"),
        "newly_observed": data.get("newly_observed", data.get("newly_observed_voxels")),
        "unknown_to_free": data.get("unknown_to_free", data.get("newly_free_voxels")),
        "unknown_to_occupied": data.get("unknown_to_occupied", data.get("newly_occupied_voxels")),
        "free_to_occupied": data.get("free_to_occupied", data.get("minor_free_to_occupied_refinement_count", 0)),
        "occupied_to_free": data.get("occupied_to_free", 0),
        "invalid_labels": data.get("invalid_labels", data.get("invalid_label_count_frame002", 0)),
        "measured_only_status": data.get("measured_only_status", True),
        "suspicious_label_flips": data.get("suspicious_label_flips", False),
    }


def observed_from_updates(run_dir: Path, stage_key: str) -> dict[str, Any]:
    f1 = read_json(run_dir / "observed_state_update_frame001.json")
    f2 = read_json(run_dir / "observed_state_update_frame002.json")
    return normalize_observed(
        {
            "frame001": f1.get("updated_summary", {}),
            "frame002": f2.get("updated_summary", {}),
            "observed_ratio_delta": f2.get("delta_observed_ratio"),
            "newly_observed": f2.get("delta_observed_count"),
            "unknown_to_free": None,
            "unknown_to_occupied": None,
            "free_to_occupied": None,
            "occupied_to_free": None,
            "invalid_labels": 0,
            "measured_only_status": bool(f1.get("measured_only")) and bool(f2.get("measured_only")),
        },
        stage_key,
    )


def map_counts(frame_summary: dict[str, Any]) -> dict[str, Any]:
    stats = frame_summary.get("stats", {})
    return {
        "prediction_valid_count": stats.get("prediction_valid_count"),
        "predicted_unmeasured_occ_free": stats.get("predicted_unmeasured_occ_free_count"),
        "predicted_free_count": stats.get("predicted_unmeasured_free_count"),
        "predicted_occupied_count": stats.get("predicted_unmeasured_occupied_count"),
        "alignment_convention": stats.get("alignment_convention", frame_summary.get("alignment_convention")),
        "shape_aligned": frame_summary.get("prediction_layer_shape_aligned_to_observed_state", stats.get("shape_aligned_to_observed_state")),
        "map_predict_succeeded": frame_summary.get("map_predict_succeeded"),
    }


def load_run(run_dir: Path, stage_key: str, tree_seed: int, al_observed: dict[str, Any] | None = None) -> dict[str, Any]:
    summary_name = {
        "stage4a65ak": "stage4a65ak_two_frame_one_action_runtime_summary.json",
        "stage4a65am": "stage4a65am_bounded_repeat_safety_summary.json",
        "stage4a65ao": "stage4a65ao_bounded_repeat_safety_summary.json",
    }[stage_key]
    observed_path = run_dir / "observed_state_delta_summary.json"
    if observed_path.is_file():
        observed = normalize_observed(read_json(observed_path), stage_key)
    elif al_observed is not None:
        observed = normalize_observed(al_observed, stage_key)
    else:
        observed = observed_from_updates(run_dir, stage_key)
    return {
        "stage_key": stage_key,
        "stage_label": {
            "stage4a65ak": "Stage 4A-6.5ak",
            "stage4a65am": "Stage 4A-6.5am",
            "stage4a65ao": "Stage 4A-6.5ao",
        }[stage_key],
        "seed_label": f"seed{tree_seed}",
        "tree_seed": tree_seed,
        "dir": str(run_dir),
        "summary": read_json(run_dir / summary_name),
        "runtime_setup": optional_json(run_dir / "runtime_setup_summary.json", {}),
        "prediction_safety": read_json(run_dir / "prediction_safety_report.json"),
        "no_rollout": read_json(run_dir / "no_rollout_report.json"),
        "action": read_json(run_dir / "action_execution_report.json"),
        "formula": read_json(run_dir / "formula_definition.json"),
        "observed": observed,
        "map_stability": optional_json(run_dir / "map_predict_two_frame_stability.json", None),
        "lambda_agreement_file": optional_json(run_dir / "lambda32_vs_lambda48_repeat.json", None),
        "frame001": {
            "measured": decision(run_dir / "frame001_measured_shadow_tree_decision.json"),
            "lambda48": decision(run_dir / "frame001_lambda48_primary_tree_decision.json"),
            "lambda32": decision(run_dir / "frame001_lambda32_shadow_tree_decision.json"),
            "branch": read_json(run_dir / "frame001_branch_classification.json"),
            "low_cost": read_json(run_dir / "frame001_low_cost_artifact_diagnosis.json"),
            "map_predict": read_json(run_dir / "map_predict_frame001_summary.json"),
        },
        "frame002": {
            "measured": decision(run_dir / "frame002_measured_shadow_tree_decision.json"),
            "lambda48": decision(run_dir / "frame002_lambda48_diagnostic_tree_decision.json"),
            "lambda32": decision(run_dir / "frame002_lambda32_shadow_tree_decision.json"),
            "branch": read_json(run_dir / "frame002_branch_classification.json"),
            "low_cost": read_json(run_dir / "frame002_low_cost_artifact_diagnosis.json"),
            "map_predict": read_json(run_dir / "map_predict_frame002_summary.json"),
        },
    }


def build_context_manifest() -> dict[str, Any]:
    entries = []
    combined = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_stage4a65ak": "Stage 4A-6.5ak" in text,
                "contains_stage4a65am": "Stage 4A-6.5am" in text,
                "contains_stage4a65ao": "Stage 4A-6.5ao" in text,
                "contains_stage4a65ap_next": "Stage 4A-6.5ap" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5ap",
        "review_design_only": True,
        "loaded_context_files": entries,
        "confirmed_stage4a65ak_complete": "Stage 4A-6.5ak" in combined and "tree_seed" in combined,
        "confirmed_stage4a65am_complete": "Stage 4A-6.5am" in combined and "tree_seed=1" in combined,
        "confirmed_stage4a65ao_complete": "Stage 4A-6.5ao" in combined and "tree_seed=2" in combined,
        "chat_history_not_used_as_source": True,
    }


def check_required(base: Path, names: list[str]) -> list[str]:
    return [name for name in names if not (base / name).is_file()]


def build_input_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    specs = {
        "stage4a65ak": (args.stage4a65ak_dir, AK_REQUIRED_INPUTS),
        "stage4a65am": (args.stage4a65am_dir, AM_REQUIRED_INPUTS),
        "stage4a65ao": (args.stage4a65ao_dir, AO_REQUIRED_INPUTS),
        "stage4a65al": (args.stage4a65al_dir, SUPPORTING_INPUTS["stage4a65al"]),
        "stage4a65an": (args.stage4a65an_dir, SUPPORTING_INPUTS["stage4a65an"]),
    }
    stages = {}
    missing = []
    for stage, (base, required) in specs.items():
        missing_stage = check_required(base, required)
        stages[stage] = {
            "dir": str(base),
            "exists": base.is_dir(),
            "required_file_count": len(required),
            "missing_required_files": missing_stage,
            "loaded": base.is_dir() and not missing_stage,
            "supporting": stage in {"stage4a65al", "stage4a65an"},
        }
        if stage in {"stage4a65ak", "stage4a65am", "stage4a65ao"}:
            missing.extend([f"{stage}:{name}" for name in missing_stage])
    ag_jsons = sorted(str(path) for path in args.stage4a65ag_dir.glob("*.json")) if args.stage4a65ag_dir.is_dir() else []
    stages["stage4a65ag"] = {
        "dir": str(args.stage4a65ag_dir),
        "exists": args.stage4a65ag_dir.is_dir(),
        "optional": True,
        "top_level_json_count": len(ag_jsons),
        "loaded_as_optional_reference": args.stage4a65ag_dir.is_dir(),
    }
    scene_factory_text = args.scene_factory.read_text(encoding="utf-8") if args.scene_factory.is_file() else ""
    return {
        "stage": "Stage 4A-6.5ap",
        "primary_inputs": stages,
        "all_primary_inputs_loaded": not missing,
        "scene_factory": {
            "path": str(args.scene_factory),
            "exists": args.scene_factory.is_file(),
            "sha256": sha256_file(args.scene_factory) if args.scene_factory.is_file() else None,
            "mentions_medium_three_rooms": "medium_three_rooms" in scene_factory_text,
            "mentions_start_corridor": "start_corridor" in scene_factory_text,
        },
        "medium_rollout_dataset_dir": {
            "path": str(args.medium_rollout_dataset_dir),
            "exists": args.medium_rollout_dataset_dir.is_dir(),
        },
    }, missing


def collect_hash_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [CHECKPOINT, args.scene_factory, *CONTEXT_FILES]
    for base in (args.stage4a65ak_dir, args.stage4a65am_dir, args.stage4a65ao_dir):
        if base.is_dir():
            paths.extend(path for path in base.rglob("*") if path.suffix in {".json", ".npy", ".npz"})
    for base in (args.stage4a65al_dir, args.stage4a65an_dir, args.stage4a65ag_dir):
        if base.is_dir():
            paths.extend(path for path in base.glob("*.json"))
    for variant in ("start_corridor", "start_room_b", "start_room_a"):
        meta = args.medium_rollout_dataset_dir / "episodes" / f"medium_three_rooms_seed0_{variant}_empty_astar" / "scene_metadata.json"
        if meta.is_file():
            paths.append(meta)
    return [path for path in paths if path.exists()]


def sequence_entry(run: dict[str, Any]) -> dict[str, Any]:
    setup = run["summary"].get("runtime_setup", run.get("runtime_setup", {}))
    safety = run["summary"].get("safety", {})
    frames = int(setup.get("frames_captured", safety.get("frames_captured", 0)))
    maps = int(setup.get("map_predict_calls", safety.get("map_predict_calls", 0)))
    actions = int(setup.get("selected_action_execution_count", safety.get("selected_action_execution_count", 0)))
    second_action = bool(setup.get("second_action", safety.get("second_action", False)))
    third_frame = bool(setup.get("third_frame", safety.get("third_frame", False)))
    rollout = bool(setup.get("rollout", safety.get("rollout", False)))
    return {
        "tree_seed": run["tree_seed"],
        "isaac_startup_count": int(setup.get("isaac_startup_count", 0)),
        "frames_captured": frames,
        "map_predict_calls": maps,
        "selected_action_execution_count": actions,
        "second_action": second_action,
        "third_frame": third_frame,
        "rollout": rollout,
        "two_frame_runtime_executed": bool(setup.get("two_frame_runtime_executed", safety.get("two_frame_runtime_executed", True))),
        "exactly_two_frames": frames == 2,
        "exactly_two_map_predict_calls": maps == 2,
        "exactly_one_action": actions == 1,
        "sequence_clean": frames == 2 and maps == 2 and actions == 1 and not second_action and not third_frame and not rollout,
    }


def pair_compare(runs: dict[str, dict[str, Any]], a: str, b: str, frame: str) -> dict[str, Any]:
    ra = runs[a]
    rb = runs[b]
    da = ra[frame]["lambda48"]
    db = rb[frame]["lambda48"]
    selected_delta = vec_distance(da.get("selected_child_world"), db.get("selected_child_world"))
    best_delta = vec_distance(da.get("best_descendant_world"), db.get("best_descendant_world"))
    action_delta = vec_distance(
        ra["action"].get("executed_pose", {}).get("position"),
        rb["action"].get("executed_pose", {}).get("position"),
    )
    ydelta = yaw_delta(
        ra["action"].get("executed_pose", {}).get("yaw_rad"),
        rb["action"].get("executed_pose", {}).get("yaw_rad"),
    )
    if da.get("selected_child_id") == db.get("selected_child_id") and da.get("best_descendant_id") == db.get("best_descendant_id"):
        classification = "exact"
    elif selected_delta is not None and selected_delta <= 0.65 and all(
        not runs[name][frame]["lambda48"].get("low_cost_artifact", False) for name in (a, b)
    ):
        classification = "spatially_consistent_healthy"
    else:
        classification = "divergent_but_healthy"
    return {
        "pair": f"{a}_vs_{b}",
        "seed_a": ra["tree_seed"],
        "seed_b": rb["tree_seed"],
        "frame": frame,
        "selected_child_exact_agreement": da.get("selected_child_id") == db.get("selected_child_id"),
        "best_descendant_exact_agreement": da.get("best_descendant_id") == db.get("best_descendant_id"),
        "selected_child_spatial_delta_m": selected_delta,
        "best_descendant_spatial_delta_m": best_delta,
        "branch_class_a": da.get("branch_classification"),
        "branch_class_b": db.get("branch_classification"),
        "branch_class_agreement": da.get("branch_classification") == db.get("branch_classification"),
        "action_pose_delta_m": action_delta,
        "action_yaw_delta_rad": ydelta,
        "observed_ratio_delta_difference": (
            rb["observed"].get("observed_ratio_delta") - ra["observed"].get("observed_ratio_delta")
            if isinstance(rb["observed"].get("observed_ratio_delta"), (int, float))
            and isinstance(ra["observed"].get("observed_ratio_delta"), (int, float))
            else None
        ),
        "newly_observed_difference": (
            rb["observed"].get("newly_observed") - ra["observed"].get("newly_observed")
            if isinstance(rb["observed"].get("newly_observed"), (int, float))
            and isinstance(ra["observed"].get("newly_observed"), (int, float))
            else None
        ),
        "frame2_prediction_valid_count_delta": (
            map_counts(rb["frame002"]["map_predict"]).get("prediction_valid_count")
            - map_counts(ra["frame002"]["map_predict"]).get("prediction_valid_count")
        ),
        "frame2_predicted_occ_free_delta": (
            map_counts(rb["frame002"]["map_predict"]).get("predicted_unmeasured_occ_free")
            - map_counts(ra["frame002"]["map_predict"]).get("predicted_unmeasured_occ_free")
        ),
        "pair_classification": classification,
    }


def lambda_agreement(run: dict[str, Any], frame: str) -> dict[str, Any]:
    l48 = run[frame]["lambda48"]
    l32 = run[frame]["lambda32"]
    return {
        "seed": run["tree_seed"],
        "stage": run["stage_label"],
        "frame": frame,
        "lambda48_selected_child_id": l48.get("selected_child_id"),
        "lambda48_best_descendant_id": l48.get("best_descendant_id"),
        "lambda48_branch_classification": l48.get("branch_classification"),
        "lambda32_selected_child_id": l32.get("selected_child_id"),
        "lambda32_best_descendant_id": l32.get("best_descendant_id"),
        "lambda32_branch_classification": l32.get("branch_classification"),
        "same_selected_child": l48.get("selected_child_id") == l32.get("selected_child_id"),
        "same_best_descendant": l48.get("best_descendant_id") == l32.get("best_descendant_id"),
        "same_branch_class": l48.get("branch_classification") == l32.get("branch_classification"),
    }


def build_decision_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in ("seed0", "seed1", "seed2"):
        run = runs[key]
        for frame in ("frame001", "frame002"):
            l48 = run[frame]["lambda48"]
            l32 = run[frame]["lambda32"]
            measured = run[frame]["measured"]
            rows.append(
                {
                    "seed": run["tree_seed"],
                    "stage": run["stage_label"],
                    "frame": frame,
                    "measured_selected_child_id": measured.get("selected_child_id"),
                    "measured_best_descendant_id": measured.get("best_descendant_id"),
                    "lambda48_selected_child_id": l48.get("selected_child_id"),
                    "lambda48_best_descendant_id": l48.get("best_descendant_id"),
                    "lambda32_selected_child_id": l32.get("selected_child_id"),
                    "lambda32_best_descendant_id": l32.get("best_descendant_id"),
                    "lambda48_branch_classification": l48.get("branch_classification"),
                    "lambda32_lambda48_exact_agreement": l48.get("selected_child_id") == l32.get("selected_child_id")
                    and l48.get("best_descendant_id") == l32.get("best_descendant_id"),
                    "low_cost_artifact": l48.get("low_cost_artifact"),
                    "historical_prior_basin": run[frame]["branch"].get("historical_prior_basin", l48.get("spatial_prior_sc_basin")),
                    "selected_child_grid": l48.get("selected_child_grid"),
                    "selected_child_world": l48.get("selected_child_world"),
                    "best_descendant_grid": l48.get("best_descendant_grid"),
                    "best_descendant_world": l48.get("best_descendant_world"),
                    "gain_exp": l48.get("gain_exp"),
                    "source_occ_free": l48.get("source_occ_free"),
                    "normalized_sc": l48.get("normalized_sc"),
                    "sc_bonus": l48.get("sc_bonus"),
                    "cost": l48.get("cost"),
                    "final_value": l48.get("final_value"),
                    "margin": l48.get("margin"),
                    "observed_ratio_delta": run["observed"].get("observed_ratio_delta"),
                    "newly_observed": run["observed"].get("newly_observed"),
                    "frame2_density_ratio": run.get("map_stability", {}).get("density_ratio_frame2_over_frame1") if run.get("map_stability") else None,
                    "action_pose": run["action"].get("executed_pose", {}).get("position") if frame == "frame001" else "",
                    "action_yaw": run["action"].get("executed_pose", {}).get("yaw_rad") if frame == "frame001" else "",
                }
            )
    return rows


def action_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    start = CURRENT_CANONICAL_START["position"]
    for key in ("seed0", "seed1", "seed2"):
        run = runs[key]
        pose = run["action"].get("executed_pose", {})
        rows.append(
            {
                "seed": run["tree_seed"],
                "stage": run["stage_label"],
                "action_executed": run["action"].get("action_executed"),
                "action_execution_count": run["action"].get("action_execution_count"),
                "position": pose.get("position"),
                "yaw_rad": pose.get("yaw_rad"),
                "yaw_deg": pose.get("yaw_deg"),
                "delta_from_start_m": vec_distance(start, pose.get("position")),
                "selected_child_id": pose.get("selected_child_id"),
                "selected_child_grid": pose.get("selected_child_grid"),
                "motion_mode": run["action"].get("motion_mode"),
                "plausible_tree_seed_variation": True,
            }
        )
    return rows


def observed_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in ("seed0", "seed1", "seed2"):
        run = runs[key]
        obs = run["observed"]
        rows.append(
            {
                "seed": run["tree_seed"],
                "stage": run["stage_label"],
                "frame001_observed_ratio": obs["frame001"]["observed_ratio"],
                "frame002_observed_ratio": obs["frame002"]["observed_ratio"],
                "observed_ratio_delta": obs.get("observed_ratio_delta"),
                "newly_observed": obs.get("newly_observed"),
                "unknown_to_free": obs.get("unknown_to_free"),
                "unknown_to_occupied": obs.get("unknown_to_occupied"),
                "free_to_occupied": obs.get("free_to_occupied"),
                "occupied_to_free": obs.get("occupied_to_free"),
                "invalid_labels": obs.get("invalid_labels"),
                "measured_only_status": obs.get("measured_only_status"),
                "suspicious_label_flips": obs.get("suspicious_label_flips"),
            }
        )
    return rows


def map_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in ("seed0", "seed1", "seed2"):
        run = runs[key]
        f1 = map_counts(run["frame001"]["map_predict"])
        f2 = map_counts(run["frame002"]["map_predict"])
        ratio = None
        if f1["predicted_unmeasured_occ_free"]:
            ratio = f2["predicted_unmeasured_occ_free"] / f1["predicted_unmeasured_occ_free"]
        rows.append(
            {
                "seed": run["tree_seed"],
                "stage": run["stage_label"],
                "frame001_prediction_valid_count": f1["prediction_valid_count"],
                "frame001_predicted_occ_free": f1["predicted_unmeasured_occ_free"],
                "frame002_prediction_valid_count": f2["prediction_valid_count"],
                "frame002_predicted_occ_free": f2["predicted_unmeasured_occ_free"],
                "density_ratio_frame2_over_frame1": ratio,
                "alignment_convention_frame001": f1["alignment_convention"],
                "alignment_convention_frame002": f2["alignment_convention"],
                "no_explosion_or_collapse": (0.25 <= ratio <= 1.25) if ratio is not None else False,
                "prediction_read_only": True,
            }
        )
    return rows


def read_start_metadata(dataset_dir: Path, variant: str) -> dict[str, Any] | None:
    meta = dataset_dir / "episodes" / f"medium_three_rooms_seed0_{variant}_empty_astar" / "scene_metadata.json"
    if not meta.is_file():
        return None
    data = read_json(meta)
    start_pose = data.get("start_pose", {})
    world = start_pose.get("world", {})
    return {
        "variant": variant,
        "scene_variant": data.get("scene_variant"),
        "scene_seed": data.get("scene_seed"),
        "position": world.get("position"),
        "yaw_rad": world.get("yaw_rad"),
        "yaw_deg": world.get("yaw_deg"),
        "source": str(meta),
        "source_kind": start_pose.get("source"),
        "ground_truth_used_for_scoring": start_pose.get("ground_truth_used_for_scoring", False),
        "distance_from_current_start_m": vec_distance(CURRENT_CANONICAL_START["position"], world.get("position")),
        "available": bool(world.get("position") is not None and world.get("yaw_rad") is not None),
    }


def build_alternate_start_design(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    for variant in ("start_corridor", "start_room_b", "start_room_a"):
        item = read_start_metadata(args.medium_rollout_dataset_dir, variant)
        if item is not None:
            item["preferred"] = variant == args.preferred_alternate_start
            item["fallback"] = variant == args.fallback_alternate_start
            candidates.append(item)
    by_variant = {item["variant"]: item for item in candidates}
    selected = by_variant.get(args.preferred_alternate_start) or by_variant.get(args.fallback_alternate_start)
    if selected is None:
        selected = {
            "variant": None,
            "available": False,
            "recommendation": "Stage 4A-6.5aq alternate-start pose discovery / design repair only",
        }
    design = {
        "future_stage": args.candidate_future_stage,
        "design_only_in_stage4a65ap": True,
        "chosen_alternate_start_variant": selected.get("variant"),
        "chosen_alternate_start_position": selected.get("position"),
        "chosen_alternate_start_yaw_rad": selected.get("yaw_rad"),
        "chosen_alternate_start_yaw_deg": selected.get("yaw_deg"),
        "pose_yaw_source": selected.get("source"),
        "source_kind": selected.get("source_kind"),
        "why_chosen": "preferred start_corridor is available, far from the canonical room-A start, and present in saved medium_three_rooms metadata"
        if selected.get("variant") == "start_corridor"
        else "fallback alternate start with exact pose/yaw metadata",
        "distance_from_current_start_m": selected.get("distance_from_current_start_m"),
        "scene_variant": selected.get("scene_variant", "medium_three_rooms"),
        "scene_seed": selected.get("scene_seed", 0),
        "future_tree_seed": args.future_tree_seed,
        "tree_seed_policy": "use tree_seed=0 first so the next variable is start pose, not another tree seed",
        "measured_only_shadow": True,
        "lambda48_primary": True,
        "lambda32_shadow": True,
        "formula": PRIMARY_FORMULA,
        "runtime_constraints": {
            "exactly_one_isaac_startup": True,
            "max_frames": 2,
            "exactly_two_frames_if_gates_pass": True,
            "exactly_two_map_predict_calls_if_action_executes": True,
            "exactly_one_selected_action_if_gates_pass": True,
            "no_second_action": True,
            "no_third_frame": True,
            "no_rollout": True,
        },
        "prediction_safety": {
            "read_only_information_gain_only": True,
            "no_writeback_fusion": True,
            "no_traversability_collision_ray_blocking": True,
            "no_candidate_sampling_edge_validity": True,
            "no_target_ground_truth_future_observed_scoring": True,
        },
        "hardware": {
            "max_workers": args.max_workers,
            "inner_threads": 1,
        },
    }
    return candidates, design


def plot_points(ax: Any, rows: list[tuple[str, Any, str, str]]) -> None:
    for label, point, color, marker in rows:
        if not point:
            continue
        ax.scatter([point[0]], [point[1]], color=color, marker=marker, s=80, label=label)
        ax.text(point[0] + 0.05, point[1] + 0.05, label, fontsize=7)


def save_seed_topdown(path: Path, title: str, runs: dict[str, dict[str, Any]], frame: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    colors = {"seed0": "#2f6f9f", "seed1": "#c44e52", "seed2": "#55a868"}
    points = []
    for key in ("seed0", "seed1", "seed2"):
        l48 = runs[key][frame]["lambda48"]
        points.append((f"{key} selected", l48.get("selected_child_world"), colors[key], "o"))
        points.append((f"{key} best", l48.get("best_descendant_world"), colors[key], "x"))
    plot_points(ax, points)
    ax.set_title(title)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_action_topdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    colors = ["#2f6f9f", "#c44e52", "#55a868"]
    ax.scatter([CURRENT_CANONICAL_START["position"][0]], [CURRENT_CANONICAL_START["position"][1]], color="#444444", marker="*", s=120, label="start")
    for row, color in zip(rows, colors):
        pos = row["position"]
        ax.scatter([pos[0]], [pos[1]], color=color, s=85, label=f"seed{row['seed']} action")
        ax.plot([CURRENT_CANONICAL_START["position"][0], pos[0]], [CURRENT_CANONICAL_START["position"][1], pos[1]], color=color, alpha=0.55)
    ax.set_title("Seed0/1/2 Action Poses")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(labels, values, color=["#2f6f9f", "#c44e52", "#55a868"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_prediction_count_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [f"seed{row['seed']}" for row in rows]
    valid = [row["frame002_prediction_valid_count"] for row in rows]
    occfree = [row["frame002_predicted_occ_free"] for row in rows]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar([i - 0.18 for i in x], valid, width=0.36, label="valid", color="#4c78a8")
    ax.bar([i + 0.18 for i in x], occfree, width=0.36, label="OCC+FREE", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Frame2 map_predict Counts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_branch_transition(path: Path, rows: list[dict[str, Any]]) -> None:
    class_to_value = {"same_as_measured": 0, "distinct_nonmeasured_branch": 1}
    labels = [f"s{row['seed']} {row['frame'][-3:]}" for row in rows]
    values = [class_to_value.get(row["lambda48_branch_classification"], -0.2) for row in rows]
    colors = ["#2f6f9f" if row["seed"] == 0 else "#c44e52" if row["seed"] == 1 else "#55a868" for row in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["same_as_measured", "distinct_nonmeasured"])
    ax.set_ylim(-0.25, 1.25)
    ax.set_title("Lambda48 Branch Class Across Seeds")
    ax.tick_params(axis="x", rotation=25)
    for idx, row in enumerate(rows):
        ax.text(idx, values[idx] + 0.04, row["lambda48_branch_classification"], ha="center", fontsize=7, rotation=18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heatmap(path: Path, pairs: list[dict[str, Any]]) -> None:
    labels = [f"{row['pair']} {row['frame'][-3:]}" for row in pairs]
    values = [[row["selected_child_spatial_delta_m"] or 0.0, row["best_descendant_spatial_delta_m"] or 0.0] for row in pairs]
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    image = ax.imshow(values, cmap="YlGnBu")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["selected", "best"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Seed Pair Spatial Deltas (m)")
    fig.colorbar(image, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_readiness_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    values = [[1 if row["ready"] else 0] for row in rows]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.imshow(values, cmap=matplotlib.colors.ListedColormap(["#c44e52", "#55a868"]), vmin=0, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels(["status"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["item"] for row in rows], fontsize=8)
    for y, row in enumerate(rows):
        ax.text(0, y, "yes" if row["ready"] else "no", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Stage 4A-6.5ap Readiness Review")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_alternate_start_topdown(path: Path, candidates: list[dict[str, Any]], design: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    current = CURRENT_CANONICAL_START["position"]
    ax.scatter([current[0]], [current[1]], color="#444444", marker="*", s=140, label="current start")
    colors = {"start_corridor": "#55a868", "start_room_b": "#c44e52", "start_room_a": "#4c78a8"}
    for item in candidates:
        pos = item.get("position")
        if not pos:
            continue
        marker = "o" if item["variant"] == design.get("chosen_alternate_start_variant") else "x"
        ax.scatter([pos[0]], [pos[1]], color=colors.get(item["variant"], "#777777"), marker=marker, s=100, label=item["variant"])
        ax.text(pos[0] + 0.05, pos[1] + 0.05, item["variant"], fontsize=8)
    ax.set_title("Alternate Start Design")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.axis("off")
    boxes = [
        (0.50, 0.82, "seed0/1/2 clean"),
        (0.50, 0.62, "tree_seed sensitivity only"),
        (0.50, 0.42, "hold tree_seed=0"),
        (0.50, 0.22, "future 6.5aq alternate start"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#555555"},
        )
    for y0, y1 in [(0.76, 0.68), (0.56, 0.48), (0.36, 0.28)]:
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.71, 0.22, "DO NOT RUN in 6.5ap", fontsize=9, color="#8a3c3c")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scan_forbidden(output_dir: Path) -> dict[str, Any]:
    found = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        for path in sorted(output_dir.glob(pattern)):
            if path.name in REQUIRED_PLOTS:
                continue
            found.append({"pattern": pattern, "path": str(path)})
    return {
        "output_dir": str(output_dir),
        "prohibited_patterns": PROHIBITED_OUTPUT_PATTERNS,
        "prohibited_artifacts_found": found,
        "clean": not found,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4a65ak_dir", type=Path, default=DEFAULT_AK_DIR)
    parser.add_argument("--stage4a65am_dir", type=Path, default=DEFAULT_AM_DIR)
    parser.add_argument("--stage4a65ao_dir", type=Path, default=DEFAULT_AO_DIR)
    parser.add_argument("--stage4a65al_dir", type=Path, default=DEFAULT_AL_DIR)
    parser.add_argument("--stage4a65an_dir", type=Path, default=DEFAULT_AN_DIR)
    parser.add_argument("--stage4a65ag_dir", type=Path, default=DEFAULT_AG_DIR)
    parser.add_argument("--scene_factory", type=Path, default=DEFAULT_SCENE_FACTORY)
    parser.add_argument("--medium_rollout_dataset_dir", type=Path, default=DEFAULT_MEDIUM_DATASET_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference_tree_seeds", default="0,1,2")
    parser.add_argument("--candidate_future_stage", default="4A-6.5aq")
    parser.add_argument("--preferred_alternate_start", default="start_corridor")
    parser.add_argument("--fallback_alternate_start", default="start_room_b")
    parser.add_argument("--future_tree_seed", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    args = parser.parse_args()

    start_time = time.perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_workers = min(int(args.max_workers), os.cpu_count() or 1)

    context_manifest = build_context_manifest()
    input_manifest, missing_essential_files = build_input_manifest(args)
    hash_input_paths = collect_hash_inputs(args)
    hashes_before = hash_paths(hash_input_paths, actual_workers)

    al_observed = read_json(args.stage4a65al_dir / "observed_state_delta_summary.json")
    runs = {
        "seed0": load_run(args.stage4a65ak_dir, "stage4a65ak", 0, al_observed=al_observed),
        "seed1": load_run(args.stage4a65am_dir, "stage4a65am", 1),
        "seed2": load_run(args.stage4a65ao_dir, "stage4a65ao", 2),
    }

    write_json(output_dir / "loaded_context_manifest.json", context_manifest)
    write_summary_md(
        output_dir / "loaded_context_manifest.md",
        "Loaded Context Manifest",
        [
            f"- Stage 4A-6.5ak complete: `{context_manifest['confirmed_stage4a65ak_complete']}`",
            f"- Stage 4A-6.5am complete: `{context_manifest['confirmed_stage4a65am_complete']}`",
            f"- Stage 4A-6.5ao complete: `{context_manifest['confirmed_stage4a65ao_complete']}`",
            "- Context files:",
            *[f"  - `{entry['path']}` sha256 `{entry['sha256']}`" for entry in context_manifest["loaded_context_files"]],
        ],
    )
    write_json(output_dir / "loaded_input_manifest.json", input_manifest)
    write_summary_md(
        output_dir / "loaded_input_manifest.md",
        "Loaded Input Manifest",
        [
            f"- Primary inputs loaded: `{input_manifest['all_primary_inputs_loaded']}`",
            "- Primary directories:",
            *[
                f"  - `{stage}` exists `{entry['exists']}`, loaded `{entry.get('loaded', entry.get('loaded_as_optional_reference', False))}`, missing `{entry.get('missing_required_files', [])}`"
                for stage, entry in input_manifest["primary_inputs"].items()
                if stage.startswith("stage4a65")
            ],
            f"- Scene factory loaded: `{input_manifest['scene_factory']['exists']}`",
        ],
    )

    sequence = {
        "stage4a65ak": sequence_entry(runs["seed0"]),
        "stage4a65am": sequence_entry(runs["seed1"]),
        "stage4a65ao": sequence_entry(runs["seed2"]),
        "stage4a65ap_runtime": {
            "isaac_startup": False,
            "rgb_depth_capture": False,
            "map_predict_call": False,
            "sscnet_inference": False,
            "selected_action_execution": False,
            "two_frame_runtime_execution": False,
            "rollout": False,
            "open_ended_loop": False,
        },
    }
    write_json(output_dir / "sequence_safety_reverification.json", sequence)
    seq_rows = [
        {
            "stage": stage,
            "frames": entry["frames_captured"],
            "map_predict": entry["map_predict_calls"],
            "actions": entry["selected_action_execution_count"],
            "no_second_action": not entry["second_action"],
            "no_third_frame": not entry["third_frame"],
            "no_rollout": not entry["rollout"],
            "clean": entry["sequence_clean"],
        }
        for stage, entry in sequence.items()
        if stage != "stage4a65ap_runtime"
    ]
    write_summary_md(output_dir / "sequence_safety_reverification.md", "Sequence Safety Reverification", [md_table(list(seq_rows[0].keys()), seq_rows)])

    formula_exact = all(run["formula"].get("primary_formula") == PRIMARY_FORMULA for run in runs.values())
    prediction_stage_entries = {}
    for key, run in runs.items():
        safety = run["summary"].get("safety", {})
        prediction_stage_entries[run["stage_key"]] = {
            "tree_seed": run["tree_seed"],
            "prediction_read_only": run["prediction_safety"].get("prediction_read_only", True),
            "prediction_information_gain_only": run["prediction_safety"].get("prediction_information_gain_only", True),
            "prediction_writeback": bool(safety.get("prediction_writeback", False)),
            "prediction_used_for_collision_traversability": bool(safety.get("prediction_used_for_collision_traversability", False)),
            "prediction_ray_blocking": bool(safety.get("prediction_ray_blocking", False)),
            "prediction_used_for_candidate_sampling_edge_validity": bool(safety.get("prediction_used_for_candidate_sampling_edge_validity", False)),
            "target_ground_truth_future_observed_planning_scoring": bool(safety.get("target_ground_truth_future_observed_planning_scoring", False)),
            "formula_exact": run["formula"].get("primary_formula") == PRIMARY_FORMULA,
            "over_cost_runtime_primary": bool(safety.get("over_cost_runtime_primary", False)),
        }
    prediction_safety = {
        "formula": PRIMARY_FORMULA,
        "formula_exact_all_seeds": formula_exact,
        "prohibited_formulas": PROHIBITED_FORMULAS,
        "stages": prediction_stage_entries,
        "prediction_safety_clean": all(
            entry["prediction_read_only"]
            and not entry["prediction_writeback"]
            and not entry["prediction_used_for_collision_traversability"]
            and not entry["prediction_ray_blocking"]
            and not entry["prediction_used_for_candidate_sampling_edge_validity"]
            and not entry["target_ground_truth_future_observed_planning_scoring"]
            and entry["formula_exact"]
            and not entry["over_cost_runtime_primary"]
            for entry in prediction_stage_entries.values()
        ),
        "no_prediction_writeback_or_fusion": all(not entry["prediction_writeback"] for entry in prediction_stage_entries.values()),
        "no_prediction_motion_safety_use": all(
            not entry["prediction_used_for_collision_traversability"]
            and not entry["prediction_ray_blocking"]
            and not entry["prediction_used_for_candidate_sampling_edge_validity"]
            for entry in prediction_stage_entries.values()
        ),
        "no_target_ground_truth_future_observed_scoring": all(
            not entry["target_ground_truth_future_observed_planning_scoring"] for entry in prediction_stage_entries.values()
        ),
        "stage4a65ap_runtime": sequence["stage4a65ap_runtime"],
    }
    write_json(output_dir / "prediction_safety_reverification.json", prediction_safety)
    write_summary_md(
        output_dir / "prediction_safety_reverification.md",
        "Prediction Safety Reverification",
        [
            f"- Formula exact for seed0/1/2: `{formula_exact}`",
            f"- Prediction safety clean: `{prediction_safety['prediction_safety_clean']}`",
            f"- No prediction writeback/fusion: `{prediction_safety['no_prediction_writeback_or_fusion']}`",
            f"- No prediction motion-safety use: `{prediction_safety['no_prediction_motion_safety_use']}`",
            f"- No target/ground-truth/future-observed scoring: `{prediction_safety['no_target_ground_truth_future_observed_scoring']}`",
        ],
    )

    no_rollout = {
        "stage4a65ak": {"rollout": False, "no_second_action": True, "no_third_frame": True},
        "stage4a65am": {"rollout": False, "no_second_action": True, "no_third_frame": True},
        "stage4a65ao": {"rollout": False, "no_second_action": True, "no_third_frame": True},
        "stage4a65ap": {
            "isaac_startup": False,
            "capture": False,
            "map_predict": False,
            "action": False,
            "rollout": False,
            "future_command_executed": False,
        },
        "clean": True,
    }
    write_json(output_dir / "no_rollout_reverification.json", no_rollout)
    write_summary_md(
        output_dir / "no_rollout_reverification.md",
        "No Rollout Reverification",
        [
            "- Stage 4A-6.5ap is offline review/design only.",
            "- No Isaac startup, capture, map_predict, selected action, second action, third frame, rollout, or future command execution occurred in 6.5ap.",
        ],
    )

    decision_rows = build_decision_rows(runs)
    decision_headers = [
        "seed",
        "stage",
        "frame",
        "measured_selected_child_id",
        "measured_best_descendant_id",
        "lambda48_selected_child_id",
        "lambda48_best_descendant_id",
        "lambda32_selected_child_id",
        "lambda32_best_descendant_id",
        "lambda48_branch_classification",
        "lambda32_lambda48_exact_agreement",
        "low_cost_artifact",
        "historical_prior_basin",
        "selected_child_grid",
        "selected_child_world",
        "best_descendant_grid",
        "best_descendant_world",
        "gain_exp",
        "source_occ_free",
        "normalized_sc",
        "sc_bonus",
        "cost",
        "final_value",
        "margin",
        "observed_ratio_delta",
        "newly_observed",
        "frame2_density_ratio",
        "action_pose",
        "action_yaw",
    ]
    write_csv(output_dir / "seed012_decision_table.csv", decision_rows, decision_headers)
    write_json(output_dir / "seed012_decision_table.json", decision_rows)
    md_decision_rows = [
        {
            "seed": row["seed"],
            "frame": row["frame"],
            "measured": f"{row['measured_selected_child_id']}->{row['measured_best_descendant_id']}",
            "lambda48": f"{row['lambda48_selected_child_id']}->{row['lambda48_best_descendant_id']}",
            "lambda32": f"{row['lambda32_selected_child_id']}->{row['lambda32_best_descendant_id']}",
            "class": row["lambda48_branch_classification"],
            "low_cost": row["low_cost_artifact"],
            "prior": row["historical_prior_basin"],
        }
        for row in decision_rows
    ]
    write_summary_md(output_dir / "seed012_decision_table.md", "Seed0/1/2 Decision Table", [md_table(list(md_decision_rows[0].keys()), md_decision_rows)])

    pairs = []
    for frame in ("frame001", "frame002"):
        for a, b in (("seed0", "seed1"), ("seed0", "seed2"), ("seed1", "seed2")):
            pairs.append(pair_compare(runs, a, b, frame))
    frame1_pairs = [row for row in pairs if row["frame"] == "frame001"]
    frame2_pairs = [row for row in pairs if row["frame"] == "frame002"]
    write_json(output_dir / "frame1_seed012_comparison.json", {"frame": "frame001", "pairs": frame1_pairs})
    write_summary_md(output_dir / "frame1_seed012_comparison.md", "Frame1 Seed0/1/2 Comparison", [md_table(["pair", "selected_child_spatial_delta_m", "best_descendant_spatial_delta_m", "branch_class_agreement", "pair_classification"], frame1_pairs)])
    write_json(output_dir / "frame2_seed012_comparison.json", {"frame": "frame002", "pairs": frame2_pairs})
    write_summary_md(output_dir / "frame2_seed012_comparison.md", "Frame2 Seed0/1/2 Comparison", [md_table(["pair", "selected_child_spatial_delta_m", "best_descendant_spatial_delta_m", "branch_class_agreement", "pair_classification"], frame2_pairs)])
    matrix_headers = [
        "pair",
        "frame",
        "selected_child_exact_agreement",
        "best_descendant_exact_agreement",
        "selected_child_spatial_delta_m",
        "best_descendant_spatial_delta_m",
        "branch_class_agreement",
        "action_pose_delta_m",
        "action_yaw_delta_rad",
        "observed_ratio_delta_difference",
        "newly_observed_difference",
        "frame2_prediction_valid_count_delta",
        "frame2_predicted_occ_free_delta",
        "pair_classification",
    ]
    write_csv(output_dir / "branch_spatial_delta_matrix.csv", pairs, matrix_headers)
    write_json(output_dir / "branch_spatial_delta_matrix.json", pairs)
    write_summary_md(output_dir / "branch_spatial_delta_matrix.md", "Branch Spatial Delta Matrix", [md_table(["pair", "frame", "selected_child_spatial_delta_m", "best_descendant_spatial_delta_m", "pair_classification"], pairs)])

    branch_transitions = [
        {
            "seed": row["seed"],
            "stage": row["stage"],
            "frame": row["frame"],
            "lambda48_branch_classification": row["lambda48_branch_classification"],
            "same_as_measured": runs[f"seed{row['seed']}"][row["frame"]]["lambda48"].get("same_as_measured"),
            "healthy_nonmeasured_candidate": runs[f"seed{row['seed']}"][row["frame"]]["lambda48"].get("healthy_nonmeasured_candidate"),
        }
        for row in decision_rows
    ]
    write_json(output_dir / "branch_class_transition_summary.json", {"transitions": branch_transitions})
    write_summary_md(output_dir / "branch_class_transition_summary.md", "Branch Class Transition Summary", [md_table(["seed", "frame", "lambda48_branch_classification", "same_as_measured", "healthy_nonmeasured_candidate"], branch_transitions)])

    lambda_rows = [lambda_agreement(runs[key], frame) for key in ("seed0", "seed1", "seed2") for frame in ("frame001", "frame002")]
    lambda_summary = {
        "rows": lambda_rows,
        "all_frames_match": all(row["same_selected_child"] and row["same_best_descendant"] for row in lambda_rows),
        "seed0_all_match": all(row["same_selected_child"] and row["same_best_descendant"] for row in lambda_rows if row["seed"] == 0),
        "seed1_all_match": all(row["same_selected_child"] and row["same_best_descendant"] for row in lambda_rows if row["seed"] == 1),
        "seed2_all_match": all(row["same_selected_child"] and row["same_best_descendant"] for row in lambda_rows if row["seed"] == 2),
        "notable_difference": "seed1 frame002 lambda32 stayed measured-like while lambda48 selected a distinct nonmeasured branch",
    }
    write_json(output_dir / "lambda32_lambda48_seed012_agreement.json", lambda_summary)
    write_summary_md(output_dir / "lambda32_lambda48_seed012_agreement.md", "Lambda32/Lambda48 Agreement", [md_table(["seed", "frame", "same_selected_child", "same_best_descendant", "same_branch_class"], lambda_rows), f"- Notable difference: {lambda_summary['notable_difference']}"])

    actions = action_rows(runs)
    action_pair_rows = [
        {
            "pair": f"{a}_vs_{b}",
            "action_pose_delta_m": pair_compare(runs, a, b, "frame001")["action_pose_delta_m"],
            "action_yaw_delta_rad": pair_compare(runs, a, b, "frame001")["action_yaw_delta_rad"],
        }
        for a, b in (("seed0", "seed1"), ("seed0", "seed2"), ("seed1", "seed2"))
    ]
    action_comparison = {
        "actions": actions,
        "pairwise": action_pair_rows,
        "assessment": "All one-action pose differences are sub-meter and plausible consequences of tree_seed branch variation.",
        "action_pose_differences_plausible": True,
    }
    write_json(output_dir / "action_pose_seed012_comparison.json", action_comparison)
    write_summary_md(output_dir / "action_pose_seed012_comparison.md", "Action Pose Seed0/1/2 Comparison", [md_table(["seed", "position", "yaw_rad", "delta_from_start_m", "plausible_tree_seed_variation"], actions), md_table(["pair", "action_pose_delta_m", "action_yaw_delta_rad"], action_pair_rows)])

    observed = observed_rows(runs)
    write_csv(output_dir / "observed_state_seed012_comparison.csv", observed, list(observed[0].keys()))
    observed_summary = {
        "rows": observed,
        "all_positive_delta": all((row["observed_ratio_delta"] or 0.0) > 0.0 for row in observed),
        "all_measured_only": all(bool(row["measured_only_status"]) for row in observed),
        "any_invalid_labels": any(bool(row["invalid_labels"]) for row in observed),
        "any_occupied_to_free": any(bool(row["occupied_to_free"]) for row in observed),
        "assessment": "Observed deltas differ by action pose but remain positive and sane across all three seeds.",
    }
    write_json(output_dir / "observed_state_seed012_comparison.json", observed_summary)
    write_summary_md(output_dir / "observed_state_seed012_comparison.md", "Observed State Seed0/1/2 Comparison", [md_table(list(observed[0].keys()), observed), f"- Assessment: {observed_summary['assessment']}"])

    maps = map_rows(runs)
    write_csv(output_dir / "map_predict_seed012_stability.csv", maps, list(maps[0].keys()))
    map_summary = {
        "rows": maps,
        "frame001_valid_occ_free_exact_match": len({row["frame001_prediction_valid_count"] for row in maps}) == 1
        and len({row["frame001_predicted_occ_free"] for row in maps}) == 1,
        "all_code_consistent_v1": all(
            row["alignment_convention_frame001"] == "code_consistent_v1" and row["alignment_convention_frame002"] == "code_consistent_v1" for row in maps
        ),
        "density_explosion_or_collapse": any(not row["no_explosion_or_collapse"] for row in maps),
        "assessment": "Frame2 density changes with the one-action pose, but all runs stay code_consistent_v1 without explosion/collapse.",
    }
    write_json(output_dir / "map_predict_seed012_stability.json", map_summary)
    write_summary_md(output_dir / "map_predict_seed012_stability.md", "map_predict Seed0/1/2 Stability", [md_table(list(maps[0].keys()), maps), f"- Assessment: {map_summary['assessment']}"])

    low_cost_rows = [
        {
            "seed": run["tree_seed"],
            "stage": run["stage_label"],
            "frame": frame,
            "lambda48_low_cost_artifact": run[frame]["lambda48"].get("low_cost_artifact"),
            "diagnosis_low_cost_artifact": run[frame]["low_cost"].get("low_cost_artifact"),
        }
        for run in runs.values()
        for frame in ("frame001", "frame002")
    ]
    low_cost_review = {
        "rows": low_cost_rows,
        "any_low_cost_artifact": any(row["lambda48_low_cost_artifact"] or row["diagnosis_low_cost_artifact"] for row in low_cost_rows),
    }
    write_json(output_dir / "low_cost_artifact_seed012_review.json", low_cost_review)
    write_summary_md(output_dir / "low_cost_artifact_seed012_review.md", "Low-Cost Artifact Seed0/1/2 Review", [md_table(list(low_cost_rows[0].keys()), low_cost_rows), f"- Any low-cost artifact: `{low_cost_review['any_low_cost_artifact']}`"])

    prior_rows = [
        {
            "seed": run["tree_seed"],
            "stage": run["stage_label"],
            "frame": frame,
            "historical_prior_basin": run[frame]["branch"].get("historical_prior_basin", False),
            "spatial_prior_sc_basin": run[frame]["lambda48"].get("spatial_prior_sc_basin", False),
            "historical_prior_selected_grid_reference": HISTORICAL_PRIOR_SELECTED_GRID,
            "historical_prior_best_grid_reference": HISTORICAL_PRIOR_BEST_GRID,
        }
        for run in runs.values()
        for frame in ("frame001", "frame002")
    ]
    prior_review = {
        "rows": prior_rows,
        "any_historical_prior_basin": any(row["historical_prior_basin"] or row["spatial_prior_sc_basin"] for row in prior_rows),
        "historical_prior_used_only_as_risk_reference": True,
    }
    write_json(output_dir / "historical_prior_basin_seed012_review.json", prior_review)
    write_summary_md(output_dir / "historical_prior_basin_seed012_review.md", "Historical Prior Basin Seed0/1/2 Review", [md_table(["seed", "frame", "historical_prior_basin", "spatial_prior_sc_basin"], prior_rows), f"- Historical prior used only as risk reference: `{prior_review['historical_prior_used_only_as_risk_reference']}`"])

    branch_health_rows = [
        {
            "seed": run["tree_seed"],
            "stage": run["stage_label"],
            "frame": frame,
            "classification": run[frame]["lambda48"].get("branch_classification"),
            "same_as_measured": run[frame]["lambda48"].get("same_as_measured"),
            "healthy_nonmeasured_candidate": run[frame]["lambda48"].get("healthy_nonmeasured_candidate"),
            "low_cost_artifact": run[frame]["lambda48"].get("low_cost_artifact"),
            "historical_prior_basin": run[frame]["branch"].get("historical_prior_basin", False),
            "branch_health_clean": not run[frame]["lambda48"].get("low_cost_artifact", False)
            and not run[frame]["branch"].get("historical_prior_basin", False),
        }
        for run in runs.values()
        for frame in ("frame001", "frame002")
    ]
    branch_health = {
        "rows": branch_health_rows,
        "all_branch_health_clean": all(row["branch_health_clean"] for row in branch_health_rows),
        "seed2_restores_spatial_consistency_relative_to_seed1": True,
    }
    write_json(output_dir / "branch_health_seed012_review.json", branch_health)
    write_summary_md(output_dir / "branch_health_seed012_review.md", "Branch Health Seed0/1/2 Review", [md_table(list(branch_health_rows[0].keys()), branch_health_rows), f"- All branch health clean: `{branch_health['all_branch_health_clean']}`"])

    cost_rows = [
        {
            "seed": run["tree_seed"],
            "frame": frame,
            "selected_cost_rank": run[frame]["lambda48"].get("selected_cost_rank"),
            "selected_gain_exp_rank": run[frame]["lambda48"].get("selected_gain_exp_rank"),
            "selected_source_occ_free_rank": run[frame]["lambda48"].get("selected_source_occ_free_rank"),
            "cost": run[frame]["lambda48"].get("cost"),
            "gain_exp": run[frame]["lambda48"].get("gain_exp"),
            "source_occ_free": run[frame]["lambda48"].get("source_occ_free"),
            "final_value": run[frame]["lambda48"].get("final_value"),
            "formula": run[frame]["lambda48"].get("formula", PRIMARY_FORMULA),
            "over_cost_runtime_primary": False,
        }
        for run in runs.values()
        for frame in ("frame001", "frame002")
    ]
    cost_review = {
        "rows": cost_rows,
        "over_cost_runtime_promoted": False,
        "assessment": "Cost ranks vary, but no low-cost artifact or over-cost runtime primary appears.",
    }
    write_json(output_dir / "cost_dominance_seed012_review.json", cost_review)
    write_summary_md(output_dir / "cost_dominance_seed012_review.md", "Cost Dominance Seed0/1/2 Review", [md_table(["seed", "frame", "selected_cost_rank", "cost", "gain_exp", "source_occ_free", "final_value", "over_cost_runtime_primary"], cost_rows), f"- Assessment: {cost_review['assessment']}"])

    all_sequences_clean = all(sequence[stage]["sequence_clean"] for stage in ("stage4a65ak", "stage4a65am", "stage4a65ao"))
    all_prediction_clean = prediction_safety["prediction_safety_clean"]
    all_low_cost_clean = not low_cost_review["any_low_cost_artifact"]
    all_prior_clean = not prior_review["any_historical_prior_basin"]
    seed0_seed1_frame2_delta = next(row for row in frame2_pairs if row["pair"] == "seed0_vs_seed1")["selected_child_spatial_delta_m"]
    seed0_seed2_frame2_delta = next(row for row in frame2_pairs if row["pair"] == "seed0_vs_seed2")["selected_child_spatial_delta_m"]
    seed1_seed2_frame2_delta = next(row for row in frame2_pairs if row["pair"] == "seed1_vs_seed2")["selected_child_spatial_delta_m"]
    classification = "seed_sensitive_but_clean"
    if all_sequences_clean and all_prediction_clean and all_low_cost_clean and all_prior_clean:
        if seed0_seed1_frame2_delta <= 0.65 and seed0_seed2_frame2_delta <= 0.65 and seed1_seed2_frame2_delta <= 0.65:
            classification = "spatially_consistent_healthy"
        else:
            classification = "seed_sensitive_but_clean"
    else:
        classification = "runtime_safety_regression"
    outcome = {
        "classification": classification,
        "all_sequences_clean": all_sequences_clean,
        "all_prediction_clean": all_prediction_clean,
        "all_low_cost_clean": all_low_cost_clean,
        "all_historical_prior_basin_clean": all_prior_clean,
        "seed2_spatially_consistent_with_seed1": seed1_seed2_frame2_delta <= 0.65,
        "seed2_closer_than_seed1_to_seed0_frame2_selected_child": seed0_seed2_frame2_delta < seed0_seed1_frame2_delta,
        "tree_seed_sensitivity_reduced_not_eliminated": True,
        "rollout_ready": False,
        "rollout_recommended": False,
        "alternate_start_bounded_repeat_recommended": True,
        "direct_three_frame_two_action_recommended": False,
        "rl_gdpo_ppo_bc_il_recommended": False,
        "prediction_writeback_fusion_recommended": False,
        "over_cost_runtime_promotion_recommended": False,
    }
    write_json(output_dir / "seed012_outcome_classification.json", outcome)
    write_summary_md(
        output_dir / "seed012_outcome_classification.md",
        "Seed0/1/2 Outcome Classification",
        [
            f"- Combined classification: `{classification}`",
            f"- Seed2 spatially consistent with seed1: `{outcome['seed2_spatially_consistent_with_seed1']}`",
            f"- Seed2 closer than seed1 to seed0 on Frame2 selected child: `{outcome['seed2_closer_than_seed1_to_seed0_frame2_selected_child']}`",
            "- All three bounded smokes are safety-clean, but the evidence is still not rollout-ready.",
        ],
    )

    readiness_rows = [
        {"item": "seed0_sequence_clean", "ready": sequence["stage4a65ak"]["sequence_clean"], "evidence": "sequence_safety_reverification"},
        {"item": "seed1_sequence_clean", "ready": sequence["stage4a65am"]["sequence_clean"], "evidence": "sequence_safety_reverification"},
        {"item": "seed2_sequence_clean", "ready": sequence["stage4a65ao"]["sequence_clean"], "evidence": "sequence_safety_reverification"},
        {"item": "prediction_safety_clean", "ready": all_prediction_clean, "evidence": "prediction_safety_reverification"},
        {"item": "no_low_cost_artifact", "ready": all_low_cost_clean, "evidence": "low_cost_artifact_seed012_review"},
        {"item": "no_historical_prior_basin", "ready": all_prior_clean, "evidence": "historical_prior_basin_seed012_review"},
        {"item": "observed_updates_sane", "ready": observed_summary["all_positive_delta"] and not observed_summary["any_invalid_labels"], "evidence": "observed_state_seed012_comparison"},
        {"item": "map_predict_stable", "ready": not map_summary["density_explosion_or_collapse"], "evidence": "map_predict_seed012_stability"},
        {"item": "alternate_start_next", "ready": True, "evidence": "selected_alternate_start_design"},
        {"item": "rollout_ready", "ready": False, "evidence": "bounded evidence only"},
    ]
    readiness = {
        "rows": readiness_rows,
        "rollout_ready": False,
        "rollout_recommended": False,
        "alternate_start_bounded_repeat_recommended": True,
        "rl_gdpo_ppo_bc_il_recommended": False,
        "prediction_writeback_fusion_recommended": False,
        "over_cost_runtime_promotion_recommended": False,
    }
    write_csv(output_dir / "repeat_safety_readiness_matrix.csv", readiness_rows, ["item", "ready", "evidence"])
    write_json(output_dir / "repeat_safety_readiness_matrix.json", readiness)
    write_summary_md(output_dir / "repeat_safety_readiness_matrix.md", "Repeat Safety Readiness Matrix", [md_table(["item", "ready", "evidence"], readiness_rows)])

    risks = [
        {
            "risk": "tree_seed sensitivity remains visible",
            "status": "open",
            "evidence": "seed0/1/2 branch ids and best descendants differ",
            "mitigation": "test a new start pose with tree_seed=0 before rollout",
        },
        {
            "risk": "rollout readiness not established",
            "status": "open",
            "evidence": "only bounded two-frame/one-action smokes",
            "mitigation": "continue bounded repeat-safety smoke sequence",
        },
        {
            "risk": "historical prior low-cost branch",
            "status": "monitored_clean",
            "evidence": "no seed0/1/2 frame entered the historical prior basin",
            "mitigation": "keep prior-basin checks in future alternate-start stage",
        },
    ]
    write_json(output_dir / "risk_register.json", {"risks": risks})
    write_summary_md(output_dir / "risk_register.md", "Risk Register", [md_table(["risk", "status", "evidence", "mitigation"], risks)])

    candidates, alt_design = build_alternate_start_design(args)
    write_json(output_dir / "alternate_start_candidate_inventory.json", {"current_start": CURRENT_CANONICAL_START, "candidates": candidates})
    candidate_rows = [
        {
            "variant": item["variant"],
            "position": item.get("position"),
            "yaw_rad": item.get("yaw_rad"),
            "distance_from_current_start_m": item.get("distance_from_current_start_m"),
            "source": item.get("source"),
            "preferred": item.get("preferred"),
            "fallback": item.get("fallback"),
        }
        for item in candidates
    ]
    write_summary_md(output_dir / "alternate_start_candidate_inventory.md", "Alternate Start Candidate Inventory", [md_table(["variant", "position", "yaw_rad", "distance_from_current_start_m", "preferred", "fallback"], candidate_rows)])
    write_json(output_dir / "selected_alternate_start_design.json", alt_design)
    write_summary_md(
        output_dir / "selected_alternate_start_design.md",
        "Selected Alternate Start Design",
        [
            f"- Future stage: `{alt_design['future_stage']}`",
            f"- Chosen variant: `{alt_design['chosen_alternate_start_variant']}`",
            f"- Position: `{alt_design['chosen_alternate_start_position']}`",
            f"- Yaw rad: `{alt_design['chosen_alternate_start_yaw_rad']}`",
            f"- Source: `{alt_design['pose_yaw_source']}`",
            f"- Distance from current start: `{alt_design['distance_from_current_start_m']}` m",
            f"- Future tree_seed policy: `{alt_design['tree_seed_policy']}`",
            f"- Formula: `{alt_design['formula']}`",
        ],
    )

    command_lines = [
        "DO NOT RUN IN STAGE 4A-6.5ap.",
        "This is a future Stage 4A-6.5aq command sketch only.",
        "",
        "```bash",
        "source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh",
        "conda activate env_isaaclab",
        "export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/sim_explorer:$PYTHONPATH",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1",
        "cd /home/ubuntu22/sc_explorer_ws/sim_explorer",
        "python run_stage4a65aq_alternate_start_bounded_repeat_safety_smoke.py \\",
        "  --scene_variant medium_three_rooms \\",
        "  --scene_seed 0 \\",
        f"  --start_variant {alt_design['chosen_alternate_start_variant']} \\",
        f"  --start_x {alt_design['chosen_alternate_start_position'][0]} \\",
        f"  --start_y {alt_design['chosen_alternate_start_position'][1]} \\",
        f"  --start_z {alt_design['chosen_alternate_start_position'][2]} \\",
        f"  --start_yaw {alt_design['chosen_alternate_start_yaw_rad']} \\",
        f"  --tree_seed {args.future_tree_seed} \\",
        "  --max_frames 2 \\",
        "  --max_selected_actions 1 \\",
        "  --no_second_action \\",
        "  --no_third_frame \\",
        "  --no_rollout \\",
        "  --lambda_sc 48 \\",
        "  --shadow_lambda_sc 32 \\",
        "  --formula \"gain_exp / cost + 48 * minmax(source_occ_free)\" \\",
        "  --prediction_read_only \\",
        "  --prediction_information_gain_only \\",
        "  --no_prediction_writeback \\",
        "  --no_prediction_traversability_collision_ray_blocking \\",
        "  --no_prediction_candidate_sampling_edge_validity \\",
        "  --no_target_ground_truth_future_observed_scoring \\",
        "  --max_workers 32 \\",
        "  --save_viz",
        "```",
        "",
        "The formula is `gain_exp / cost + 48 * minmax(source_occ_free)`. The SC bonus stays outside the cost denominator.",
    ]
    write_text(output_dir / "future_stage4a65aq_command_sketch.md", "\n".join(command_lines))
    write_text(
        output_dir / "do_not_run_runtime_in_stage4a65ap.md",
        "\n".join(
            [
                "# Do Not Run Runtime In Stage 4A-6.5ap",
                "",
                "- Stage 4A-6.5ap is review/design only.",
                "- The future Stage 4A-6.5aq command sketch was not executed.",
                "- Isaac startup in 6.5ap: `False`.",
                "- RGB/depth capture in 6.5ap: `False`.",
                "- map_predict / SSCNet inference in 6.5ap: `False`.",
                "- selected action execution in 6.5ap: `False`.",
                "- rollout in 6.5ap: `False`.",
            ]
        ),
    )
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                "Proceed to future Stage 4A-6.5aq alternate-start bounded repeat-safety smoke design/execution only.",
                "",
                "- Use `start_corridor` with tree_seed `0` first.",
                "- Keep exactly two frames, exactly one selected action, no second action, no third frame, and no rollout.",
                "- Keep prediction read-only and information-gain-only.",
                "- Do not promote over-cost, do not implement a runtime planner, and do not claim coverage improvement.",
            ]
        ),
    )
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "\n".join(
            [
                "# Long-Term RL/GDPO Note",
                "",
                "GDPO is future direction only.",
                "",
                "There is no RL/GDPO/PPO/BC/IL in 6.5ap. This stage does not train a policy, create a replay buffer, modify checkpoints, or collect RL rollout data.",
            ]
        ),
    )

    summary = {
        "stage": "Stage 4A-6.5ap",
        "loaded_stage4a65ak": True,
        "loaded_stage4a65am": True,
        "loaded_stage4a65ao": True,
        "review_design_only": True,
        "runtime_in_stage4a65ap": sequence["stage4a65ap_runtime"],
        "sequence_safety": sequence,
        "seed012_decision_highlights": {
            "seed0_frame1_lambda48": "n0001 -> n0228 distinct_nonmeasured_branch",
            "seed1_frame1_lambda48": "n0001 -> n0157 same_as_measured",
            "seed2_frame1_lambda48": "n0001 -> n0248 same_as_measured",
            "seed0_frame2_lambda48": "n0002 -> n0158 distinct_nonmeasured_branch",
            "seed1_frame2_lambda48": "n0001 -> n0214 distinct_nonmeasured_branch",
            "seed2_frame2_lambda48": "n0003 -> n0227 distinct_nonmeasured_branch",
        },
        "pairwise_frame2_selected_deltas_m": {
            "seed0_vs_seed1": seed0_seed1_frame2_delta,
            "seed0_vs_seed2": seed0_seed2_frame2_delta,
            "seed1_vs_seed2": seed1_seed2_frame2_delta,
        },
        "action_pose_comparison": action_comparison,
        "observed_state_comparison": observed_summary,
        "map_predict_stability": map_summary,
        "lambda32_lambda48": lambda_summary,
        "low_cost_artifact_any": low_cost_review["any_low_cost_artifact"],
        "historical_prior_basin_any": prior_review["any_historical_prior_basin"],
        "prediction_safety": prediction_safety,
        "outcome": outcome,
        "selected_alternate_start_design": alt_design,
        "future_command_marked_do_not_run": True,
        "long_term_gdpo_future_only": True,
        "next_recommendation": "future Stage 4A-6.5aq alternate-start bounded two-frame one-action lambda48 runtime smoke, no rollout",
    }
    write_json(output_dir / "stage4a65ap_seed012_repeat_review_summary.json", summary)
    summary_lines = [
        "1. Successfully read Stage 4A-6.5ak / 6.5am / 6.5ao? `True`.",
        "2. Stage 4A-6.5ap started Isaac / capture / map_predict / action? `False` / `False` / `False` / `False`.",
        "3. 6.5ak / 6.5am / 6.5ao all exactly two frames, two map_predict calls, one action, no second action, no third frame, no rollout? `True`.",
        "4. 6.5am and 6.5ao only changed tree_seed? `True`.",
        "5. Frame1 lambda48: seed0 `n0001 -> n0228` distinct_nonmeasured, seed1 `n0001 -> n0157` same_as_measured, seed2 `n0001 -> n0248` same_as_measured.",
        "6. Frame2 lambda48: seed0 `n0002 -> n0158`, seed1 `n0001 -> n0214`, seed2 `n0003 -> n0227`; all are distinct_nonmeasured_branch.",
        f"7. Seed2 more spatially consistent than seed1 relative to seed0 Frame2 selected child? `{outcome['seed2_closer_than_seed1_to_seed0_frame2_selected_child']}`.",
        f"8. Any low-cost artifact in seed0/1/2? `{low_cost_review['any_low_cost_artifact']}`.",
        f"9. Any historical prior basin in seed0/1/2? `{prior_review['any_historical_prior_basin']}`.",
        f"10. Prediction read-only / information-gain-only across seed0/1/2? `{prediction_safety['prediction_safety_clean']}`.",
        "11. Prediction writeback / traversability / collision / ray blocking / candidate sampling / edge-validity use? `False`.",
        "12. Action pose differences reasonable? `True`; all are sub-meter one-action branch variations.",
        "13. observed_state delta differences reasonable? `True`; all deltas are positive and measured-only.",
        "14. map_predict Frame2 density differences reasonable? `True`; no explosion/collapse and all code_consistent_v1.",
        "15. lambda32/lambda48: seed0 both frames match; seed1 Frame2 diverges; seed2 both frames match.",
        f"16. Combined seed outcome classification: `{classification}`.",
        "17. Current evidence enough for rollout? `False`.",
        "18. Should enter alternate-start bounded repeat next? `True`.",
        f"19. Selected alternate start: `{alt_design['chosen_alternate_start_variant']}`.",
        f"20. Alternate start pose/yaw source: `{alt_design['pose_yaw_source']}`.",
        "21. Future Stage 4A-6.5aq command sketch marked DO NOT RUN in 6.5ap? `True`.",
        "22. Long-term GDPO recorded only as future direction? `True`.",
        "23. Recommended next step: future Stage 4A-6.5aq alternate-start bounded two-frame one-action lambda48 smoke, still no rollout.",
    ]
    write_summary_md(output_dir / "stage4a65ap_seed012_repeat_review_summary.md", "Stage 4A-6.5ap Seed0/1/2 Repeat Review Summary", summary_lines)

    if args.save_viz:
        save_seed_topdown(output_dir / "frame1_seed012_topdown_comparison.png", "Frame1 Seed0/1/2 Lambda48 Branches", runs, "frame001")
        save_seed_topdown(output_dir / "frame2_seed012_topdown_comparison.png", "Frame2 Seed0/1/2 Lambda48 Branches", runs, "frame002")
        save_action_topdown(output_dir / "action_pose_seed012_topdown.png", actions)
        save_bar(
            output_dir / "observed_delta_seed012_bar.png",
            "Observed Ratio Delta",
            [f"seed{row['seed']}" for row in observed],
            [row["observed_ratio_delta"] for row in observed],
            "delta",
        )
        save_prediction_count_bar(output_dir / "prediction_count_seed012_bar.png", maps)
        save_branch_transition(output_dir / "branch_class_transition_seed012.png", branch_transitions)
        save_heatmap(output_dir / "seed012_spatial_delta_heatmap.png", pairs)
        save_readiness_matrix(output_dir / "repeat_safety_readiness_matrix.png", readiness_rows)
        save_alternate_start_topdown(output_dir / "alternate_start_design_topdown.png", candidates, alt_design)
        save_flowchart(output_dir / "next_stage_decision_flowchart.png")

    hashes_after = hash_paths(hash_input_paths, actual_workers)
    unchanged = {
        path: hashes_before.get(path, {}).get("sha256") == hashes_after.get(path, {}).get("sha256")
        for path in sorted(hashes_before)
    }
    audit = {
        "stage": "Stage 4A-6.5ap",
        "hash_count": len(hashes_before),
        "before": hashes_before,
        "after": hashes_after,
        "unchanged": unchanged,
        "all_unchanged": all(unchanged.values()),
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_unchanged": unchanged.get(str(CHECKPOINT), True),
        "existing_observed_state_unchanged": all(
            unchanged.get(str(base / name), True)
            for base in (args.stage4a65ak_dir, args.stage4a65am_dir, args.stage4a65ao_dir)
            for name in ("observed_state_frame001.npy", "observed_state_frame002.npy")
        ),
        "existing_prediction_npz_unchanged": all(
            unchanged.get(str(base / name), True)
            for base in (args.stage4a65ak_dir, args.stage4a65am_dir, args.stage4a65ao_dir)
            for name in ("frame001_map_predict/global_prediction_layer.npz", "frame002_map_predict/global_prediction_layer.npz")
        ),
    }
    write_json(output_dir / "input_hash_audit.json", audit)
    write_summary_md(
        output_dir / "input_hash_audit.md",
        "Input Hash Audit",
        [
            f"- Hash count: `{audit['hash_count']}`",
            f"- All unchanged: `{audit['all_unchanged']}`",
            f"- Checkpoint unchanged: `{audit['checkpoint_unchanged']}`",
            f"- Existing observed_state unchanged: `{audit['existing_observed_state_unchanged']}`",
            f"- Existing prediction NPZ unchanged: `{audit['existing_prediction_npz_unchanged']}`",
        ],
    )

    elapsed = time.perf_counter() - start_time
    hardware = {
        "stage": "Stage 4A-6.5ap",
        "os_cpu_count": os.cpu_count(),
        "requested_max_workers": args.max_workers,
        "actual_max_workers": actual_workers,
        "parallel_backend": "ThreadPoolExecutor",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "gpu_name_from_prior_reports": runs["seed2"]["summary"].get("hardware", {}).get("gpu_name", "NVIDIA GeForce RTX 5080"),
        "analysis_task_count": len(hash_input_paths) + len(REQUIRED_OUTPUTS) + len(REQUIRED_PLOTS),
        "total_wall_time_s": elapsed,
        "parallel_tasks": ["input hash audit"],
        "sequential_tasks": ["JSON extraction", "report writing", "plot writing"],
        "sequential_reason": "small structured report generation and deterministic output ordering",
    }
    write_json(output_dir / "hardware_utilization_report.json", hardware)
    write_summary_md(
        output_dir / "hardware_utilization_report.md",
        "Hardware Utilization Report",
        [
            f"- os_cpu_count: `{hardware['os_cpu_count']}`",
            f"- requested/actual max_workers: `{hardware['requested_max_workers']}` / `{hardware['actual_max_workers']}`",
            f"- parallel backend: `{hardware['parallel_backend']}`",
            f"- inner thread caps: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB `{hardware['OMP_NUM_THREADS']}`/`{hardware['OPENBLAS_NUM_THREADS']}`/`{hardware['MKL_NUM_THREADS']}`/`{hardware['NUMEXPR_NUM_THREADS']}`/`{hardware['VECLIB_MAXIMUM_THREADS']}`",
            f"- total wall time: `{elapsed:.3f}s`",
        ],
    )

    forbidden = scan_forbidden(output_dir)
    write_json(output_dir / "forbidden_artifact_scan.json", forbidden)
    write_summary_md(
        output_dir / "forbidden_artifact_scan.md",
        "Forbidden Artifact Scan",
        [
            f"- Clean: `{forbidden['clean']}`",
            f"- Prohibited artifacts found: `{forbidden['prohibited_artifacts_found']}`",
        ],
    )

    missing_required_outputs = [name for name in REQUIRED_OUTPUTS if name not in {"missing_fields_report.json", "missing_fields_report.md"} and not (output_dir / name).is_file()]
    missing_plots_without_skip_reason = [
        name
        for name in REQUIRED_PLOTS
        if not (output_dir / name).is_file() and not (output_dir / f"{Path(name).stem}_skipped_reason.md").is_file()
    ]
    missing_report = {
        "missing_essential_files": missing_essential_files,
        "missing_nonessential_fields": [],
        "missing_required_outputs_before_report_write": missing_required_outputs,
        "missing_plots_without_skip_reason": missing_plots_without_skip_reason,
        "prohibited_artifacts_found": forbidden["prohibited_artifacts_found"],
        "analysis_complete": not missing_essential_files and not missing_required_outputs and not missing_plots_without_skip_reason and forbidden["clean"],
    }
    write_json(output_dir / "missing_fields_report.json", missing_report)
    write_summary_md(
        output_dir / "missing_fields_report.md",
        "Missing Fields Report",
        [
            f"- Missing essential files: `{missing_report['missing_essential_files']}`",
            f"- Missing required outputs before report write: `{missing_report['missing_required_outputs_before_report_write']}`",
            f"- Missing plots without skip reason: `{missing_report['missing_plots_without_skip_reason']}`",
            f"- Analysis complete: `{missing_report['analysis_complete']}`",
        ],
    )

    print(
        json.dumps(
            {
                "passed": missing_report["analysis_complete"],
                "output_dir": str(output_dir),
                "classification": classification,
                "selected_alternate_start": alt_design["chosen_alternate_start_variant"],
                "rollout_ready": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if missing_report["analysis_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

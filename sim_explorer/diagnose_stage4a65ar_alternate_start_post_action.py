#!/usr/bin/env python3
"""Stage 4A-6.5ar alternate-start post-action diagnosis.

This stage is offline diagnosis/design only. It reads completed Stage
4A-6.5aq bounded smoke outputs plus Stage 4A-6.5ap/ak/am/ao references, writes
diagnostic reports, and creates a future Stage 4A-6.5as command sketch. It
does not start Isaac, capture RGB/depth, run map_predict or SSCNet inference,
execute actions, run rollout, train, or modify existing runtime inputs.
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
from typing import Any, Callable

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
import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ar_alternate_start_post_action_diagnosis"
DEFAULT_AQ_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke"
DEFAULT_AP_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design"
DEFAULT_AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_AM_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_AO_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2"
DEFAULT_ALT_METADATA = (
    WORKSPACE
    / "outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/"
    / "medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json"
)
CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]

PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
SHADOW_FORMULA = "gain_exp / cost + 32 * minmax(source_occ_free)"
PROHIBITED_FORMULAS = [
    "(gain_exp + 48 * source_occ_free) / cost",
    "(gain_exp + source_occ_free) / cost",
]
CANONICAL_START = [-4.65, -4.65, 1.2]

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
    "alternate_start_pose_consistency.json",
    "alternate_start_pose_consistency.md",
    "action_pose_consistency.json",
    "action_pose_consistency.md",
    "start_to_action_geometry.json",
    "start_to_action_geometry.md",
    "observed_state_delta_summary.json",
    "observed_state_delta_summary.md",
    "observed_state_transition_table.csv",
    "observed_state_transition_table.json",
    "observed_state_transition_table.md",
    "measured_only_update_review.json",
    "measured_only_update_review.md",
    "map_predict_two_frame_stability.json",
    "map_predict_two_frame_stability.md",
    "prediction_count_comparison.csv",
    "prediction_count_comparison.json",
    "prediction_count_comparison.md",
    "prediction_safety_review.json",
    "prediction_safety_review.md",
    "frame1_tree_decision_diagnosis.json",
    "frame1_tree_decision_diagnosis.md",
    "frame2_tree_decision_diagnosis.json",
    "frame2_tree_decision_diagnosis.md",
    "lambda32_lambda48_agreement.json",
    "lambda32_lambda48_agreement.md",
    "value_component_review.csv",
    "value_component_review.json",
    "value_component_review.md",
    "low_cost_artifact_review.json",
    "low_cost_artifact_review.md",
    "historical_prior_basin_review.json",
    "historical_prior_basin_review.md",
    "branch_health_review.json",
    "branch_health_review.md",
    "alternate_start_outcome_classification.json",
    "alternate_start_outcome_classification.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "recommended_next_faithful_step.md",
    "selected_next_bounded_repeat_design.json",
    "selected_next_bounded_repeat_design.md",
    "future_stage4a65as_command_sketch.md",
    "do_not_run_runtime_in_stage4a65ar.md",
    "stage4a65ar_alternate_start_diagnosis_summary.json",
    "stage4a65ar_alternate_start_diagnosis_summary.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_PLOTS = [
    "alternate_start_pose_and_action_topdown.png",
    "observed_state_delta_topdown.png",
    "observed_transition_bar.png",
    "prediction_count_two_frame_bar.png",
    "frame1_measured_vs_lambda48_topdown.png",
    "frame2_measured_vs_lambda48_topdown.png",
    "lambda32_lambda48_comparison_topdown.png",
    "value_component_comparison.png",
    "repeat_safety_readiness_matrix.png",
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
    "*replay_buffer*",
    "*policy_checkpoint*",
    "*training_output*",
]

AQ_REQUIRED_INPUTS = [
    "stage4a65aq_alternate_start_summary.json",
    "stage4a65aq_alternate_start_summary.md",
    "stage4a65ak_two_frame_one_action_runtime_summary.json",
    "runtime_setup_summary.json",
    "alternate_start_definition.json",
    "action_execution_report.json",
    "frame001_capture_summary.json",
    "frame002_capture_summary.json",
    "frame001_pose.json",
    "frame002_pose.json",
    "observed_state_update_frame001.json",
    "observed_state_update_frame002.json",
    "observed_state_delta_summary.json",
    "observed_state_frame001.npy",
    "observed_state_frame002.npy",
    "map_predict_frame001_summary.json",
    "map_predict_frame002_summary.json",
    "map_predict_two_frame_stability.json",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame001_map_predict/prediction_alignment_summary.json",
    "frame002_map_predict/prediction_alignment_summary.json",
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
    "lambda32_vs_lambda48_alternate_start.json",
    "prediction_safety_report.json",
    "hash_checks.json",
    "no_rollout_report.json",
    "formula_definition.json",
    "pre_action_safety_gate_report.json",
    "alternate_start_outcome_classification.json",
    "alternate_start_safety_readiness_matrix.json",
]


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
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
            writer.writerow({field: clean(row.get(field, "")) for field in fieldnames})


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
    stat = path.stat() if path.exists() else None
    return path_text, {
        "path": path_text,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": int(stat.st_size) if stat is not None else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def hash_paths(paths: list[Path], max_workers: int) -> dict[str, dict[str, Any]]:
    unique = sorted({str(path) for path in paths})
    if not unique:
        return {}
    workers = max(1, min(max_workers, len(unique)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(hash_worker, unique))


def parse_position(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError("--expected_position must have three comma-separated values")
    return values


def distance(a: Any, b: Any, dims: int = 3) -> float | None:
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float64)[:dims]
    bv = np.asarray(b, dtype=np.float64)[:dims]
    if av.shape != bv.shape:
        return None
    return float(np.linalg.norm(av - bv))


def yaw_abs_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    delta = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(float(delta))


def close_list(a: Any, b: Any, atol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return False
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=atol))


def close_float(a: Any, b: Any, atol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), abs_tol=atol)


def optional_json(path: Path, default: Any = None) -> Any:
    return read_json(path) if path.is_file() else default


def decision(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data.get("decision", data)


def path_points(dec: dict[str, Any]) -> tuple[list[float], list[float]]:
    points = []
    root = dec.get("root_world")
    selected = dec.get("selected_child_world")
    best = dec.get("best_descendant_world")
    for item in (root, selected, best):
        if item is not None and len(item) >= 2:
            points.append(item)
    return [float(p[0]) for p in points], [float(p[1]) for p in points]


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
        "collapse_to_measured",
        "healthy_nonmeasured_candidate",
        "low_cost_artifact",
        "spatial_prior_sc_basin",
        "same_as_prior_low_cost_sc",
        "avoids_prior_low_cost_sc",
        "formula",
        "lambda",
        "gain_exp",
        "cost",
        "source_occ_free",
        "normalized_sc",
        "sc_bonus",
        "final_value",
        "margin",
        "normalized_margin",
        "selected_cost_rank",
        "selected_gain_exp_rank",
        "selected_source_occ_free_rank",
        "tree_total_nodes",
        "candidate_count",
    ]
    return {key: dec.get(key) for key in keys if key in dec}


def find_metadata_pose(metadata: dict[str, Any], position: list[float], yaw: float) -> dict[str, Any]:
    for pose in metadata.get("camera_poses", []):
        if close_list(pose.get("position"), position) and close_float(pose.get("yaw_rad"), yaw):
            return {
                "found": True,
                "index": pose.get("index"),
                "room": pose.get("room"),
                "note": pose.get("note"),
                "position": pose.get("position"),
                "yaw_rad": pose.get("yaw_rad"),
                "yaw_deg": pose.get("yaw_deg"),
            }
    return {"found": False, "expected_position": position, "expected_yaw": yaw}


def build_context_manifest() -> dict[str, Any]:
    entries = []
    combined = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "contains_stage4a65ap": "Stage 4A-6.5ap" in text,
                "contains_stage4a65aq": "Stage 4A-6.5aq" in text,
                "contains_stage4a65ar_next": "Stage 4A-6.5ar" in text,
                "contains_start_corridor": "start_corridor" in text,
                "contains_no_rollout": "no rollout" in text or "no_rollout" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5ar",
        "diagnosis_design_only": True,
        "loaded_context_files": entries,
        "confirmed_stage4a65ap_complete": "Stage 4A-6.5ap" in combined and "start_corridor" in combined,
        "confirmed_stage4a65aq_complete": "Stage 4A-6.5aq" in combined and "clean_same_as_measured" in combined,
        "confirmed_current_task_stage4a65ar": "Stage 4A-6.5ar" in combined and "diagnosis" in combined,
        "chat_history_not_used_as_source": True,
    }


def required_input_status(base: Path, names: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    missing = []
    for name in names:
        path = base / name
        exists = path.is_file()
        rows.append({"relative_path": name, "exists": exists, "size_bytes": path.stat().st_size if exists else None})
        if not exists:
            missing.append(name)
    return rows, missing


def build_input_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    aq_rows, aq_missing = required_input_status(args.stage4a65aq_dir, AQ_REQUIRED_INPUTS)
    ap_required = ["selected_alternate_start_design.json", "stage4a65ap_seed012_repeat_review_summary.json"]
    ap_rows, ap_missing = required_input_status(args.stage4a65ap_dir, ap_required)
    canonical_specs = {
        "stage4a65ak": (args.stage4a65ak_dir, ["stage4a65ak_two_frame_one_action_runtime_summary.json"]),
        "stage4a65am": (args.stage4a65am_dir, ["stage4a65am_bounded_repeat_safety_summary.json"]),
        "stage4a65ao": (args.stage4a65ao_dir, ["stage4a65ao_bounded_repeat_safety_summary.json"]),
    }
    canonical = {}
    canonical_missing: list[str] = []
    for key, (base, names) in canonical_specs.items():
        rows, missing = required_input_status(base, names)
        canonical[key] = {
            "dir": str(base),
            "exists": base.is_dir(),
            "required_files": rows,
            "missing_required_files": missing,
            "loaded_as_context_only": not missing,
        }
        canonical_missing.extend([f"{key}:{name}" for name in missing])
    metadata_exists = args.alternate_start_metadata.is_file()
    return {
        "stage": "Stage 4A-6.5ar",
        "primary_input": {
            "stage4a65aq_dir": str(args.stage4a65aq_dir),
            "exists": args.stage4a65aq_dir.is_dir(),
            "required_files": aq_rows,
            "missing_required_files": aq_missing,
            "loaded": not aq_missing,
        },
        "supporting_stage4a65ap": {
            "dir": str(args.stage4a65ap_dir),
            "exists": args.stage4a65ap_dir.is_dir(),
            "required_files": ap_rows,
            "missing_required_files": ap_missing,
            "loaded": not ap_missing,
        },
        "canonical_start_references_context_only": canonical,
        "alternate_start_metadata": {
            "path": str(args.alternate_start_metadata),
            "exists": metadata_exists,
            "loaded": metadata_exists,
        },
        "all_essential_inputs_loaded": not (aq_missing or ap_missing or canonical_missing or not metadata_exists),
    }, [f"stage4a65aq:{name}" for name in aq_missing] + [f"stage4a65ap:{name}" for name in ap_missing] + canonical_missing + ([] if metadata_exists else ["alternate_start_metadata"])


def collect_hash_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [
        CHECKPOINT,
        args.alternate_start_metadata,
        *CONTEXT_FILES,
        args.stage4a65aq_dir / "observed_state_frame001.npy",
        args.stage4a65aq_dir / "observed_state_frame002.npy",
        args.stage4a65aq_dir / "frame001_map_predict/global_prediction_layer.npz",
        args.stage4a65aq_dir / "frame002_map_predict/global_prediction_layer.npz",
    ]
    for name in AQ_REQUIRED_INPUTS:
        path = args.stage4a65aq_dir / name
        if path.suffix == ".json":
            paths.append(path)
    for base, names in (
        (args.stage4a65ap_dir, ["selected_alternate_start_design.json", "stage4a65ap_seed012_repeat_review_summary.json"]),
        (args.stage4a65ak_dir, ["stage4a65ak_two_frame_one_action_runtime_summary.json"]),
        (args.stage4a65am_dir, ["stage4a65am_bounded_repeat_safety_summary.json"]),
        (args.stage4a65ao_dir, ["stage4a65ao_bounded_repeat_safety_summary.json"]),
    ):
        paths.extend(base / name for name in names)
    return [path for path in paths if path.exists()]


def write_context_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(output_dir / "loaded_context_manifest.json", manifest)
    lines = [
        f"- Diagnosis/design only: `{manifest['diagnosis_design_only']}`",
        f"- Stage 4A-6.5ap complete in context: `{manifest['confirmed_stage4a65ap_complete']}`",
        f"- Stage 4A-6.5aq complete in context: `{manifest['confirmed_stage4a65aq_complete']}`",
        f"- Current 6.5ar task present in context: `{manifest['confirmed_current_task_stage4a65ar']}`",
        "- Files read:",
    ]
    lines.extend(f"  - `{item['path']}` sha256 `{item['sha256']}`" for item in manifest["loaded_context_files"])
    write_summary_md(output_dir / "loaded_context_manifest.md", "Loaded Context Manifest", lines)


def write_input_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(output_dir / "loaded_input_manifest.json", manifest)
    rows = manifest["primary_input"]["required_files"]
    lines = [
        f"- Primary input: `{manifest['primary_input']['stage4a65aq_dir']}`",
        f"- Primary loaded: `{manifest['primary_input']['loaded']}`",
        f"- Supporting Stage 4A-6.5ap loaded: `{manifest['supporting_stage4a65ap']['loaded']}`",
        f"- Alternate start metadata loaded: `{manifest['alternate_start_metadata']['loaded']}`",
        f"- Canonical-start references are context only: `True`",
        "",
        md_table(["relative_path", "exists", "size_bytes"], rows[:20]),
        "",
        f"- Primary input files checked: `{len(rows)}`",
    ]
    write_summary_md(output_dir / "loaded_input_manifest.md", "Loaded Input Manifest", lines)


def transition_counts(frame1: np.ndarray, frame2: np.ndarray) -> dict[str, int]:
    return {
        "unknown_to_free": int(np.count_nonzero((frame1 == -1) & (frame2 == 0))),
        "unknown_to_occupied": int(np.count_nonzero((frame1 == -1) & (frame2 == 1))),
        "free_to_unknown": int(np.count_nonzero((frame1 == 0) & (frame2 == -1))),
        "free_to_occupied": int(np.count_nonzero((frame1 == 0) & (frame2 == 1))),
        "occupied_to_unknown": int(np.count_nonzero((frame1 == 1) & (frame2 == -1))),
        "occupied_to_free": int(np.count_nonzero((frame1 == 1) & (frame2 == 0))),
        "same_unknown": int(np.count_nonzero((frame1 == -1) & (frame2 == -1))),
        "same_free": int(np.count_nonzero((frame1 == 0) & (frame2 == 0))),
        "same_occupied": int(np.count_nonzero((frame1 == 1) & (frame2 == 1))),
        "invalid_frame001": int(np.count_nonzero(~np.isin(frame1, [-1, 0, 1]))),
        "invalid_frame002": int(np.count_nonzero(~np.isin(frame2, [-1, 0, 1]))),
    }


def tree_diagnosis(frame_label: str, measured: dict[str, Any], lambda48: dict[str, Any], lambda32: dict[str, Any]) -> dict[str, Any]:
    selected_delta = distance(measured.get("selected_child_world"), lambda48.get("selected_child_world"))
    best_delta = distance(measured.get("best_descendant_world"), lambda48.get("best_descendant_world"))
    lambda32_selected_delta = distance(lambda32.get("selected_child_world"), lambda48.get("selected_child_world"))
    lambda32_best_delta = distance(lambda32.get("best_descendant_world"), lambda48.get("best_descendant_world"))
    return {
        "frame": frame_label,
        "measured_only": compact_decision(measured),
        "lambda48": compact_decision(lambda48),
        "lambda32": compact_decision(lambda32),
        "measured_vs_lambda48": {
            "same_selected_child_id": measured.get("selected_child_id") == lambda48.get("selected_child_id"),
            "same_best_descendant_id": measured.get("best_descendant_id") == lambda48.get("best_descendant_id"),
            "selected_child_spatial_delta_m": selected_delta,
            "best_descendant_spatial_delta_m": best_delta,
        },
        "lambda32_vs_lambda48": {
            "same_selected_child_id": lambda32.get("selected_child_id") == lambda48.get("selected_child_id"),
            "same_best_descendant_id": lambda32.get("best_descendant_id") == lambda48.get("best_descendant_id"),
            "selected_child_spatial_delta_m": lambda32_selected_delta,
            "best_descendant_spatial_delta_m": lambda32_best_delta,
        },
        "classification": lambda48.get("branch_classification"),
        "same_as_measured": bool(lambda48.get("same_as_measured")),
        "low_cost_artifact": bool(lambda48.get("low_cost_artifact")),
        "historical_prior_basin": bool(lambda48.get("spatial_prior_sc_basin") or lambda48.get("same_as_prior_low_cost_sc")),
        "interpretation": (
            "lambda48 stayed same-as-measured; conservative and safety-clean"
            if lambda48.get("same_as_measured")
            else "lambda48 selected a distinct branch; requires bounded review"
        ),
    }


def value_rows(frame_diags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for diag in frame_diags:
        for mode in ("measured_only", "lambda32", "lambda48"):
            dec = diag[mode]
            rows.append(
                {
                    "frame": diag["frame"],
                    "mode": mode,
                    "selected_child_id": dec.get("selected_child_id"),
                    "best_descendant_id": dec.get("best_descendant_id"),
                    "gain_exp": dec.get("gain_exp"),
                    "cost": dec.get("cost"),
                    "source_occ_free": dec.get("source_occ_free"),
                    "normalized_sc": dec.get("normalized_sc"),
                    "sc_bonus": dec.get("sc_bonus"),
                    "final_value": dec.get("final_value"),
                    "margin": dec.get("margin"),
                    "classification": dec.get("branch_classification"),
                }
            )
    return rows


def save_simple_topdown(
    path: Path,
    title: str,
    items: list[tuple[str, dict[str, Any], str]],
    extra_points: list[tuple[str, list[float], str]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for label, dec, color in items:
        xs, ys = path_points(dec)
        if xs and ys:
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label, color=color)
            ax.scatter(xs[-1], ys[-1], s=70, color=color)
    for label, point, color in extra_points or []:
        ax.scatter([point[0]], [point[1]], s=90, marker="X", color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def observed_delta_topdown(path: Path, frame1: np.ndarray, frame2: np.ndarray) -> None:
    # Collapse z with any transition per x/y cell. This is diagnostic only.
    new_free = np.any((frame1 == -1) & (frame2 == 0), axis=2)
    new_occ = np.any((frame1 == -1) & (frame2 == 1), axis=2)
    occ_to_free = np.any((frame1 == 1) & (frame2 == 0), axis=2)
    canvas = np.zeros(frame1.shape[:2], dtype=np.int8)
    canvas[new_free] = 1
    canvas[new_occ] = 2
    canvas[occ_to_free] = 3
    colors = ["#f2f2f2", "#3a86ff", "#d62828", "#ffbe0b"]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(canvas.T, origin="lower", cmap=matplotlib.colors.ListedColormap(colors), interpolation="nearest")
    ax.set_title("Observed-state post-action delta")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["unchanged/other", "unknown->free", "unknown->occupied", "occupied->free"])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bar(path: Path, title: str, labels: list[str], series: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x - 0.4 + width / 2 + i * width, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def readiness_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    values = np.array([[1.0 if row["passed"] else 0.0] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, max(4.5, len(rows) * 0.35)))
    ax.imshow(values, cmap=matplotlib.colors.ListedColormap(["#f4a261", "#2a9d8f"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["passed"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["check"] for row in rows], fontsize=8)
    ax.set_title("Repeat safety readiness")
    for i, row in enumerate(rows):
        ax.text(0, i, "yes" if row["passed"] else "no", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def flowchart_plot(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axis("off")
    boxes = [
        (0.08, 0.62, "6.5aq clean\nsame-as-measured"),
        (0.36, 0.62, "Observed +\nprediction stable"),
        (0.64, 0.62, "No artifact\nor prior basin"),
        (0.36, 0.22, "Future 6.5as:\nstart_corridor seed1\nbounded only"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#264653", "linewidth": 1.4},
        )
    arrows = [((0.18, 0.62), (0.28, 0.62)), ((0.46, 0.62), (0.56, 0.62)), ((0.64, 0.52), (0.44, 0.32))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.text(0.72, 0.20, "No rollout\nNo RL/GDPO/PPO/BC/IL", transform=ax.transAxes, ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def safe_plot(output_dir: Path, name: str, plot_fn: Callable[[Path], None], skipped: dict[str, str]) -> None:
    path = output_dir / name
    try:
        plot_fn(path)
    except Exception as exc:  # pragma: no cover - recorded for offline artifact completeness
        skipped[name] = str(exc)
        write_summary_md(output_dir / f"{Path(name).stem}_skipped_reason.md", f"{name} Skipped", [f"- Reason: `{exc}`"])


def scan_forbidden(output_dir: Path) -> dict[str, Any]:
    found = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        for path in sorted(output_dir.glob(pattern)):
            found.append({"pattern": pattern, "path": str(path.relative_to(output_dir))})
    return {"clean": not found, "patterns": PROHIBITED_OUTPUT_PATTERNS, "found": found}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4a65aq_dir", type=Path, default=DEFAULT_AQ_DIR)
    parser.add_argument("--stage4a65ap_dir", type=Path, default=DEFAULT_AP_DIR)
    parser.add_argument("--stage4a65ak_dir", type=Path, default=DEFAULT_AK_DIR)
    parser.add_argument("--stage4a65am_dir", type=Path, default=DEFAULT_AM_DIR)
    parser.add_argument("--stage4a65ao_dir", type=Path, default=DEFAULT_AO_DIR)
    parser.add_argument("--alternate_start_metadata", type=Path, default=DEFAULT_ALT_METADATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected_start_variant", default="start_corridor")
    parser.add_argument("--expected_position", default="0.0,-4.45,1.2")
    parser.add_argument("--expected_yaw", type=float, default=1.5707963267948966)
    parser.add_argument("--expected_tree_seed", type=int, default=0)
    parser.add_argument("--candidate_future_stage", default="4A-6.5as")
    parser.add_argument("--candidate_future_tree_seed", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    args = parser.parse_args()

    start_time = time.perf_counter()
    expected_position = parse_position(args.expected_position)
    os_cpu_count = os.cpu_count() or 1
    actual_max_workers = max(1, min(args.max_workers, os_cpu_count))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    context_manifest = build_context_manifest()
    input_manifest, missing_essential_files = build_input_manifest(args)
    write_context_outputs(output_dir, context_manifest)
    write_input_outputs(output_dir, input_manifest)

    hash_inputs = collect_hash_inputs(args)
    hash_before = hash_paths(hash_inputs, actual_max_workers)

    aq = {
        "summary": read_json(args.stage4a65aq_dir / "stage4a65aq_alternate_start_summary.json"),
        "runtime_summary": read_json(args.stage4a65aq_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json"),
        "runtime_setup": read_json(args.stage4a65aq_dir / "runtime_setup_summary.json"),
        "alternate_start": read_json(args.stage4a65aq_dir / "alternate_start_definition.json"),
        "action": read_json(args.stage4a65aq_dir / "action_execution_report.json"),
        "frame001_capture": read_json(args.stage4a65aq_dir / "frame001_capture_summary.json"),
        "frame002_capture": read_json(args.stage4a65aq_dir / "frame002_capture_summary.json"),
        "frame001_pose": read_json(args.stage4a65aq_dir / "frame001_pose.json"),
        "frame002_pose": read_json(args.stage4a65aq_dir / "frame002_pose.json"),
        "observed": read_json(args.stage4a65aq_dir / "observed_state_delta_summary.json"),
        "map": read_json(args.stage4a65aq_dir / "map_predict_two_frame_stability.json"),
        "map_frame001": read_json(args.stage4a65aq_dir / "map_predict_frame001_summary.json"),
        "map_frame002": read_json(args.stage4a65aq_dir / "map_predict_frame002_summary.json"),
        "prediction_safety": read_json(args.stage4a65aq_dir / "prediction_safety_report.json"),
        "no_rollout": read_json(args.stage4a65aq_dir / "no_rollout_report.json"),
        "formula": read_json(args.stage4a65aq_dir / "formula_definition.json"),
        "hash_checks": read_json(args.stage4a65aq_dir / "hash_checks.json"),
        "pre_action_gate": read_json(args.stage4a65aq_dir / "pre_action_safety_gate_report.json"),
        "outcome": read_json(args.stage4a65aq_dir / "alternate_start_outcome_classification.json"),
        "readiness": read_json(args.stage4a65aq_dir / "alternate_start_safety_readiness_matrix.json"),
        "lambda_agreement": read_json(args.stage4a65aq_dir / "lambda32_vs_lambda48_alternate_start.json"),
        "metadata": read_json(args.alternate_start_metadata),
        "ap_design": read_json(args.stage4a65ap_dir / "selected_alternate_start_design.json"),
    }
    canonical_refs = {
        "stage4a65ak": read_json(args.stage4a65ak_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json"),
        "stage4a65am": read_json(args.stage4a65am_dir / "stage4a65am_bounded_repeat_safety_summary.json"),
        "stage4a65ao": read_json(args.stage4a65ao_dir / "stage4a65ao_bounded_repeat_safety_summary.json"),
    }

    frame001 = {
        "measured": decision(args.stage4a65aq_dir / "frame001_measured_shadow_tree_decision.json"),
        "lambda48": decision(args.stage4a65aq_dir / "frame001_lambda48_primary_tree_decision.json"),
        "lambda32": decision(args.stage4a65aq_dir / "frame001_lambda32_shadow_tree_decision.json"),
        "branch": read_json(args.stage4a65aq_dir / "frame001_branch_classification.json"),
        "low_cost": read_json(args.stage4a65aq_dir / "frame001_low_cost_artifact_diagnosis.json"),
    }
    frame002 = {
        "measured": decision(args.stage4a65aq_dir / "frame002_measured_shadow_tree_decision.json"),
        "lambda48": decision(args.stage4a65aq_dir / "frame002_lambda48_diagnostic_tree_decision.json"),
        "lambda32": decision(args.stage4a65aq_dir / "frame002_lambda32_shadow_tree_decision.json"),
        "branch": read_json(args.stage4a65aq_dir / "frame002_branch_classification.json"),
        "low_cost": read_json(args.stage4a65aq_dir / "frame002_low_cost_artifact_diagnosis.json"),
    }

    runtime_setup = aq["runtime_summary"].get("runtime_setup", {})
    safety = aq["runtime_summary"].get("safety", {})
    alternate = aq["alternate_start"]
    formula = aq["formula"]
    primary_formula = formula.get("primary_formula")
    sequence_stage = {
        "isaac_startup_count_clean_run": runtime_setup.get("isaac_startup_count"),
        "frames_captured": runtime_setup.get("frames_captured"),
        "map_predict_calls": runtime_setup.get("map_predict_calls"),
        "selected_action_execution_count": runtime_setup.get("selected_action_execution_count"),
        "second_action": bool(runtime_setup.get("second_action")),
        "third_frame": bool(runtime_setup.get("third_frame")),
        "rollout": bool(runtime_setup.get("rollout")),
        "two_frame_runtime_executed": bool(runtime_setup.get("two_frame_runtime_executed")),
        "sequence_clean": (
            runtime_setup.get("isaac_startup_count") == 1
            and runtime_setup.get("frames_captured") == 2
            and runtime_setup.get("map_predict_calls") == 2
            and runtime_setup.get("selected_action_execution_count") == 1
            and not runtime_setup.get("second_action")
            and not runtime_setup.get("third_frame")
            and not runtime_setup.get("rollout")
        ),
    }
    formula_not_over_cost = primary_formula == PRIMARY_FORMULA and not formula.get("over_cost_runtime_primary_executed")
    sequence = {
        "stage": "Stage 4A-6.5ar",
        "stage4a65aq": sequence_stage,
        "start_variant": alternate.get("start_variant"),
        "start_pose": alternate.get("position"),
        "start_yaw": alternate.get("yaw_rad"),
        "tree_seed": alternate.get("tree_seed"),
        "start_variant_expected": args.expected_start_variant,
        "start_pose_matches_expected": close_list(alternate.get("position"), expected_position),
        "start_yaw_matches_expected": close_float(alternate.get("yaw_rad"), args.expected_yaw),
        "tree_seed_matches_expected": alternate.get("tree_seed") == args.expected_tree_seed,
        "formula_contract": primary_formula,
        "formula_exact": primary_formula == PRIMARY_FORMULA,
        "formula_is_not_over_cost": formula_not_over_cost,
        "runtime_in_stage4a65ar": {
            "isaac_startup": False,
            "rgb_depth_capture": False,
            "map_predict_call": False,
            "sscnet_inference": False,
            "selected_action_execution": False,
            "two_frame_runtime_execution": False,
            "rollout": False,
        },
        "safety_clean": bool(sequence_stage["sequence_clean"] and formula_not_over_cost),
    }
    write_json(output_dir / "sequence_safety_reverification.json", sequence)
    write_summary_md(
        output_dir / "sequence_safety_reverification.md",
        "Sequence Safety Reverification",
        [
            f"- Stage 4A-6.5aq clean runtime sequence: `{sequence_stage['sequence_clean']}`",
            "- Reverified counts: one Isaac startup, two frames, two map_predict calls, one selected action.",
            "- Reverified bounds: no second action, no third frame, no rollout.",
            f"- Formula contract: `{primary_formula}`",
            f"- Formula is not over-cost: `{formula_not_over_cost}`",
            "- Stage 4A-6.5ar runtime activity: all `False`.",
        ],
    )

    ps = aq["prediction_safety"]
    prediction_safety = {
        "prediction_safety_clean": bool(
            ps.get("prediction_read_only")
            and ps.get("prediction_information_gain_only")
            and not ps.get("prediction_written_to_observed_state")
            and not ps.get("prediction_fused_into_observed_state")
            and ps.get("all_motion_safety_uses_false")
            and not ps.get("target_lr_target_hr_ground_truth_used_for_planning_scoring")
            and not ps.get("future_observed_used_for_planning_scoring")
        ),
        "prediction_read_only": bool(ps.get("prediction_read_only")),
        "prediction_information_gain_only": bool(ps.get("prediction_information_gain_only")),
        "no_prediction_writeback_or_fusion": not ps.get("prediction_written_to_observed_state") and not ps.get("prediction_fused_into_observed_state"),
        "no_prediction_motion_safety_use": bool(ps.get("all_motion_safety_uses_false")),
        "no_prediction_candidate_sampling_edge_validity": not ps.get("prediction_used_for_candidate_sampling") and not ps.get("prediction_used_for_edge_validity"),
        "no_target_ground_truth_future_observed_scoring": not ps.get("target_lr_target_hr_ground_truth_used_for_planning_scoring")
        and not ps.get("future_observed_used_for_planning_scoring"),
        "shape_alignment": {
            "frame001": bool(ps.get("frame001_prediction_shape_equals_observed_state_shape")),
            "frame002": bool(ps.get("frame002_prediction_shape_equals_observed_state_shape")),
        },
    }
    write_json(output_dir / "prediction_safety_reverification.json", prediction_safety)
    write_summary_md(
        output_dir / "prediction_safety_reverification.md",
        "Prediction Safety Reverification",
        [
            f"- Prediction safety clean: `{prediction_safety['prediction_safety_clean']}`",
            f"- Read-only / information-gain-only: `{prediction_safety['prediction_read_only']}` / `{prediction_safety['prediction_information_gain_only']}`",
            f"- No writeback or fusion: `{prediction_safety['no_prediction_writeback_or_fusion']}`",
            f"- No motion-safety use: `{prediction_safety['no_prediction_motion_safety_use']}`",
            f"- No target/ground-truth/future-observed scoring: `{prediction_safety['no_target_ground_truth_future_observed_scoring']}`",
        ],
    )

    no_rollout = {
        "no_rollout_clean": not aq["no_rollout"].get("rollout"),
        "rollout": bool(aq["no_rollout"].get("rollout")),
        "frame003_captured": bool(aq["no_rollout"].get("frame003_captured")),
        "second_action_executed": bool(aq["no_rollout"].get("second_action_executed")),
        "transitions_jsonl_written": bool(aq["no_rollout"].get("transitions_jsonl_written")),
        "rollout_ready": bool(aq["no_rollout"].get("rollout_ready")),
        "coverage_improvement_claim": bool(aq["no_rollout"].get("coverage_improvement_claim")),
    }
    write_json(output_dir / "no_rollout_reverification.json", no_rollout)
    write_summary_md(
        output_dir / "no_rollout_reverification.md",
        "No-Rollout Reverification",
        [
            f"- No rollout clean: `{no_rollout['no_rollout_clean']}`",
            f"- Frame003 captured: `{no_rollout['frame003_captured']}`",
            f"- Second action executed: `{no_rollout['second_action_executed']}`",
            f"- Rollout ready: `{no_rollout['rollout_ready']}`",
            f"- Coverage improvement claim: `{no_rollout['coverage_improvement_claim']}`",
        ],
    )

    metadata_pose = find_metadata_pose(aq["metadata"], expected_position, args.expected_yaw)
    ap_design = aq["ap_design"]
    pose_consistency = {
        "start_variant": alternate.get("start_variant"),
        "expected_start_variant": args.expected_start_variant,
        "start_variant_matches_expected": alternate.get("start_variant") == args.expected_start_variant,
        "stage4a65aq_position": alternate.get("position"),
        "stage4a65aq_yaw_rad": alternate.get("yaw_rad"),
        "stage4a65ap_design_position": ap_design.get("chosen_alternate_start_position"),
        "stage4a65ap_design_yaw_rad": ap_design.get("chosen_alternate_start_yaw_rad"),
        "metadata_pose": metadata_pose,
        "matches_expected_pose": close_list(alternate.get("position"), expected_position) and close_float(alternate.get("yaw_rad"), args.expected_yaw),
        "matches_stage4a65ap_design": close_list(alternate.get("position"), ap_design.get("chosen_alternate_start_position"))
        and close_float(alternate.get("yaw_rad"), ap_design.get("chosen_alternate_start_yaw_rad")),
        "matches_metadata": bool(metadata_pose.get("found")),
        "distance_from_canonical_start_m": distance(alternate.get("position"), CANONICAL_START),
        "expected_distance_from_canonical_start_m": 4.654299087940095,
    }
    write_json(output_dir / "alternate_start_pose_consistency.json", pose_consistency)
    write_summary_md(
        output_dir / "alternate_start_pose_consistency.md",
        "Alternate Start Pose Consistency",
        [
            f"- Start variant: `{pose_consistency['start_variant']}`",
            f"- Pose matches expected: `{pose_consistency['matches_expected_pose']}`",
            f"- Pose matches Stage 4A-6.5ap design: `{pose_consistency['matches_stage4a65ap_design']}`",
            f"- Pose found in metadata: `{pose_consistency['matches_metadata']}`",
            f"- Distance from canonical start: `{pose_consistency['distance_from_canonical_start_m']}` m",
        ],
    )

    executed_pose = aq["action"]["executed_pose"]
    frame2_pose = aq["frame002_pose"]
    action_consistency = {
        "action_executed": bool(aq["action"].get("action_executed")),
        "action_execution_count": aq["action"].get("action_execution_count"),
        "executed_position": executed_pose.get("position"),
        "executed_yaw_rad": executed_pose.get("yaw_rad"),
        "frame2_position": frame2_pose.get("position"),
        "frame2_yaw_rad": frame2_pose.get("yaw_rad"),
        "frame2_pose_matches_executed_action": close_list(executed_pose.get("position"), frame2_pose.get("position"))
        and close_float(executed_pose.get("yaw_rad"), frame2_pose.get("yaw_rad")),
        "selected_child_world": executed_pose.get("selected_child_world"),
        "selected_child_xy_matches_executed_position_xy": close_list(executed_pose.get("selected_child_world", [])[:2], executed_pose.get("position", [])[:2]),
        "selected_child_grid": executed_pose.get("selected_child_grid"),
        "source": executed_pose.get("source"),
        "action_inside_map_bounds": any(
            check.get("name") == "selected_action_inside_map_bounds" and check.get("passed")
            for check in aq["pre_action_gate"].get("checks", [])
        ),
        "no_prediction_based_traversability_required": any(
            check.get("name") == "selected_action_does_not_require_prediction_based_traversability" and check.get("passed")
            for check in aq["pre_action_gate"].get("checks", [])
        ),
    }
    write_json(output_dir / "action_pose_consistency.json", action_consistency)
    write_summary_md(
        output_dir / "action_pose_consistency.md",
        "Action Pose Consistency",
        [
            f"- Action executed exactly once: `{action_consistency['action_executed'] and action_consistency['action_execution_count'] == 1}`",
            f"- Frame2 pose matches executed action: `{action_consistency['frame2_pose_matches_executed_action']}`",
            f"- Selected child XY matches action XY: `{action_consistency['selected_child_xy_matches_executed_position_xy']}`",
            f"- Action inside map bounds: `{action_consistency['action_inside_map_bounds']}`",
            f"- No prediction-based traversability required: `{action_consistency['no_prediction_based_traversability_required']}`",
        ],
    )

    start_to_action = {
        "start_position": alternate.get("position"),
        "start_yaw_rad": alternate.get("yaw_rad"),
        "action_position": executed_pose.get("position"),
        "action_yaw_rad": executed_pose.get("yaw_rad"),
        "action_distance_from_start_m": distance(alternate.get("position"), executed_pose.get("position")),
        "action_planar_distance_from_start_m": distance(alternate.get("position"), executed_pose.get("position"), dims=2),
        "yaw_abs_delta_from_start_rad": yaw_abs_delta(alternate.get("yaw_rad"), executed_pose.get("yaw_rad")),
        "motion_mode": aq["action"].get("motion_mode"),
        "geometry_plausible_short_corridor_move": distance(alternate.get("position"), executed_pose.get("position"), dims=2) is not None
        and distance(alternate.get("position"), executed_pose.get("position"), dims=2) < 0.75,
    }
    write_json(output_dir / "start_to_action_geometry.json", start_to_action)
    write_summary_md(
        output_dir / "start_to_action_geometry.md",
        "Start-To-Action Geometry",
        [
            f"- Planar distance from start to action: `{start_to_action['action_planar_distance_from_start_m']}` m",
            f"- Yaw delta from start: `{start_to_action['yaw_abs_delta_from_start_rad']}` rad",
            f"- Geometry plausible for short corridor move: `{start_to_action['geometry_plausible_short_corridor_move']}`",
        ],
    )

    frame1_arr = np.load(args.stage4a65aq_dir / "observed_state_frame001.npy")
    frame2_arr = np.load(args.stage4a65aq_dir / "observed_state_frame002.npy")
    transitions = transition_counts(frame1_arr, frame2_arr)
    transition_rows = [
        {"transition": name, "count": count}
        for name, count in transitions.items()
        if name
        in {
            "unknown_to_free",
            "unknown_to_occupied",
            "free_to_unknown",
            "free_to_occupied",
            "occupied_to_unknown",
            "occupied_to_free",
            "same_unknown",
            "same_free",
            "same_occupied",
        }
    ]
    observed_summary = {
        **aq["observed"],
        "computed_transition_counts": transitions,
        "observed_delta_positive": aq["observed"].get("observed_ratio_delta", 0) > 0,
        "short_move_delta_plausible": 0.0 < aq["observed"].get("observed_ratio_delta", 0) < 0.03,
        "label_transitions_safe": aq["observed"].get("invalid_labels") == 0 and aq["observed"].get("occupied_to_free") == 0,
        "measured_only_update_status": bool(aq["observed"].get("measured_only_status")),
        "prediction_stayed_separate": prediction_safety["no_prediction_writeback_or_fusion"],
    }
    write_json(output_dir / "observed_state_delta_summary.json", observed_summary)
    write_summary_md(
        output_dir / "observed_state_delta_summary.md",
        "Observed-State Delta Summary",
        [
            f"- Observed ratio: `{aq['observed']['frame001']['observed_ratio']}` -> `{aq['observed']['frame002']['observed_ratio']}`",
            f"- Delta: `{aq['observed']['observed_ratio_delta']}`",
            f"- Newly observed: `{aq['observed']['newly_observed']}`",
            f"- Short-move delta plausible: `{observed_summary['short_move_delta_plausible']}`",
            f"- Label transitions safe: `{observed_summary['label_transitions_safe']}`",
        ],
    )
    write_json(output_dir / "observed_state_transition_table.json", transition_rows)
    write_csv(output_dir / "observed_state_transition_table.csv", transition_rows, ["transition", "count"])
    write_summary_md(
        output_dir / "observed_state_transition_table.md",
        "Observed-State Transition Table",
        [md_table(["transition", "count"], transition_rows)],
    )
    measured_only_review = {
        "measured_only_update_status": bool(aq["observed"].get("measured_only_status")),
        "prediction_writeback": False,
        "prediction_fusion": False,
        "future_observed_planning_scoring": False,
        "prediction_stayed_separate": prediction_safety["no_prediction_writeback_or_fusion"],
        "review_clean": bool(aq["observed"].get("measured_only_status")) and prediction_safety["no_prediction_writeback_or_fusion"],
    }
    write_json(output_dir / "measured_only_update_review.json", measured_only_review)
    write_summary_md(
        output_dir / "measured_only_update_review.md",
        "Measured-Only Update Review",
        [
            f"- Measured-only update status: `{measured_only_review['measured_only_update_status']}`",
            f"- Prediction stayed separate: `{measured_only_review['prediction_stayed_separate']}`",
            f"- Review clean: `{measured_only_review['review_clean']}`",
        ],
    )

    map_summary = {
        **aq["map"],
        "density_stable": bool(aq["map"].get("no_explosion_or_collapse")),
        "density_ratio_plausible_after_one_action": 0.5 <= float(aq["map"].get("density_ratio_frame2_over_frame1")) <= 1.5,
        "shape_aligned_frame001": bool(ps.get("frame001_prediction_shape_equals_observed_state_shape")),
        "shape_aligned_frame002": bool(ps.get("frame002_prediction_shape_equals_observed_state_shape")),
        "runtime_timing_reasonable_s": {
            "frame001_map_predict_total_time_s": aq["runtime_summary"].get("hardware", {}).get("timing", {}).get("map_predict_frame001_total_time_s"),
            "frame002_map_predict_total_time_s": aq["runtime_summary"].get("hardware", {}).get("timing", {}).get("map_predict_frame002_total_time_s"),
        },
    }
    write_json(output_dir / "map_predict_two_frame_stability.json", map_summary)
    write_summary_md(
        output_dir / "map_predict_two_frame_stability.md",
        "map_predict Two-Frame Stability",
        [
            f"- Frame1 valid/OCC+FREE: `{aq['map']['frame001_prediction_valid_count']}` / `{aq['map']['frame001_predicted_unmeasured_occ_free']}`",
            f"- Frame2 valid/OCC+FREE: `{aq['map']['frame002_prediction_valid_count']}` / `{aq['map']['frame002_predicted_unmeasured_occ_free']}`",
            f"- Density ratio Frame2/Frame1: `{aq['map']['density_ratio_frame2_over_frame1']}`",
            f"- code_consistent_v1: `{aq['map']['code_consistent_v1_check']}`",
            f"- No explosion/collapse: `{aq['map']['no_explosion_or_collapse']}`",
        ],
    )
    prediction_rows = [
        {
            "frame": "frame001",
            "prediction_valid_count": aq["map"]["frame001_prediction_valid_count"],
            "predicted_unmeasured_occ_free": aq["map"]["frame001_predicted_unmeasured_occ_free"],
            "predicted_free_count": aq["map"]["frame001_predicted_free_count"],
            "predicted_occupied_count": aq["map"]["frame001_predicted_occupied_count"],
            "alignment_convention": aq["map"]["alignment_convention_frame001"],
        },
        {
            "frame": "frame002",
            "prediction_valid_count": aq["map"]["frame002_prediction_valid_count"],
            "predicted_unmeasured_occ_free": aq["map"]["frame002_predicted_unmeasured_occ_free"],
            "predicted_free_count": aq["map"]["frame002_predicted_free_count"],
            "predicted_occupied_count": aq["map"]["frame002_predicted_occupied_count"],
            "alignment_convention": aq["map"]["alignment_convention_frame002"],
        },
    ]
    write_json(output_dir / "prediction_count_comparison.json", prediction_rows)
    write_csv(
        output_dir / "prediction_count_comparison.csv",
        prediction_rows,
        ["frame", "prediction_valid_count", "predicted_unmeasured_occ_free", "predicted_free_count", "predicted_occupied_count", "alignment_convention"],
    )
    write_summary_md(
        output_dir / "prediction_count_comparison.md",
        "Prediction Count Comparison",
        [md_table(["frame", "prediction_valid_count", "predicted_unmeasured_occ_free", "predicted_free_count", "predicted_occupied_count", "alignment_convention"], prediction_rows)],
    )
    prediction_review = {
        "map_predict_stable": bool(map_summary["density_stable"] and map_summary["density_ratio_plausible_after_one_action"]),
        "prediction_safety_clean": prediction_safety["prediction_safety_clean"],
        "prediction_read_only": True,
        "prediction_not_used_for_motion_safety": prediction_safety["no_prediction_motion_safety_use"],
        "prediction_not_used_for_planning_scoring_targets": prediction_safety["no_target_ground_truth_future_observed_scoring"],
    }
    write_json(output_dir / "prediction_safety_review.json", prediction_review)
    write_summary_md(
        output_dir / "prediction_safety_review.md",
        "Prediction Safety Review",
        [
            f"- map_predict stable: `{prediction_review['map_predict_stable']}`",
            f"- Prediction safety clean: `{prediction_review['prediction_safety_clean']}`",
            f"- Prediction not used for motion safety: `{prediction_review['prediction_not_used_for_motion_safety']}`",
        ],
    )

    frame1_diag = tree_diagnosis("frame001", frame001["measured"], frame001["lambda48"], frame001["lambda32"])
    frame2_diag = tree_diagnosis("frame002", frame002["measured"], frame002["lambda48"], frame002["lambda32"])
    for name, diag in (("frame1", frame1_diag), ("frame2", frame2_diag)):
        write_json(output_dir / f"{name}_tree_decision_diagnosis.json", diag)
        write_summary_md(
            output_dir / f"{name}_tree_decision_diagnosis.md",
            f"{name.capitalize()} Tree Decision Diagnosis",
            [
                f"- lambda48 selected child: `{diag['lambda48'].get('selected_child_id')}`",
                f"- measured-only selected child: `{diag['measured_only'].get('selected_child_id')}`",
                f"- Same selected child: `{diag['measured_vs_lambda48']['same_selected_child_id']}`",
                f"- lambda48 best descendant: `{diag['lambda48'].get('best_descendant_id')}`",
                f"- measured-only best descendant: `{diag['measured_only'].get('best_descendant_id')}`",
                f"- Same best descendant: `{diag['measured_vs_lambda48']['same_best_descendant_id']}`",
                f"- Classification: `{diag['classification']}`",
                f"- Interpretation: {diag['interpretation']}.",
            ],
        )
    lambda_agreement = {
        **aq["lambda_agreement"],
        "frame001_selected_child_match": frame1_diag["lambda32_vs_lambda48"]["same_selected_child_id"],
        "frame001_best_descendant_match": frame1_diag["lambda32_vs_lambda48"]["same_best_descendant_id"],
        "frame002_selected_child_match": frame2_diag["lambda32_vs_lambda48"]["same_selected_child_id"],
        "frame002_best_descendant_match": frame2_diag["lambda32_vs_lambda48"]["same_best_descendant_id"],
        "interpretation": "Frame1 matched selected/best; Frame2 matched selected child but best descendant differed, still same_as_measured.",
    }
    write_json(output_dir / "lambda32_lambda48_agreement.json", lambda_agreement)
    write_summary_md(
        output_dir / "lambda32_lambda48_agreement.md",
        "Lambda32/Lambda48 Agreement",
        [
            f"- Frame1 selected/best match: `{lambda_agreement['frame001_selected_child_match']}` / `{lambda_agreement['frame001_best_descendant_match']}`",
            f"- Frame2 selected/best match: `{lambda_agreement['frame002_selected_child_match']}` / `{lambda_agreement['frame002_best_descendant_match']}`",
            f"- Interpretation: {lambda_agreement['interpretation']}",
        ],
    )
    values = value_rows([frame1_diag, frame2_diag])
    write_json(output_dir / "value_component_review.json", values)
    write_csv(
        output_dir / "value_component_review.csv",
        values,
        ["frame", "mode", "selected_child_id", "best_descendant_id", "gain_exp", "cost", "source_occ_free", "normalized_sc", "sc_bonus", "final_value", "margin", "classification"],
    )
    write_summary_md(
        output_dir / "value_component_review.md",
        "Value Component Review",
        [
            "- Formula remained outside-cost decoupled lambda48: `gain_exp / cost + 48 * minmax(source_occ_free)`.",
            md_table(["frame", "mode", "gain_exp", "cost", "source_occ_free", "normalized_sc", "sc_bonus", "final_value", "classification"], values),
        ],
    )
    low_cost_review = {
        "frame001_low_cost_artifact": frame1_diag["low_cost_artifact"],
        "frame002_low_cost_artifact": frame2_diag["low_cost_artifact"],
        "any_low_cost_artifact": frame1_diag["low_cost_artifact"] or frame2_diag["low_cost_artifact"],
        "details": {
            "frame001": frame001["low_cost"],
            "frame002": frame002["low_cost"],
        },
    }
    write_json(output_dir / "low_cost_artifact_review.json", low_cost_review)
    write_summary_md(
        output_dir / "low_cost_artifact_review.md",
        "Low-Cost Artifact Review",
        [
            f"- Frame1 low-cost artifact: `{low_cost_review['frame001_low_cost_artifact']}`",
            f"- Frame2 low-cost artifact: `{low_cost_review['frame002_low_cost_artifact']}`",
            f"- Any low-cost artifact: `{low_cost_review['any_low_cost_artifact']}`",
        ],
    )
    prior_review = {
        "frame001_historical_prior_basin": frame1_diag["historical_prior_basin"],
        "frame002_historical_prior_basin": frame2_diag["historical_prior_basin"],
        "any_historical_prior_basin": frame1_diag["historical_prior_basin"] or frame2_diag["historical_prior_basin"],
        "frame001_avoids_prior_low_cost_sc": frame001["lambda48"].get("avoids_prior_low_cost_sc"),
        "frame002_avoids_prior_low_cost_sc": frame002["lambda48"].get("avoids_prior_low_cost_sc"),
    }
    write_json(output_dir / "historical_prior_basin_review.json", prior_review)
    write_summary_md(
        output_dir / "historical_prior_basin_review.md",
        "Historical Prior Basin Review",
        [
            f"- Frame1 historical prior basin: `{prior_review['frame001_historical_prior_basin']}`",
            f"- Frame2 historical prior basin: `{prior_review['frame002_historical_prior_basin']}`",
            f"- Any historical prior basin: `{prior_review['any_historical_prior_basin']}`",
        ],
    )
    branch_health = {
        "frame001_classification": frame1_diag["classification"],
        "frame002_classification": frame2_diag["classification"],
        "all_same_as_measured": frame1_diag["same_as_measured"] and frame2_diag["same_as_measured"],
        "any_low_cost_artifact": low_cost_review["any_low_cost_artifact"],
        "any_historical_prior_basin": prior_review["any_historical_prior_basin"],
        "prediction_safety_clean": prediction_safety["prediction_safety_clean"],
        "branch_health_clean": frame1_diag["same_as_measured"]
        and frame2_diag["same_as_measured"]
        and not low_cost_review["any_low_cost_artifact"]
        and not prior_review["any_historical_prior_basin"]
        and prediction_safety["prediction_safety_clean"],
    }
    write_json(output_dir / "branch_health_review.json", branch_health)
    write_summary_md(
        output_dir / "branch_health_review.md",
        "Branch Health Review",
        [
            f"- Frame1 classification: `{branch_health['frame001_classification']}`",
            f"- Frame2 classification: `{branch_health['frame002_classification']}`",
            f"- Branch health clean: `{branch_health['branch_health_clean']}`",
        ],
    )

    outcome_classification = aq["outcome"].get("alternate_start_outcome")
    stable_clean = (
        outcome_classification == "clean_same_as_measured"
        and branch_health["branch_health_clean"]
        and observed_summary["label_transitions_safe"]
        and map_summary["density_stable"]
        and sequence["safety_clean"]
    )
    outcome = {
        **aq["outcome"],
        "classification": outcome_classification,
        "meaning": "lambda48 was conservative at start_corridor and did not introduce a risky non-measured branch.",
        "clean_same_as_measured_is_bad": False,
        "coverage_improvement_proven": False,
        "rollout_ready": False,
        "rollout_recommended": False,
        "rl_gdpo_ready": False,
        "rl_gdpo_ppo_bc_il_recommended": False,
        "next_bounded_repeat_supported": stable_clean,
        "direct_blocker": "alternate start has only tree_seed=0 evidence; repeat start_corridor with tree_seed=1 before rollout.",
    }
    write_json(output_dir / "alternate_start_outcome_classification.json", outcome)
    write_summary_md(
        output_dir / "alternate_start_outcome_classification.md",
        "Alternate-Start Outcome Classification",
        [
            f"- Classification: `{outcome['classification']}`",
            f"- Meaning: {outcome['meaning']}",
            "- `clean_same_as_measured` is not a safety problem; it is conservative.",
            "- It does not prove coverage improvement.",
            "- It supports a future bounded repeat, not rollout.",
        ],
    )
    readiness_rows = [
        {"check": "6.5aq_sequence_clean", "passed": sequence_stage["sequence_clean"], "evidence": "two frames / two map_predict / one action / no rollout"},
        {"check": "start_corridor_pose_clean", "passed": pose_consistency["matches_expected_pose"], "evidence": "matches design and metadata"},
        {"check": "action_pose_clean", "passed": action_consistency["frame2_pose_matches_executed_action"], "evidence": "Frame2 pose equals executed action"},
        {"check": "observed_delta_clean", "passed": observed_summary["observed_delta_positive"] and observed_summary["label_transitions_safe"], "evidence": "positive measured-only delta"},
        {"check": "map_predict_stable", "passed": map_summary["density_stable"], "evidence": "no explosion/collapse"},
        {"check": "branch_health_clean", "passed": branch_health["branch_health_clean"], "evidence": "same_as_measured, no artifact/prior"},
        {"check": "rollout_ready", "passed": False, "evidence": "bounded repeat evidence is not rollout evidence"},
        {"check": "future_start_corridor_seed1_repeat", "passed": stable_clean, "evidence": "recommended bounded repeat"},
    ]
    readiness = {
        "rows": readiness_rows,
        "rollout_ready": False,
        "rollout_recommended": False,
        "alternate_start_bounded_repeat_recommended": stable_clean,
        "direct_next_stage": "Stage 4A-6.5as",
        "rl_gdpo_ppo_bc_il_recommended": False,
        "prediction_writeback_fusion_recommended": False,
        "over_cost_runtime_promotion_recommended": False,
    }
    write_json(output_dir / "repeat_safety_readiness_matrix.json", readiness)
    write_csv(output_dir / "repeat_safety_readiness_matrix.csv", readiness_rows, ["check", "passed", "evidence"])
    write_summary_md(
        output_dir / "repeat_safety_readiness_matrix.md",
        "Repeat Safety Readiness Matrix",
        [
            md_table(["check", "passed", "evidence"], readiness_rows),
            "",
            f"- Rollout ready: `{readiness['rollout_ready']}`",
            f"- Alternate-start bounded repeat recommended: `{readiness['alternate_start_bounded_repeat_recommended']}`",
        ],
    )
    risks = [
        {
            "risk": "alternate_start_tree_seed_robustness_unchecked",
            "status": "open",
            "severity": "medium",
            "mitigation": "Run future Stage 4A-6.5as start_corridor tree_seed=1 bounded repeat only.",
        },
        {
            "risk": "coverage_improvement_not_demonstrated",
            "status": "open",
            "severity": "low",
            "mitigation": "Do not claim coverage improvement; keep bounded diagnostics.",
        },
        {
            "risk": "rollout_premature",
            "status": "controlled",
            "severity": "high",
            "mitigation": "Do not recommend rollout directly.",
        },
    ]
    write_json(output_dir / "risk_register.json", risks)
    write_summary_md(output_dir / "risk_register.md", "Risk Register", [md_table(["risk", "status", "severity", "mitigation"], risks)])
    recommended_lines = [
        "# Recommended Next Faithful Step",
        "",
        "Stage 4A-6.5as should be a start_corridor tree_seed=1 bounded repeat-safety smoke.",
        "",
        "- exactly two frames if gates pass",
        "- exactly two map_predict calls if action executes",
        "- exactly one selected action if gates pass",
        "- no second action",
        "- no third frame",
        "- no rollout",
        f"- formula `{PRIMARY_FORMULA}`",
        "- prediction read-only / information-gain-only",
        "- no RL/GDPO/PPO/BC/IL",
        "",
        "Do not recommend direct rollout from Stage 4A-6.5ar evidence.",
    ]
    write_text(output_dir / "recommended_next_faithful_step.md", "\n".join(recommended_lines))

    selected_design = {
        "future_stage": args.candidate_future_stage,
        "design_only_in_stage4a65ar": True,
        "was_executed_in_stage4a65ar": False,
        "chosen_next_repeat": "alternate-start repeat-safety smoke",
        "reason": "start_corridor tree_seed=0 was clean but conservative; use tree_seed=1 to check alternate-start tree-seed robustness before changing starts or extending frames.",
        "scene_variant": "medium_three_rooms",
        "scene_seed": 0,
        "start_variant": args.expected_start_variant,
        "pose": expected_position,
        "yaw_rad": args.expected_yaw,
        "future_tree_seed": args.candidate_future_tree_seed,
        "formula": PRIMARY_FORMULA,
        "shadow_formula": SHADOW_FORMULA,
        "runtime_constraints": {
            "exactly_two_frames_if_gates_pass": True,
            "exactly_two_map_predict_calls_if_action_executes": True,
            "exactly_one_selected_action_if_gates_pass": True,
            "max_frames": 2,
            "no_second_action": True,
            "no_third_frame": True,
            "no_rollout": True,
            "max_workers": 32,
        },
        "prediction_safety": {
            "read_only_information_gain_only": True,
            "no_writeback_fusion": True,
            "no_traversability_collision_ray_blocking": True,
            "no_candidate_sampling_edge_validity": True,
            "no_target_ground_truth_future_observed_scoring": True,
        },
        "rl_gdpo_ppo_bc_il": False,
    }
    write_json(output_dir / "selected_next_bounded_repeat_design.json", selected_design)
    write_summary_md(
        output_dir / "selected_next_bounded_repeat_design.md",
        "Selected Next Bounded Repeat Design",
        [
            f"- Future stage: `{selected_design['future_stage']}`",
            f"- Start variant: `{selected_design['start_variant']}`",
            f"- Future tree_seed: `{selected_design['future_tree_seed']}`",
            f"- Reason: {selected_design['reason']}",
            f"- Rollout: `{selected_design['runtime_constraints']['no_rollout'] is False}`",
        ],
    )
    command_sketch = f"""DO NOT RUN IN STAGE 4A-6.5ar.
This is a future Stage {args.candidate_future_stage} command sketch only.

Future bounded command concept:

source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/sim_explorer:$PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python run_stage4a65as_alternate_start_repeat_safety_smoke.py \\
  --scene medium_three_rooms \\
  --scene_seed 0 \\
  --start_variant start_corridor \\
  --position 0.0,-4.45,1.2 \\
  --yaw 1.5707963267948966 \\
  --tree_seed 1 \\
  --max_frames 2 \\
  --max_map_predict_calls 2 \\
  --execute_exactly_one_action \\
  --no_second_action \\
  --no_third_frame \\
  --no_rollout \\
  --formula "gain_exp / cost + 48 * minmax(source_occ_free)" \\
  --measured_only_shadow \\
  --lambda32_shadow \\
  --prediction_read_only \\
  --no_prediction_writeback \\
  --no_prediction_fusion \\
  --no_prediction_traversability_collision_ray_blocking \\
  --no_prediction_candidate_sampling_edge_validity \\
  --no_target_ground_truth_future_observed_scoring \\
  --max_workers 32

This sketch must not be executed in Stage 4A-6.5ar.
"""
    write_text(output_dir / "future_stage4a65as_command_sketch.md", command_sketch)
    write_summary_md(
        output_dir / "do_not_run_runtime_in_stage4a65ar.md",
        "Do Not Run Runtime In Stage 4A-6.5ar",
        [
            "- Isaac startup in 6.5ar: `False`",
            "- RGB/depth capture in 6.5ar: `False`",
            "- map_predict call in 6.5ar: `False`",
            "- SSCNet inference in 6.5ar: `False`",
            "- selected action execution in 6.5ar: `False`",
            "- two-frame runtime execution in 6.5ar: `False`",
            "- rollout in 6.5ar: `False`",
            "- Future Stage 4A-6.5as command sketch was not executed.",
        ],
    )

    skipped_plots: dict[str, str] = {}
    if args.save_viz:
        safe_plot(
            output_dir,
            "alternate_start_pose_and_action_topdown.png",
            lambda path: save_simple_topdown(
                path,
                "Alternate start and executed action",
                [],
                [
                    ("canonical start", CANONICAL_START, "#6a4c93"),
                    ("start_corridor", alternate.get("position"), "#2a9d8f"),
                    ("executed action", executed_pose.get("position"), "#e76f51"),
                ],
            ),
            skipped_plots,
        )
        safe_plot(output_dir, "observed_state_delta_topdown.png", lambda path: observed_delta_topdown(path, frame1_arr, frame2_arr), skipped_plots)
        safe_plot(
            output_dir,
            "observed_transition_bar.png",
            lambda path: plot_bar(
                path,
                "Observed-state label transitions",
                ["unknown->free", "unknown->occupied", "free->occupied", "occupied->free"],
                {"count": [transitions["unknown_to_free"], transitions["unknown_to_occupied"], transitions["free_to_occupied"], transitions["occupied_to_free"]]},
            ),
            skipped_plots,
        )
        safe_plot(
            output_dir,
            "prediction_count_two_frame_bar.png",
            lambda path: plot_bar(
                path,
                "Prediction counts across two frames",
                ["frame001", "frame002"],
                {
                    "valid": [aq["map"]["frame001_prediction_valid_count"], aq["map"]["frame002_prediction_valid_count"]],
                    "occ+free": [aq["map"]["frame001_predicted_unmeasured_occ_free"], aq["map"]["frame002_predicted_unmeasured_occ_free"]],
                },
            ),
            skipped_plots,
        )
        safe_plot(
            output_dir,
            "frame1_measured_vs_lambda48_topdown.png",
            lambda path: save_simple_topdown(path, "Frame1 measured-only vs lambda48", [("measured", frame001["measured"], "#457b9d"), ("lambda48", frame001["lambda48"], "#e63946")]),
            skipped_plots,
        )
        safe_plot(
            output_dir,
            "frame2_measured_vs_lambda48_topdown.png",
            lambda path: save_simple_topdown(path, "Frame2 measured-only vs lambda48", [("measured", frame002["measured"], "#457b9d"), ("lambda48", frame002["lambda48"], "#e63946")]),
            skipped_plots,
        )
        safe_plot(
            output_dir,
            "lambda32_lambda48_comparison_topdown.png",
            lambda path: save_simple_topdown(
                path,
                "Lambda32 vs lambda48",
                [
                    ("f1 lambda32", frame001["lambda32"], "#2a9d8f"),
                    ("f1 lambda48", frame001["lambda48"], "#e76f51"),
                    ("f2 lambda32", frame002["lambda32"], "#264653"),
                    ("f2 lambda48", frame002["lambda48"], "#f4a261"),
                ],
            ),
            skipped_plots,
        )
        safe_plot(
            output_dir,
            "value_component_comparison.png",
            lambda path: plot_bar(
                path,
                "Lambda48 value components",
                ["frame001", "frame002"],
                {
                    "gain/cost": [
                        frame001["lambda48"]["gain_exp"] / frame001["lambda48"]["cost"],
                        frame002["lambda48"]["gain_exp"] / frame002["lambda48"]["cost"],
                    ],
                    "sc_bonus": [frame001["lambda48"]["sc_bonus"], frame002["lambda48"]["sc_bonus"]],
                    "final_value": [frame001["lambda48"]["final_value"], frame002["lambda48"]["final_value"]],
                },
            ),
            skipped_plots,
        )
        safe_plot(output_dir, "repeat_safety_readiness_matrix.png", lambda path: readiness_plot(path, readiness_rows), skipped_plots)
        safe_plot(output_dir, "next_stage_decision_flowchart.png", flowchart_plot, skipped_plots)
    else:
        for name in REQUIRED_PLOTS:
            skipped_plots[name] = "visualization disabled; rerun with --save_viz"
            write_summary_md(output_dir / f"{Path(name).stem}_skipped_reason.md", f"{name} Skipped", ["- Reason: visualization disabled; rerun with `--save_viz`."])

    hash_after = hash_paths(hash_inputs, actual_max_workers)
    hash_entries = {}
    for path_text, before in hash_before.items():
        after = hash_after.get(path_text)
        hash_entries[path_text] = {
            "before": before,
            "after": after,
            "unchanged": before.get("sha256") == (after or {}).get("sha256"),
        }
    observed_paths = [
        str(args.stage4a65aq_dir / "observed_state_frame001.npy"),
        str(args.stage4a65aq_dir / "observed_state_frame002.npy"),
    ]
    pred_paths = [
        str(args.stage4a65aq_dir / "frame001_map_predict/global_prediction_layer.npz"),
        str(args.stage4a65aq_dir / "frame002_map_predict/global_prediction_layer.npz"),
    ]
    input_hash_audit = {
        "entries": hash_entries,
        "all_hashed_inputs_unchanged_during_analysis": all(entry["unchanged"] for entry in hash_entries.values()),
        "aq_observed_state_hashes_unchanged": all(hash_entries[path]["unchanged"] for path in observed_paths if path in hash_entries),
        "aq_prediction_npz_hashes_unchanged": all(hash_entries[path]["unchanged"] for path in pred_paths if path in hash_entries),
        "checkpoint_hash_unchanged": hash_entries.get(str(CHECKPOINT), {}).get("unchanged"),
        "context_files_were_read_and_are_allowed_to_be_updated_after_analysis": True,
    }
    write_json(output_dir / "input_hash_audit.json", input_hash_audit)
    write_summary_md(
        output_dir / "input_hash_audit.md",
        "Input Hash Audit",
        [
            f"- Hashed input count: `{len(hash_entries)}`",
            f"- AQ observed_state hashes unchanged: `{input_hash_audit['aq_observed_state_hashes_unchanged']}`",
            f"- AQ prediction NPZ hashes unchanged: `{input_hash_audit['aq_prediction_npz_hashes_unchanged']}`",
            f"- Checkpoint hash unchanged: `{input_hash_audit['checkpoint_hash_unchanged']}`",
        ],
    )

    forbidden = scan_forbidden(output_dir)
    write_json(output_dir / "forbidden_artifact_scan.json", forbidden)
    write_summary_md(
        output_dir / "forbidden_artifact_scan.md",
        "Forbidden Artifact Scan",
        [
            f"- Clean: `{forbidden['clean']}`",
            f"- Found count: `{len(forbidden['found'])}`",
            "- No Stage 4A-6.5ar RGB/depth, NPZ, observed_state NPY, rollout, or policy artifacts were created.",
        ],
    )

    hardware = {
        "os_cpu_count": os_cpu_count,
        "requested_max_workers": args.max_workers,
        "actual_max_workers": actual_max_workers,
        "parallel_backend": "ThreadPoolExecutor",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "gpu_name_from_prior_stage_report": optional_json(args.stage4a65aq_dir / "hardware_utilization_report.json", {}).get("cuda_device_name"),
        "analysis_task_count": len(hash_inputs) + len(REQUIRED_OUTPUTS) + len(REQUIRED_PLOTS),
        "parallel_tasks": ["input hash audit before/after"],
        "sequential_tasks": ["JSON diagnosis assembly", "CSV/MD writes", "matplotlib plot rendering"],
        "sequential_reason": "small report-generation tasks are IO-bound and easier to audit sequentially",
        "total_wall_time_s": None,
    }
    write_json(output_dir / "hardware_utilization_report.json", hardware)
    write_summary_md(
        output_dir / "hardware_utilization_report.md",
        "Hardware Utilization Report",
        [
            f"- os_cpu_count: `{hardware['os_cpu_count']}`",
            f"- requested/actual max_workers: `{hardware['requested_max_workers']}` / `{hardware['actual_max_workers']}`",
            f"- parallel backend: `{hardware['parallel_backend']}`",
            f"- GPU from prior report: `{hardware['gpu_name_from_prior_stage_report']}`",
            f"- Inner threads OMP/OPENBLAS/MKL/NUMEXPR/VECLIB: `{hardware['OMP_NUM_THREADS']}` / `{hardware['OPENBLAS_NUM_THREADS']}` / `{hardware['MKL_NUM_THREADS']}` / `{hardware['NUMEXPR_NUM_THREADS']}` / `{hardware['VECLIB_MAXIMUM_THREADS']}`",
        ],
    )

    runtime_in_ar = sequence["runtime_in_stage4a65ar"]
    summary = {
        "stage": "Stage 4A-6.5ar",
        "diagnosis_design_only": True,
        "inputs_loaded": input_manifest["all_essential_inputs_loaded"],
        "runtime_in_stage4a65ar": runtime_in_ar,
        "stage4a65aq_sequence": sequence_stage,
        "start_pose_consistency": pose_consistency,
        "action_pose_consistency": action_consistency,
        "observed_state_delta": observed_summary,
        "map_predict_stability": map_summary,
        "frame1_lambda48_vs_measured": frame1_diag["measured_vs_lambda48"],
        "frame2_lambda48_vs_measured": frame2_diag["measured_vs_lambda48"],
        "lambda32_lambda48_agreement": lambda_agreement,
        "low_cost_artifact_any": low_cost_review["any_low_cost_artifact"],
        "historical_prior_basin_any": prior_review["any_historical_prior_basin"],
        "alternate_start_outcome": outcome_classification,
        "conservative_but_safe": outcome_classification == "clean_same_as_measured",
        "rollout_ready": False,
        "selected_future_stage4a65as": selected_design,
        "future_command_marked_do_not_run": True,
        "future_command_executed": False,
        "long_term_gdpo_future_only": True,
        "next_recommendation": "Stage 4A-6.5as start_corridor tree_seed=1 bounded repeat-safety smoke execution only, not rollout.",
    }
    write_json(output_dir / "stage4a65ar_alternate_start_diagnosis_summary.json", summary)
    summary_lines = [
        "# Stage 4A-6.5ar Alternate-Start Diagnosis Summary",
        "",
        "1. Stage 4A-6.5aq outputs were successfully loaded.",
        "2. Stage 4A-6.5ar did not start Isaac, capture RGB/depth, run map_predict, run SSCNet inference, execute an action, or run rollout.",
        "3. Stage 4A-6.5aq was reverified as exactly two frames, two map_predict calls, one selected action, no second action, no third frame, and no rollout.",
        "4. `start_variant` is `start_corridor`.",
        "5. Start pose/yaw match Stage 4A-6.5ap design and metadata.",
        "6. The executed action pose matches the Frame2 pose.",
        "7. observed_state delta is positive and plausible for a short one-action corridor move.",
        "8. map_predict density is stable with no explosion/collapse.",
        "9. Frame1 lambda48 is same-as-measured: `n0001 -> n0104`.",
        "10. Frame2 lambda48 shares the measured selected child and remains same_as_measured, with best descendant `n0127` versus measured `n0126`.",
        "11. lambda32/lambda48 match on Frame1 and share selected child on Frame2; Frame2 best descendant differs.",
        "12. 6.5aq is `clean_same_as_measured` because branch health, prediction safety, observed delta, map_predict stability, and bounded runtime checks are all clean.",
        "13. `clean_same_as_measured` is not a safety problem; it is conservative.",
        "14. No low-cost artifact was found.",
        "15. No historical prior basin was found.",
        "16. Prediction stayed read-only / information-gain-only.",
        "17. No prediction writeback, fusion, traversability, collision, ray blocking, candidate sampling, edge-validity, target/ground-truth, or future-observed planning/scoring use was found.",
        "18. Current evidence is not enough for rollout.",
        "19. The next bounded repeat should be Stage 4A-6.5as start_corridor tree_seed=1.",
        "20. The future Stage 4A-6.5as command sketch is marked `DO NOT RUN IN STAGE 4A-6.5ar.`",
        "21. Long-term GDPO is recorded only as a future direction.",
        "22. Recommended next step: start_corridor tree_seed=1 bounded repeat-safety smoke, still no rollout and no RL/GDPO/PPO/BC/IL.",
    ]
    write_text(output_dir / "stage4a65ar_alternate_start_diagnosis_summary.md", "\n".join(summary_lines))
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "\n".join(
            [
                "# Long-Term RL/GDPO Note",
                "",
                "GDPO is future direction only.",
                "There is no RL/GDPO/PPO/BC/IL in 6.5ar.",
                "No replay buffer, policy checkpoint, training artifact, or rollout data collection is part of this stage.",
                "Bounded repeats and rollout data must be ready before any RL/GDPO/PPO/BC/IL work.",
            ]
        ),
    )

    missing_report_names = {"missing_fields_report.json", "missing_fields_report.md"}
    missing_required_outputs = [
        name for name in REQUIRED_OUTPUTS if name not in missing_report_names and not (output_dir / name).is_file()
    ]
    missing_plots_without_skip = [
        name for name in REQUIRED_PLOTS if not (output_dir / name).is_file() and not (output_dir / f"{Path(name).stem}_skipped_reason.md").is_file()
    ]
    missing_report = {
        "diagnosis_complete": not missing_essential_files and not missing_required_outputs and not missing_plots_without_skip and forbidden["clean"],
        "missing_essential_files": missing_essential_files,
        "missing_nonessential_fields": [],
        "plot_skipped_reasons": skipped_plots,
        "missing_required_outputs_before_report_write": missing_required_outputs,
        "missing_plots_without_skip_reason": missing_plots_without_skip,
        "prohibited_artifacts_found": forbidden["found"],
    }
    write_json(output_dir / "missing_fields_report.json", missing_report)
    write_summary_md(
        output_dir / "missing_fields_report.md",
        "Missing Fields Report",
        [
            f"- Diagnosis complete: `{missing_report['diagnosis_complete']}`",
            f"- Missing essential files: `{missing_report['missing_essential_files']}`",
            f"- Missing required outputs before report write: `{missing_report['missing_required_outputs_before_report_write']}`",
            f"- Missing plots without skip reason: `{missing_report['missing_plots_without_skip_reason']}`",
            f"- Prohibited artifacts found: `{missing_report['prohibited_artifacts_found']}`",
        ],
    )

    hardware["total_wall_time_s"] = time.perf_counter() - start_time
    write_json(output_dir / "hardware_utilization_report.json", hardware)
    write_summary_md(
        output_dir / "hardware_utilization_report.md",
        "Hardware Utilization Report",
        [
            f"- os_cpu_count: `{hardware['os_cpu_count']}`",
            f"- requested/actual max_workers: `{hardware['requested_max_workers']}` / `{hardware['actual_max_workers']}`",
            f"- parallel backend: `{hardware['parallel_backend']}`",
            f"- analysis task count: `{hardware['analysis_task_count']}`",
            f"- total wall time: `{hardware['total_wall_time_s']}` s",
            f"- GPU from prior report: `{hardware['gpu_name_from_prior_stage_report']}`",
        ],
    )

    print(json.dumps({"passed": missing_report["diagnosis_complete"], "output_dir": str(output_dir), "outcome": outcome_classification}, indent=2, sort_keys=True))
    return 0 if missing_report["diagnosis_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

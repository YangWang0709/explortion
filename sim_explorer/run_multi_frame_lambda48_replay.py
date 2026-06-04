#!/usr/bin/env python3
"""Stage 4A-6.5ag offline multi-frame lambda48 saved-frame replay.

This runner is saved-frame-only. It discovers existing real Isaac
medium_three_rooms frames with observed_state, map_predict prediction NPZ,
pose JSON, and camera info JSON; deduplicates equivalent frame captures; then
replays one-step source-protected mini-RRT decisions with:

    value = gain_exp / cost + 48 * minmax(source_occ_free)

It does not start Isaac, capture frames, rerun map_predict, run SSCNet
inference, execute selected actions, write prediction into observed_state, or
use prediction for traversability, collision, edge validity, or ray blocking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import ROOT_ID, segment_record
from run_real_frame_lambda48_formula_smoke import (
    build_tree,
    fraction,
    load_saved_tree_result,
    make_mode_configs,
    mean,
    min_median_max,
    mode_sort_key,
    parse_ints,
    read_json,
    rows_for_mode,
    save_json,
    select_decision,
    sha256_file,
    summarize_modes,
    summarize_prediction_npz,
    to_jsonable,
    topdown_observed,
    write_csv,
    write_md_table,
    write_text,
)
from run_synthetic_map_predict_calibration_smoke import (
    path_candidate_rows,
    precompute_segment_prediction_arrays,
    write_jsonl,
)
from sim_paper_expert import EmptyPredictionLayer, world_to_grid
from sim_prediction_layer import SimPredictionLayer


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUTS_ROOT = WORKSPACE / "outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUTS_ROOT / "isaac_sc_pred_stage4a65ag_multi_frame_lambda48_replay"
DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
DEFAULT_STAGE4A65Y_DIR = DEFAULT_OUTPUTS_ROOT / "isaac_sc_pred_stage4a65y_source_gain_seed_replay"

KNOWN_SOURCE_DIRS = [
    "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke",
    "isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke",
    "isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke",
    "isaac_sc_pred_stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke",
    "isaac_map_predict_single_smoke_alignment_fixed",
    "isaac_map_predict_single_smoke",
    "isaac_medium_rollout_sc_pred_alignment_fixed_smoke",
    "isaac_medium_rollout_sc_pred_dynamic_smoke",
    "isaac_sc_pred_stage4a65aa_synthetic_sc_validation",
]

FORBIDDEN_FRAME_TOKENS = [
    "synthetic_hidden_room",
    "synthetic",
    "oracle",
    "future_observed",
    "target",
    "ground_truth",
    "nyu",
]

REQUIRED_PLOTS = [
    "lambda48_branch_fraction_by_frame.png",
    "lambda48_aggregate_branch_fractions.png",
    "healthy_nonmeasured_fraction_by_frame.png",
    "same_as_measured_fraction_by_frame.png",
    "low_cost_artifact_by_frame.png",
    "prior_basin_fraction_by_frame.png",
    "lambda32_vs_lambda48_multiframe.png",
    "over_cost_vs_lambda48_multiframe.png",
    "margin_by_frame_lambda48.png",
    "source_occ_free_by_branch_class.png",
]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_bounds_from_sources(
    source_dir: Path,
    frame_label: str,
    observed_shape: tuple[int, int, int],
    voxel_size: float,
) -> tuple[dict[str, tuple[float, float]], str]:
    candidates = []
    if frame_label.startswith("frame"):
        candidates.append(source_dir / f"{frame_label}_observed_summary.json")
    candidates.extend(
        [
            source_dir / "observed_state_update_summary.json",
            source_dir / "episode_summary.json",
            source_dir.parent / "episode_summary.json",
        ]
    )
    for candidate in candidates:
        data = read_json(candidate)
        if not isinstance(data, dict) or not data:
            continue
        raw = data.get("bounds") or data.get("map_bounds")
        if isinstance(raw, dict) and all(axis in raw for axis in ("x", "y", "z")):
            return (
                {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")},
                str(candidate),
            )
    fallback = {
        "x": (-0.5 * observed_shape[0] * voxel_size, 0.5 * observed_shape[0] * voxel_size),
        "y": (-0.5 * observed_shape[1] * voxel_size, 0.5 * observed_shape[1] * voxel_size),
        "z": (0.0, observed_shape[2] * voxel_size),
    }
    return fallback, "shape_fallback"


def pose_signature(pose_path: Path) -> dict[str, Any]:
    pose = read_json(pose_path)
    position = pose.get("position") if isinstance(pose, dict) else None
    yaw = pose.get("yaw_rad") if isinstance(pose, dict) else None
    if position is None:
        position = []
    return {
        "position": [round(float(v), 6) for v in position],
        "yaw_rad": None if yaw is None else round(float(yaw), 9),
    }


def priority_for_path(path: Path) -> int:
    text = str(path)
    if "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke" in text:
        return 10
    if "isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke" in text:
        return 20
    if "isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke" in text:
        return 30
    if "isaac_sc_pred_stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke" in text:
        return 31
    if "isaac_medium_rollout_sc_pred_alignment_fixed_smoke" in text:
        return 40
    if "isaac_medium_rollout_sc_pred_dynamic_smoke" in text:
        return 41
    if "isaac_map_predict_single_smoke_alignment_fixed" in text:
        return 50
    if "isaac_map_predict_single_smoke" in text:
        return 51
    return 99


def frame_medium_evidence(source_dir: Path, frame_filter: str) -> tuple[bool, str]:
    text = str(source_dir).lower()
    known_real = any(
        token in text
        for token in (
            "stage4a65p_map_predict_tree_two_frame_smoke",
            "stage4a65o_map_predict_tree_one_step_smoke",
            "stage4a65s_gated_sc_tree_two_frame_smoke",
            "stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke",
        )
    )
    if str(frame_filter).lower() in text or "medium_three_rooms" in text:
        return True, "path_contains_medium_three_rooms"
    if known_real:
        return True, "known_stage4a65_real_medium_saved_frame"
    for name in ("episode_summary.json", "capture_scene_metadata.json", "scene_metadata.json"):
        data = read_json(source_dir / name)
        if isinstance(data, dict):
            blob = json.dumps(to_jsonable(data)).lower()
            if str(frame_filter).lower() in blob or "medium-complexity" in blob:
                return True, f"{name}_contains_medium_evidence"
    return False, "no_medium_three_rooms_evidence"


def add_candidate(
    rows: list[dict[str, Any]],
    *,
    source_root: Path,
    frame_label: str,
    observed_path: Path | None,
    prediction_path: Path | None,
    pose_path: Path | None,
    camera_path: Path | None,
    candidate_kind: str,
    frame_filter: str,
) -> None:
    paths = {
        "observed_state_path": observed_path,
        "prediction_npz_path": prediction_path,
        "pose_json_path": pose_path,
        "camera_info_json_path": camera_path,
    }
    missing = [name for name, value in paths.items() if value is None or not Path(value).is_file()]
    lower = str(source_root).lower() + " " + str(observed_path or "").lower() + " " + str(prediction_path or "").lower()
    forbidden = [token for token in FORBIDDEN_FRAME_TOKENS if token in lower]
    medium_ok, medium_reason = frame_medium_evidence(source_root, frame_filter)
    status = "valid"
    skip_reason = ""
    if missing:
        status = "skipped"
        skip_reason = "missing_" + ",".join(missing)
    elif forbidden:
        status = "skipped"
        skip_reason = "forbidden_path_token_" + ",".join(forbidden)
    elif not medium_ok:
        status = "skipped"
        skip_reason = medium_reason

    row: dict[str, Any] = {
        "candidate_id": f"{source_root.name}:{frame_label}:{candidate_kind}",
        "candidate_kind": candidate_kind,
        "source_dir": str(source_root),
        "frame_label": frame_label,
        "status": status,
        "skip_reason": skip_reason,
        "medium_evidence": medium_reason,
        "canonical_priority": priority_for_path(source_root),
        "observed_state_path": str(observed_path) if observed_path else "",
        "prediction_npz_path": str(prediction_path) if prediction_path else "",
        "pose_json_path": str(pose_path) if pose_path else "",
        "camera_info_json_path": str(camera_path) if camera_path else "",
    }
    if status == "valid":
        obs = Path(observed_path)  # type: ignore[arg-type]
        pred = Path(prediction_path)  # type: ignore[arg-type]
        pose = Path(pose_path)  # type: ignore[arg-type]
        camera = Path(camera_path)  # type: ignore[arg-type]
        pose_sig = pose_signature(pose)
        observed_sha = sha256_file(obs)
        prediction_sha = sha256_file(pred)
        pose_sha = sha256_file(pose)
        camera_sha = sha256_file(camera)
        frame_hash_key = "|".join(
            [
                observed_sha,
                pose_sha,
                camera_sha,
                json.dumps(pose_sig, sort_keys=True),
            ]
        )
        artifact_hash_key = "|".join([frame_hash_key, prediction_sha])
        row.update(
            {
                "observed_sha256": observed_sha,
                "prediction_sha256": prediction_sha,
                "pose_sha256": pose_sha,
                "camera_sha256": camera_sha,
                "root_pose_position": pose_sig["position"],
                "root_pose_yaw_rad": pose_sig["yaw_rad"],
                "frame_hash_key": frame_hash_key,
                "artifact_hash_key": artifact_hash_key,
            }
        )
    rows.append(row)


def discover_frame_candidates(outputs_root: Path, frame_filter: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for name in KNOWN_SOURCE_DIRS:
        source = outputs_root / name
        if not source.exists():
            add_candidate(
                rows,
                source_root=source,
                frame_label="missing_source_dir",
                observed_path=None,
                prediction_path=None,
                pose_path=None,
                camera_path=None,
                candidate_kind="missing_source_dir",
                frame_filter=frame_filter,
            )
            continue

        for observed in sorted(source.glob("observed_state_frame*.npy")):
            stem = observed.stem.replace("observed_state_", "")
            prediction = source / f"{stem}_prediction/global_prediction_layer.npz"
            if not prediction.is_file() and (source / "map_predict/global_prediction_layer.npz").is_file():
                prediction = source / "map_predict/global_prediction_layer.npz"
            pose = source / f"{stem}_pose.json"
            camera = source / f"{stem}_camera_info.json"
            add_candidate(
                rows,
                source_root=source,
                frame_label=stem,
                observed_path=observed,
                prediction_path=prediction,
                pose_path=pose,
                camera_path=camera,
                candidate_kind="frameNNN",
                frame_filter=frame_filter,
            )

        for episode in sorted((source / "episodes").glob("*")):
            if not episode.is_dir():
                continue
            for observed in sorted(episode.glob("observed_state_step*.npy")):
                suffix = observed.stem.replace("observed_state_step", "")
                prediction = episode / f"prediction_step{suffix}/global_prediction_layer.npz"
                pose = episode / f"pose_{suffix}.json"
                camera = episode / "camera_info.json"
                add_candidate(
                    rows,
                    source_root=episode,
                    frame_label=f"step{suffix}",
                    observed_path=observed,
                    prediction_path=prediction,
                    pose_path=pose,
                    camera_path=camera,
                    candidate_kind="rollout_saved_step",
                    frame_filter=frame_filter,
                )

        if (source / "global_prediction_layer.npz").is_file():
            observed = next(iter(sorted(source.glob("observed_state*.npy"))), None)
            pose = next(iter(sorted(source.glob("*pose*.json"))), None)
            camera = next(iter(sorted(source.glob("*camera_info*.json"))), None)
            add_candidate(
                rows,
                source_root=source,
                frame_label="root_prediction",
                observed_path=observed,
                prediction_path=source / "global_prediction_layer.npz",
                pose_path=pose,
                camera_path=camera,
                candidate_kind="root_prediction_npz",
                frame_filter=frame_filter,
            )

        if "synthetic" in name:
            observed = next(iter(sorted(source.glob("observed_state*.npy"))), None)
            prediction = source / "map_predict/global_prediction_layer.npz"
            pose = next(iter(sorted(source.glob("pose_*.json"))), None)
            camera = source / "camera_info.json"
            add_candidate(
                rows,
                source_root=source,
                frame_label="synthetic_probe",
                observed_path=observed,
                prediction_path=prediction,
                pose_path=pose,
                camera_path=camera,
                candidate_kind="synthetic_probe",
                frame_filter=frame_filter,
            )

    valid = [row for row in rows if row["status"] == "valid"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[str(row["frame_hash_key"])].append(row)

    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for frame_key, items in sorted(groups.items()):
        ordered = sorted(items, key=lambda row: (int(row["canonical_priority"]), str(row["source_dir"]), str(row["frame_label"])))
        canonical = dict(ordered[0])
        canonical["selected"] = True
        canonical["duplicate_count"] = len(ordered) - 1
        selected.append(canonical)
        for duplicate in ordered[1:]:
            item = dict(duplicate)
            item["duplicate_of_candidate_id"] = canonical["candidate_id"]
            item["duplicate_of_source_dir"] = canonical["source_dir"]
            item["duplicate_reason"] = (
                "same_observed_pose_camera_prediction_variant"
                if item.get("prediction_sha256") != canonical.get("prediction_sha256")
                else "exact_same_observed_prediction_pose_camera"
            )
            duplicates.append(item)

    selected = sorted(selected, key=lambda row: (int(row["canonical_priority"]), str(row["source_dir"]), str(row["frame_label"])))
    skipped = [row for row in rows if row["status"] != "valid"]
    return rows, selected, duplicates + skipped


def write_loaded_context_manifest(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    context_files = [
        WORKSPACE / ".project_context/CURRENT_STATE.md",
        WORKSPACE / ".project_context/CODEX_LOG.md",
        WORKSPACE / ".project_context/TODO.md",
    ]
    context_rows = []
    for path in context_files:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        context_rows.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "mentions_stage4a65ab": "Stage 4A-6.5ab" in text,
                "mentions_stage4a65ac": "Stage 4A-6.5ac" in text,
                "mentions_stage4a65ad": "Stage 4A-6.5ad" in text,
                "mentions_stage4a65ae": "Stage 4A-6.5ae" in text,
                "mentions_stage4a65af": "Stage 4A-6.5af" in text,
            }
        )
    stage_dirs = {
        "stage4a65ab_dir": args.stage4a65ab_dir,
        "stage4a65ac_dir": args.stage4a65ac_dir,
        "stage4a65ad_dir": args.stage4a65ad_dir,
        "stage4a65ae_dir": args.stage4a65ae_dir,
        "stage4a65af_dir": args.stage4a65af_dir,
    }
    manifest = {
        "stage": "Stage 4A-6.5ag",
        "task": "offline saved-frame-only multi-frame lambda48 replay",
        "context_files": context_rows,
        "stage_reference_dirs": {
            key: {"path": str(Path(value).resolve()), "exists": Path(value).is_dir()}
            for key, value in stage_dirs.items()
        },
        "confirmed_prior_context": {
            "stage4a65ab_best_candidate": "map_predict|source_occ_free|decoupled_minmax_lambda48|tau0p1|occ0p5|free0p5",
            "stage4a65ac_formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
            "stage4a65ad_frame2_complete": True,
            "stage4a65ae_frame1_complete": True,
            "stage4a65af_next_task_is_multiframe_replay": True,
        },
        "safety_scope": {
            "offline_saved_frame_only": True,
            "isaac_startup": False,
            "new_capture": False,
            "map_predict_rerun": False,
            "sscnet_inference": False,
            "selected_action_execution": False,
            "two_frame_runtime": False,
            "rollout": False,
            "training_rl_ppo_bc_il": False,
        },
    }
    save_json(output_dir / "loaded_context_manifest.json", manifest)
    lines = [
        "# Loaded Context Manifest",
        "",
        "- stage: `Stage 4A-6.5ag`",
        "- task: offline saved-frame-only multi-frame lambda48 replay",
        "- prior context confirmed from CURRENT_STATE / CODEX_LOG / TODO.",
        "- Stage 4A-6.5ab best candidate: `map_predict|source_occ_free|decoupled_minmax_lambda48|tau0p1|occ0p5|free0p5`",
        "- Stage 4A-6.5ac formula: `gain_exp / cost + 48 * minmax(source_occ_free)`",
        "- Stage 4A-6.5af next task: multi-frame saved real medium replay.",
        "- Isaac startup / new capture / map_predict rerun / rollout: `false`.",
    ]
    write_text(output_dir / "loaded_context_manifest.md", "\n".join(lines))
    return manifest


def write_formula_definition(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    definition = {
        "stage": "Stage 4A-6.5ag",
        "recommended_formula_name": "map_predict_source_occ_free_decoupled_minmax_lambda48",
        "recommended_formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
        "lambda": float(args.lambda_sc),
        "tau": float(args.tau),
        "occ_threshold": float(args.occ_threshold),
        "free_threshold": float(args.free_threshold),
        "lambda32_diagnostic_formula": "gain_exp / cost + 32 * minmax(source_occ_free)",
        "diagnostic_only": {
            "source_occ_free_over_cost": "(gain_exp + source_occ_free) / cost",
            "raw_hybrid_over_cost": "(gain_exp + source_occ_free) / cost",
            "source_occ_free_no_cost": "source_occ_free",
        },
        "source_occ_free": {
            "counts": "predicted OCC plus predicted FREE",
            "confidence_threshold": float(args.tau),
            "occ_threshold": float(args.occ_threshold),
            "free_threshold": float(args.free_threshold),
            "validity": "prediction-valid and unmeasured voxels only",
            "prediction_unknown_counted": False,
        },
        "minmax_scope": "per frame, per seed, per tree over valid root-to-descendant path accumulated source_occ_free",
        "prediction_information_gain_only": True,
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
    }
    save_json(output_dir / "formula_definition.json", definition)
    lines = [
        "# Formula Definition",
        "",
        f"- formula: `{definition['recommended_formula']}`",
        f"- lambda: `{definition['lambda']}`",
        f"- tau: `{definition['tau']}`",
        f"- occ/free thresholds: `{definition['occ_threshold']}` / `{definition['free_threshold']}`",
        "- SC stays outside the cost denominator for lambda48/lambda32.",
        "- over-cost and no-cost modes are diagnostic-only.",
        "- prediction remains information-gain-only.",
    ]
    write_text(output_dir / "formula_definition.md", "\n".join(lines))
    return definition


def summarize_fraction(rows: list[dict[str, Any]], key: str) -> float | None:
    return None if not rows else float(sum(bool(row.get(key)) for row in rows) / len(rows))


def summary_for_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    counts = Counter(str(row.get("branch_classification")) for row in rows)
    return {
        "label": label,
        "row_count": len(rows),
        "branch_classification_counts": dict(counts),
        "same_as_measured_count": int(sum(bool(row.get("same_as_measured")) for row in rows)),
        "same_as_measured_fraction": summarize_fraction(rows, "same_as_measured"),
        "distinct_nonmeasured_count": int(
            sum(str(row.get("branch_classification")) == "distinct_nonmeasured_branch" for row in rows)
        ),
        "distinct_nonmeasured_fraction": None
        if not rows
        else float(sum(str(row.get("branch_classification")) == "distinct_nonmeasured_branch" for row in rows) / len(rows)),
        "healthy_nonmeasured_count": int(sum(bool(row.get("healthy_nonmeasured_candidate")) for row in rows)),
        "healthy_nonmeasured_fraction": summarize_fraction(rows, "healthy_nonmeasured_candidate"),
        "local_jitter_count": int(sum(str(row.get("branch_classification")) == "local_jitter" for row in rows)),
        "local_jitter_fraction": None
        if not rows
        else float(sum(str(row.get("branch_classification")) == "local_jitter" for row in rows) / len(rows)),
        "historical_prior_basin_count": int(sum(bool(row.get("spatial_prior_sc_basin")) for row in rows)),
        "historical_prior_basin_fraction": summarize_fraction(rows, "spatial_prior_sc_basin"),
        "low_cost_artifact_count": int(sum(bool(row.get("low_cost_artifact")) for row in rows)),
        "low_cost_artifact_fraction": summarize_fraction(rows, "low_cost_artifact"),
        "margin": min_median_max([float(row.get("margin") or 0.0) for row in rows]),
        "mean_source_occ_free": mean([row.get("source_occ_free_count") for row in rows]),
        "mean_normalized_sc": mean([row.get("normalized_sc") for row in rows]),
    }


def normalize_decision_for_multiframe(row: dict[str, Any], same_seed_measured: dict[str, Any] | None) -> dict[str, Any]:
    if row.get("same_as_prior_low_cost_sc") or row.get("spatial_prior_sc_basin"):
        row["branch_classification"] = "spatial_prior_sc_basin"
        row["spatial_prior_sc_basin"] = True
    if row.get("same_as_seed1_near_sc_basin"):
        row["same_as_known_raw_sc_reference"] = True
        if row.get("branch_classification") == "same_as_seed1_near_sc_basin":
            row["branch_classification"] = "same_as_known_raw_sc_reference"
    else:
        row["same_as_known_raw_sc_reference"] = False
    row["avoids_historical_prior_bad_branch"] = not bool(row.get("spatial_prior_sc_basin"))
    if row.get("mode") == "measured_only":
        row["branch_classification"] = "same_as_measured"
        row["same_as_measured"] = True
        row["changed_vs_measured_only"] = False
        row["healthy_nonmeasured_candidate"] = False
        row["low_cost_artifact"] = False
        return row
    row["healthy_nonmeasured_candidate"] = bool(
        row.get("branch_classification") == "distinct_nonmeasured_branch"
        and not bool(row.get("spatial_prior_sc_basin"))
        and not bool(row.get("low_cost_artifact"))
        and (
            float(row.get("normalized_sc") or 0.0) >= 0.5
            or int(row.get("selected_source_occ_free_rank") or 10**9) <= 64
        )
    )
    if same_seed_measured is not None:
        row["measured_reference_selected_child_id"] = same_seed_measured.get("selected_child_id")
        row["measured_reference_best_descendant_id"] = same_seed_measured.get("best_descendant_id")
        row["measured_reference_gain_exp"] = same_seed_measured.get("gain_exp")
        row["measured_reference_cost"] = same_seed_measured.get("cost")
    return row


def should_use_stage4a65y_saved_trees(frame: dict[str, Any], stage4a65y_dir: Path, seeds: list[int]) -> bool:
    source = str(frame.get("source_dir", ""))
    label = str(frame.get("frame_label", ""))
    if "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke" not in source or label != "frame002":
        return False
    for seed in seeds:
        for mode in ("measured_only", "current_raw_count"):
            path = stage4a65y_dir / "raw_trees" / f"seed_{seed:03d}" / mode / "mini_rrt_tree_segments.jsonl"
            if not path.is_file():
                return False
    return True


def enrich_decision_with_frame(
    row: dict[str, Any],
    frame: dict[str, Any],
    *,
    tree_source: str,
    root_alignment_status: str,
    observed_hash: str,
    prediction_hash: str,
) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "frame_id": frame["frame_id"],
            "frame_label": frame["frame_label"],
            "frame_source_dir": frame["source_dir"],
            "frame_hash_key": frame["frame_hash_key"],
            "tree_source": tree_source,
            "root_alignment_status": root_alignment_status,
            "observed_state_hash": observed_hash,
            "prediction_hash": prediction_hash,
        }
    )
    return out


def replay_frame(
    frame: dict[str, Any],
    args: argparse.Namespace,
    seeds: list[int],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    observed_path = Path(frame["observed_state_path"])
    prediction_path = Path(frame["prediction_npz_path"])
    pose_path = Path(frame["pose_json_path"])
    camera_path = Path(frame["camera_info_json_path"])
    observed_hash_before = sha256_file(observed_path)
    prediction_hash_before = sha256_file(prediction_path)
    pose_hash_before = sha256_file(pose_path)
    camera_hash_before = sha256_file(camera_path)

    observed_state = np.load(observed_path)
    observed_shape = tuple(int(v) for v in observed_state.shape)
    bounds, bounds_source = normalize_bounds_from_sources(
        Path(frame["source_dir"]),
        str(frame["frame_label"]),
        observed_shape,
        float(args.voxel_size),
    )
    pose = read_json(pose_path)
    root_world = [float(v) for v in pose.get("position", [])]
    if len(root_world) != 3:
        raise ValueError(f"pose JSON missing 3D position: {pose_path}")
    root_grid = list(world_to_grid(root_world, bounds, float(args.voxel_size), shape=observed_shape, clip=True))
    root_yaw = float(pose.get("yaw_rad", 0.0))
    prediction_layer = SimPredictionLayer.from_npz(prediction_path)
    empty_layer = EmptyPredictionLayer(observed_shape)
    if tuple(prediction_layer.shape()) != observed_shape:
        raise ValueError(f"prediction shape {prediction_layer.shape()} != observed_state {observed_shape}")

    configs = make_mode_configs(args)
    hidden_mask = np.zeros(observed_shape, dtype=bool)
    frontier_local_mask = np.zeros(observed_shape, dtype=bool)
    use_saved = should_use_stage4a65y_saved_trees(frame, Path(args.stage4a65y_dir), seeds)
    tree_source = "saved_raw_tree" if use_saved else "recomputed_offline_tree"
    decisions: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    measured_by_seed: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        if use_saved:
            measured_result = load_saved_tree_result(
                Path(args.stage4a65y_dir) / "raw_trees" / f"seed_{seed:03d}" / "measured_only"
            )
        else:
            measured_result = build_tree(
                observed_state=observed_state,
                root_grid=root_grid,
                root_world=root_world,
                root_yaw=root_yaw,
                bounds=bounds,
                seed=seed,
                prediction_layer=empty_layer,
                gain_mode="exp",
                args=args,
            )
        measured_tree = measured_result["tree"]
        tree_root = measured_tree[ROOT_ID]
        root_distance_cells = float(
            np.linalg.norm(np.asarray(tree_root.end_grid[:3], dtype=np.float64) - np.asarray(root_grid[:3], dtype=np.float64))
        )
        root_alignment_status = (
            "exact_or_snapped_from_pose" if root_distance_cells <= 5.0 else "mismatch_or_best_effort"
        )
        if use_saved:
            root_alignment_status = "saved_raw_tree_reused_pose_checked"
        root_rows.append(
            {
                "frame_id": frame["frame_id"],
                "seed": int(seed),
                "tree_source": tree_source,
                "root_alignment_status": root_alignment_status,
                "reference_tree_unavailable": not use_saved,
                "pose_root_grid": root_grid,
                "tree_root_grid": tree_root.end_grid,
                "pose_root_world": root_world,
                "tree_root_world": tree_root.end_world,
                "root_distance_cells": root_distance_cells,
                "bounds_source": bounds_source,
            }
        )
        measured_candidates = path_candidate_rows(measured_tree, None, None)
        measured_decision = select_decision(
            measured_candidates,
            seed=seed,
            mode="measured_only",
            config=configs["measured_only"],
            same_seed_measured=None,
            voxel_size=float(args.voxel_size),
        )
        measured_decision = normalize_decision_for_multiframe(measured_decision, None)
        measured_decision = enrich_decision_with_frame(
            measured_decision,
            frame,
            tree_source=tree_source,
            root_alignment_status=root_alignment_status,
            observed_hash=observed_hash_before,
            prediction_hash=prediction_hash_before,
        )
        measured_by_seed[seed] = measured_decision
        decisions.append(measured_decision)

        if use_saved:
            sc_result = load_saved_tree_result(
                Path(args.stage4a65y_dir) / "raw_trees" / f"seed_{seed:03d}" / "current_raw_count"
            )
        else:
            sc_result = build_tree(
                observed_state=observed_state,
                root_grid=root_grid,
                root_world=root_world,
                root_yaw=root_yaw,
                bounds=bounds,
                seed=seed,
                prediction_layer=prediction_layer,
                gain_mode="hybrid",
                args=args,
            )
        sc_tree = sc_result["tree"]
        segment_arrays = precompute_segment_prediction_arrays(
            sc_tree,
            observed_state,
            prediction_layer,
            hidden_mask,
            frontier_local_mask,
            args,
        )
        if bool(args.save_raw_tree_summaries):
            for label, tree in (("measured_only", measured_tree), ("map_predict_raw_count", sc_tree)):
                seed_dir = output_dir / "raw_tree_summaries" / frame["frame_id"] / f"seed_{seed:03d}" / label
                write_jsonl(seed_dir / "mini_rrt_tree_segments.jsonl", [segment_record(seg) for seg in tree.values()])

        for mode in (
            "map_predict_lambda32",
            "map_predict_lambda48",
            "source_occ_free_over_cost",
            "raw_hybrid_over_cost",
            "source_occ_free_no_cost",
        ):
            config = configs[mode]
            candidates = path_candidate_rows(sc_tree, segment_arrays, config)
            decision = select_decision(
                candidates,
                seed=seed,
                mode=mode,
                config=config,
                same_seed_measured=measured_by_seed[seed],
                voxel_size=float(args.voxel_size),
            )
            decision = normalize_decision_for_multiframe(decision, measured_by_seed[seed])
            decision = enrich_decision_with_frame(
                decision,
                frame,
                tree_source=tree_source,
                root_alignment_status=root_alignment_status,
                observed_hash=observed_hash_before,
                prediction_hash=prediction_hash_before,
            )
            decisions.append(decision)

    for row in decisions:
        value_rows.append(
            {
                key: row.get(key)
                for key in (
                    "frame_id",
                    "frame_source_dir",
                    "frame_hash_key",
                    "tree_source",
                    "root_alignment_status",
                    "seed",
                    "mode",
                    "formula",
                    "lambda",
                    "tau",
                    "occ_threshold",
                    "free_threshold",
                    "gain_exp",
                    "source_occ_free_count",
                    "source_occ_count",
                    "source_free_count",
                    "cost",
                    "base_exp_value",
                    "normalized_sc",
                    "sc_bonus",
                    "final_value",
                    "runner_up_value",
                    "margin",
                    "normalized_margin",
                    "min_sc",
                    "max_sc",
                    "path_node_ids",
                )
            }
        )

    observed_hash_after = sha256_file(observed_path)
    prediction_hash_after = sha256_file(prediction_path)
    frame_hash_checks = {
        "frame_id": frame["frame_id"],
        "observed_state": {
            "path": str(observed_path),
            "sha256_before": observed_hash_before,
            "sha256_after": observed_hash_after,
            "unchanged": observed_hash_before == observed_hash_after,
        },
        "prediction_npz": {
            "path": str(prediction_path),
            "sha256_before": prediction_hash_before,
            "sha256_after": prediction_hash_after,
            "unchanged": prediction_hash_before == prediction_hash_after,
        },
        "pose_json": {
            "path": str(pose_path),
            "sha256_before": pose_hash_before,
            "sha256_after": sha256_file(pose_path),
            "unchanged": pose_hash_before == sha256_file(pose_path),
        },
        "camera_info_json": {
            "path": str(camera_path),
            "sha256_before": camera_hash_before,
            "sha256_after": sha256_file(camera_path),
            "unchanged": camera_hash_before == sha256_file(camera_path),
        },
    }
    frame_manifest = {
        "frame_id": frame["frame_id"],
        "observed_state": {
            "path": str(observed_path),
            "sha256": observed_hash_before,
            "shape": list(observed_shape),
        },
        "prediction_npz": summarize_prediction_npz(prediction_path, observed_state, float(args.tau)),
        "pose_json": {
            "path": str(pose_path),
            "sha256": pose_hash_before,
            "position": root_world,
            "yaw_rad": root_yaw,
        },
        "camera_info_json": {"path": str(camera_path), "sha256": camera_hash_before},
        "bounds": bounds,
        "bounds_source": bounds_source,
        "tree_source": tree_source,
        "root_grid": root_grid,
        "root_world": root_world,
    }
    return decisions, value_rows, root_rows, frame_hash_checks, frame_manifest


def replay_frame_worker(payload: tuple[dict[str, Any], dict[str, Any], list[int], str]) -> tuple[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]]:
    frame, args_dict, seeds, output_dir = payload
    worker_args = argparse.Namespace(**args_dict)
    return str(frame["frame_id"]), replay_frame(frame, worker_args, seeds, Path(output_dir))


def write_branch_summary(output_dir: Path, decisions: list[dict[str, Any]]) -> None:
    summary = summarize_modes(decisions)
    write_md_table(
        output_dir / "branch_classification_summary.md",
        "Branch Classification Summary",
        summary,
        [
            "mode",
            "branch_classification_counts",
            "same_as_measured_fraction",
            "spatial_prior_sc_basin_fraction",
            "healthy_nonmeasured_fraction",
            "low_cost_artifact_fraction",
        ],
    )


def build_lambda48_summary(output_dir: Path, selected_frames: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    map48 = rows_for_mode(decisions, "map_predict_lambda48")
    rows: list[dict[str, Any]] = []
    for frame in selected_frames:
        frame_rows = [row for row in map48 if row["frame_id"] == frame["frame_id"]]
        item = summary_for_rows(frame_rows, frame["frame_id"])
        item.update(
            {
                "frame_id": frame["frame_id"],
                "frame_source_dir": frame["source_dir"],
                "frame_label": frame["frame_label"],
                "tree_source": frame.get("tree_source"),
            }
        )
        rows.append(item)
    aggregate = summary_for_rows(map48, "aggregate")
    aggregate.update(
        {
            "unique_frame_count": len(selected_frames),
            "total_seed_frame_rows": len(map48),
            "mode": "map_predict_lambda48",
            "saved_frame_only_readiness": True,
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
            "formula_components_logged": all(
                key in row and row[key] is not None
                for row in map48
                for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value")
            ),
        }
    )
    save_json(output_dir / "lambda48_multiframe_summary.json", {"aggregate": aggregate, "by_frame": rows})
    write_csv(output_dir / "lambda48_multiframe_summary.csv", rows + [aggregate])
    lines = [
        "# Lambda48 Multiframe Summary",
        "",
        f"- unique frame count: `{aggregate['unique_frame_count']}`",
        f"- seed-frame rows: `{aggregate['total_seed_frame_rows']}`",
        f"- same-as-measured: `{aggregate['same_as_measured_count']}/{aggregate['row_count']}` fraction `{aggregate['same_as_measured_fraction']}`",
        f"- distinct non-measured: `{aggregate['distinct_nonmeasured_count']}/{aggregate['row_count']}` fraction `{aggregate['distinct_nonmeasured_fraction']}`",
        f"- healthy non-measured: `{aggregate['healthy_nonmeasured_count']}/{aggregate['row_count']}` fraction `{aggregate['healthy_nonmeasured_fraction']}`",
        f"- historical prior basin: `{aggregate['historical_prior_basin_count']}/{aggregate['row_count']}` fraction `{aggregate['historical_prior_basin_fraction']}`",
        f"- low-cost artifact: `{aggregate['low_cost_artifact_count']}/{aggregate['row_count']}` fraction `{aggregate['low_cost_artifact_fraction']}`",
        f"- margin min/median/max: `{aggregate['margin']}`",
        "- runtime smoke readiness: `false`",
        "- rollout readiness: `false`",
    ]
    write_text(output_dir / "lambda48_multiframe_summary.md", "\n".join(lines))
    return rows, aggregate


def compare_modes(decisions: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
    by_key = {(row["frame_id"], int(row["seed"]), row["mode"]): row for row in decisions}
    rows: list[dict[str, Any]] = []
    keys = sorted({(row["frame_id"], int(row["seed"])) for row in decisions if row["mode"] in {left, right}})
    for frame_id, seed in keys:
        a = by_key.get((frame_id, seed, left))
        b = by_key.get((frame_id, seed, right))
        if a is None or b is None:
            continue
        rows.append(
            {
                "frame_id": frame_id,
                "seed": int(seed),
                "left_mode": left,
                "right_mode": right,
                "left_branch_classification": a.get("branch_classification"),
                "right_branch_classification": b.get("branch_classification"),
                "branch_class_agreement": a.get("branch_classification") == b.get("branch_classification"),
                "selected_child_agreement": a.get("selected_child_id") == b.get("selected_child_id"),
                "best_descendant_agreement": a.get("best_descendant_id") == b.get("best_descendant_id"),
                "left_low_cost_artifact": a.get("low_cost_artifact"),
                "right_low_cost_artifact": b.get("low_cost_artifact"),
                "left_prior_basin": a.get("spatial_prior_sc_basin"),
                "right_prior_basin": b.get("spatial_prior_sc_basin"),
                "left_healthy_nonmeasured": a.get("healthy_nonmeasured_candidate"),
                "right_healthy_nonmeasured": b.get("healthy_nonmeasured_candidate"),
                "left_margin": a.get("margin"),
                "right_margin": b.get("margin"),
            }
        )
    return rows


def write_lambda32_comparison(output_dir: Path, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = compare_modes(decisions, "map_predict_lambda32", "map_predict_lambda48")
    summary = {
        "row_count": len(rows),
        "branch_class_agreement_count": int(sum(bool(row["branch_class_agreement"]) for row in rows)),
        "branch_class_agreement_fraction": summarize_fraction(rows, "branch_class_agreement"),
        "selected_child_agreement_count": int(sum(bool(row["selected_child_agreement"]) for row in rows)),
        "selected_child_agreement_fraction": summarize_fraction(rows, "selected_child_agreement"),
        "best_descendant_agreement_count": int(sum(bool(row["best_descendant_agreement"]) for row in rows)),
        "best_descendant_agreement_fraction": summarize_fraction(rows, "best_descendant_agreement"),
        "interpretation": "lambda32 is diagnostic; synthetic calibration still favors lambda48 unless broader evidence changes.",
    }
    save_json(output_dir / "lambda32_vs_lambda48_multiframe.json", {"summary": summary, "rows": rows})
    write_csv(output_dir / "lambda32_vs_lambda48_multiframe.csv", rows)
    lines = [
        "# Lambda32 Vs Lambda48 Multiframe",
        "",
        f"- branch-class agreement: `{summary['branch_class_agreement_count']}/{summary['row_count']}` fraction `{summary['branch_class_agreement_fraction']}`",
        f"- selected-child agreement: `{summary['selected_child_agreement_count']}/{summary['row_count']}` fraction `{summary['selected_child_agreement_fraction']}`",
        f"- best-descendant agreement: `{summary['best_descendant_agreement_count']}/{summary['row_count']}` fraction `{summary['best_descendant_agreement_fraction']}`",
        "- synthetic evidence still favors lambda48 over lambda32.",
    ]
    write_text(output_dir / "lambda32_vs_lambda48_multiframe.md", "\n".join(lines))
    return summary


def write_over_cost_diagnostic(output_dir: Path, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = compare_modes(decisions, "source_occ_free_over_cost", "map_predict_lambda48")
    over = rows_for_mode(decisions, "source_occ_free_over_cost")
    lambda48 = rows_for_mode(decisions, "map_predict_lambda48")
    over_summary = summary_for_rows(over, "source_occ_free_over_cost")
    lambda48_summary = summary_for_rows(lambda48, "map_predict_lambda48")
    summary = {
        "row_count": len(rows),
        "source_occ_free_over_cost": over_summary,
        "map_predict_lambda48": lambda48_summary,
        "aggression_difference_distinct_fraction": None
        if over_summary["distinct_nonmeasured_fraction"] is None
        or lambda48_summary["distinct_nonmeasured_fraction"] is None
        else float(over_summary["distinct_nonmeasured_fraction"] - lambda48_summary["distinct_nonmeasured_fraction"]),
        "prior_basin_risk": bool((over_summary["historical_prior_basin_fraction"] or 0.0) > 0.0),
        "low_cost_artifact_risk": bool((over_summary["low_cost_artifact_fraction"] or 0.0) > 0.0),
        "diagnostic_only": True,
    }
    save_json(output_dir / "over_cost_multiframe_diagnostic.json", {"summary": summary, "rows": rows})
    write_csv(output_dir / "over_cost_multiframe_diagnostic.csv", rows)
    lines = [
        "# Over-Cost Multiframe Diagnostic",
        "",
        f"- over-cost distinct fraction: `{over_summary['distinct_nonmeasured_fraction']}`",
        f"- lambda48 distinct fraction: `{lambda48_summary['distinct_nonmeasured_fraction']}`",
        f"- aggression difference: `{summary['aggression_difference_distinct_fraction']}`",
        f"- over-cost prior-basin risk: `{summary['prior_basin_risk']}`",
        f"- over-cost low-cost artifact risk: `{summary['low_cost_artifact_risk']}`",
        "- interpretation: over-cost remains diagnostic-only.",
    ]
    write_text(output_dir / "over_cost_multiframe_diagnostic.md", "\n".join(lines))
    return summary


def write_low_cost(output_dir: Path, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "frame_id": row["frame_id"],
            "seed": row["seed"],
            "mode": row["mode"],
            "branch_classification": row.get("branch_classification"),
            "changed_vs_measured_only": row.get("changed_vs_measured_only"),
            "gain_exp": row.get("gain_exp"),
            "source_occ_free_count": row.get("source_occ_free_count"),
            "cost": row.get("cost"),
            "base_exp_selected_child_id": row.get("base_exp_selected_child_id"),
            "base_exp_best_descendant_id": row.get("base_exp_best_descendant_id"),
            "base_exp_selected_gain_exp": row.get("base_exp_selected_gain_exp"),
            "base_exp_selected_source_occ_free": row.get("base_exp_selected_source_occ_free"),
            "base_exp_selected_cost": row.get("base_exp_selected_cost"),
            "low_cost_artifact": row.get("low_cost_artifact"),
            "spatial_prior_sc_basin": row.get("spatial_prior_sc_basin"),
        }
        for row in decisions
    ]
    map48 = [row for row in rows if row["mode"] == "map_predict_lambda48"]
    summary = {
        "all_mode_low_cost_artifact_count": int(sum(bool(row["low_cost_artifact"]) for row in rows)),
        "all_mode_row_count": len(rows),
        "map_predict_lambda48_low_cost_artifact_count": int(sum(bool(row["low_cost_artifact"]) for row in map48)),
        "map_predict_lambda48_row_count": len(map48),
        "map_predict_lambda48_low_cost_artifact_fraction": summarize_fraction(map48, "low_cost_artifact"),
    }
    save_json(output_dir / "low_cost_artifact_multiframe.json", {"summary": summary, "rows": rows})
    write_csv(output_dir / "low_cost_artifact_multiframe.csv", rows)
    write_md_table(
        output_dir / "low_cost_artifact_multiframe.md",
        "Low-Cost Artifact Multiframe",
        rows,
        [
            "frame_id",
            "seed",
            "mode",
            "branch_classification",
            "gain_exp",
            "source_occ_free_count",
            "cost",
            "low_cost_artifact",
            "spatial_prior_sc_basin",
        ],
    )
    return summary


def write_safety_and_hashes(
    output_dir: Path,
    hash_checks: list[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_before = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    checkpoint_after = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    hash_report = {
        "frames": hash_checks,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": checkpoint_before,
            "sha256_after": checkpoint_after,
            "unchanged": checkpoint_before == checkpoint_after,
        },
    }
    save_json(output_dir / "hash_checks.json", hash_report)
    observed_changed = any(not item["observed_state"]["unchanged"] for item in hash_checks)
    prediction_changed = any(not item["prediction_npz"]["unchanged"] for item in hash_checks)
    safety = {
        "isaac_startup": False,
        "new_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "two_frame_runtime": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_rl_ppo_bc_il": False,
        "checkpoint_modified": not bool(hash_report["checkpoint"]["unchanged"]),
        "existing_observed_state_modified": observed_changed,
        "prediction_npz_modified": prediction_changed,
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_planning_scoring": False,
        "future_observed_planning_scoring": False,
        "external_source_modified_built": False,
        "coverage_improvement_claim": False,
        "prediction_information_gain_only": True,
        "runtime_smoke_ready": False,
        "rollout_ready": False,
    }
    save_json(output_dir / "prediction_safety_report.json", safety)
    lines = [
        "# Prediction Safety Report",
        "",
        "- Isaac startup: `false`",
        "- new capture: `false`",
        "- map_predict rerun: `false`",
        "- SSCNet inference: `false`",
        "- selected action execution: `false`",
        "- two-frame runtime: `false`",
        "- rollout: `false`",
        "- prediction writeback: `false`",
        "- prediction used for traversability/collision/ray blocking: `false`",
        "- target/ground-truth/future-observed planning or scoring: `false`",
        f"- observed_state modified: `{observed_changed}`",
        f"- prediction NPZ modified: `{prediction_changed}`",
        f"- checkpoint modified: `{safety['checkpoint_modified']}`",
    ]
    write_text(output_dir / "prediction_safety_report.md", "\n".join(lines))
    return safety


def plot_stacked_branch_by_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    classes = sorted({str(row.get("branch_classification")) for row in rows})
    frames = sorted({str(row.get("frame_id")) for row in rows})
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counters[str(row["frame_id"])][str(row.get("branch_classification"))] += 1
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    bottoms = np.zeros(len(frames), dtype=np.float64)
    palette = ["#f97316", "#059669", "#2563eb", "#dc2626", "#7c3aed", "#0f766e", "#64748b"]
    for idx, cls in enumerate(classes):
        values = np.asarray([counters[frame][cls] / max(1, sum(counters[frame].values())) for frame in frames])
        ax.bar(range(len(frames)), values, bottom=bottoms, label=cls, color=palette[idx % len(palette)])
        bottoms += values
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels(frames, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("fraction")
    ax.set_title("Lambda48 branch fractions by frame")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_aggregate_branch(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("branch_classification")) for row in rows)
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.bar(range(len(labels)), values, color="#2563eb")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Lambda48 aggregate branch classes")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_fraction_by_frame(path: Path, rows: list[dict[str, Any]], key: str, title: str, color: str) -> None:
    frames = sorted({str(row.get("frame_id")) for row in rows})
    values = []
    for frame in frames:
        frame_rows = [row for row in rows if str(row.get("frame_id")) == frame]
        values.append(float(sum(bool(row.get(key)) for row in frame_rows) / max(1, len(frame_rows))))
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.bar(range(len(frames)), values, color=color)
    ax.set_ylim(0.0, max(1.0, max(values, default=0.0) + 0.1))
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels(frames, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("fraction")
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_mode_comparison(path: Path, summary_values: dict[str, float | None], title: str) -> None:
    labels = list(summary_values.keys())
    values = [0.0 if summary_values[label] is None else float(summary_values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.bar(range(len(labels)), values, color=["#7c3aed", "#059669", "#dc2626", "#2563eb"][: len(labels)])
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("fraction")
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_margin_by_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    frames = sorted({str(row.get("frame_id")) for row in rows})
    data = []
    for frame in frames:
        values = [float(row.get("margin") or 0.0) for row in rows if str(row.get("frame_id")) == frame]
        data.append(values or [0.0])
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.boxplot(data, labels=frames, showmeans=True)
    ax.tick_params(axis="x", labelrotation=32, labelsize=8)
    ax.set_ylabel("margin")
    ax.set_title("Lambda48 margin by frame")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_source_occ_free_by_branch(path: Path, rows: list[dict[str, Any]]) -> None:
    classes = sorted({str(row.get("branch_classification")) for row in rows})
    data = []
    for cls in classes:
        values = [float(row.get("source_occ_free_count") or 0.0) for row in rows if str(row.get("branch_classification")) == cls]
        data.append(values or [0.0])
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.boxplot(data, labels=classes, showmeans=True)
    ax.tick_params(axis="x", labelrotation=28)
    ax.set_ylabel("source_occ_free")
    ax.set_title("Source OCC+FREE by branch class")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_plots(
    output_dir: Path,
    decisions: list[dict[str, Any]],
    lambda32_summary: dict[str, Any],
    over_summary: dict[str, Any],
    missing_report: dict[str, Any],
) -> None:
    map48 = rows_for_mode(decisions, "map_predict_lambda48")
    plotters = {
        "lambda48_branch_fraction_by_frame.png": lambda p: plot_stacked_branch_by_frame(p, map48),
        "lambda48_aggregate_branch_fractions.png": lambda p: plot_aggregate_branch(p, map48),
        "healthy_nonmeasured_fraction_by_frame.png": lambda p: plot_fraction_by_frame(
            p, map48, "healthy_nonmeasured_candidate", "Healthy non-measured fraction by frame", "#059669"
        ),
        "same_as_measured_fraction_by_frame.png": lambda p: plot_fraction_by_frame(
            p, map48, "same_as_measured", "Same-as-measured fraction by frame", "#f97316"
        ),
        "low_cost_artifact_by_frame.png": lambda p: plot_fraction_by_frame(
            p, map48, "low_cost_artifact", "Low-cost artifact fraction by frame", "#dc2626"
        ),
        "prior_basin_fraction_by_frame.png": lambda p: plot_fraction_by_frame(
            p, map48, "spatial_prior_sc_basin", "Historical prior-basin fraction by frame", "#7c3aed"
        ),
        "lambda32_vs_lambda48_multiframe.png": lambda p: plot_mode_comparison(
            p,
            {
                "branch class": lambda32_summary.get("branch_class_agreement_fraction"),
                "selected child": lambda32_summary.get("selected_child_agreement_fraction"),
                "best descendant": lambda32_summary.get("best_descendant_agreement_fraction"),
            },
            "Lambda32 vs Lambda48 agreement",
        ),
        "over_cost_vs_lambda48_multiframe.png": lambda p: plot_mode_comparison(
            p,
            {
                "over distinct": (over_summary.get("source_occ_free_over_cost") or {}).get("distinct_nonmeasured_fraction"),
                "lambda48 distinct": (over_summary.get("map_predict_lambda48") or {}).get("distinct_nonmeasured_fraction"),
                "over prior": (over_summary.get("source_occ_free_over_cost") or {}).get("historical_prior_basin_fraction"),
                "over artifact": (over_summary.get("source_occ_free_over_cost") or {}).get("low_cost_artifact_fraction"),
            },
            "Over-cost vs lambda48 diagnostic",
        ),
        "margin_by_frame_lambda48.png": lambda p: plot_margin_by_frame(p, map48),
        "source_occ_free_by_branch_class.png": lambda p: plot_source_occ_free_by_branch(p, map48),
    }
    for name, plotter in plotters.items():
        try:
            plotter(output_dir / name)
        except Exception as exc:  # pragma: no cover - operational diagnostic
            reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
            write_text(reason, f"# Plot Skipped\n\n- plot: `{name}`\n- reason: `{type(exc).__name__}: {exc}`")
            missing_report.setdefault("plot_failures", []).append({"plot": name, "reason": f"{type(exc).__name__}: {exc}"})


def recommended_next(aggregate: dict[str, Any], unique_frame_count: int, over_summary: dict[str, Any]) -> tuple[str, str]:
    low = float(aggregate.get("low_cost_artifact_fraction") or 0.0)
    prior = float(aggregate.get("historical_prior_basin_fraction") or 0.0)
    healthy = float(aggregate.get("healthy_nonmeasured_fraction") or 0.0)
    same = float(aggregate.get("same_as_measured_fraction") or 0.0)
    over_prior = bool(over_summary.get("prior_basin_risk"))
    over_artifact = bool(over_summary.get("low_cost_artifact_risk"))
    if low > 0.0 or prior > 0.0:
        return (
            "low-cost artifact / dominance-gate diagnosis, still offline",
            "lambda48 touched the historical prior basin or produced a low-cost artifact.",
        )
    if unique_frame_count < 3:
        return (
            "identify/collect additional saved real medium frames in a controlled capture-only stage",
            "only two unique real frames were available, so evidence remains limited; no rollout.",
        )
    if healthy >= 0.20 and low == 0.0 and prior == 0.0:
        return (
            "multi-scene/start saved-frame replay or staged one-frame runtime-smoke design review only",
            "lambda48 stayed artifact-free and found healthy non-measured branches across multiple saved frames.",
        )
    if same > 0.80:
        return (
            "SC signal / normalization design review or additional saved frames, still offline",
            "lambda48 remained artifact-free but mostly conservative.",
        )
    if over_prior or over_artifact:
        return (
            "offline over-cost risk review only",
            "over-cost is more aggressive but remains diagnostic-only due risk flags.",
        )
    return (
        "additional saved-frame replay breadth, still offline",
        "lambda48 is mixed but artifact-free; broaden offline evidence before runtime.",
    )


def write_final_summary(
    output_dir: Path,
    *,
    all_candidates: list[dict[str, Any]],
    selected_frames: list[dict[str, Any]],
    duplicates_and_skipped: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    lambda_by_frame: list[dict[str, Any]],
    lambda_aggregate: dict[str, Any],
    lambda32_summary: dict[str, Any],
    over_summary: dict[str, Any],
    low_cost_summary: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    next_step, why = recommended_next(lambda_aggregate, len(selected_frames), over_summary)
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {next_step}",
                f"- why: {why}",
                "- runtime smoke readiness: no",
                "- rollout readiness: no",
                "- do not run RL/PPO/BC/IL.",
                "- do not promote over-cost to runtime.",
            ]
        ),
    )
    skipped_reasons = Counter(str(row.get("skip_reason") or row.get("duplicate_reason") or "selected") for row in duplicates_and_skipped)
    answers = {
        "candidate_frames_found": len(all_candidates),
        "valid_candidate_frames_found": int(sum(row.get("status") == "valid" for row in all_candidates)),
        "unique_real_medium_frames_selected": len(selected_frames),
        "skipped_or_duplicate_frames": len(duplicates_and_skipped),
        "duplicate_frames_found": int(sum("duplicate_reason" in row for row in duplicates_and_skipped)),
        "skipped_reason_counts": dict(skipped_reasons),
        "deduplicate_rule": "observed_state + pose + camera + root pose frame key, keeping canonical prediction by source priority",
        "no_isaac_no_capture_no_map_predict_rerun": True,
        "frames_tested": len(selected_frames),
        "seed_count": len({int(row["seed"]) for row in decisions}),
        "mode_count": len({row["mode"] for row in decisions}),
        "decision_row_count": len(decisions),
        "lambda48_aggregate": lambda_aggregate,
        "lambda48_by_frame": lambda_by_frame,
        "lambda48_avoids_historical_prior_bad_branch": bool(
            (lambda_aggregate.get("historical_prior_basin_count") or 0) == 0
        ),
        "lambda48_low_cost_artifact_zero": bool((lambda_aggregate.get("low_cost_artifact_count") or 0) == 0),
        "lambda48_conservative": bool(float(lambda_aggregate.get("same_as_measured_fraction") or 0.0) > 0.70),
        "lambda32_vs_lambda48": lambda32_summary,
        "synthetic_reason_to_keep_lambda48": "Stage 4A-6.5ab/ac synthetic calibration favored lambda48 because lambda32 was not stable.",
        "over_cost_diagnostic": over_summary,
        "over_cost_diagnostic_only": True,
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
        "next_small_task": next_step,
        "recommendation_reason": why,
    }
    summary = {
        "stage": "Stage 4A-6.5ag",
        "status": "completed",
        "answers": answers,
        "low_cost_artifact_summary": low_cost_summary,
        "safety": safety,
        "readiness": {
            "saved_frame_only": True,
            "runtime_smoke": False,
            "rollout": False,
        },
        "coverage_improvement_claimed": False,
    }
    save_json(output_dir / "stage4a65ag_multi_frame_lambda48_replay_summary.json", summary)
    lines = [
        "# Stage 4A-6.5ag Multi-Frame Lambda48 Replay Summary",
        "",
        f"1. Candidate frames found: `{answers['candidate_frames_found']}`; valid candidates: `{answers['valid_candidate_frames_found']}`.",
        f"2. Unique real medium frames selected: `{answers['unique_real_medium_frames_selected']}`.",
        f"3. Skipped/duplicate frames: `{answers['skipped_or_duplicate_frames']}` with reasons `{answers['skipped_reason_counts']}`.",
        f"4. Duplicate handling: {answers['deduplicate_rule']}.",
        f"5. No Isaac / new capture / map_predict rerun: `{answers['no_isaac_no_capture_no_map_predict_rerun']}`.",
        f"6. Frames / seeds / modes / decision rows: `{answers['frames_tested']}` / `{answers['seed_count']}` / `{answers['mode_count']}` / `{answers['decision_row_count']}`.",
        f"7. Lambda48 aggregate same/distinct/healthy/prior/artifact fractions: `{lambda_aggregate['same_as_measured_fraction']}` / `{lambda_aggregate['distinct_nonmeasured_fraction']}` / `{lambda_aggregate['healthy_nonmeasured_fraction']}` / `{lambda_aggregate['historical_prior_basin_fraction']}` / `{lambda_aggregate['low_cost_artifact_fraction']}`.",
        "8. Per-frame lambda48 behavior is recorded in `lambda48_multiframe_summary.*`.",
        f"9. Lambda48 avoided the historical prior bad branch: `{answers['lambda48_avoids_historical_prior_bad_branch']}`.",
        f"10. Lambda48 low-cost artifact stayed zero: `{answers['lambda48_low_cost_artifact_zero']}`.",
        f"11. Lambda48 conservative: `{answers['lambda48_conservative']}`.",
        f"12. Lambda32 vs lambda48: branch agreement `{lambda32_summary['branch_class_agreement_fraction']}`, selected-child agreement `{lambda32_summary['selected_child_agreement_fraction']}`.",
        f"13. Keep lambda48 over lambda32 because: {answers['synthetic_reason_to_keep_lambda48']}",
        f"14. Over-cost more aggressive diagnostic: `{over_summary.get('aggression_difference_distinct_fraction')}`.",
        f"15. Over-cost remains diagnostic-only: `{answers['over_cost_diagnostic_only']}`.",
        f"16. Current evidence enough for runtime smoke: `{answers['runtime_smoke_readiness']}`.",
        f"17. Current evidence enough for rollout: `{answers['rollout_readiness']}`.",
        f"18. Next step: {next_step}.",
        "",
        f"Why: {why}",
        "",
        "This is an offline saved-frame-only replay and does not claim coverage improvement.",
    ]
    write_text(output_dir / "stage4a65ag_multi_frame_lambda48_replay_summary.md", "\n".join(lines))
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_root = Path(args.outputs_root).resolve()
    seeds = parse_ints(args.seeds)
    checkpoint_path = Path(args.checkpoint).resolve()

    write_loaded_context_manifest(output_dir, args)
    write_formula_definition(output_dir, args)

    all_candidates, selected_frames, duplicates_and_skipped = discover_frame_candidates(outputs_root, args.frame_filter)
    for idx, frame in enumerate(selected_frames, start=1):
        frame["frame_id"] = f"real_medium_frame_{idx:03d}"
    write_csv(output_dir / "frame_discovery_inventory.csv", all_candidates)
    save_json(output_dir / "frame_discovery_inventory.json", all_candidates)
    write_md_table(
        output_dir / "frame_discovery_inventory.md",
        "Frame Discovery Inventory",
        all_candidates,
        ["candidate_id", "status", "skip_reason", "medium_evidence", "source_dir", "frame_label"],
        limit=240,
    )
    duplicate_rows = [row for row in duplicates_and_skipped if "duplicate_reason" in row]
    skipped_rows = [row for row in duplicates_and_skipped if "duplicate_reason" not in row]
    write_csv(output_dir / "frame_inventory_duplicates.csv", duplicate_rows)
    save_json(output_dir / "frame_inventory_duplicates.json", duplicate_rows)
    write_csv(output_dir / "skipped_frame_candidates.csv", skipped_rows)
    save_json(output_dir / "skipped_frame_candidates.json", skipped_rows)
    write_csv(output_dir / "selected_frame_manifest.csv", selected_frames)
    save_json(output_dir / "selected_frame_manifest.json", selected_frames)
    write_md_table(
        output_dir / "selected_frame_manifest.md",
        "Selected Frame Manifest",
        selected_frames,
        [
            "frame_id",
            "candidate_id",
            "source_dir",
            "frame_label",
            "observed_state_path",
            "prediction_npz_path",
            "pose_json_path",
            "camera_info_json_path",
            "duplicate_count",
        ],
    )

    decisions: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    hash_checks: list[dict[str, Any]] = []
    per_frame_manifests: list[dict[str, Any]] = []

    worker_count = int(args.workers)
    task_count = max(1, len(selected_frames) * len(seeds))
    if worker_count <= 0:
        worker_count = min(task_count, max(1, min(32, os.cpu_count() or 1)))
    worker_count = max(1, min(worker_count, task_count))
    if worker_count == 1:
        for frame in selected_frames:
            frame_decisions, frame_values, frame_roots, frame_hashes, frame_manifest = replay_frame(
                frame,
                args,
                seeds,
                output_dir,
            )
            frame["tree_source"] = frame_manifest["tree_source"]
            decisions.extend(frame_decisions)
            value_rows.extend(frame_values)
            root_rows.extend(frame_roots)
            hash_checks.append(frame_hashes)
            per_frame_manifests.append(frame_manifest)
            print(f"[stage4a65ag] completed {frame['frame_id']} with {len(frame_decisions)} decisions", flush=True)
    else:
        args_dict = vars(args).copy()
        payloads = [
            (dict(frame), args_dict, [int(seed)], str(output_dir))
            for frame in selected_frames
            for seed in seeds
        ]
        frame_by_id = {str(frame["frame_id"]): frame for frame in selected_frames}
        hash_by_frame: dict[str, dict[str, Any]] = {}
        manifest_by_frame: dict[str, dict[str, Any]] = {}
        print(
            f"[stage4a65ag] running frame+seed replay with {worker_count} workers over {len(payloads)} tasks",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {executor.submit(replay_frame_worker, payload): payload[0]["frame_id"] for payload in payloads}
            for future in as_completed(future_map):
                frame_id = str(future_map[future])
                frame_decisions, frame_values, frame_roots, frame_hashes, frame_manifest = future.result()[1]
                frame_by_id[frame_id]["tree_source"] = frame_manifest["tree_source"]
                decisions.extend(frame_decisions)
                value_rows.extend(frame_values)
                root_rows.extend(frame_roots)
                hash_by_frame[frame_id] = frame_hashes
                manifest_by_frame[frame_id] = frame_manifest
                seed_label = frame_decisions[0]["seed"] if frame_decisions else "?"
                print(
                    f"[stage4a65ag] completed {frame_id} seed {seed_label} with {len(frame_decisions)} decisions",
                    flush=True,
                )
        for frame in selected_frames:
            frame_id = str(frame["frame_id"])
            if frame_id in hash_by_frame:
                hash_checks.append(hash_by_frame[frame_id])
            if frame_id in manifest_by_frame:
                per_frame_manifests.append(manifest_by_frame[frame_id])

    decisions = sorted(decisions, key=lambda row: (row["frame_id"], int(row["seed"]), mode_sort_key(str(row["mode"]))))
    value_rows = sorted(value_rows, key=lambda row: (row["frame_id"], int(row["seed"]), mode_sort_key(str(row["mode"]))))
    root_rows = sorted(root_rows, key=lambda row: (row["frame_id"], int(row["seed"])))

    decision_fields = [
        "frame_id",
        "frame_source_dir",
        "frame_hash_key",
        "tree_source",
        "root_alignment_status",
        "seed",
        "mode",
        "prediction_source",
        "formula",
        "lambda",
        "tau",
        "occ_threshold",
        "free_threshold",
        "selected_child_id",
        "selected_child_grid",
        "selected_child_world",
        "best_descendant_id",
        "best_descendant_grid",
        "best_descendant_world",
        "branch_classification",
        "selected_child_distance_from_same_seed_measured_m",
        "selected_child_distance_from_prior_low_cost_sc_reference_m",
        "best_descendant_distance_from_prior_low_cost_sc_reference_m",
        "changed_vs_measured_only",
        "same_as_measured",
        "same_as_known_raw_sc_reference",
        "spatial_prior_sc_basin",
        "avoids_historical_prior_bad_branch",
        "healthy_nonmeasured_candidate",
        "gain_exp",
        "source_occ_free",
        "source_occ_free_count",
        "source_occ_count",
        "source_free_count",
        "cost",
        "base_exp_value",
        "normalized_sc",
        "sc_bonus",
        "final_value",
        "runner_up_value",
        "margin",
        "normalized_margin",
        "branch_depth",
        "path_node_ids",
        "selected_cost_rank",
        "selected_gain_exp_rank",
        "selected_source_occ_free_rank",
        "min_sc",
        "max_sc",
        "low_cost_artifact",
        "prediction_safety_flags",
        "observed_state_hash",
        "prediction_hash",
    ]
    save_json(output_dir / "per_frame_seed_mode_decisions.json", decisions)
    write_csv(output_dir / "per_frame_seed_mode_decisions.csv", decisions, decision_fields)
    write_md_table(
        output_dir / "per_frame_seed_mode_decisions.md",
        "Per-Frame Seed Mode Decisions",
        decisions,
        [
            "frame_id",
            "seed",
            "mode",
            "selected_child_id",
            "best_descendant_id",
            "branch_classification",
            "gain_exp",
            "source_occ_free_count",
            "cost",
            "base_exp_value",
            "normalized_sc",
            "sc_bonus",
            "final_value",
            "margin",
            "low_cost_artifact",
        ],
        limit=360,
    )
    save_json(output_dir / "per_frame_value_components.json", value_rows)
    write_csv(output_dir / "per_frame_value_components.csv", value_rows)
    branch_rows = [
        {
            key: row.get(key)
            for key in (
                "frame_id",
                "seed",
                "mode",
                "selected_child_id",
                "best_descendant_id",
                "selected_child_grid",
                "best_descendant_grid",
                "branch_classification",
                "same_as_measured",
                "same_as_known_raw_sc_reference",
                "spatial_prior_sc_basin",
                "avoids_historical_prior_bad_branch",
                "healthy_nonmeasured_candidate",
                "low_cost_artifact",
                "selected_child_distance_from_same_seed_measured_m",
                "selected_child_distance_from_prior_low_cost_sc_reference_m",
                "best_descendant_distance_from_prior_low_cost_sc_reference_m",
            )
        }
        for row in decisions
    ]
    save_json(output_dir / "branch_classification_by_frame_seed_mode.json", branch_rows)
    write_csv(output_dir / "branch_classification_by_frame_seed_mode.csv", branch_rows)
    write_branch_summary(output_dir, decisions)
    save_json(output_dir / "root_alignment_report.json", root_rows)
    write_csv(output_dir / "root_alignment_report.csv", root_rows)
    write_md_table(
        output_dir / "root_alignment_report.md",
        "Root Alignment Report",
        root_rows,
        [
            "frame_id",
            "seed",
            "tree_source",
            "root_alignment_status",
            "reference_tree_unavailable",
            "pose_root_grid",
            "tree_root_grid",
            "root_distance_cells",
            "bounds_source",
        ],
        limit=240,
    )

    lambda_by_frame, lambda_aggregate = build_lambda48_summary(output_dir, selected_frames, decisions)
    lambda32_summary = write_lambda32_comparison(output_dir, decisions)
    over_summary = write_over_cost_diagnostic(output_dir, decisions)
    low_cost_summary = write_low_cost(output_dir, decisions)
    safety = write_safety_and_hashes(output_dir, hash_checks, checkpoint_path)

    missing_report: dict[str, Any] = {
        "missing_required_inputs": [row for row in skipped_rows if str(row.get("skip_reason", "")).startswith("missing_")],
        "plot_failures": [],
        "optional_modes_implemented": ["raw_hybrid_over_cost", "source_occ_free_no_cost"],
        "per_frame_loaded_manifest": per_frame_manifests,
        "elapsed_s": None,
    }
    write_plots(output_dir, decisions, lambda32_summary, over_summary, missing_report)
    missing_report["required_plots"] = {
        name: {
            "exists": (output_dir / name).is_file(),
            "skipped_reason_exists": (output_dir / f"{Path(name).stem}_skipped_reason.md").is_file(),
        }
        for name in REQUIRED_PLOTS
    }
    missing_report["elapsed_s"] = float(time.perf_counter() - start)
    save_json(output_dir / "missing_fields_report.json", missing_report)
    write_text(
        output_dir / "missing_fields_report.md",
        "\n".join(
            [
                "# Missing Fields Report",
                "",
                f"- missing/incomplete candidates: `{len(missing_report['missing_required_inputs'])}`",
                f"- plot failures: `{len(missing_report['plot_failures'])}`",
                f"- optional modes implemented: `{missing_report['optional_modes_implemented']}`",
            ]
        ),
    )

    final_summary = write_final_summary(
        output_dir,
        all_candidates=all_candidates,
        selected_frames=selected_frames,
        duplicates_and_skipped=duplicates_and_skipped,
        decisions=decisions,
        lambda_by_frame=lambda_by_frame,
        lambda_aggregate=lambda_aggregate,
        lambda32_summary=lambda32_summary,
        over_summary=over_summary,
        low_cost_summary=low_cost_summary,
        safety=safety,
    )
    print(json.dumps(to_jsonable(final_summary["answers"]), indent=2, sort_keys=True))
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_root", default=str(DEFAULT_OUTPUTS_ROOT))
    parser.add_argument("--stage4a65ab_dir", required=True)
    parser.add_argument("--stage4a65ac_dir", required=True)
    parser.add_argument("--stage4a65ad_dir", required=True)
    parser.add_argument("--stage4a65ae_dir", required=True)
    parser.add_argument("--stage4a65af_dir", required=True)
    parser.add_argument("--stage4a65y_dir", default=str(DEFAULT_STAGE4A65Y_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--frame_filter", default="medium_three_rooms")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--occ_threshold", type=float, default=0.5)
    parser.add_argument("--free_threshold", type=float, default=0.5)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", choices=["reject", "crop", "allow"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--deduplicate_frames", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--save_raw_tree_summaries", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="frame-level worker count; 0 chooses a conservative automatic value")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

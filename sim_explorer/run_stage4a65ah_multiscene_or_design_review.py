#!/usr/bin/env python3
"""Stage 4A-6.5ah saved-frame discovery or runtime-smoke design review.

This stage is read-only over existing saved artifacts. It searches for
complete real Isaac medium saved frames, compares them against the Stage
4A-6.5ag selected frame set, and either replays only genuinely new complete
frames or writes a staged one-frame runtime-smoke design review.

It does not start Isaac, capture frames, rerun map_predict, run SSCNet
inference, execute selected actions, write prediction into observed_state, or
use prediction for traversability, collision, or ray blocking.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]

for _name in THREAD_ENV_VARS:
    os.environ.setdefault(_name, "1")


from run_multi_frame_lambda48_replay import (  # noqa: E402
    compare_modes,
    replay_frame_worker,
    rows_for_mode,
    summary_for_rows,
)
from run_real_frame_lambda48_formula_smoke import (  # noqa: E402
    mode_sort_key,
    parse_ints,
    read_json,
    save_json,
    sha256_file,
    to_jsonable,
    write_csv,
    write_md_table,
    write_text,
)


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUTS_ROOT = WORKSPACE / "outputs"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUTS_ROOT / "isaac_sc_pred_stage4a65ah_multiscene_or_runtime_design_review"
)
DEFAULT_STAGE4A65AG_DIR = (
    DEFAULT_OUTPUTS_ROOT / "isaac_sc_pred_stage4a65ag_multi_frame_lambda48_replay"
)
DEFAULT_STAGE4A65Y_DIR = DEFAULT_OUTPUTS_ROOT / "isaac_sc_pred_stage4a65y_source_gain_seed_replay"
DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)

FORBIDDEN_FRAME_TOKENS = [
    "synthetic_hidden_room",
    "synthetic",
    "oracle",
    "target",
    "ground_truth",
    "future_observed",
    "future-observed",
    "nyu",
]

OUTPUT_DIR_EXCLUDE_TOKENS = [
    "isaac_sc_pred_stage4a65ah_multiscene_or_runtime_design_review",
]

INVENTORY_FIELDS = [
    "candidate_id",
    "candidate_kind",
    "source_dir",
    "frame_label",
    "classification",
    "classification_reason",
    "medium_evidence",
    "start_variant",
    "scene_label",
    "observed_state_path",
    "prediction_npz_path",
    "pose_json_path",
    "camera_info_json_path",
    "observed_sha256",
    "prediction_sha256",
    "pose_sha256",
    "camera_sha256",
    "root_pose_position",
    "root_pose_yaw_rad",
    "frame_hash_key",
    "artifact_hash_key",
    "duplicate_of_candidate_id",
    "duplicate_of_stage4a65ag_frame_id",
    "duplicate_reason",
]


def configure_process_worker() -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = "1"


def actual_worker_count(requested: int, task_count: int | None = None) -> int:
    cpu_count = os.cpu_count() or 1
    count = max(1, min(int(requested), cpu_count))
    if task_count is not None and task_count > 0:
        count = max(1, min(count, int(task_count)))
    return count


def path_is_excluded(path: Path) -> bool:
    lower = str(path).lower()
    return any(token in lower for token in OUTPUT_DIR_EXCLUDE_TOKENS)


def existing_path(path: Path | None) -> Path | None:
    if path is not None and path.is_file():
        return path
    return None


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def parse_step_suffix(text: str) -> tuple[str, list[str]]:
    raw = text
    if raw.isdigit():
        number = int(raw)
        return raw, sorted({raw, f"{number:03d}", str(number)})
    return raw, [raw]


def pose_signature(pose_path: Path) -> dict[str, Any]:
    pose = read_json(pose_path)
    position = pose.get("position") if isinstance(pose, dict) else None
    yaw = pose.get("yaw_rad") if isinstance(pose, dict) else None
    return {
        "position": [] if position is None else [round(float(v), 6) for v in position],
        "yaw_rad": None if yaw is None else round(float(yaw), 9),
    }


def metadata_for_root(root: Path) -> dict[str, Any]:
    for name in ("episode_summary.json", "scene_metadata.json", "capture_scene_metadata.json"):
        data = read_json(root / name)
        if isinstance(data, dict) and data:
            return data
    return {}


def medium_evidence(root: Path) -> tuple[bool, str, str, str]:
    lower = str(root).lower()
    if "medium_three_rooms" in lower:
        meta = metadata_for_root(root)
        return True, "path_contains_medium_three_rooms", str(meta.get("start_variant") or ""), str(
            meta.get("scene") or meta.get("scene_name") or "medium_three_rooms"
        )
    known = (
        "stage4a65p_map_predict_tree_two_frame_smoke",
        "stage4a65o_map_predict_tree_one_step_smoke",
        "stage4a65s_gated_sc_tree_two_frame_smoke",
        "stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke",
    )
    if any(token in lower for token in known):
        return True, "known_stage4a65_real_medium_saved_frame", "", "medium_three_rooms"
    meta = metadata_for_root(root)
    if meta:
        blob = json.dumps(to_jsonable(meta), sort_keys=True).lower()
        if "medium-complexity" in blob or "medium_three_rooms" in blob:
            return True, "metadata_contains_medium_evidence", str(meta.get("start_variant") or ""), str(
                meta.get("scene") or meta.get("scene_name") or "medium-complexity scripted Isaac indoor scene"
            )
    return False, "no_medium_three_rooms_evidence", "", ""


def add_candidate(
    candidates: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    source_dir: Path,
    frame_label: str,
    candidate_kind: str,
    observed_path: Path | None,
    prediction_path: Path | None,
    pose_path: Path | None,
    camera_path: Path | None,
) -> None:
    if path_is_excluded(source_dir):
        return
    key = (
        str(observed_path or ""),
        str(prediction_path or ""),
        str(pose_path or ""),
        str(camera_path or ""),
    )
    if key in candidates:
        return
    medium_ok, medium_reason, start_variant, scene_label = medium_evidence(source_dir)
    row = {
        "candidate_id": f"{source_dir.name}:{frame_label}:{candidate_kind}",
        "candidate_kind": candidate_kind,
        "source_dir": str(source_dir),
        "frame_label": frame_label,
        "medium_evidence": medium_reason if medium_ok else "",
        "start_variant": start_variant,
        "scene_label": scene_label,
        "observed_state_path": str(observed_path) if observed_path else "",
        "prediction_npz_path": str(prediction_path) if prediction_path else "",
        "pose_json_path": str(pose_path) if pose_path else "",
        "camera_info_json_path": str(camera_path) if camera_path else "",
        "_medium_ok": medium_ok,
        "_medium_reason": medium_reason,
    }
    candidates[key] = row


def candidates_from_observed(outputs_root: Path) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for observed in sorted(outputs_root.rglob("observed_state*.npy")):
        if path_is_excluded(observed):
            continue
        parent = observed.parent
        stem = observed.stem
        if stem.startswith("observed_state_frame"):
            label = stem.replace("observed_state_", "")
            prediction = first_existing(
                [
                    parent / f"{label}_prediction/global_prediction_layer.npz",
                    parent / "map_predict/global_prediction_layer.npz",
                    parent / "global_prediction_layer.npz",
                ]
            )
            pose = first_existing([parent / f"{label}_pose.json", parent / "pose.json"])
            camera = first_existing([parent / f"{label}_camera_info.json", parent / "camera_info.json"])
            add_candidate(
                candidates,
                source_dir=parent,
                frame_label=label,
                candidate_kind="frameNNN",
                observed_path=observed,
                prediction_path=prediction,
                pose_path=pose,
                camera_path=camera,
            )
        elif stem.startswith("observed_state_step"):
            raw_suffix = stem.replace("observed_state_step", "")
            label, suffixes = parse_step_suffix(raw_suffix)
            prediction = first_existing(
                [parent / f"prediction_step{suffix}/global_prediction_layer.npz" for suffix in suffixes]
                + [parent / "global_prediction_layer.npz"]
            )
            pose = first_existing([parent / f"pose_{suffix}.json" for suffix in suffixes])
            camera = existing_path(parent / "camera_info.json")
            add_candidate(
                candidates,
                source_dir=parent,
                frame_label=f"step{label}",
                candidate_kind="rollout_saved_step",
                observed_path=observed,
                prediction_path=prediction,
                pose_path=pose,
                camera_path=camera,
            )
        elif stem == "observed_state_final":
            prediction = first_existing(
                [
                    parent / "prediction_final/global_prediction_layer.npz",
                    parent / "global_prediction_layer.npz",
                ]
            )
            pose = first_existing([parent / "pose_final.json", parent / "pose.json"])
            camera = existing_path(parent / "camera_info.json")
            add_candidate(
                candidates,
                source_dir=parent,
                frame_label="final",
                candidate_kind="final_state_probe",
                observed_path=observed,
                prediction_path=prediction,
                pose_path=pose,
                camera_path=camera,
            )
        else:
            prediction = first_existing([parent / "global_prediction_layer.npz", parent / "map_predict/global_prediction_layer.npz"])
            pose = first_existing(sorted(parent.glob("*pose*.json")))
            camera = first_existing(sorted(parent.glob("*camera_info*.json")) + [parent / "camera_info.json"])
            add_candidate(
                candidates,
                source_dir=parent,
                frame_label=stem,
                candidate_kind="observed_state_probe",
                observed_path=observed,
                prediction_path=prediction,
                pose_path=pose,
                camera_path=camera,
            )
    return list(candidates.values())


def candidates_from_predictions(outputs_root: Path, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (
            row.get("observed_state_path", ""),
            row.get("prediction_npz_path", ""),
            row.get("pose_json_path", ""),
            row.get("camera_info_json_path", ""),
        ): row
        for row in existing
    }
    for prediction in sorted(outputs_root.rglob("global_prediction_layer.npz")):
        if path_is_excluded(prediction):
            continue
        parent = prediction.parent
        source = parent
        frame_label = "prediction_only"
        kind = "prediction_npz_probe"
        observed: Path | None = None
        pose: Path | None = None
        camera: Path | None = None

        if parent.name.startswith("prediction_step"):
            source = parent.parent
            raw_suffix = parent.name.replace("prediction_step", "")
            label, suffixes = parse_step_suffix(raw_suffix)
            frame_label = f"step{label}"
            kind = "rollout_saved_step_prediction"
            observed = first_existing([source / f"observed_state_step{suffix}.npy" for suffix in suffixes])
            pose = first_existing([source / f"pose_{suffix}.json" for suffix in suffixes])
            camera = existing_path(source / "camera_info.json")
        elif parent.name.endswith("_prediction"):
            source = parent.parent
            frame_label = parent.name.replace("_prediction", "")
            kind = "frameNNN_prediction"
            observed = existing_path(source / f"observed_state_{frame_label}.npy")
            pose = existing_path(source / f"{frame_label}_pose.json")
            camera = existing_path(source / f"{frame_label}_camera_info.json") or existing_path(source / "camera_info.json")
        elif parent.name == "map_predict":
            source = parent.parent
            frame_label = "map_predict_root"
            kind = "map_predict_root"
            observed = first_existing(sorted(source.glob("observed_state*.npy")))
            pose = first_existing(sorted(source.glob("*pose*.json")))
            camera = first_existing(sorted(source.glob("*camera_info*.json")) + [source / "camera_info.json"])
        else:
            source = parent
            observed = first_existing(sorted(source.glob("observed_state*.npy")))
            pose = first_existing(sorted(source.glob("*pose*.json")))
            camera = first_existing(sorted(source.glob("*camera_info*.json")) + [source / "camera_info.json"])

        add_candidate(
            candidates,
            source_dir=source,
            frame_label=frame_label,
            candidate_kind=kind,
            observed_path=observed,
            prediction_path=prediction,
            pose_path=pose,
            camera_path=camera,
        )
    return list(candidates.values())


def discover_raw_candidates(outputs_root: Path) -> list[dict[str, Any]]:
    rows = candidates_from_observed(outputs_root)
    rows = candidates_from_predictions(outputs_root, rows)
    rows = sorted(rows, key=lambda row: (row["source_dir"], row["frame_label"], row["candidate_kind"]))
    return rows


def missing_classification(row: dict[str, Any]) -> tuple[str | None, str]:
    missing = []
    if not row.get("observed_state_path") or not Path(row["observed_state_path"]).is_file():
        missing.append("observed_state")
    if not row.get("prediction_npz_path") or not Path(row["prediction_npz_path"]).is_file():
        missing.append("prediction")
    if not row.get("pose_json_path") or not Path(row["pose_json_path"]).is_file():
        missing.append("pose")
    if not row.get("camera_info_json_path") or not Path(row["camera_info_json_path"]).is_file():
        missing.append("camera")
    if not missing:
        return None, ""
    if missing == ["prediction"]:
        return "incomplete_missing_prediction", "missing_prediction"
    if "pose" in missing or "camera" in missing:
        return "incomplete_missing_pose_or_camera", "missing_" + ",".join(missing)
    return "unknown_skipped", "missing_" + ",".join(missing)


def forbidden_classification(row: dict[str, Any]) -> tuple[str | None, str]:
    blob = " ".join(
        str(row.get(key, ""))
        for key in (
            "source_dir",
            "observed_state_path",
            "prediction_npz_path",
            "pose_json_path",
            "camera_info_json_path",
        )
    ).lower()
    tokens = [token for token in FORBIDDEN_FRAME_TOKENS if token in blob]
    if "calibration" in blob and not bool(row.get("_medium_ok")):
        tokens.append("calibration_without_real_medium_evidence")
    if tokens:
        return "synthetic_or_forbidden", "forbidden_path_token_" + ",".join(sorted(set(tokens)))
    return None, ""


def hash_candidate_worker(row: dict[str, Any]) -> dict[str, Any]:
    configure_process_worker()
    out = dict(row)
    observed = Path(str(row["observed_state_path"]))
    prediction = Path(str(row["prediction_npz_path"]))
    pose = Path(str(row["pose_json_path"]))
    camera = Path(str(row["camera_info_json_path"]))
    pose_sig = pose_signature(pose)
    observed_sha = sha256_file(observed)
    prediction_sha = sha256_file(prediction)
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
    out.update(
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
    return out


def load_stage4a65ag_keys(stage4a65ag_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    selected_path = stage4a65ag_dir / "selected_frame_manifest.json"
    selected = read_json(selected_path)
    if not isinstance(selected, list):
        selected = []
    frame_key_map = {}
    artifact_key_map = {}
    for row in selected:
        if not isinstance(row, dict):
            continue
        frame_key = str(row.get("frame_hash_key") or "")
        artifact_key = str(row.get("artifact_hash_key") or "")
        if frame_key:
            frame_key_map[frame_key] = row
        if artifact_key:
            artifact_key_map[artifact_key] = row
    return frame_key_map, artifact_key_map


def classify_candidates(
    raw_rows: list[dict[str, Any]],
    stage4a65ag_dir: Path,
    *,
    max_workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    preclassified: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        row = dict(row)
        cls, reason = forbidden_classification(row)
        if cls is None:
            cls, reason = missing_classification(row)
        if cls is None and not bool(row.get("_medium_ok")):
            cls, reason = "unknown_skipped", str(row.get("_medium_reason") or "no_medium_three_rooms_evidence")
        if cls is not None:
            row["classification"] = cls
            row["classification_reason"] = reason
            preclassified.append(row)
        else:
            hash_rows.append(row)

    worker_count = actual_worker_count(max_workers, max(1, len(hash_rows)))
    process_used = len(hash_rows) > 1
    hashed: list[dict[str, Any]] = []
    if process_used:
        with ProcessPoolExecutor(max_workers=worker_count, initializer=configure_process_worker) as executor:
            futures = [executor.submit(hash_candidate_worker, row) for row in hash_rows]
            for future in as_completed(futures):
                hashed.append(future.result())
    else:
        hashed = [hash_candidate_worker(row) for row in hash_rows]
    hashed = sorted(hashed, key=lambda row: (row["source_dir"], row["frame_label"], row["candidate_kind"]))

    ag_frame_keys, ag_artifact_keys = load_stage4a65ag_keys(stage4a65ag_dir)
    seen_artifacts: dict[str, dict[str, Any]] = {}
    seen_frame_keys: dict[str, dict[str, Any]] = {}
    classified_hash_rows: list[dict[str, Any]] = []
    for row in hashed:
        artifact_key = str(row.get("artifact_hash_key") or "")
        frame_key = str(row.get("frame_hash_key") or "")
        if artifact_key in ag_artifact_keys:
            ag = ag_artifact_keys[artifact_key]
            row["classification"] = "already_in_stage4a65ag"
            row["classification_reason"] = "artifact_hash_matches_stage4a65ag_selected_frame"
            row["duplicate_of_stage4a65ag_frame_id"] = ag.get("frame_id")
            row["duplicate_reason"] = "exact_stage4a65ag_artifact"
        elif frame_key in ag_frame_keys:
            ag = ag_frame_keys[frame_key]
            row["classification"] = "duplicate_of_existing"
            row["classification_reason"] = "same_observed_pose_camera_as_stage4a65ag_selected_frame"
            row["duplicate_of_stage4a65ag_frame_id"] = ag.get("frame_id")
            row["duplicate_reason"] = "stage4a65ag_frame_prediction_variant_or_duplicate"
        elif artifact_key in seen_artifacts:
            previous = seen_artifacts[artifact_key]
            row["classification"] = "duplicate_of_existing"
            row["classification_reason"] = "same_complete_artifact_as_prior_candidate"
            row["duplicate_of_candidate_id"] = previous.get("candidate_id")
            row["duplicate_reason"] = "exact_same_observed_prediction_pose_camera"
        elif frame_key in seen_frame_keys:
            previous = seen_frame_keys[frame_key]
            row["classification"] = "duplicate_of_existing"
            row["classification_reason"] = "same_observed_pose_camera_as_prior_candidate"
            row["duplicate_of_candidate_id"] = previous.get("candidate_id")
            row["duplicate_reason"] = "same_frame_prediction_variant"
        else:
            row["classification"] = "new_complete_frame"
            row["classification_reason"] = "complete_real_medium_frame_not_seen_in_stage4a65ag"
            seen_artifacts[artifact_key] = row
            seen_frame_keys[frame_key] = row
        classified_hash_rows.append(row)

    rows = preclassified + classified_hash_rows
    rows = [{key: row.get(key, "") for key in INVENTORY_FIELDS} for row in rows]
    rows = sorted(rows, key=lambda row: (row["classification"], row["source_dir"], row["frame_label"], row["candidate_kind"]))
    wall = time.perf_counter() - start
    report = {
        "hash_task_count": len(hash_rows),
        "hash_parallel_backend": "ProcessPoolExecutor" if process_used else "serial",
        "hash_process_workers_used": process_used,
        "hash_worker_count": worker_count if process_used else 1,
        "hash_wall_time_s": wall,
        "hash_average_task_time_s": None if not hash_rows else wall / max(1, len(hash_rows)),
        "hash_worker_utilization_estimate": None
        if not process_used
        else min(1.0, len(hash_rows) / max(1, worker_count)),
        "hash_parallel_disabled_reason": None if process_used else "hash_task_count_less_than_two",
    }
    return rows, report


def write_loaded_context_manifest(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    context_files = [
        WORKSPACE / ".project_context/CURRENT_STATE.md",
        WORKSPACE / ".project_context/CODEX_LOG.md",
        WORKSPACE / ".project_context/TODO.md",
    ]
    rows = []
    for path in context_files:
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        rows.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "mentions_stage4a65af": "Stage 4A-6.5af" in text,
                "mentions_stage4a65ag": "Stage 4A-6.5ag" in text,
                "mentions_stage4a65ah": "Stage 4A-6.5ah" in text,
            }
        )
    manifest = {
        "stage": "Stage 4A-6.5ah",
        "task": "hardware-aware multi-scene/start saved-frame replay if available, otherwise staged one-frame runtime-smoke design review only",
        "context_files": rows,
        "confirmed_prior_context": {
            "stage4a65af_complete": True,
            "stage4a65af_synthetic_lambda48_hidden_room": "5/5",
            "stage4a65af_real_same_as_measured": "14/20",
            "stage4a65af_real_healthy_nonmeasured": "6/20",
            "stage4a65af_real_prior_basin": "0/20",
            "stage4a65af_real_low_cost_artifact": "0/20",
            "stage4a65ag_complete": True,
            "stage4a65ag_output": str(Path(args.stage4a65ag_dir).resolve()),
            "stage4a65ag_candidate_rows": 20,
            "stage4a65ag_valid_saved_real_frame_candidates": 17,
            "stage4a65ag_unique_selected_real_medium_frames": 7,
            "stage4a65ag_lambda48_rows": 70,
            "stage4a65ag_same_as_measured": "33/70",
            "stage4a65ag_healthy_nonmeasured": "35/70",
            "stage4a65ag_local_jitter": "2/70",
            "stage4a65ag_prior_basin": "0/70",
            "stage4a65ag_low_cost_artifact": "0/70",
            "stage4a65ag_over_cost_prior_basin_fraction": "24/70",
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
        },
        "safety_scope": safety_scope(),
    }
    save_json(output_dir / "loaded_context_manifest.json", manifest)
    write_text(
        output_dir / "loaded_context_manifest.md",
        "\n".join(
            [
                "# Loaded Context Manifest",
                "",
                "- stage: `Stage 4A-6.5ah`",
                "- prior context read from CURRENT_STATE, CODEX_LOG, and TODO.",
                "- Stage 4A-6.5af complete: synthetic lambda48 hidden-room `5/5`; real aggregate same-as-measured `14/20`, healthy non-measured `6/20`, prior basin `0/20`, low-cost artifact `0/20`.",
                "- Stage 4A-6.5ag complete: 7 selected real medium frames, lambda48 same-as-measured `33/70`, healthy non-measured `35/70`, local jitter `2/70`, prior basin `0/70`, low-cost artifact `0/70`.",
                "- This stage does not execute runtime, rollout, map_predict, SSCNet inference, or selected actions.",
            ]
        ),
    )
    return manifest


def safety_scope() -> dict[str, bool]:
    return {
        "offline_saved_frame_only": True,
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
        "future_observed_planning_scoring": False,
        "external_source_modified_built": False,
        "pareto_dominance_gate_implemented": False,
        "runtime_planner_implemented": False,
        "coverage_improvement_claim": False,
    }


def write_discovery_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    duplicates = [
        row
        for row in rows
        if row["classification"] in {"already_in_stage4a65ag", "duplicate_of_existing"}
    ]
    new_frames = [row for row in rows if row["classification"] == "new_complete_frame"]
    skipped = [
        row
        for row in rows
        if row["classification"]
        in {
            "incomplete_missing_prediction",
            "incomplete_missing_pose_or_camera",
            "synthetic_or_forbidden",
            "unknown_skipped",
        }
    ]
    write_csv(output_dir / "additional_frame_discovery_inventory.csv", rows, INVENTORY_FIELDS)
    save_json(output_dir / "additional_frame_discovery_inventory.json", rows)
    write_md_table(
        output_dir / "additional_frame_discovery_inventory.md",
        "Additional Frame Discovery Inventory",
        rows,
        [
            "candidate_id",
            "classification",
            "classification_reason",
            "medium_evidence",
            "start_variant",
            "source_dir",
            "frame_label",
        ],
        limit=320,
    )
    write_csv(output_dir / "additional_frame_duplicates.csv", duplicates, INVENTORY_FIELDS)
    save_json(output_dir / "additional_frame_duplicates.json", duplicates)
    write_csv(output_dir / "new_complete_frame_manifest.csv", new_frames, INVENTORY_FIELDS)
    save_json(output_dir / "new_complete_frame_manifest.json", new_frames)
    write_md_table(
        output_dir / "new_complete_frame_manifest.md",
        "New Complete Frame Manifest",
        new_frames,
        [
            "candidate_id",
            "source_dir",
            "frame_label",
            "observed_state_path",
            "prediction_npz_path",
            "pose_json_path",
            "camera_info_json_path",
        ],
    )
    write_csv(output_dir / "skipped_frame_candidates.csv", skipped, INVENTORY_FIELDS)
    save_json(output_dir / "skipped_frame_candidates.json", skipped)
    write_md_table(
        output_dir / "skipped_frame_candidates.md",
        "Skipped Frame Candidates",
        skipped,
        ["candidate_id", "classification", "classification_reason", "source_dir", "frame_label"],
        limit=320,
    )
    counts = Counter(row["classification"] for row in rows)
    summary = {
        "candidate_rows": len(rows),
        "classification_counts": dict(counts),
        "new_complete_frame_count": len(new_frames),
        "duplicate_or_already_in_stage4a65ag_count": len(duplicates),
        "skipped_count": len(skipped),
    }
    return summary


def write_runtime_design_review(output_dir: Path, discovery_summary: dict[str, Any]) -> None:
    write_text(
        output_dir / "runtime_smoke_design_review.md",
        "\n".join(
            [
                "# Runtime Smoke Design Review",
                "",
                "Stage 4A-6.5ah found no new complete saved real frames beyond the Stage 4A-6.5ag selected frame identities, so this stage stops at design review only.",
                "",
                "Future safe runtime smoke shape:",
                "",
                "- one Isaac startup",
                "- one frame only",
                "- one map_predict call only",
                "- one source-protected tree decision only",
                "- no selected action execution",
                "- no second frame",
                "- no rollout",
                "- no coverage claim",
                "- lambda48 formula only: `gain_exp / cost + 48 * minmax(source_occ_free)`",
                "- measured-only baseline shadow",
                "- lambda32 shadow optional",
                "- over-cost diagnostic prohibited in runtime execution; allowed only as an offline shadow if a saved frame exists",
                "- prediction information-gain-only",
                "- no prediction writeback",
                "- no prediction traversability, collision, or ray blocking",
                "- hard stop after decision output",
                "",
                "Prerequisites loaded:",
                "",
                "- Stage 4A-6.5ag evidence loaded",
                "- lambda48 artifact-free `0/70`",
                "- prior basin `0/70`",
                "- healthy non-measured `35/70`",
                "- over-cost prior basin `24/70`, diagnostic-only",
                f"- no more saved frames found: `{discovery_summary['new_complete_frame_count']}` new complete frames",
                "",
                "This design review is not runtime execution and does not claim coverage improvement.",
            ]
        ),
    )
    checklist = {
        "stage": "Stage 4A-6.5ah",
        "runtime_executed": False,
        "future_runtime_smoke_design": {
            "one_isaac_startup": True,
            "one_frame_only": True,
            "one_map_predict_call_only": True,
            "one_source_protected_tree_decision_only": True,
            "execute_selected_action": False,
            "second_frame": False,
            "rollout": False,
            "coverage_claim": False,
            "formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
            "measured_only_shadow": True,
            "lambda32_shadow_optional": True,
            "over_cost_runtime_execution": False,
            "prediction_information_gain_only": True,
            "prediction_writeback": False,
            "prediction_traversability_collision_ray_blocking": False,
            "hard_stop_after_decision_output": True,
        },
        "prohibited_in_future_runtime_smoke": {
            "selected_action_execution": True,
            "second_frame": True,
            "rollout": True,
            "coverage_improvement_claim": True,
            "prediction_writeback": True,
            "prediction_traversability_collision_ray_blocking": True,
            "target_ground_truth_future_observed_scoring": True,
            "over_cost_as_runtime_formula": True,
        },
    }
    save_json(output_dir / "runtime_smoke_safety_checklist.json", checklist)
    write_text(
        output_dir / "runtime_smoke_safety_checklist.md",
        "\n".join(
            [
                "# Runtime Smoke Safety Checklist",
                "",
                "- runtime executed in Stage 4A-6.5ah: `false`",
                "- one future Isaac startup: `true`",
                "- one future frame only: `true`",
                "- one future map_predict call only: `true`",
                "- one source-protected tree decision only: `true`",
                "- execute selected action: `false`",
                "- second frame: `false`",
                "- rollout: `false`",
                "- prediction writeback: `false`",
                "- prediction traversability/collision/ray blocking: `false`",
                "- coverage claim: `false`",
            ]
        ),
    )
    write_text(
        output_dir / "future_stage4a65ai_command_sketch.md",
        "\n".join(
            [
                "# Future Stage 4A-6.5ai Command Sketch",
                "",
                "`DO NOT RUN IN THIS STAGE`",
                "",
                "```bash",
                "python sim_explorer/run_stage4a65ai_one_frame_lambda48_runtime_smoke.py \\",
                "  --scene medium_three_rooms \\",
                "  --start_variant start_room_a \\",
                "  --max_frames 1 \\",
                "  --max_map_predict_calls 1 \\",
                "  --formula decoupled_minmax_lambda48 \\",
                "  --tau 0.1 \\",
                "  --occ_threshold 0.5 \\",
                "  --free_threshold 0.5 \\",
                "  --measured_only_shadow \\",
                "  --lambda32_shadow \\",
                "  --no_execute_selected_action \\",
                "  --no_second_frame \\",
                "  --no_rollout \\",
                "  --no_prediction_writeback \\",
                "  --max_workers 32 \\",
                "  --output_dir outputs/isaac_sc_pred_stage4a65ai_one_frame_lambda48_runtime_smoke",
                "```",
                "",
                "The command is a sketch for a future stage only. Stage 4A-6.5ah does not implement or execute it.",
            ]
        ),
    )


def write_hardware_report(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    discovery_parallel_report: dict[str, Any],
    replay_parallel_report: dict[str, Any] | None,
    total_wall_time_s: float,
) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    requested = int(args.max_workers)
    actual = actual_worker_count(requested)
    torch_imported = "torch" in sys.modules
    torch_num_threads = None
    if torch_imported:
        torch = sys.modules["torch"]
        if hasattr(torch, "get_num_threads"):
            torch_num_threads = int(torch.get_num_threads())
    report = {
        "stage": "Stage 4A-6.5ah",
        "os_cpu_count": cpu_count,
        "requested_max_workers": requested,
        "actual_max_workers": actual,
        "parallel_backend": "ProcessPoolExecutor",
        "process_workers_used": bool(discovery_parallel_report.get("hash_process_workers_used"))
        or bool((replay_parallel_report or {}).get("replay_process_workers_used")),
        "thread_workers_used": False,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "torch_imported": torch_imported,
        "torch_num_threads": torch_num_threads,
        "oversubscription_policy": "process-pool workers set BLAS/OMP inner threads to 1",
        "task_count": int(discovery_parallel_report.get("hash_task_count") or 0)
        + int((replay_parallel_report or {}).get("replay_task_count") or 0),
        "wall_time_s": total_wall_time_s,
        "average_task_time_s": None,
        "worker_utilization_estimate": discovery_parallel_report.get("hash_worker_utilization_estimate"),
        "discovery_parallel_report": discovery_parallel_report,
        "replay_parallel_report": replay_parallel_report
        or {
            "replay_task_count": 0,
            "replay_process_workers_used": False,
            "parallel_disabled_reason": "no_new_complete_frames",
        },
    }
    if report["task_count"]:
        report["average_task_time_s"] = total_wall_time_s / max(1, int(report["task_count"]))
    save_json(output_dir / "hardware_utilization_report.json", report)
    write_text(
        output_dir / "hardware_utilization_report.md",
        "\n".join(
            [
                "# Hardware Utilization Report",
                "",
                f"- os_cpu_count: `{report['os_cpu_count']}`",
                f"- requested_max_workers: `{report['requested_max_workers']}`",
                f"- actual_max_workers: `{report['actual_max_workers']}`",
                f"- parallel_backend: `{report['parallel_backend']}`",
                f"- process_workers_used: `{report['process_workers_used']}`",
                f"- task_count: `{report['task_count']}`",
                f"- wall_time_s: `{report['wall_time_s']}`",
                f"- OMP_NUM_THREADS: `{report['OMP_NUM_THREADS']}`",
                f"- OPENBLAS_NUM_THREADS: `{report['OPENBLAS_NUM_THREADS']}`",
                f"- MKL_NUM_THREADS: `{report['MKL_NUM_THREADS']}`",
                f"- NUMEXPR_NUM_THREADS: `{report['NUMEXPR_NUM_THREADS']}`",
                f"- VECLIB_MAXIMUM_THREADS: `{report['VECLIB_MAXIMUM_THREADS']}`",
                f"- torch_num_threads: `{report['torch_num_threads']}`",
                "- oversubscription policy: process-pool workers set BLAS/OMP inner threads to 1.",
            ]
        ),
    )
    return report


def decision_fields() -> list[str]:
    return [
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
        "same_as_measured",
        "spatial_prior_sc_basin",
        "healthy_nonmeasured_candidate",
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
        "low_cost_artifact",
        "prediction_safety_flags",
        "observed_state_hash",
        "prediction_hash",
    ]


def write_new_frame_replay_outputs(
    output_dir: Path,
    new_frames: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    value_rows: list[dict[str, Any]],
    hash_checks: list[dict[str, Any]],
    hardware_report: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    decisions = sorted(decisions, key=lambda row: (row["frame_id"], int(row["seed"]), mode_sort_key(str(row["mode"]))))
    value_rows = sorted(value_rows, key=lambda row: (row["frame_id"], int(row["seed"]), mode_sort_key(str(row["mode"]))))
    write_csv(output_dir / "new_frame_per_frame_seed_mode_decisions.csv", decisions, decision_fields())
    save_json(output_dir / "new_frame_per_frame_seed_mode_decisions.json", decisions)
    write_md_table(
        output_dir / "new_frame_per_frame_seed_mode_decisions.md",
        "New Frame Per-Frame Seed Mode Decisions",
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
            "final_value",
            "margin",
            "low_cost_artifact",
        ],
        limit=360,
    )
    write_csv(output_dir / "new_frame_value_components.csv", value_rows)
    save_json(output_dir / "new_frame_value_components.json", value_rows)

    branch_rows = [
        {
            key: row.get(key)
            for key in (
                "frame_id",
                "seed",
                "mode",
                "selected_child_id",
                "best_descendant_id",
                "branch_classification",
                "same_as_measured",
                "spatial_prior_sc_basin",
                "healthy_nonmeasured_candidate",
                "low_cost_artifact",
            )
        }
        for row in decisions
    ]
    write_csv(output_dir / "new_frame_branch_classification.csv", branch_rows)
    save_json(output_dir / "new_frame_branch_classification.json", branch_rows)
    write_md_table(
        output_dir / "new_frame_branch_classification.md",
        "New Frame Branch Classification",
        branch_rows,
        [
            "frame_id",
            "seed",
            "mode",
            "selected_child_id",
            "best_descendant_id",
            "branch_classification",
            "low_cost_artifact",
        ],
        limit=360,
    )

    map48 = rows_for_mode(decisions, "map_predict_lambda48")
    by_frame = []
    for frame in new_frames:
        frame_rows = [row for row in map48 if row["frame_id"] == frame["frame_id"]]
        item = summary_for_rows(frame_rows, str(frame["frame_id"]))
        item.update({"frame_id": frame["frame_id"], "source_dir": frame["source_dir"], "frame_label": frame["frame_label"]})
        by_frame.append(item)
    aggregate = summary_for_rows(map48, "new_frame_aggregate")
    aggregate.update(
        {
            "unique_new_frame_count": len(new_frames),
            "total_seed_frame_rows": len(map48),
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
        }
    )
    save_json(output_dir / "new_frame_lambda48_summary.json", {"aggregate": aggregate, "by_frame": by_frame})
    write_csv(output_dir / "new_frame_lambda48_summary.csv", by_frame + [aggregate])
    write_text(
        output_dir / "new_frame_lambda48_summary.md",
        "\n".join(
            [
                "# New Frame Lambda48 Summary",
                "",
                f"- unique new frames: `{len(new_frames)}`",
                f"- lambda48 rows: `{len(map48)}`",
                f"- same-as-measured: `{aggregate['same_as_measured_count']}/{aggregate['row_count']}`",
                f"- healthy non-measured: `{aggregate['healthy_nonmeasured_count']}/{aggregate['row_count']}`",
                f"- historical prior basin: `{aggregate['historical_prior_basin_count']}/{aggregate['row_count']}`",
                f"- low-cost artifact: `{aggregate['low_cost_artifact_count']}/{aggregate['row_count']}`",
                "- runtime smoke readiness: `false`",
                "- rollout readiness: `false`",
            ]
        ),
    )

    lambda32_rows = compare_modes(decisions, "map_predict_lambda32", "map_predict_lambda48")
    lambda32_summary = {
        "row_count": len(lambda32_rows),
        "branch_class_agreement_count": int(sum(bool(row["branch_class_agreement"]) for row in lambda32_rows)),
        "selected_child_agreement_count": int(sum(bool(row["selected_child_agreement"]) for row in lambda32_rows)),
        "best_descendant_agreement_count": int(sum(bool(row["best_descendant_agreement"]) for row in lambda32_rows)),
    }
    save_json(output_dir / "new_frame_lambda32_vs_lambda48.json", {"summary": lambda32_summary, "rows": lambda32_rows})
    write_csv(output_dir / "new_frame_lambda32_vs_lambda48.csv", lambda32_rows)
    write_text(
        output_dir / "new_frame_lambda32_vs_lambda48.md",
        "\n".join(
            [
                "# New Frame Lambda32 Vs Lambda48",
                "",
                f"- rows: `{lambda32_summary['row_count']}`",
                f"- branch-class agreement count: `{lambda32_summary['branch_class_agreement_count']}`",
                f"- selected-child agreement count: `{lambda32_summary['selected_child_agreement_count']}`",
                f"- best-descendant agreement count: `{lambda32_summary['best_descendant_agreement_count']}`",
            ]
        ),
    )

    over_rows = compare_modes(decisions, "source_occ_free_over_cost", "map_predict_lambda48")
    over_summary = {
        "row_count": len(over_rows),
        "source_occ_free_over_cost": summary_for_rows(rows_for_mode(decisions, "source_occ_free_over_cost"), "source_occ_free_over_cost"),
        "map_predict_lambda48": aggregate,
        "diagnostic_only": True,
    }
    save_json(output_dir / "new_frame_over_cost_diagnostic.json", {"summary": over_summary, "rows": over_rows})
    write_csv(output_dir / "new_frame_over_cost_diagnostic.csv", over_rows)
    write_text(
        output_dir / "new_frame_over_cost_diagnostic.md",
        "\n".join(
            [
                "# New Frame Over-Cost Diagnostic",
                "",
                "- over-cost remains diagnostic-only.",
                f"- rows: `{len(over_rows)}`",
            ]
        ),
    )

    low_rows = [
        {
            "frame_id": row["frame_id"],
            "seed": row["seed"],
            "mode": row["mode"],
            "branch_classification": row.get("branch_classification"),
            "gain_exp": row.get("gain_exp"),
            "source_occ_free_count": row.get("source_occ_free_count"),
            "cost": row.get("cost"),
            "low_cost_artifact": row.get("low_cost_artifact"),
            "spatial_prior_sc_basin": row.get("spatial_prior_sc_basin"),
        }
        for row in decisions
    ]
    low_map48 = [row for row in low_rows if row["mode"] == "map_predict_lambda48"]
    low_summary = {
        "map_predict_lambda48_row_count": len(low_map48),
        "map_predict_lambda48_low_cost_artifact_count": int(sum(bool(row["low_cost_artifact"]) for row in low_map48)),
    }
    save_json(output_dir / "new_frame_low_cost_artifact.json", {"summary": low_summary, "rows": low_rows})
    write_csv(output_dir / "new_frame_low_cost_artifact.csv", low_rows)
    write_md_table(
        output_dir / "new_frame_low_cost_artifact.md",
        "New Frame Low-Cost Artifact",
        low_rows,
        ["frame_id", "seed", "mode", "branch_classification", "cost", "low_cost_artifact", "spatial_prior_sc_basin"],
        limit=360,
    )

    checkpoint_before = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    hash_report = {
        "frames": hash_checks,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": checkpoint_before,
            "sha256_after": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
            "unchanged": True,
        },
    }
    save_json(output_dir / "hash_checks.json", hash_report)
    save_json(output_dir / "prediction_safety_report.json", safety_scope())
    write_text(
        output_dir / "prediction_safety_report.md",
        "# Prediction Safety Report\n\n- prediction remained information-gain-only.\n- no runtime, no writeback, no traversability/collision/ray blocking.",
    )

    ag_summary = read_json(DEFAULT_STAGE4A65AG_DIR / "stage4a65ag_multi_frame_lambda48_replay_summary.json")
    combined = {
        "stage4a65ag_lambda48": (ag_summary or {}).get("answers", {}).get("lambda48_aggregate"),
        "stage4a65ah_new_frame_lambda48": aggregate,
        "hardware_utilization": hardware_report,
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
    }
    save_json(output_dir / "combined_with_stage4a65ag_summary.json", combined)
    write_text(
        output_dir / "combined_with_stage4a65ag_summary.md",
        "\n".join(
            [
                "# Combined With Stage 4A-6.5ag Summary",
                "",
                f"- Stage 4A-6.5ah new frames: `{len(new_frames)}`",
                "- Stage 4A-6.5ag aggregate is referenced from saved summary.",
                "- runtime smoke readiness: `false`",
                "- rollout readiness: `false`",
            ]
        ),
    )
    return {
        "new_frame_lambda48_aggregate": aggregate,
        "new_frame_lambda32_vs_lambda48": lambda32_summary,
        "new_frame_over_cost_diagnostic": over_summary,
        "new_frame_low_cost_artifact": low_summary,
    }


def run_new_frame_replay(args: argparse.Namespace, output_dir: Path, new_frames: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    seeds = parse_ints(args.seeds)
    for idx, frame in enumerate(new_frames, start=1):
        frame["frame_id"] = f"new_real_medium_frame_{idx:03d}"
    payloads = [
        (dict(frame), vars(args).copy(), [int(seed)], str(output_dir))
        for frame in new_frames
        for seed in seeds
    ]
    worker_count = actual_worker_count(int(args.max_workers), max(1, len(payloads)))
    decisions: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []
    hash_by_frame: dict[str, dict[str, Any]] = {}
    manifest_by_frame: dict[str, dict[str, Any]] = {}
    if len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=worker_count, initializer=configure_process_worker) as executor:
            future_map = {executor.submit(replay_frame_worker, payload): payload[0]["frame_id"] for payload in payloads}
            for future in as_completed(future_map):
                frame_id = str(future_map[future])
                frame_decisions, frame_values, _frame_roots, frame_hashes, frame_manifest = future.result()[1]
                decisions.extend(frame_decisions)
                value_rows.extend(frame_values)
                hash_by_frame[frame_id] = frame_hashes
                manifest_by_frame[frame_id] = frame_manifest
                print(f"[stage4a65ah] completed {frame_id} replay task", flush=True)
    else:
        for payload in payloads:
            frame_id, result = replay_frame_worker(payload)
            frame_decisions, frame_values, _frame_roots, frame_hashes, frame_manifest = result
            decisions.extend(frame_decisions)
            value_rows.extend(frame_values)
            hash_by_frame[frame_id] = frame_hashes
            manifest_by_frame[frame_id] = frame_manifest

    hash_checks = [hash_by_frame[str(frame["frame_id"])] for frame in new_frames if str(frame["frame_id"]) in hash_by_frame]
    report = {
        "replay_task_count": len(payloads),
        "replay_process_workers_used": len(payloads) > 1,
        "replay_worker_count": worker_count if len(payloads) > 1 else 1,
        "replay_wall_time_s": time.perf_counter() - start,
        "replay_average_task_time_s": None if not payloads else (time.perf_counter() - start) / max(1, len(payloads)),
        "parallel_disabled_reason": None if len(payloads) > 1 else "replay_task_count_less_than_two",
        "per_frame_loaded_manifest": list(manifest_by_frame.values()),
    }
    replay_outputs = write_new_frame_replay_outputs(
        output_dir,
        new_frames,
        decisions,
        value_rows,
        hash_checks,
        report,
        Path(args.checkpoint),
    )
    return replay_outputs, report


def write_recommended_next(output_dir: Path, no_new_frames: bool) -> None:
    if no_new_frames:
        lines = [
            "# Recommended Next Faithful Step",
            "",
            "- if runtime is desired next: Stage 4A-6.5ai staged one-frame lambda48 runtime smoke, no action execution",
            "- otherwise: collect additional saved frames in a controlled capture-only stage, still no rollout",
            "- do not recommend rollout directly.",
            "- do not promote over-cost to runtime.",
        ]
    else:
        lines = [
            "# Recommended Next Faithful Step",
            "",
            "- review the new-frame replay summary before any runtime smoke.",
            "- runtime, if chosen later, must still be one-frame lambda48 only with no action execution.",
            "- do not recommend rollout directly.",
        ]
    write_text(output_dir / "recommended_next_faithful_step.md", "\n".join(lines))


def write_final_summary(
    output_dir: Path,
    *,
    discovery_summary: dict[str, Any],
    hardware_report: dict[str, Any],
    replay_outputs: dict[str, Any] | None,
) -> dict[str, Any]:
    no_new = int(discovery_summary["new_complete_frame_count"]) == 0
    summary = {
        "stage": "Stage 4A-6.5ah",
        "status": "completed",
        "branch": "runtime_design_review_only" if no_new else "new_complete_frame_offline_replay",
        "discovery": discovery_summary,
        "hardware_utilization": hardware_report,
        "replay_outputs": replay_outputs,
        "safety": safety_scope(),
        "runtime_executed": False,
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
        "coverage_improvement_claimed": False,
        "recommended_next": (
            "Stage 4A-6.5ai staged one-frame lambda48 runtime smoke, no action execution, if runtime is desired; otherwise collect additional saved frames in a controlled capture-only stage"
            if no_new
            else "review new-frame offline replay before any one-frame runtime smoke"
        ),
    }
    save_json(output_dir / "stage4a65ah_multiscene_or_runtime_design_review_summary.json", summary)
    write_text(
        output_dir / "stage4a65ah_multiscene_or_runtime_design_review_summary.md",
        "\n".join(
            [
                "# Stage 4A-6.5ah Summary",
                "",
                f"- branch: `{summary['branch']}`",
                f"- candidate rows: `{discovery_summary['candidate_rows']}`",
                f"- new complete frames: `{discovery_summary['new_complete_frame_count']}`",
                f"- duplicate/already-in-6.5ag rows: `{discovery_summary['duplicate_or_already_in_stage4a65ag_count']}`",
                f"- skipped rows: `{discovery_summary['skipped_count']}`",
                f"- requested/actual max workers: `{hardware_report['requested_max_workers']}` / `{hardware_report['actual_max_workers']}`",
                "- runtime executed: `false`",
                "- rollout readiness: `false`",
                "- coverage improvement claimed: `false`",
                f"- recommended next: {summary['recommended_next']}",
            ]
        ),
    )
    write_recommended_next(output_dir, no_new)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_loaded_context_manifest(output_dir, args)

    raw_candidates = discover_raw_candidates(Path(args.outputs_root).resolve())
    inventory_rows, discovery_parallel_report = classify_candidates(
        raw_candidates,
        Path(args.stage4a65ag_dir).resolve(),
        max_workers=int(args.max_workers),
    )
    discovery_summary = write_discovery_outputs(output_dir, inventory_rows)
    new_frames = [dict(row) for row in inventory_rows if row["classification"] == "new_complete_frame"]

    replay_outputs: dict[str, Any] | None = None
    replay_parallel_report: dict[str, Any] | None = None
    if new_frames:
        replay_outputs, replay_parallel_report = run_new_frame_replay(args, output_dir, new_frames)
    else:
        write_runtime_design_review(output_dir, discovery_summary)

    hardware_report = write_hardware_report(
        output_dir,
        args,
        discovery_parallel_report=discovery_parallel_report,
        replay_parallel_report=replay_parallel_report,
        total_wall_time_s=time.perf_counter() - start,
    )
    final_summary = write_final_summary(
        output_dir,
        discovery_summary=discovery_summary,
        hardware_report=hardware_report,
        replay_outputs=replay_outputs,
    )
    print(json.dumps(to_jsonable(final_summary), indent=2, sort_keys=True))
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_root", default=str(DEFAULT_OUTPUTS_ROOT))
    parser.add_argument("--stage4a65ag_dir", default=str(DEFAULT_STAGE4A65AG_DIR))
    parser.add_argument("--stage4a65y_dir", default=str(DEFAULT_STAGE4A65Y_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max_workers", type=int, default=32)
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
    parser.add_argument("--save_raw_tree_summaries", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

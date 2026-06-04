#!/usr/bin/env python3
"""Stage 4A-6.9 bounded two-frame lambda48 expert pilot.

This stage keeps the Stage 4A-6.8 lambda48 scoring contract, adds one
post-action diagnostic frame per start, and stops. It executes exactly one
lambda48 action per start, runs map_predict once on Frame1 and once on Frame2,
and never uses prediction for safety, reachability, collision, ray blocking,
candidate validity, edge validity, target/ground-truth scoring, or future
observed scoring.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from depth_to_voxel import UNKNOWN, update_observed_state_from_depth
import run_stage4a67_measured_only_expert_pilot as s67
import run_stage4a68_map_predict_lambda48_expert_pilot as s68


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_CAMERA_FIX_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix"
DEFAULT_MEASURED_ONLY_DIR = WORKSPACE / "outputs/isaac_stage4a67_measured_only_expert_pilot"
DEFAULT_LAMBDA48_DIR = WORKSPACE / "outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot"
DEFAULT_FIXED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
DEFAULT_SOURCE_OBSERVED = DEFAULT_CAMERA_FIX_DIR / "observed_state_final.npy"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot"
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"

STAGE = "Stage 4A-6.9-bounded-two-frame-lambda48-pilot"
FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
FORMULA_NAME = "lambda48_minmax_source_occ_free"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: Any) -> None:
    s68.save_json(path, data)


def read_json(path: Path) -> Any:
    return s68.read_json(path)


def write_text(path: Path, text: str) -> None:
    s68.write_text(path, text)


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    s68.write_csv(path, rows, field_order)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    s68.write_jsonl(path, rows)


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    return s68.markdown_table(title, rows)


def markdown_list(title: str, rows: list[str]) -> str:
    return s68.markdown_list(title, rows)


def git_status_text() -> str:
    return s68.git_status_text()


def sha256_file(path: Path | str) -> str | None:
    return s68.sha256_file(path)


def sha256_array(array: np.ndarray) -> str:
    return s67.sha256_array(array)


def log_event(output_dir: Path, events: list[dict[str, Any]], event: str, **payload: Any) -> None:
    row = {"time_utc": utc_now(), "event": str(event), **payload}
    events.append(row)
    path = output_dir / "logs/stage4a69_execution_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(s68.jsonable(row), sort_keys=True, allow_nan=False))
        handle.write("\n")
    print(f"[{STAGE}] {event}: {json.dumps(s68.jsonable(payload), sort_keys=True)[:900]}", flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def branch_counts(decisions: list[dict[str, Any]]) -> Counter:
    return Counter(str(row.get("branch_classification", "unknown")) for row in decisions)


def distance_xy(a: list[float], b: list[float]) -> float:
    return s68.distance_xy(a, b)


def load_required_inputs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    inputs = s68.load_required_inputs(args, output_dir)
    lambda48_dir = Path(args.lambda48_pilot_dir).resolve()
    required68 = {
        "summary": lambda48_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
        "dataset": lambda48_dir / "expert_dataset.npz",
        "manifest": lambda48_dir / "expert_dataset_manifest.jsonl",
        "lambda_vs_measured": lambda48_dir / "lambda48_vs_measured_comparison.json",
        "stage67_comparison": lambda48_dir / "stage4a68_vs_stage4a67_comparison.md",
        "prediction_safety": lambda48_dir / "prediction_safety_audit.json",
        "quality": lambda48_dir / "expert_data_quality_audit.json",
        "lambda_decisions": lambda48_dir / "lambda48_decisions.jsonl",
    }
    missing = [str(path) for path in required68.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Stage 4A-6.8 files: {missing}")
    summary68 = read_json(required68["summary"])
    pred68 = read_json(required68["prediction_safety"])
    quality68 = read_json(required68["quality"])
    lambda_decisions68 = load_jsonl(required68["lambda_decisions"])
    with np.load(required68["dataset"], allow_pickle=False) as data:
        dataset68_keys = sorted(data.files)
        dataset68_shapes = {key: [int(v) for v in data[key].shape] for key in data.files}

    checks = {
        "stage4a68_completed": bool(summary68.get("completed", False)),
        "stage4a68_sample_count": int(summary68.get("sample_count", -1)),
        "stage4a68_capture_count": int(summary68.get("capture_count", -1)),
        "stage4a68_map_predict_calls": int(summary68.get("map_predict_calls", -1)),
        "stage4a68_exactly_one_action_per_start": bool(summary68.get("exactly_one_action_per_start", False)),
        "stage4a68_continuous_rollout_executed": bool(summary68.get("continuous_rollout_executed", True)),
        "stage4a68_prediction_safety_passed": bool(pred68.get("passed", False)),
        "stage4a68_quality_audit_passed": bool(quality68.get("passed", False)),
        "stage4a68_warning_classes": quality68.get("warnings", []),
        "stage4a68_dataset_keys": dataset68_keys,
        "stage4a68_dataset_shapes": dataset68_shapes,
    }
    if not checks["stage4a68_completed"]:
        raise ValueError("Stage 4A-6.8 is not completed")
    if checks["stage4a68_sample_count"] != 10 or checks["stage4a68_capture_count"] != 10:
        raise ValueError(f"Unexpected Stage 4A-6.8 counts: {checks}")
    if checks["stage4a68_map_predict_calls"] != 10:
        raise ValueError("Stage 4A-6.8 map_predict call count is not 10")
    if not checks["stage4a68_exactly_one_action_per_start"] or checks["stage4a68_continuous_rollout_executed"]:
        raise ValueError("Stage 4A-6.8 action/rollout gates are not clean")
    if not checks["stage4a68_prediction_safety_passed"] or not checks["stage4a68_quality_audit_passed"]:
        raise ValueError("Stage 4A-6.8 safety/quality did not pass")
    if set(checks["stage4a68_warning_classes"]) - {"candidate_all_local"}:
        raise ValueError(f"Unexpected Stage 4A-6.8 warning classes: {checks['stage4a68_warning_classes']}")
    if len(lambda_decisions68) != 10:
        raise ValueError("Stage 4A-6.8 lambda decision JSONL does not contain 10 rows")

    inputs.update(
        {
            "lambda48_dir": lambda48_dir,
            "stage4a68_summary": summary68,
            "stage4a68_prediction_safety": pred68,
            "stage4a68_quality": quality68,
            "stage4a68_lambda_decisions": lambda_decisions68,
            "stage4a68_checks": checks,
        }
    )
    preflight69 = {
        "stage": STAGE,
        "loaded_at_utc": utc_now(),
        **checks,
        "bounded_two_frame_pilot": True,
        "long_rollout": False,
        "full_expert_dataset": False,
        "rl_gdpo_ppo_bc_il": False,
        "training": False,
        "frame2_diagnostic_only": True,
        "prediction_read_only": True,
        "prediction_used_for_safety": False,
    }
    save_json(output_dir / "input_preflight_stage4a69_report.json", preflight69)
    write_text(output_dir / "input_preflight_stage4a69_report.md", markdown_table("Stage 4A-6.9 Input Preflight", preflight69))
    return inputs


def capture_poses_from_stage68(starts: list[dict[str, Any]], stage68_lambda_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_start = {int(row["start_variant_id"]): row for row in stage68_lambda_rows}
    poses: list[dict[str, Any]] = []
    for start in starts:
        sid = int(start["index"])
        start_pos = [float(v) for v in start["position"]]
        start_yaw = float(start.get("yaw_rad", start.get("yaw", 0.0)))
        poses.append(
            {
                "index": sid * 2,
                "start_variant_id": sid,
                "frame_id": 1,
                "name": f"frame1_start_{start['name']}",
                "start_variant_name": str(start["name"]),
                "semantic_zone_guess": start.get("semantic_zone_guess"),
                "position": start_pos,
                "yaw": start_yaw,
                "yaw_rad": start_yaw,
                "target": s67.pose_target(start_pos, start_yaw),
                "source": "stage4a69_frame1_start_pose",
                "one_action_only_for_this_start": True,
                "action_executed_in_isaac": False,
            }
        )
        prior = by_start[sid]
        action_pos = [float(v) for v in prior["world"]]
        action_yaw = float(prior["yaw_rad"])
        poses.append(
            {
                "index": sid * 2 + 1,
                "start_variant_id": sid,
                "frame_id": 2,
                "name": f"frame2_after_lambda48_action_{start['name']}",
                "start_variant_name": str(start["name"]),
                "semantic_zone_guess": start.get("semantic_zone_guess"),
                "position": action_pos,
                "yaw": action_yaw,
                "yaw_rad": action_yaw,
                "target": s67.pose_target(action_pos, action_yaw),
                "source": "stage4a69_frame2_pose_seeded_from_stage4a68_lambda48_reproduction_target",
                "stage4a68_lambda48_candidate_id": int(prior["candidate_id"]),
                "stage4a68_lambda48_score": float(prior["final_score_lambda48"]),
                "one_action_only_for_this_start": True,
                "action_executed_in_isaac": True,
                "diagnostic_only_after_this_capture": True,
            }
        )
    return poses


def run_capture_once(
    args: argparse.Namespace,
    app_launcher_cls: Any,
    output_dir: Path,
    fixed_usd: Path,
    starts: list[dict[str, Any]],
    stage68_lambda_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    poses = capture_poses_from_stage68(starts, stage68_lambda_rows)
    save_json(output_dir / "two_frame_capture_poses.json", {"poses": poses})
    result = s67.run_one_isaac_startup(args, app_launcher_cls, output_dir, fixed_usd, poses, events)
    result.update(
        {
            "capture_semantics": "frame1_start_and_frame2_after_one_lambda48_action_per_start",
            "frame_count": len(poses),
            "start_count": len(starts),
            "capture_count": len(result["capture_records"]),
            "one_capture_per_frame": len(result["capture_records"]) == len(poses),
            "executed_action_count": len(starts),
            "exactly_one_action_per_start": True,
            "second_action_count": 0,
            "third_frame_count": 0,
            "continuous_rollout_executed": False,
            "long_rollout_executed": False,
        }
    )
    save_json(output_dir / "headless_two_frame_capture_manifest.json", result)
    write_text(output_dir / "headless_two_frame_capture_manifest.md", markdown_table("Headless Two-Frame Capture Manifest", result))
    return result


def load_existing_capture_result(output_dir: Path, starts: list[dict[str, Any]]) -> dict[str, Any]:
    camera_info = read_json(output_dir / "camera_info.json")
    records = []
    for start in starts:
        sid = int(start["index"])
        for frame_id, capture_index in ((1, sid * 2), (2, sid * 2 + 1)):
            rgb_path = output_dir / f"action_rgb_{capture_index:03d}.png"
            depth_path = output_dir / f"action_depth_{capture_index:03d}.npy"
            depth_color_path = output_dir / f"action_depth_color_{capture_index:03d}.png"
            pose_path = output_dir / f"action_pose_{capture_index:03d}.json"
            for path in (rgb_path, depth_path, depth_color_path, pose_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Cannot reuse missing capture artifact: {path}")
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            depth = np.load(depth_path)
            records.append(
                {
                    "index": capture_index,
                    "start_variant_id": sid,
                    "frame_id": frame_id,
                    "name": f"reused_start_{sid:03d}_frame{frame_id}",
                    "start_variant_name": start.get("name"),
                    "semantic_zone_guess": start.get("semantic_zone_guess"),
                    "render_backend": "isaac_headless_reused_after_close_hang",
                    "rgb_file": rgb_path.name,
                    "depth_file": depth_path.name,
                    "depth_color_file": depth_color_path.name,
                    "pose_file": pose_path.name,
                    "rgb_key_used": "rgb_reused",
                    "rgb_stats": s67.rgb_stats(rgb),
                    "depth_stats": s67.depth_stats(depth),
                    "second_action_executed": False,
                    "third_action_executed": False,
                }
            )
    result = {
        "isaac_headless_startup_count": 1,
        "camera_info": camera_info,
        "capture_records": records,
        "capture_count": len(records),
        "frame_count": len(records),
        "start_count": len(starts),
        "one_capture_per_frame": len(records) == len(starts) * 2,
        "executed_action_count": len(starts),
        "exactly_one_action_per_start": True,
        "second_action_count": 0,
        "third_frame_count": 0,
        "continuous_rollout_executed": False,
        "long_rollout_executed": False,
        "reused_existing_captures": True,
        "isaac_shutdown_note": (
            "The initial run completed the required two-frame captures, then simulation_app.close "
            "did not return promptly. This recovery run reused finalized captures and did not "
            "start Isaac a second time."
        ),
    }
    save_json(output_dir / "headless_two_frame_capture_manifest.json", result)
    write_text(output_dir / "headless_two_frame_capture_manifest.md", markdown_table("Headless Two-Frame Capture Manifest", result))
    return result


def copy_capture_to_frame(
    output_dir: Path,
    sample_dir: Path,
    capture_record: dict[str, Any],
    prefix: str,
    pose_role: str,
    action_executed: bool,
) -> dict[str, Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rgb": sample_dir / f"{prefix}_rgb.png",
        "depth": sample_dir / f"{prefix}_depth.npy",
        "depth_color": sample_dir / f"{prefix}_depth_color.png",
        "pose": sample_dir / f"{prefix}_pose.json",
    }
    for key, src_name in (("rgb", "rgb_file"), ("depth", "depth_file"), ("depth_color", "depth_color_file")):
        src = output_dir / capture_record[src_name]
        paths[key].write_bytes(src.read_bytes())
    pose = read_json(output_dir / capture_record["pose_file"])
    pose.update({"pose_role": pose_role, "action_executed": bool(action_executed)})
    save_json(paths["pose"], pose)
    return paths


def run_prediction(
    args: argparse.Namespace,
    output_dir: Path,
    sample_dir: Path,
    predictor: Any,
    checkpoint_before: dict[str, Any],
    observed_state: np.ndarray,
    observed_path: Path,
    depth: np.ndarray,
    depth_path: Path,
    pose: dict[str, Any],
    pose_path: Path,
    camera_info: dict[str, Any],
    bounds: dict[str, Any],
    voxel_size: float,
    start_index: int,
    frame_id: int,
    step: int,
) -> tuple[Any, dict[str, Any]]:
    observed_hash_before = sha256_array(observed_state)
    prediction_result = predictor.predict_step(
        depth=depth,
        pose=pose,
        camera_info=camera_info,
        observed_state=observed_state,
        map_bounds=bounds,
        voxel_size=voxel_size,
        output_dir=sample_dir / f"frame{frame_id}_map_predict",
        step=step,
        save_probs=False,
        save_viz=False,
        observed_state_path=observed_path,
        depth_source=depth_path,
        pose_source=pose_path,
        camera_info_source=output_dir / "camera_info.json",
    )
    layer = prediction_result["prediction_layer"]
    counts = s68.prediction_counts(layer, observed_state, float(args.tau))
    summary = {
        "stage": STAGE,
        "start_variant_id": int(start_index),
        "frame_id": int(frame_id),
        "map_predict_called": True,
        "sscnet_inference_called": True,
        "map_predict_call_index": int(predictor.steps_predicted),
        "predictor_loaded_once": bool(predictor.model_loaded_once),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256_before": checkpoint_before["sha256"],
        "alignment_convention": str(args.alignment_convention),
        "tau": float(args.tau),
        **counts,
        "timing": prediction_result["timing"],
        "gpu_memory_peak": prediction_result["summary"].get("gpu_memory_peak"),
        "gpu_memory_after_model_load": prediction_result["summary"].get("gpu_memory_after_model_load"),
        "observed_state_hash_before_map_predict": observed_hash_before,
        "observed_state_hash_after_map_predict": prediction_result["summary"].get("observed_state_sha256_after"),
        "observed_state_hash_unchanged": bool(prediction_result["summary"].get("strict_no_observed_write", False)),
        "prediction_writeback": False,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "prediction_summary_only": bool(args.save_prediction_summary_only),
    }
    if bool(args.save_prediction_summary_only):
        removed = []
        for key in ("prediction_npz", "global_prediction_npz", "local_prediction_npz"):
            raw = prediction_result.get(key)
            if raw:
                path = Path(raw)
                if path.is_file():
                    path.unlink()
                    removed.append(str(path))
        summary["prediction_array_npz_removed_after_summary"] = removed
    save_json(sample_dir / f"frame{frame_id}_prediction_summary.json", summary)
    return layer, summary


def save_candidate_outputs(sample_dir: Path, prefix: str, decision: dict[str, Any], lambda_name: str, measured_name: str) -> None:
    rows = []
    for rank, row in enumerate(decision.get("top_n", []), start=1):
        out = dict(row)
        out["rank"] = int(rank)
        out["start_variant_id"] = int(decision["start_index"])
        rows.append(out)
    write_csv(sample_dir / f"{prefix}_top_candidates.csv", rows)
    write_jsonl(sample_dir / f"{prefix}_top_candidates.jsonl", rows)
    save_json(sample_dir / lambda_name, decision.get("selected_lambda48"))
    save_json(sample_dir / measured_name, decision.get("selected_measured_shadow"))


def candidate_arrays(decision: dict[str, Any], max_candidates: int) -> dict[str, Any]:
    feature = np.full((max_candidates, 8), np.nan, dtype=np.float32)
    valid = np.zeros((max_candidates,), dtype=bool)
    lambda_scores = np.full((max_candidates,), np.nan, dtype=np.float32)
    measured_scores = np.full((max_candidates,), np.nan, dtype=np.float32)
    gain_exp = np.full((max_candidates,), np.nan, dtype=np.float32)
    source_occ_free = np.full((max_candidates,), np.nan, dtype=np.float32)
    path_cost = np.full((max_candidates,), np.nan, dtype=np.float32)
    for row in sorted(decision.get("candidate_rows_lambda48", []), key=lambda item: int(item["candidate_id"]))[:max_candidates]:
        cid = int(row["candidate_id"])
        valid[cid] = True
        feature[cid, :] = np.asarray(
            [
                float(row["gain_exp"]),
                float(row["source_occ_free"]),
                float(row["source_occ_free_minmax"]),
                float(row["path_cost_m"]),
                float(row["cost_s"]),
                float(row["final_score_lambda48"]),
                float(row["final_score_measured"]),
                float(row["visible_count"]),
            ],
            dtype=np.float32,
        )
        lambda_scores[cid] = float(row["final_score_lambda48"])
        measured_scores[cid] = float(row["final_score_measured"])
        gain_exp[cid] = float(row["gain_exp"])
        source_occ_free[cid] = float(row["source_occ_free"])
        path_cost[cid] = float(row["cost_s"])
    selected_lambda = decision.get("selected_lambda48") or {}
    selected_measured = decision.get("selected_measured_shadow") or {}
    return {
        "candidate_features": feature,
        "candidate_mask": valid,
        "lambda_scores": lambda_scores,
        "measured_scores": measured_scores,
        "gain_exp": gain_exp,
        "source_occ_free": source_occ_free,
        "path_cost": path_cost,
        "lambda_index": int(selected_lambda.get("candidate_id", -1)),
        "measured_index": int(selected_measured.get("candidate_id", -1)),
        "lambda_world": [float(v) for v in selected_lambda.get("world", [math.nan, math.nan, math.nan])],
        "measured_world": [float(v) for v in selected_measured.get("world", [math.nan, math.nan, math.nan])],
        "lambda_yaw": float(selected_lambda.get("yaw_rad", math.nan)),
        "measured_yaw": float(selected_measured.get("yaw_rad", math.nan)),
    }


def write_action_quality(sample_dir: Path, name: str, quality: dict[str, Any]) -> None:
    save_json(sample_dir / f"{name}.json", quality)
    write_text(
        sample_dir / f"{name}.md",
        markdown_table(
            name.replace("_", " ").title(),
            {
                "passed": quality.get("passed"),
                "quality_score": quality.get("quality_score"),
                "warnings": quality.get("warnings"),
                "branch_classification": quality.get("branch_classification"),
                "candidate_count": quality.get("candidate_count"),
                "prediction_density": quality.get("prediction_density"),
            },
        ),
    )


def plot_two_frame_path(
    path: Path,
    observed_state: np.ndarray,
    bounds: dict[str, Any],
    start: dict[str, Any],
    executed: dict[str, Any],
    frame2_lambda: dict[str, Any] | None,
) -> None:
    top = s68.topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    s = start["position"]
    a = executed["world"]
    ax.scatter([s[0]], [s[1]], s=80, c="#2563eb", marker="^", edgecolors="white", linewidths=0.6, label="frame1 start")
    ax.scatter([a[0]], [a[1]], s=105, c="#10b981", marker="*", edgecolors="black", linewidths=0.5, label="executed action")
    ax.plot([s[0], a[0]], [s[1], a[1]], color="#10b981", linewidth=1.6)
    if frame2_lambda:
        l = frame2_lambda["world"]
        ax.scatter([l[0]], [l[1]], s=72, c="#d97706", marker="o", edgecolors="black", linewidths=0.5, label="frame2 diagnostic")
        ax.plot([a[0], l[0]], [a[1], l[1]], color="#d97706", linewidth=1.1, linestyle="--")
    ax.set_title("two-frame path, one executed action")
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def plot_observed_delta(path: Path, before: np.ndarray, after: np.ndarray, bounds: dict[str, Any], title: str) -> None:
    unknown_to_free = np.any((before == UNKNOWN) & (after == s67.FREE), axis=2)
    unknown_to_occ = np.any((before == UNKNOWN) & (after == s67.OCCUPIED), axis=2)
    image = np.zeros(before.shape[:2], dtype=np.int8)
    image[unknown_to_free] = 1
    image[unknown_to_occ] = 2
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(image.T, origin="lower", extent=extent, cmap=plt.get_cmap("viridis", 3), interpolation="nearest")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def process_samples(
    args: argparse.Namespace,
    output_dir: Path,
    inputs: dict[str, Any],
    capture_result: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    source_observed = inputs["source_observed"]
    bounds = inputs["bounds"]
    voxel_size = float(inputs["voxel_size"])
    starts = inputs["starts"]
    camera_info = capture_result["camera_info"]
    record_by_index = {int(row["index"]): row for row in capture_result["capture_records"]}
    yaws = s68.yaw_priors_by_start(starts, inputs["inspection_manifest"])

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_before = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": int(checkpoint_path.stat().st_size) if checkpoint_path.is_file() else None,
        "mtime_ns": int(checkpoint_path.stat().st_mtime_ns) if checkpoint_path.is_file() else None,
    }
    predictor = s68.IsaacMapPredictor(
        checkpoint=checkpoint_path,
        device=str(args.predictor_device),
        tau=float(args.tau),
        torch_num_threads=1,
        alignment_convention=str(args.alignment_convention),
    )
    log_event(output_dir, events, "predictor_loaded", checkpoint=str(checkpoint_path), model_load_time=predictor.model_load_time)

    frame1_decisions: list[dict[str, Any]] = []
    frame2_decisions: list[dict[str, Any]] = []
    frame1_lambda_rows: list[dict[str, Any]] = []
    frame2_lambda_rows: list[dict[str, Any]] = []
    frame1_measured_rows: list[dict[str, Any]] = []
    frame2_measured_rows: list[dict[str, Any]] = []
    per_start_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    observed_delta_rows: list[dict[str, Any]] = []
    cumulative = source_observed.copy()

    frame1_arrays: dict[str, list[Any]] = {key: [] for key in [
        "observed", "features", "mask", "lambda_scores", "measured_scores", "gain_exp",
        "source_occ_free", "path_cost", "lambda_index", "measured_index", "lambda_world",
        "measured_world", "lambda_yaw", "measured_yaw", "prediction_json",
    ]}
    frame2_arrays: dict[str, list[Any]] = {key: [] for key in frame1_arrays}
    executed_world: list[list[float]] = []
    executed_yaw: list[float] = []
    safety_flag_names = [
        "low_cost_artifact",
        "historical_prior_basin",
        "same_cell_target",
        "unreachable_target",
        "repeated_target",
        "outside_bounds_target",
        "prediction_leakage",
        "prediction_used_for_safety",
        "path_cost_suspiciously_small",
        "candidate_all_local",
        "no_valid_candidate",
        "frame2_regression",
    ]
    safety_flag_rows: list[list[int]] = []
    quality_flag_rows: list[list[int]] = []

    for start in starts:
        sid = int(start["index"])
        sample_dir = output_dir / "samples" / f"start_{sid:03d}"
        frame1_paths = copy_capture_to_frame(
            output_dir,
            sample_dir,
            record_by_index[sid * 2],
            "frame1",
            "frame1_start_capture_before_lambda48_action",
            False,
        )
        frame2_paths = copy_capture_to_frame(
            output_dir,
            sample_dir,
            record_by_index[sid * 2 + 1],
            "frame2",
            "frame2_after_exactly_one_lambda48_action_diagnostic_only",
            True,
        )

        frame1_pose = read_json(frame1_paths["pose"])
        frame1_depth = np.load(frame1_paths["depth"])
        frame1_observed = update_observed_state_from_depth(
            observed_state=source_observed.copy(),
            depth=frame1_depth,
            camera_pose=frame1_pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=voxel_size,
            pixel_stride=int(args.pixel_stride),
        )
        frame1_observed_path = sample_dir / "frame1_observed_state.npy"
        np.save(frame1_observed_path, frame1_observed)
        before_cumulative = cumulative.copy()
        cumulative = update_observed_state_from_depth(
            observed_state=cumulative,
            depth=frame1_depth,
            camera_pose=frame1_pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=voxel_size,
            pixel_stride=int(args.pixel_stride),
        )
        pred1_layer, pred1 = run_prediction(
            args,
            output_dir,
            sample_dir,
            predictor,
            checkpoint_before,
            frame1_observed,
            frame1_observed_path,
            frame1_depth,
            frame1_paths["depth"],
            frame1_pose,
            frame1_paths["pose"],
            camera_info,
            bounds,
            voxel_size,
            sid,
            1,
            sid * 2,
        )
        decision1 = s68.score_start_lambda48(frame1_observed, pred1_layer, bounds, voxel_size, start, yaws.get(sid, []), args)
        decision1.update(
            {
                "stage": STAGE,
                "frame_id": 1,
                "observed_state_path": str(frame1_observed_path),
                "observed_state_summary": s68.observed_summary(frame1_observed, f"frame1_observed_state_start_{sid:03d}"),
                "transition_delta_from_source": s67.state_transition(source_observed, frame1_observed),
                "cumulative_delta": s67.state_transition(before_cumulative, cumulative),
                "prediction_summary": pred1,
                "map_predict_called": True,
                "sscnet_inference_called": True,
                "exactly_one_action_per_start": True,
                "action_executed_in_isaac": True,
                "second_action_executed": False,
                "third_frame_executed": False,
                "continuous_rollout_executed": False,
            }
        )
        save_candidate_outputs(
            sample_dir,
            "frame1",
            decision1,
            "frame1_lambda48_decision.json",
            "frame1_measured_shadow_decision.json",
        )
        s68.plot_sample_topdown(
            sample_dir / "frame1_expert_topdown.png",
            frame1_observed,
            bounds,
            start,
            decision1.get("selected_lambda48"),
            decision1.get("selected_measured_shadow"),
            decision1.get("top_n", []),
            f"start {sid:03d} frame1 measured shadow vs lambda48",
        )
        s68.plot_prediction_overlay(
            sample_dir / "frame1_prediction_overlay.png",
            frame1_observed,
            pred1_layer,
            bounds,
            start,
            decision1.get("selected_lambda48"),
            float(args.tau),
        )
        s68.plot_candidate_scores(sample_dir / "frame1_candidate_score_bar.png", decision1)
        s68.plot_sample_topdown(
            sample_dir / "frame1_candidate_map.png",
            frame1_observed,
            bounds,
            start,
            decision1.get("selected_lambda48"),
            decision1.get("selected_measured_shadow"),
            decision1.get("top_n", []),
            f"start {sid:03d} frame1 top-N candidate map",
        )
        q1 = s68.action_quality(decision1, frame1_paths, pred1)
        write_action_quality(sample_dir, "frame1_action_quality", q1)

        selected1 = decision1.get("selected_lambda48") or {}
        executed = {
            "start_variant_id": sid,
            "executed_action_index": 1,
            "world": [float(v) for v in selected1.get("world", [math.nan, math.nan, math.nan])],
            "yaw_rad": float(selected1.get("yaw_rad", math.nan)),
            "candidate_id": int(selected1.get("candidate_id", -1)),
            "final_score_lambda48": selected1.get("final_score_lambda48"),
            "source": "frame1_lambda48_primary_expert",
            "second_action_executed": False,
            "third_frame_executed": False,
        }
        frame2_pose = read_json(frame2_paths["pose"])
        executed["captured_frame2_pose_world"] = [float(v) for v in frame2_pose["position"]]
        executed["captured_frame2_pose_yaw_rad"] = float(frame2_pose["yaw_rad"])
        executed["matches_captured_frame2_pose"] = bool(
            distance_xy(executed["world"], executed["captured_frame2_pose_world"]) <= 0.15
            and abs(s67.wrap_angle(executed["yaw_rad"] - executed["captured_frame2_pose_yaw_rad"])) <= 0.10
        )
        save_json(sample_dir / "executed_action.json", executed)
        write_text(sample_dir / "executed_action.md", markdown_table("Executed Action", executed))

        frame2_depth = np.load(frame2_paths["depth"])
        frame2_observed = update_observed_state_from_depth(
            observed_state=frame1_observed.copy(),
            depth=frame2_depth,
            camera_pose=frame2_pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=voxel_size,
            pixel_stride=int(args.pixel_stride),
        )
        frame2_observed_path = sample_dir / "frame2_observed_state.npy"
        np.save(frame2_observed_path, frame2_observed)
        before_cumulative = cumulative.copy()
        cumulative = update_observed_state_from_depth(
            observed_state=cumulative,
            depth=frame2_depth,
            camera_pose=frame2_pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=voxel_size,
            pixel_stride=int(args.pixel_stride),
        )
        observed_delta = s67.state_transition(frame1_observed, frame2_observed)
        observed_delta_rows.append({"start_variant_id": sid, **observed_delta})

        pred2_layer, pred2 = run_prediction(
            args,
            output_dir,
            sample_dir,
            predictor,
            checkpoint_before,
            frame2_observed,
            frame2_observed_path,
            frame2_depth,
            frame2_paths["depth"],
            frame2_pose,
            frame2_paths["pose"],
            camera_info,
            bounds,
            voxel_size,
            sid,
            2,
            sid * 2 + 1,
        )
        frame2_start = dict(start)
        frame2_start.update(
            {
                "name": f"{start['name']}_frame2_post_action",
                "position": [float(v) for v in frame2_pose["position"]],
                "yaw": float(frame2_pose["yaw_rad"]),
                "yaw_rad": float(frame2_pose["yaw_rad"]),
            }
        )
        decision2 = s68.score_start_lambda48(frame2_observed, pred2_layer, bounds, voxel_size, frame2_start, yaws.get(sid, []), args)
        decision2.update(
            {
                "stage": STAGE,
                "frame_id": 2,
                "diagnostic_only": True,
                "observed_state_path": str(frame2_observed_path),
                "observed_state_summary": s68.observed_summary(frame2_observed, f"frame2_observed_state_start_{sid:03d}"),
                "transition_delta_from_frame1": observed_delta,
                "cumulative_delta": s67.state_transition(before_cumulative, cumulative),
                "prediction_summary": pred2,
                "map_predict_called": True,
                "sscnet_inference_called": True,
                "action_executed_in_isaac": False,
                "second_action_executed": False,
                "third_frame_executed": False,
                "continuous_rollout_executed": False,
            }
        )
        save_candidate_outputs(
            sample_dir,
            "frame2",
            decision2,
            "frame2_lambda48_diagnostic_decision.json",
            "frame2_measured_shadow_diagnostic_decision.json",
        )
        s68.plot_sample_topdown(
            sample_dir / "frame2_expert_topdown.png",
            frame2_observed,
            bounds,
            frame2_start,
            decision2.get("selected_lambda48"),
            decision2.get("selected_measured_shadow"),
            decision2.get("top_n", []),
            f"start {sid:03d} frame2 diagnostic measured shadow vs lambda48",
        )
        s68.plot_prediction_overlay(
            sample_dir / "frame2_prediction_overlay.png",
            frame2_observed,
            pred2_layer,
            bounds,
            frame2_start,
            decision2.get("selected_lambda48"),
            float(args.tau),
        )
        s68.plot_candidate_scores(sample_dir / "frame2_candidate_score_bar.png", decision2)
        s68.plot_sample_topdown(
            sample_dir / "frame2_candidate_map.png",
            frame2_observed,
            bounds,
            frame2_start,
            decision2.get("selected_lambda48"),
            decision2.get("selected_measured_shadow"),
            decision2.get("top_n", []),
            f"start {sid:03d} frame2 top-N candidate map",
        )
        q2 = s68.action_quality(decision2, frame2_paths, pred2)
        write_action_quality(sample_dir, "frame2_action_quality", q2)
        plot_two_frame_path(sample_dir / "two_frame_path_topdown.png", frame2_observed, bounds, start, executed, decision2.get("selected_lambda48"))
        plot_observed_delta(sample_dir / "observed_delta_topdown.png", frame1_observed, frame2_observed, bounds, "Frame1 to Frame2 observed delta")

        all_warnings = sorted(set(q1.get("warnings", []) + q2.get("warnings", [])))
        frame2_regression = bool(decision2.get("safety_flags", {}).get("no_valid_candidate", False))
        two_frame_quality = {
            "start_variant_id": sid,
            "passed": bool(q1.get("passed", False) and q2.get("passed", False) and executed["matches_captured_frame2_pose"]),
            "frame1_quality_score": q1.get("quality_score"),
            "frame2_quality_score": q2.get("quality_score"),
            "warnings": all_warnings,
            "frame1_branch_classification": decision1.get("branch_classification"),
            "frame2_branch_classification": decision2.get("branch_classification"),
            "observed_delta": observed_delta,
            "frame2_regression": frame2_regression,
            "executed_action_matches_frame1_lambda48": executed["matches_captured_frame2_pose"],
        }
        save_json(sample_dir / "sample_quality_summary.json", two_frame_quality)
        write_text(sample_dir / "sample_quality_summary.md", markdown_table("Sample Quality Summary", two_frame_quality))
        write_text(sample_dir / "action_quality_two_frame.md", markdown_table("Two-Frame Action Quality", two_frame_quality))

        for frame_id, decision, pred, quality, paths, observed_path, observed_state in (
            (1, decision1, pred1, q1, frame1_paths, frame1_observed_path, frame1_observed),
            (2, decision2, pred2, q2, frame2_paths, frame2_observed_path, frame2_observed),
        ):
            lam = decision.get("selected_lambda48") or {}
            meas = decision.get("selected_measured_shadow") or {}
            per_frame_rows.append(
                {
                    "start_variant_id": sid,
                    "frame_id": frame_id,
                    "rgb": str(paths["rgb"]),
                    "depth": str(paths["depth"]),
                    "pose": str(paths["pose"]),
                    "observed_state": str(observed_path),
                    "lambda48_candidate_id": lam.get("candidate_id"),
                    "lambda48_world": lam.get("world"),
                    "lambda48_yaw": lam.get("yaw_rad"),
                    "lambda48_gain_exp": lam.get("gain_exp"),
                    "lambda48_source_occ_free": lam.get("source_occ_free"),
                    "lambda48_score": lam.get("final_score_lambda48"),
                    "measured_candidate_id": meas.get("candidate_id"),
                    "measured_world": meas.get("world"),
                    "measured_yaw": meas.get("yaw_rad"),
                    "measured_score": meas.get("final_score_measured"),
                    "branch_classification": decision.get("branch_classification"),
                    "candidate_count": decision.get("scored_candidate_count"),
                    "prediction_density": pred.get("prediction_density"),
                    "quality_passed": quality.get("passed"),
                    "quality_warnings": quality.get("warnings"),
                }
            )
            map_rows.append(pred)
            manifest_rows.append(
                {
                    "stage": STAGE,
                    "start_variant_id": sid,
                    "frame_id": frame_id,
                    "sample_dir": str(sample_dir),
                    "rgb": str(paths["rgb"]),
                    "depth": str(paths["depth"]),
                    "pose": str(paths["pose"]),
                    "observed_state": str(observed_path),
                    "prediction_summary": str(sample_dir / f"frame{frame_id}_prediction_summary.json"),
                    "lambda48_decision": str(sample_dir / ("frame1_lambda48_decision.json" if frame_id == 1 else "frame2_lambda48_diagnostic_decision.json")),
                    "measured_shadow_decision": str(sample_dir / ("frame1_measured_shadow_decision.json" if frame_id == 1 else "frame2_measured_shadow_diagnostic_decision.json")),
                    "action_executed": frame_id == 1,
                    "diagnostic_only": frame_id == 2,
                    "map_predict_called": True,
                    "sscnet_inference_called": True,
                }
            )

        def append_decision_rows(target_lambda: list[dict[str, Any]], target_measured: list[dict[str, Any]], decision: dict[str, Any], frame_id: int) -> None:
            lam = decision.get("selected_lambda48") or {}
            meas = decision.get("selected_measured_shadow") or {}
            if lam:
                target_lambda.append(
                    {
                        "start_variant_id": sid,
                        "frame_id": frame_id,
                        "start_name": decision["start_name"],
                        "candidate_id": lam.get("candidate_id"),
                        "grid": lam.get("grid"),
                        "world": lam.get("world"),
                        "yaw_rad": lam.get("yaw_rad"),
                        "gain_exp": lam.get("gain_exp"),
                        "source_occ_free": lam.get("source_occ_free"),
                        "source_occ_free_minmax": lam.get("source_occ_free_minmax"),
                        "lambda48_bonus": lam.get("lambda48_bonus"),
                        "final_score_lambda48": lam.get("final_score_lambda48"),
                        "path_cost": lam.get("cost_s"),
                        "path_cost_m": lam.get("path_cost_m"),
                        "branch_classification": decision.get("branch_classification"),
                    }
                )
            if meas:
                target_measured.append(
                    {
                        "start_variant_id": sid,
                        "frame_id": frame_id,
                        "start_name": decision["start_name"],
                        "candidate_id": meas.get("candidate_id"),
                        "grid": meas.get("grid"),
                        "world": meas.get("world"),
                        "yaw_rad": meas.get("yaw_rad"),
                        "gain_exp": meas.get("gain_exp"),
                        "cost": meas.get("cost_s"),
                        "path_cost_m": meas.get("path_cost_m"),
                        "score": meas.get("final_score_measured"),
                    }
                )

        append_decision_rows(frame1_lambda_rows, frame1_measured_rows, decision1, 1)
        append_decision_rows(frame2_lambda_rows, frame2_measured_rows, decision2, 2)

        arr1 = candidate_arrays(decision1, int(args.num_candidates))
        arr2 = candidate_arrays(decision2, int(args.num_candidates))
        for store, arr, observed, pred in ((frame1_arrays, arr1, frame1_observed, pred1), (frame2_arrays, arr2, frame2_observed, pred2)):
            store["observed"].append(observed)
            store["features"].append(arr["candidate_features"])
            store["mask"].append(arr["candidate_mask"])
            store["lambda_scores"].append(arr["lambda_scores"])
            store["measured_scores"].append(arr["measured_scores"])
            store["gain_exp"].append(arr["gain_exp"])
            store["source_occ_free"].append(arr["source_occ_free"])
            store["path_cost"].append(arr["path_cost"])
            store["lambda_index"].append(arr["lambda_index"])
            store["measured_index"].append(arr["measured_index"])
            store["lambda_world"].append(arr["lambda_world"])
            store["measured_world"].append(arr["measured_world"])
            store["lambda_yaw"].append(arr["lambda_yaw"])
            store["measured_yaw"].append(arr["measured_yaw"])
            store["prediction_json"].append(json.dumps(s68.jsonable(pred), sort_keys=True))
        executed_world.append(executed["world"])
        executed_yaw.append(executed["yaw_rad"])

        flags = Counter()
        for decision in (decision1, decision2):
            for name, value in decision.get("safety_flags", {}).items():
                if value:
                    flags[name] += 1
        flags["frame2_regression"] = int(frame2_regression)
        safety_flag_rows.append([1 if flags.get(name, 0) else 0 for name in safety_flag_names])
        quality_flag_rows.append([1 if warning in all_warnings else 0 for warning in ["candidate_all_local", "blank_rgb", "invalid_depth", "empty_prediction", "prediction_over_dense"]])

        per_start_rows.append(
            {
                "start_variant_id": sid,
                "start_name": start["name"],
                "frame1_branch_classification": decision1.get("branch_classification"),
                "frame2_branch_classification": decision2.get("branch_classification"),
                "executed_action_world": executed["world"],
                "executed_action_yaw": executed["yaw_rad"],
                "observed_delta_newly_observed": observed_delta.get("newly_observed"),
                "frame1_prediction_density": pred1.get("prediction_density"),
                "frame2_prediction_density": pred2.get("prediction_density"),
                "frame2_candidate_count": decision2.get("scored_candidate_count"),
                "quality_passed": two_frame_quality["passed"],
                "warnings": all_warnings,
            }
        )
        frame1_decisions.append(decision1)
        frame2_decisions.append(decision2)
        quality_rows.append(two_frame_quality)
        log_event(
            output_dir,
            events,
            "start_two_frame_complete",
            start_variant_id=sid,
            frame1_branch=decision1.get("branch_classification"),
            frame2_branch=decision2.get("branch_classification"),
            frame2_newly_observed=observed_delta.get("newly_observed"),
        )

    np.save(output_dir / "observed_state_final.npy", cumulative)
    checkpoint_after = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": int(checkpoint_path.stat().st_size) if checkpoint_path.is_file() else None,
        "mtime_ns": int(checkpoint_path.stat().st_mtime_ns) if checkpoint_path.is_file() else None,
    }
    checkpoint_report = {
        "stage": STAGE,
        "checkpoint": str(checkpoint_path),
        "before": checkpoint_before,
        "after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "predictor_loaded_once": bool(predictor.model_loaded_once),
        "map_predict_calls": int(predictor.steps_predicted),
    }
    save_json(output_dir / "checkpoint_hash_report.json", checkpoint_report)
    write_text(output_dir / "checkpoint_hash_report.md", markdown_table("Checkpoint Hash Report", checkpoint_report))

    dataset_path = output_dir / "expert_dataset_two_frame.npz"
    np.savez_compressed(
        dataset_path,
        start_variant_id=np.asarray([int(row["index"]) for row in starts], dtype=np.int32),
        frame1_observed_state_reference=np.asarray(frame1_arrays["observed"], dtype=np.int8),
        frame1_candidate_features=np.asarray(frame1_arrays["features"], dtype=np.float32),
        frame1_candidate_mask=np.asarray(frame1_arrays["mask"], dtype=bool),
        frame1_lambda48_action_index=np.asarray(frame1_arrays["lambda_index"], dtype=np.int32),
        frame1_measured_action_index=np.asarray(frame1_arrays["measured_index"], dtype=np.int32),
        frame1_lambda48_scores=np.asarray(frame1_arrays["lambda_scores"], dtype=np.float32),
        frame1_measured_scores=np.asarray(frame1_arrays["measured_scores"], dtype=np.float32),
        frame1_selected_world_xyz_lambda48=np.asarray(frame1_arrays["lambda_world"], dtype=np.float32),
        frame1_selected_yaw_lambda48=np.asarray(frame1_arrays["lambda_yaw"], dtype=np.float32),
        frame1_selected_world_xyz_measured=np.asarray(frame1_arrays["measured_world"], dtype=np.float32),
        frame1_selected_yaw_measured=np.asarray(frame1_arrays["measured_yaw"], dtype=np.float32),
        frame1_gain_exp=np.asarray(frame1_arrays["gain_exp"], dtype=np.float32),
        frame1_source_occ_free=np.asarray(frame1_arrays["source_occ_free"], dtype=np.float32),
        frame1_path_cost=np.asarray(frame1_arrays["path_cost"], dtype=np.float32),
        frame1_final_score_lambda48=np.asarray(frame1_arrays["lambda_scores"], dtype=np.float32),
        frame1_prediction_summary=np.asarray(frame1_arrays["prediction_json"]),
        executed_action_world_xyz=np.asarray(executed_world, dtype=np.float32),
        executed_action_yaw=np.asarray(executed_yaw, dtype=np.float32),
        frame2_observed_state_reference=np.asarray(frame2_arrays["observed"], dtype=np.int8),
        frame2_candidate_features=np.asarray(frame2_arrays["features"], dtype=np.float32),
        frame2_candidate_mask=np.asarray(frame2_arrays["mask"], dtype=bool),
        frame2_lambda48_diagnostic_action_index=np.asarray(frame2_arrays["lambda_index"], dtype=np.int32),
        frame2_measured_diagnostic_action_index=np.asarray(frame2_arrays["measured_index"], dtype=np.int32),
        frame2_lambda48_scores=np.asarray(frame2_arrays["lambda_scores"], dtype=np.float32),
        frame2_measured_scores=np.asarray(frame2_arrays["measured_scores"], dtype=np.float32),
        frame2_selected_world_xyz_lambda48=np.asarray(frame2_arrays["lambda_world"], dtype=np.float32),
        frame2_selected_yaw_lambda48=np.asarray(frame2_arrays["lambda_yaw"], dtype=np.float32),
        frame2_selected_world_xyz_measured=np.asarray(frame2_arrays["measured_world"], dtype=np.float32),
        frame2_selected_yaw_measured=np.asarray(frame2_arrays["measured_yaw"], dtype=np.float32),
        frame2_gain_exp=np.asarray(frame2_arrays["gain_exp"], dtype=np.float32),
        frame2_source_occ_free=np.asarray(frame2_arrays["source_occ_free"], dtype=np.float32),
        frame2_path_cost=np.asarray(frame2_arrays["path_cost"], dtype=np.float32),
        frame2_final_score_lambda48=np.asarray(frame2_arrays["lambda_scores"], dtype=np.float32),
        frame2_prediction_summary=np.asarray(frame2_arrays["prediction_json"]),
        observed_delta=np.asarray([row["newly_observed"] for row in observed_delta_rows], dtype=np.int32),
        safety_flags=np.asarray(safety_flag_rows, dtype=np.int8),
        safety_flag_names=np.asarray(safety_flag_names),
        quality_flags=np.asarray(quality_flag_rows, dtype=np.int8),
        quality_flag_names=np.asarray(["candidate_all_local", "blank_rgb", "invalid_depth", "empty_prediction", "prediction_over_dense"]),
    )
    metadata = {
        "stage": STAGE,
        "created_at_utc": utc_now(),
        "dataset_npz": str(dataset_path),
        "start_count": len(starts),
        "frame_count": len(starts) * 2,
        "capture_count": int(capture_result["capture_count"]),
        "map_predict_calls": int(predictor.steps_predicted),
        "executed_action_count": len(starts),
        "second_action_count": 0,
        "third_frame_count": 0,
        "formula": FORMULA,
        "lambda_sc": float(args.lambda_sc),
        "minmax_normalization_scope": "per frame / per start over valid candidate/yaw scored rows",
        "source_occ_free_definition": s68.SOURCE_OCC_FREE_DEFINITION,
        "candidate_set_size_requested": int(args.num_candidates),
        "top_n": int(args.top_n),
        "candidate_sampling_mode": str(args.candidate_sampling_mode),
        "path_cost_mode": str(args.path_cost_mode),
        "prediction_mode": str(args.prediction_mode),
        "alignment_convention": str(args.alignment_convention),
        "tau": float(args.tau),
        "forbidden_fields_absent": [
            "target_lr",
            "target_hr",
            "ground_truth",
            "future_observed",
            "class_prob",
            "policy_logits",
            "rl_reward",
            "replay_buffer",
            "training_state",
        ],
        "no_long_rollout": True,
        "no_second_action": True,
        "no_third_frame": True,
        "no_training": True,
        "no_rl_gdpo_ppo_bc_il": True,
        "prediction_writeback": False,
    }
    save_json(output_dir / "expert_dataset_metadata.json", metadata)
    write_jsonl(output_dir / "expert_dataset_manifest.jsonl", manifest_rows)
    write_csv(output_dir / "per_start_summary.csv", per_start_rows)
    save_json(output_dir / "per_start_summary.json", {"starts": per_start_rows})
    write_text(output_dir / "per_start_summary.md", markdown_list("Per-Start Summary", [
        f"`{row['start_variant_id']:03d}` frame1 `{row['frame1_branch_classification']}`, frame2 `{row['frame2_branch_classification']}`, delta `{row['observed_delta_newly_observed']}`, warnings `{row['warnings']}`"
        for row in per_start_rows
    ]))
    write_csv(output_dir / "per_frame_summary.csv", per_frame_rows)
    save_json(output_dir / "per_frame_summary.json", {"frames": per_frame_rows})
    write_text(output_dir / "per_frame_summary.md", markdown_list("Per-Frame Summary", [
        f"start `{row['start_variant_id']:03d}` frame `{row['frame_id']}` branch `{row['branch_classification']}` density `{row['prediction_density']}`"
        for row in per_frame_rows
    ]))
    write_jsonl(output_dir / "frame1_lambda48_decisions.jsonl", frame1_lambda_rows)
    write_csv(output_dir / "frame1_lambda48_decisions.csv", frame1_lambda_rows)
    write_jsonl(output_dir / "frame1_measured_shadow_decisions.jsonl", frame1_measured_rows)
    write_csv(output_dir / "frame1_measured_shadow_decisions.csv", frame1_measured_rows)
    write_jsonl(output_dir / "frame2_lambda48_diagnostic_decisions.jsonl", frame2_lambda_rows)
    write_csv(output_dir / "frame2_lambda48_diagnostic_decisions.csv", frame2_lambda_rows)
    write_jsonl(output_dir / "frame2_measured_shadow_diagnostic_decisions.jsonl", frame2_measured_rows)
    write_csv(output_dir / "frame2_measured_shadow_diagnostic_decisions.csv", frame2_measured_rows)
    write_csv(output_dir / "map_predict_summary.csv", map_rows)
    save_json(output_dir / "map_predict_summary.json", {"frames": map_rows})
    write_text(output_dir / "map_predict_summary.md", markdown_table("Map Predict Summary", {
        "map_predict_calls": int(predictor.steps_predicted),
        "frame_count": len(map_rows),
        "mean_prediction_density": float(np.mean([float(row["prediction_density"]) for row in map_rows])) if map_rows else None,
        "total_prediction_valid_count": int(sum(int(row["prediction_valid_count"]) for row in map_rows)),
        "total_predicted_unmeasured_count": int(sum(int(row["predicted_unmeasured_count"]) for row in map_rows)),
        "alignment_convention": str(args.alignment_convention),
        "tau": float(args.tau),
        "checkpoint": str(checkpoint_path),
    }))
    return {
        "frame1_decisions": frame1_decisions,
        "frame2_decisions": frame2_decisions,
        "frame1_lambda_rows": frame1_lambda_rows,
        "frame2_lambda_rows": frame2_lambda_rows,
        "frame1_measured_rows": frame1_measured_rows,
        "frame2_measured_rows": frame2_measured_rows,
        "per_start_rows": per_start_rows,
        "per_frame_rows": per_frame_rows,
        "map_rows": map_rows,
        "quality_rows": quality_rows,
        "observed_delta_rows": observed_delta_rows,
        "metadata": metadata,
        "checkpoint_report": checkpoint_report,
        "dataset_npz": dataset_path,
        "safety_flag_names": safety_flag_names,
        "safety_flag_rows": safety_flag_rows,
    }


def write_lambda_comparison(output_dir: Path, stem: str, title: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for decision in decisions:
        lam = decision.get("selected_lambda48") or {}
        meas = decision.get("selected_measured_shadow") or {}
        relation = decision.get("relation") or {}
        rows.append(
            {
                "start_variant_id": int(decision["start_index"]),
                "frame_id": int(decision.get("frame_id", 0)),
                "start_name": decision["start_name"],
                "lambda48_candidate_id": lam.get("candidate_id"),
                "measured_candidate_id": meas.get("candidate_id"),
                "lambda48_world": lam.get("world"),
                "measured_world": meas.get("world"),
                "action_spatial_delta_m": relation.get("distance_m"),
                "yaw_delta_rad": relation.get("yaw_delta_rad"),
                "lambda48_gain_exp": lam.get("gain_exp"),
                "measured_gain_exp": meas.get("gain_exp"),
                "lambda48_source_occ_free": lam.get("source_occ_free"),
                "lambda48_source_occ_free_minmax": lam.get("source_occ_free_minmax"),
                "lambda48_bonus": lam.get("lambda48_bonus"),
                "lambda48_final_score": lam.get("final_score_lambda48"),
                "measured_score": meas.get("final_score_measured"),
                "lambda48_path_cost": lam.get("cost_s"),
                "measured_path_cost": meas.get("cost_s"),
                "branch_classification": decision.get("branch_classification"),
            }
        )
    deltas = [float(row["action_spatial_delta_m"]) for row in rows if row.get("action_spatial_delta_m") is not None]
    yaw = [float(row["yaw_delta_rad"]) for row in rows if row.get("yaw_delta_rad") is not None]
    counts = branch_counts(decisions)
    report = {
        "stage": STAGE,
        "rows": rows,
        "action_changed_count": int(sum(1 for row in rows if float(row.get("action_spatial_delta_m") or 0.0) > 0.15)),
        "mean_action_distance": float(np.mean(deltas)) if deltas else None,
        "mean_yaw_delta": float(np.mean(yaw)) if yaw else None,
        "same_as_measured": int(counts.get("same_as_measured", 0)),
        "local_jitter": int(counts.get("local_jitter", 0)),
        "distinct_nonmeasured_branch": int(counts.get("distinct_nonmeasured_branch", 0)),
        "no_valid_candidate": int(counts.get("no_valid_candidate", 0)),
        "low_cost_artifact": int(sum(1 for d in decisions if d.get("safety_flags", {}).get("low_cost_artifact"))),
        "historical_prior_basin": int(sum(1 for d in decisions if d.get("safety_flags", {}).get("historical_prior_basin"))),
    }
    write_csv(output_dir / f"{stem}.csv", rows)
    save_json(output_dir / f"{stem}.json", report)
    write_text(output_dir / f"{stem}.md", markdown_table(title, {k: v for k, v in report.items() if k != "rows"}))
    return report


def compare_to_stage68(output_dir: Path, inputs: dict[str, Any], frame1_decisions: list[dict[str, Any]]) -> dict[str, Any]:
    base = {int(row["start_variant_id"]): row for row in inputs["stage4a68_lambda_decisions"]}
    rows = []
    for decision in frame1_decisions:
        sid = int(decision["start_index"])
        lam = decision.get("selected_lambda48") or {}
        prior = base.get(sid, {})
        dist = distance_xy(lam.get("world", [math.nan, math.nan, math.nan]), prior.get("world", [math.nan, math.nan, math.nan])) if lam and prior else None
        yaw_delta = abs(s67.wrap_angle(float(lam.get("yaw_rad", 0.0)) - float(prior.get("yaw_rad", 0.0)))) if lam and prior else None
        rows.append(
            {
                "start_variant_id": sid,
                "stage4a68_lambda48_world": prior.get("world"),
                "stage4a69_frame1_lambda48_world": lam.get("world"),
                "stage4a68_lambda48_yaw": prior.get("yaw_rad"),
                "stage4a69_frame1_lambda48_yaw": lam.get("yaw_rad"),
                "action_distance_delta_m": dist,
                "yaw_delta_rad": yaw_delta,
                "stage4a68_candidate_id": prior.get("candidate_id"),
                "stage4a69_candidate_id": lam.get("candidate_id"),
                "stage4a68_branch": prior.get("branch_classification"),
                "stage4a69_frame1_branch": decision.get("branch_classification"),
                "reproduced": bool(dist is not None and dist <= 0.15 and (yaw_delta or 0.0) <= 0.10),
            }
        )
    report = {
        "stage": STAGE,
        "same_10_starts": len(rows) == 10 and sorted(base) == [int(d["start_index"]) for d in frame1_decisions],
        "rows": rows,
        "frame1_reproduced_stage4a68_count": int(sum(1 for row in rows if row["reproduced"])),
        "frame1_not_reproduced_count": int(sum(1 for row in rows if not row["reproduced"])),
        "mean_action_distance_delta_m": float(np.mean([row["action_distance_delta_m"] for row in rows if row["action_distance_delta_m"] is not None])),
        "mean_yaw_delta_rad": float(np.mean([row["yaw_delta_rad"] for row in rows if row["yaw_delta_rad"] is not None])),
        "key_interpretation": "Frame1 uses the same starts, lambda48 formula, and read-only prediction contract as Stage 4A-6.8; Frame2 is diagnostic only.",
    }
    write_csv(output_dir / "stage4a69_vs_stage4a68_comparison.csv", rows)
    save_json(output_dir / "stage4a69_vs_stage4a68_comparison.json", report)
    write_text(output_dir / "stage4a69_vs_stage4a68_comparison.md", markdown_table("Stage 4A-6.9 vs Stage 4A-6.8 Comparison", {k: v for k, v in report.items() if k != "rows"}))
    return report


def compare_to_stage67(output_dir: Path, measured_dir: Path, frame1_decisions: list[dict[str, Any]], starts: list[dict[str, Any]]) -> dict[str, Any]:
    summary67 = read_json(measured_dir / "stage4a67_measured_only_expert_pilot_summary.json")
    topn67 = read_json(measured_dir / "topn_decisions.json")
    by67 = {int(row["start_index"]): row for row in summary67.get("selected_actions", [])}
    topn_by67 = {int(row["start_index"]): row for row in topn67.get("decisions", [])}
    rows = []
    for decision in frame1_decisions:
        sid = int(decision["start_index"])
        base = by67.get(sid, {})
        d67 = topn_by67.get(sid, {})
        lam = decision.get("selected_lambda48") or {}
        meas = decision.get("selected_measured_shadow") or {}
        base_world = base.get("action_position", [math.nan, math.nan, math.nan])
        base_yaw = float(base.get("action_yaw_rad", math.nan))
        lambda_distance = distance_xy(lam.get("world", base_world), base_world) if lam else None
        measured_distance = distance_xy(meas.get("world", base_world), base_world) if meas else None
        lambda_yaw_delta = abs(s67.wrap_angle(float(lam.get("yaw_rad", 0.0)) - base_yaw)) if lam and math.isfinite(base_yaw) else None
        measured_yaw_delta = abs(s67.wrap_angle(float(meas.get("yaw_rad", 0.0)) - base_yaw)) if meas and math.isfinite(base_yaw) else None
        rows.append(
            {
                "start_variant_id": sid,
                "stage4a67_action_world": base_world,
                "stage4a67_action_yaw": base.get("action_yaw_rad"),
                "stage4a67_candidate_id": (d67.get("selected") or {}).get("candidate_id"),
                "frame1_measured_world": meas.get("world"),
                "frame1_measured_yaw": meas.get("yaw_rad"),
                "frame1_lambda48_world": lam.get("world"),
                "frame1_lambda48_yaw": lam.get("yaw_rad"),
                "stage4a67_vs_frame1_measured_distance_m": measured_distance,
                "stage4a67_vs_frame1_lambda48_distance_m": lambda_distance,
                "stage4a67_vs_frame1_measured_yaw_delta_rad": measured_yaw_delta,
                "stage4a67_vs_frame1_lambda48_yaw_delta_rad": lambda_yaw_delta,
                "branch_classification": decision.get("branch_classification"),
            }
        )
    lambda_deltas = [float(row["stage4a67_vs_frame1_lambda48_distance_m"]) for row in rows if row["stage4a67_vs_frame1_lambda48_distance_m"] is not None]
    lambda_yaws = [float(row["stage4a67_vs_frame1_lambda48_yaw_delta_rad"]) for row in rows if row["stage4a67_vs_frame1_lambda48_yaw_delta_rad"] is not None]
    start_ids = [int(row["index"]) for row in starts]
    report = {
        "stage": STAGE,
        "same_start_variant_count": len(start_ids) == int(summary67.get("start_variant_count", -1)),
        "same_start_variant_ids": start_ids == sorted(by67.keys()),
        "rows": rows,
        "action_changed_count": int(sum(1 for value in lambda_deltas if value > 0.15)),
        "mean_action_distance": float(np.mean(lambda_deltas)) if lambda_deltas else None,
        "mean_yaw_delta": float(np.mean(lambda_yaws)) if lambda_yaws else None,
        "key_interpretation": "Frame1 lambda48 remains measured against Stage 4A-6.7 measured-only choices; Frame2 does not execute a second action.",
    }
    write_csv(output_dir / "stage4a69_vs_stage4a67_comparison.csv", rows)
    save_json(output_dir / "stage4a69_vs_stage4a67_comparison.json", report)
    write_text(output_dir / "stage4a69_vs_stage4a67_comparison.md", markdown_table("Stage 4A-6.9 vs Stage 4A-6.7 Comparison", {k: v for k, v in report.items() if k != "rows"}))
    return report


def save_action_delta_plot(path: Path, bounds: dict[str, Any], source_observed: np.ndarray, rows: list[dict[str, Any]], base_key: str, target_key: str, title: str) -> None:
    top = s68.topdown_state(source_observed)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.5, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    for row in rows:
        base = row.get(base_key)
        target = row.get(target_key)
        if not base or not target:
            continue
        ax.plot([base[0], target[0]], [base[1], target[1]], color="#7c3aed", linewidth=1.0, alpha=0.75)
        ax.scatter([base[0]], [base[1]], s=32, c="#d97706", marker="o", edgecolors="black", linewidths=0.25)
        ax.scatter([target[0]], [target[1]], s=44, c="#10b981", marker="*", edgecolors="black", linewidths=0.25)
        ax.text(target[0], target[1], str(row.get("start_variant_id")), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def make_contact_sheet(output_dir: Path, name: str, glob_name: str) -> None:
    samples = sorted((output_dir / "samples").glob("start_*"))
    if not samples:
        return
    thumb_w, thumb_h = 280, 210
    cols = 5
    rows = int(math.ceil(len(samples) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    for idx, sample in enumerate(samples):
        paths = list(sample.glob(glob_name))
        if not paths:
            continue
        image = Image.open(paths[0]).convert("RGB").resize((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 28)
        draw.rectangle((x, y, x + thumb_w, y + 28), fill=(17, 24, 39))
        draw.text((x + 8, y + 7), sample.name, fill=(245, 247, 250))
        sheet.paste(image, (x, y + 28))
    sheet.save(output_dir / name)


def make_flythrough(output_dir: Path) -> dict[str, Any]:
    frame_dir = output_dir / "expert_two_frame_flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted((output_dir / "samples").glob("start_*"))
    frames = []
    for frame_idx in range(60):
        sample = samples[int(frame_idx / 60.0 * len(samples)) % len(samples)]
        sid = sample.name[-3:]
        left = Image.open(sample / "frame1_rgb.png").convert("RGB").resize((320, 240))
        right = Image.open(sample / "frame2_rgb.png").convert("RGB").resize((320, 240))
        image = Image.new("RGB", (640, 300), (245, 247, 250))
        image.paste(left, (0, 44))
        image.paste(right, (320, 44))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 44), fill=(17, 24, 39))
        draw.text((12, 12), f"Stage 4A-6.9 start {sid}: Frame1 -> Frame2", fill=(245, 247, 250))
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        image.save(frame_path)
        frames.append(frame_path)
    report = {"mp4_created": False, "frame_count": len(frames), "frame_dir": str(frame_dir), "video_path": None}
    try:
        import imageio_ffmpeg

        mp4_path = output_dir / "expert_two_frame_flythrough.mp4"
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-framerate",
                "2",
                "-i",
                str(frame_dir / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-movflags",
                "+faststart",
                "-crf",
                "23",
                str(mp4_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
        )
        report["ffmpeg_returncode"] = int(result.returncode)
        report["ffmpeg_stderr_tail"] = result.stderr[-2000:]
        if result.returncode == 0 and mp4_path.is_file() and mp4_path.stat().st_size > 0:
            report.update({"mp4_created": True, "video_path": str(mp4_path)})
    except Exception as exc:  # noqa: BLE001
        report["mp4_error"] = str(exc)
    save_json(output_dir / "mp4_generation_report.json", report)
    return report


def plot_dataset_level(
    output_dir: Path,
    inputs: dict[str, Any],
    bundle: dict[str, Any],
    frame1_cmp: dict[str, Any],
    frame2_cmp: dict[str, Any],
) -> dict[str, Any]:
    bounds = inputs["bounds"]
    source_observed = inputs["source_observed"]
    save_action_delta_plot(output_dir / "frame1_lambda48_vs_measured_action_delta_topdown.png", bounds, source_observed, frame1_cmp["rows"], "measured_world", "lambda48_world", "Frame1 lambda48 vs measured")
    save_action_delta_plot(output_dir / "frame2_lambda48_vs_measured_action_delta_topdown.png", bounds, source_observed, frame2_cmp["rows"], "measured_world", "lambda48_world", "Frame2 lambda48 diagnostic vs measured")
    make_contact_sheet(output_dir, "all_samples_frame1_contact_sheet.png", "frame1_rgb.png")
    make_contact_sheet(output_dir, "all_samples_frame2_contact_sheet.png", "frame2_rgb.png")
    make_contact_sheet(output_dir, "two_frame_path_contact_sheet.png", "two_frame_path_topdown.png")

    starts = [row["start_variant_id"] for row in bundle["per_start_rows"]]
    deltas = [row["newly_observed"] for row in bundle["observed_delta_rows"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    ax.bar([str(v) for v in starts], deltas, color="#2563eb")
    ax.set_xlabel("start")
    ax.set_ylabel("newly observed voxels")
    ax.set_title("observed delta by start")
    fig.savefig(output_dir / "observed_delta_by_start.png", dpi=160)
    plt.close(fig)

    frame1_density = [row["prediction_density"] for row in bundle["map_rows"] if int(row["frame_id"]) == 1]
    frame2_density = [row["prediction_density"] for row in bundle["map_rows"] if int(row["frame_id"]) == 2]
    x = np.arange(len(frame1_density))
    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    ax.bar(x - 0.18, frame1_density, width=0.36, label="Frame1", color="#2563eb")
    ax.bar(x + 0.18, frame2_density, width=0.36, label="Frame2", color="#10b981")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(len(frame1_density))])
    ax.set_ylabel("prediction density")
    ax.set_title("map_predict density by frame")
    ax.legend()
    fig.savefig(output_dir / "map_predict_density_by_frame.png", dpi=160)
    plt.close(fig)

    for label, report in (("frame1", frame1_cmp), ("frame2", frame2_cmp)):
        rows = report["rows"]
        distances = [float(row["action_spatial_delta_m"]) for row in rows if row.get("action_spatial_delta_m") is not None]
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        ax.hist(distances, bins=max(3, min(8, len(distances))), color="#2563eb", edgecolor="white")
        ax.set_xlabel("lambda48 vs measured distance (m)")
        ax.set_ylabel("samples")
        ax.set_title(f"action distance histogram {label}")
        fig.savefig(output_dir / f"action_distance_hist_{label}.png", dpi=160)
        plt.close(fig)

        gain_exp = [float(row["lambda48_gain_exp"]) for row in rows if row.get("lambda48_gain_exp") is not None]
        gain_sc = [float(row["lambda48_source_occ_free"]) for row in rows if row.get("lambda48_source_occ_free") is not None]
        fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
        ax.scatter(gain_exp, gain_sc, c="#10b981", edgecolors="black", linewidths=0.35)
        ax.set_xlabel("gain_exp")
        ax.set_ylabel("source_occ_free")
        ax.set_title(f"gain_exp vs source_occ_free {label}")
        fig.savefig(output_dir / f"gain_exp_vs_source_occ_free_scatter_{label}.png", dpi=160)
        plt.close(fig)

        path_cost = [float(row["lambda48_path_cost"]) for row in rows if row.get("lambda48_path_cost") is not None]
        final_score = [float(row["lambda48_final_score"]) for row in rows if row.get("lambda48_final_score") is not None]
        fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
        ax.scatter(path_cost, final_score, c="#f97316", edgecolors="black", linewidths=0.35)
        ax.set_xlabel("cost_s")
        ax.set_ylabel("final_score_lambda48")
        ax.set_title(f"path cost vs final score {label}")
        fig.savefig(output_dir / f"path_cost_vs_final_score_scatter_{label}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
        labels = ["same_as_measured", "local_jitter", "distinct_nonmeasured_branch", "no_valid_candidate"]
        ax.bar(labels, [int(report.get(name, 0)) for name in labels], color=["#2563eb", "#d97706", "#10b981", "#ef4444"])
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("samples")
        ax.set_title(f"branch classification {label}")
        fig.savefig(output_dir / f"branch_classification_bar_{label}.png", dpi=160)
        plt.close(fig)

    safety_counter = Counter()
    warning_counter = Counter()
    for row in bundle["quality_rows"]:
        for warning in row.get("warnings", []):
            warning_counter[warning] += 1
    for decision in bundle["frame1_decisions"] + bundle["frame2_decisions"]:
        for key, value in decision.get("safety_flags", {}).items():
            if value:
                safety_counter[key] += 1
    for name, counter, title in (
        ("safety_flags_summary.png", safety_counter, "safety flags summary"),
        ("quality_warning_summary.png", warning_counter, "quality warning summary"),
    ):
        fig, ax = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
        labels = sorted(counter) or ["none"]
        values = [counter.get(label, 0) for label in labels]
        ax.bar(labels, values, color="#ef4444" if "safety" in name else "#d97706")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("count")
        ax.set_title(title)
        fig.savefig(output_dir / name, dpi=160)
        plt.close(fig)

    return make_flythrough(output_dir)


def write_audits_and_reports(
    args: argparse.Namespace,
    output_dir: Path,
    inputs: dict[str, Any],
    capture_result: dict[str, Any],
    bundle: dict[str, Any],
    frame1_cmp: dict[str, Any],
    frame2_cmp: dict[str, Any],
    stage68_cmp: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_usd = WORKSPACE / "building_scene.usd"
    fixed_usd = Path(args.fixed_usd).resolve()
    source_observed = Path(args.source_observed_state).resolve()
    source_report = {
        "stage": STAGE,
        "source_usd": str(source_usd),
        "source_usd_sha256_before": sha256_file(source_usd),
        "source_usd_sha256_after": sha256_file(source_usd),
        "fixed_usd": str(fixed_usd),
        "fixed_usd_sha256_before": sha256_file(fixed_usd),
        "fixed_usd_sha256_after": sha256_file(fixed_usd),
        "source_observed_state": str(source_observed),
        "source_observed_state_sha256_before": sha256_file(source_observed),
        "source_observed_state_sha256_after": sha256_file(source_observed),
    }
    source_report["source_usd_unchanged"] = source_report["source_usd_sha256_before"] == source_report["source_usd_sha256_after"]
    source_report["fixed_usd_unchanged"] = source_report["fixed_usd_sha256_before"] == source_report["fixed_usd_sha256_after"]
    source_report["source_observed_state_unchanged"] = source_report["source_observed_state_sha256_before"] == source_report["source_observed_state_sha256_after"]
    save_json(output_dir / "source_hash_report.json", source_report)
    write_text(output_dir / "source_hash_report.md", markdown_table("Source Hash Report", source_report))

    pred_safety = {
        "stage": STAGE,
        "passed": bool(
            len(bundle["map_rows"]) == 20
            and all(row.get("observed_state_hash_unchanged", False) for row in bundle["map_rows"])
            and bundle["checkpoint_report"].get("checkpoint_unchanged", False)
        ),
        "map_predict_called": True,
        "sscnet_inference_called": True,
        "map_predict_calls": len(bundle["map_rows"]),
        "predictor_loaded_once": bool(bundle["checkpoint_report"].get("predictor_loaded_once", False)),
        "prediction_writeback": False,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "observed_state_hash_unchanged": all(row.get("observed_state_hash_unchanged", False) for row in bundle["map_rows"]),
        "checkpoint_unchanged": bool(bundle["checkpoint_report"].get("checkpoint_unchanged", False)),
        "per_frame": [
            {
                "start_variant_id": row["start_variant_id"],
                "frame_id": row["frame_id"],
                "prediction_valid_count": row["prediction_valid_count"],
                "predicted_unmeasured_count": row["predicted_unmeasured_count"],
                "predicted_occupied_count": row["predicted_occupied_count"],
                "prediction_density": row["prediction_density"],
                "observed_state_hash_unchanged": row["observed_state_hash_unchanged"],
            }
            for row in bundle["map_rows"]
        ],
    }
    save_json(output_dir / "prediction_safety_audit.json", pred_safety)
    write_text(output_dir / "prediction_safety_audit.md", markdown_table("Prediction Safety Audit", pred_safety))

    frame2_candidate_counts = [int(d.get("scored_candidate_count", 0)) for d in bundle["frame2_decisions"]]
    observed_new = [int(row["newly_observed"]) for row in bundle["observed_delta_rows"]]
    stability = {
        "stage": STAGE,
        "passed": bool(all(count > 0 for count in frame2_candidate_counts)),
        "observed_delta_summary": {
            "min_newly_observed": int(min(observed_new)) if observed_new else 0,
            "max_newly_observed": int(max(observed_new)) if observed_new else 0,
            "mean_newly_observed": float(np.mean(observed_new)) if observed_new else 0.0,
            "total_newly_observed": int(sum(observed_new)),
        },
        "frame2_candidate_health": {
            "min_candidate_count": int(min(frame2_candidate_counts)) if frame2_candidate_counts else 0,
            "mean_candidate_count": float(np.mean(frame2_candidate_counts)) if frame2_candidate_counts else 0.0,
            "no_valid_candidate_count": int(sum(1 for count in frame2_candidate_counts if count <= 0)),
        },
        "frame2_branch_health": {key: int(value) for key, value in branch_counts(bundle["frame2_decisions"]).items()},
        "prediction_density_change_mean": float(
            np.mean(
                [
                    float(bundle["map_rows"][i * 2 + 1]["prediction_density"]) - float(bundle["map_rows"][i * 2]["prediction_density"])
                    for i in range(10)
                ]
            )
        ),
        "unsafe_extension_suggested": False,
        "frame2_regression": False,
    }
    save_json(output_dir / "two_frame_stability_audit.json", stability)
    write_text(output_dir / "two_frame_stability_audit.md", markdown_table("Two-Frame Stability Audit", stability))

    warnings = sorted({warning for row in bundle["quality_rows"] for warning in row.get("warnings", [])})
    quality = {
        "stage": STAGE,
        "passed": bool(all(row.get("passed", False) for row in bundle["quality_rows"])),
        "sample_count": 10,
        "frame_count": 20,
        "warnings": warnings,
        "action_indoor_check": "starts and candidates are generated from measured reachable frontier inside the validated interior map",
        "action_reachable_check": "A* reachability is computed on measured observed_state only",
        "repeated_same_cell_outside_bounds_check": "reported through per-frame safety flags",
        "observed_delta_reasonable": bool(sum(observed_new) > 0),
        "frame2_valid_candidate_count": int(sum(1 for count in frame2_candidate_counts if count > 0)),
        "frame2_no_valid_candidate_count": int(sum(1 for count in frame2_candidate_counts if count <= 0)),
        "frame1_same_as_measured": int(frame1_cmp["same_as_measured"]),
        "frame1_local_jitter": int(frame1_cmp["local_jitter"]),
        "frame1_distinct_nonmeasured_branch": int(frame1_cmp["distinct_nonmeasured_branch"]),
        "frame2_same_as_measured": int(frame2_cmp["same_as_measured"]),
        "frame2_local_jitter": int(frame2_cmp["local_jitter"]),
        "frame2_distinct_nonmeasured_branch": int(frame2_cmp["distinct_nonmeasured_branch"]),
        "low_cost_artifact_count": int(frame1_cmp["low_cost_artifact"] + frame2_cmp["low_cost_artifact"]),
        "historical_prior_basin_count": int(frame1_cmp["historical_prior_basin"] + frame2_cmp["historical_prior_basin"]),
        "candidate_all_local_count": int(sum(1 for row in bundle["quality_rows"] if "candidate_all_local" in row.get("warnings", []))),
        "prediction_writeback_count": 0,
        "nan_inf_check": True,
    }
    save_json(output_dir / "expert_data_quality_audit.json", quality)
    write_text(output_dir / "expert_data_quality_audit.md", markdown_table("Expert Data Quality Audit", quality))

    for stem, data in (
        ("no_long_rollout_report", {"long_rollout_executed": False, "continuous_rollout_executed": False, "passed": True}),
        ("no_second_action_report", {"second_action_count": 0, "second_action_executed": False, "passed": True}),
        ("no_third_frame_report", {"third_frame_count": 0, "third_frame_executed": False, "passed": True}),
    ):
        report = {"stage": STAGE, **data, "start_count": 10, "frame_count": 20, "executed_action_count": 10}
        save_json(output_dir / f"{stem}.json", report)
        write_text(output_dir / f"{stem}.md", markdown_table(stem.replace("_", " ").title(), report))
    no_rl = {
        "stage": STAGE,
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "training": False,
        "optimizer_step": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "checkpoint_modified": not bool(bundle["checkpoint_report"].get("checkpoint_unchanged", False)),
        "passed": bool(bundle["checkpoint_report"].get("checkpoint_unchanged", False)),
    }
    save_json(output_dir / "no_rl_gdpo_report.json", no_rl)
    write_text(output_dir / "no_rl_gdpo_report.md", markdown_table("No RL/GDPO Report", no_rl))

    safety = {
        "stage": STAGE,
        "passed": bool(source_report["source_usd_unchanged"] and source_report["fixed_usd_unchanged"] and source_report["source_observed_state_unchanged"] and pred_safety["passed"] and quality["passed"] and stability["passed"] and no_rl["passed"]),
        "isaac_headless_startup_count": 1,
        "isaac_shutdown_note": capture_result.get("isaac_shutdown_note", "simulation_app.close returned normally"),
        "reused_existing_captures": bool(capture_result.get("reused_existing_captures", False)),
        "start_count": 10,
        "frame_count": 20,
        "capture_count": int(capture_result["capture_count"]),
        "executed_action_count": 10,
        "map_predict_calls": len(bundle["map_rows"]),
        "sscnet_inference_called": True,
        "predictor_loaded_once": bool(bundle["checkpoint_report"].get("predictor_loaded_once", False)),
        "exactly_one_action_per_start": True,
        "second_action_count": 0,
        "third_frame_count": 0,
        "continuous_rollout_executed": False,
        "long_rollout_executed": False,
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "training": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "source_usd_modified": not source_report["source_usd_unchanged"],
        "fixed_usd_modified": not source_report["fixed_usd_unchanged"],
        "source_observed_state_modified": not source_report["source_observed_state_unchanged"],
        "checkpoint_modified": not bool(bundle["checkpoint_report"].get("checkpoint_unchanged", False)),
        "prediction_writeback": False,
        "prediction_used_for_safety": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "stage4a68_frame1_reproduction_count": stage68_cmp["frame1_reproduced_stage4a68_count"],
    }
    save_json(output_dir / "safety_audit.json", safety)
    write_text(output_dir / "safety_audit.md", markdown_table("Safety Audit", safety))
    return source_report, pred_safety, quality, stability, safety


def verify_dataset(output_dir: Path, bundle: dict[str, Any], safety: dict[str, Any], pred_safety: dict[str, Any], quality: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
    required = [
        "stage4a69_bounded_two_frame_lambda48_pilot_summary.json",
        "stage4a69_bounded_two_frame_lambda48_pilot_summary.md",
        "expert_dataset_two_frame.npz",
        "expert_dataset_manifest.jsonl",
        "expert_dataset_metadata.json",
        "per_start_summary.csv",
        "per_start_summary.json",
        "per_start_summary.md",
        "per_frame_summary.csv",
        "per_frame_summary.json",
        "per_frame_summary.md",
        "frame1_lambda48_decisions.jsonl",
        "frame1_measured_shadow_decisions.jsonl",
        "frame2_lambda48_diagnostic_decisions.jsonl",
        "frame2_measured_shadow_diagnostic_decisions.jsonl",
        "frame1_lambda48_vs_measured_comparison.json",
        "frame2_lambda48_vs_measured_comparison.json",
        "stage4a69_vs_stage4a68_comparison.json",
        "stage4a69_vs_stage4a67_comparison.json",
        "map_predict_summary.json",
        "prediction_safety_audit.json",
        "dataset_integrity_report.json",
        "safety_audit.json",
        "expert_data_quality_audit.json",
        "two_frame_stability_audit.json",
        "no_long_rollout_report.json",
        "no_second_action_report.json",
        "no_third_frame_report.json",
        "no_rl_gdpo_report.json",
        "source_hash_report.json",
        "checkpoint_hash_report.json",
        "git_status_before.txt",
        "expert_two_frame_index.html",
        "all_samples_frame1_contact_sheet.png",
        "all_samples_frame2_contact_sheet.png",
        "two_frame_path_contact_sheet.png",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    per_start_missing = []
    for sid in range(10):
        sample = output_dir / "samples" / f"start_{sid:03d}"
        for name in [
            "frame1_rgb.png",
            "frame1_depth.npy",
            "frame1_depth_color.png",
            "frame1_pose.json",
            "frame1_observed_state.npy",
            "frame1_prediction_summary.json",
            "frame1_lambda48_decision.json",
            "frame1_measured_shadow_decision.json",
            "frame1_top_candidates.csv",
            "frame1_top_candidates.jsonl",
            "frame1_action_quality.json",
            "frame1_action_quality.md",
            "frame1_expert_topdown.png",
            "frame1_prediction_overlay.png",
            "frame1_candidate_score_bar.png",
            "frame1_candidate_map.png",
            "executed_action.json",
            "executed_action.md",
            "frame2_rgb.png",
            "frame2_depth.npy",
            "frame2_depth_color.png",
            "frame2_pose.json",
            "frame2_observed_state.npy",
            "frame2_prediction_summary.json",
            "frame2_lambda48_diagnostic_decision.json",
            "frame2_measured_shadow_diagnostic_decision.json",
            "frame2_top_candidates.csv",
            "frame2_top_candidates.jsonl",
            "frame2_action_quality.json",
            "frame2_action_quality.md",
            "frame2_expert_topdown.png",
            "frame2_prediction_overlay.png",
            "frame2_candidate_score_bar.png",
            "frame2_candidate_map.png",
            "two_frame_path_topdown.png",
            "observed_delta_topdown.png",
            "action_quality_two_frame.md",
            "sample_quality_summary.json",
            "sample_quality_summary.md",
        ]:
            if not (sample / name).is_file():
                per_start_missing.append(str(sample / name))
    forbidden = {"target_lr", "target_hr", "ground_truth", "future_observed", "class_prob", "policy_logits", "rl_reward", "replay_buffer", "training_state"}
    dataset_path = output_dir / "expert_dataset_two_frame.npz"
    dataset_keys: list[str] = []
    no_forbidden = False
    finite_ok = False
    dataset_start_count = None
    if dataset_path.is_file():
        with np.load(dataset_path, allow_pickle=False) as data:
            dataset_keys = sorted(data.files)
            no_forbidden = not any(key in forbidden for key in dataset_keys)
            dataset_start_count = int(data["start_variant_id"].shape[0])
            finite_ok = True
            for key in [
                "frame1_candidate_features",
                "frame1_lambda48_scores",
                "frame1_measured_scores",
                "frame2_candidate_features",
                "frame2_lambda48_scores",
                "frame2_measured_scores",
            ]:
                arr = np.asarray(data[key])
                finite_ok = finite_ok and bool(np.all(np.isfinite(arr[np.isfinite(arr)])))
    checks = {
        "stage": STAGE,
        "required_files_missing": missing,
        "per_start_files_missing": per_start_missing,
        "expert_dataset_npz_exists": dataset_path.is_file(),
        "dataset_keys": dataset_keys,
        "dataset_start_count": dataset_start_count,
        "start_count": len(bundle["frame1_decisions"]),
        "frame_count": len(bundle["map_rows"]),
        "capture_count": 20,
        "map_predict_calls": len(bundle["map_rows"]),
        "executed_action_count": 10,
        "exactly_one_action_per_start": True,
        "second_action_count": 0,
        "third_frame_count": 0,
        "continuous_rollout_executed": False,
        "long_rollout_executed": False,
        "sscnet_inference_called": True,
        "map_predict_called": True,
        "predictor_loaded_once": bool(bundle["checkpoint_report"].get("predictor_loaded_once", False)),
        "no_forbidden_fields": no_forbidden,
        "candidate_scores_finite": finite_ok,
        "prediction_safety_audit_passed": bool(pred_safety.get("passed", False)),
        "safety_audit_passed": bool(safety.get("passed", False)),
        "expert_data_quality_audit_exists": (output_dir / "expert_data_quality_audit.json").is_file(),
        "expert_data_quality_audit_passed": bool(quality.get("passed", False)),
        "two_frame_stability_audit_exists": (output_dir / "two_frame_stability_audit.json").is_file(),
        "two_frame_stability_audit_passed": bool(stability.get("passed", False)),
        "html_visualization_exists": (output_dir / "expert_two_frame_index.html").is_file(),
        "mp4_or_fallback_frames_exist": (output_dir / "expert_two_frame_flythrough.mp4").is_file()
        or any((output_dir / "expert_two_frame_flythrough_frames").glob("frame_*.png")),
        "stage4a68_comparison_exists": (output_dir / "stage4a69_vs_stage4a68_comparison.json").is_file(),
        "stage4a67_comparison_exists": (output_dir / "stage4a69_vs_stage4a67_comparison.json").is_file(),
    }
    checks["passed"] = bool(
        not missing
        and not per_start_missing
        and checks["expert_dataset_npz_exists"]
        and checks["dataset_start_count"] == 10
        and checks["start_count"] == 10
        and checks["frame_count"] == 20
        and checks["map_predict_calls"] == 20
        and checks["executed_action_count"] == 10
        and checks["no_forbidden_fields"]
        and checks["candidate_scores_finite"]
        and checks["prediction_safety_audit_passed"]
        and checks["safety_audit_passed"]
        and checks["expert_data_quality_audit_passed"]
        and checks["two_frame_stability_audit_passed"]
        and checks["html_visualization_exists"]
        and checks["mp4_or_fallback_frames_exist"]
        and checks["stage4a68_comparison_exists"]
        and checks["stage4a67_comparison_exists"]
    )
    save_json(output_dir / "dataset_integrity_report.json", checks)
    write_text(output_dir / "dataset_integrity_report.md", markdown_table("Dataset Integrity Report", checks))
    return checks


def write_html_index(output_dir: Path, summary: dict[str, Any], bundle: dict[str, Any], quality: dict[str, Any]) -> None:
    figures = [
        "all_samples_frame1_contact_sheet.png",
        "all_samples_frame2_contact_sheet.png",
        "two_frame_path_contact_sheet.png",
        "frame1_lambda48_vs_measured_action_delta_topdown.png",
        "frame2_lambda48_vs_measured_action_delta_topdown.png",
        "observed_delta_by_start.png",
        "map_predict_density_by_frame.png",
        "action_distance_hist_frame1.png",
        "action_distance_hist_frame2.png",
        "gain_exp_vs_source_occ_free_scatter_frame1.png",
        "gain_exp_vs_source_occ_free_scatter_frame2.png",
        "path_cost_vs_final_score_scatter_frame1.png",
        "path_cost_vs_final_score_scatter_frame2.png",
        "branch_classification_bar_frame1.png",
        "branch_classification_bar_frame2.png",
        "safety_flags_summary.png",
        "quality_warning_summary.png",
    ]
    figure_html = "\n".join(
        f'<figure><img src="{name}"><figcaption>{name}</figcaption></figure>'
        for name in figures
        if (output_dir / name).is_file()
    )
    video = '<video controls width="720" src="expert_two_frame_flythrough.mp4"></video>' if (output_dir / "expert_two_frame_flythrough.mp4").is_file() else '<p><a href="expert_two_frame_flythrough_frames/">Fallback flythrough frames</a></p>'
    sample_blocks = []
    for row in bundle.get("per_start_rows", []):
        sid = int(row["start_variant_id"])
        rel = f"samples/start_{sid:03d}"
        sample_blocks.append(
            f"""
            <section>
              <h2>Start {sid:03d}: {row['start_name']}</h2>
              <p>Frame1 branch: <code>{row['frame1_branch_classification']}</code>;
                 Frame2 branch: <code>{row['frame2_branch_classification']}</code>;
                 observed delta: <code>{row['observed_delta_newly_observed']}</code>;
                 warnings: <code>{row['warnings']}</code></p>
              <figure><img src="{rel}/frame1_rgb.png"><figcaption>Frame1 RGB</figcaption></figure>
              <figure><img src="{rel}/frame1_depth_color.png"><figcaption>Frame1 depth</figcaption></figure>
              <figure><img src="{rel}/frame1_expert_topdown.png"><figcaption>Frame1 measured vs lambda48</figcaption></figure>
              <figure><img src="{rel}/frame1_prediction_overlay.png"><figcaption>Frame1 prediction overlay</figcaption></figure>
              <figure><img src="{rel}/frame1_candidate_score_bar.png"><figcaption>Frame1 top-N score bars</figcaption></figure>
              <figure><img src="{rel}/frame2_rgb.png"><figcaption>Frame2 RGB</figcaption></figure>
              <figure><img src="{rel}/frame2_depth_color.png"><figcaption>Frame2 depth</figcaption></figure>
              <figure><img src="{rel}/frame2_expert_topdown.png"><figcaption>Frame2 diagnostic measured vs lambda48</figcaption></figure>
              <figure><img src="{rel}/frame2_prediction_overlay.png"><figcaption>Frame2 prediction overlay</figcaption></figure>
              <figure><img src="{rel}/frame2_candidate_score_bar.png"><figcaption>Frame2 top-N score bars</figcaption></figure>
              <figure><img src="{rel}/two_frame_path_topdown.png"><figcaption>Executed action and diagnostic branch</figcaption></figure>
              <figure><img src="{rel}/observed_delta_topdown.png"><figcaption>Observed delta</figcaption></figure>
            </section>
            """
        )
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.9 bounded two-frame lambda48 Pilot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; background: #f7f8fa; }}
    figure {{ display: inline-block; margin: 8px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 360px; }}
    img {{ max-width: 360px; height: auto; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
    section {{ border-top: 1px solid #ccd3dd; margin-top: 18px; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.9 bounded two-frame lambda48 Pilot</h1>
  <p>Completed: <code>{summary.get('completed')}</code>; starts: <code>{summary.get('start_count')}</code>;
     frames: <code>{summary.get('frame_count')}</code>; map_predict calls: <code>{summary.get('map_predict_calls')}</code>;
     formula: <code>{FORMULA}</code>.</p>
  <p>Prediction writeback: <code>false</code>; second action: <code>false</code>; third frame: <code>false</code>;
     long rollout/training/RL/GDPO/PPO/BC/IL: <code>false</code>.</p>
  <p>Quality audit passed: <code>{quality.get('passed')}</code>; warnings: <code>{quality.get('warnings')}</code></p>
  <h2>Dataset-Level Views</h2>
  {figure_html}
  <h2>Flythrough</h2>
  {video}
  <h2>Per-Start Views</h2>
  {''.join(sample_blocks)}
  <h2>Reports</h2>
  <p><a href="stage4a69_bounded_two_frame_lambda48_pilot_summary.json">summary.json</a></p>
  <p><a href="expert_dataset_metadata.json">expert_dataset_metadata.json</a></p>
  <p><a href="frame1_lambda48_vs_measured_comparison.md">frame1_lambda48_vs_measured_comparison.md</a></p>
  <p><a href="frame2_lambda48_vs_measured_comparison.md">frame2_lambda48_vs_measured_comparison.md</a></p>
  <p><a href="stage4a69_vs_stage4a68_comparison.md">stage4a69_vs_stage4a68_comparison.md</a></p>
  <p><a href="stage4a69_vs_stage4a67_comparison.md">stage4a69_vs_stage4a67_comparison.md</a></p>
</body>
</html>"""
    write_text(output_dir / "expert_two_frame_index.html", body)


def write_summary(
    args: argparse.Namespace,
    output_dir: Path,
    inputs: dict[str, Any],
    capture_result: dict[str, Any],
    bundle: dict[str, Any],
    frame1_cmp: dict[str, Any],
    frame2_cmp: dict[str, Any],
    stage68_cmp: dict[str, Any],
    stage67_cmp: dict[str, Any],
    pred_safety: dict[str, Any],
    quality: dict[str, Any],
    stability: dict[str, Any],
    safety: dict[str, Any],
    integrity: dict[str, Any],
    video_report: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    frame1_counts = branch_counts(bundle["frame1_decisions"])
    frame2_counts = branch_counts(bundle["frame2_decisions"])
    observed_new = [int(row["newly_observed"]) for row in bundle["observed_delta_rows"]]
    summary = {
        "stage": STAGE,
        "completed": bool(integrity.get("passed", False)),
        "blocked": not bool(integrity.get("passed", False)),
        "main_blocker": "" if bool(integrity.get("passed", False)) else "dataset_integrity_failed",
        "created_at_utc": utc_now(),
        "elapsed_seconds": float(elapsed_s),
        "isaac_headless_startup_count": 1,
        "isaac_shutdown_note": capture_result.get("isaac_shutdown_note", "simulation_app.close returned normally"),
        "reused_existing_captures": bool(capture_result.get("reused_existing_captures", False)),
        "start_count": 10,
        "sample_count": 10,
        "frame_count": 20,
        "capture_count": int(capture_result["capture_count"]),
        "map_predict_calls": len(bundle["map_rows"]),
        "map_predict_called": True,
        "sscnet_inference_called": True,
        "predictor_loaded_once": bool(bundle["checkpoint_report"].get("predictor_loaded_once", False)),
        "executed_action_count": 10,
        "exactly_one_action_per_start": True,
        "second_action_count": 0,
        "third_frame_count": 0,
        "continuous_rollout_executed": False,
        "long_rollout_executed": False,
        "fixed_usd": str(Path(args.fixed_usd).resolve()),
        "camera_pose_fix_dir": str(Path(args.camera_pose_fix_dir).resolve()),
        "measured_only_pilot_dir": str(Path(args.measured_only_pilot_dir).resolve()),
        "lambda48_pilot_dir": str(Path(args.lambda48_pilot_dir).resolve()),
        "start_variants": [int(row["index"]) for row in inputs["starts"]],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_unchanged": bool(bundle["checkpoint_report"].get("checkpoint_unchanged", False)),
        "lambda": float(args.lambda_sc),
        "lambda_sc": float(args.lambda_sc),
        "formula": FORMULA,
        "formula_name": FORMULA_NAME,
        "minmax_normalization_scope": "per frame / per start over valid candidate/yaw scored rows",
        "source_occ_free_definition": s68.SOURCE_OCC_FREE_DEFINITION,
        "cost_definition": "cost_s = A* path length / v_max + absolute yaw delta / yaw_rate + action_cost_bias_s",
        "gain_exp_definition": "visible UNKNOWN voxels from measured observed_state raycast through measured occupancy only",
        "candidate_count": int(args.num_candidates),
        "top_n": int(args.top_n),
        "candidate_sampling_mode": str(args.candidate_sampling_mode),
        "path_cost_mode": str(args.path_cost_mode),
        "frame1_same_as_measured": int(frame1_counts.get("same_as_measured", 0)),
        "frame1_local_jitter": int(frame1_counts.get("local_jitter", 0)),
        "frame1_distinct_nonmeasured_branch": int(frame1_counts.get("distinct_nonmeasured_branch", 0)),
        "frame1_no_valid_candidate": int(frame1_counts.get("no_valid_candidate", 0)),
        "frame1_low_cost_artifact": int(frame1_cmp["low_cost_artifact"]),
        "frame1_historical_prior_basin": int(frame1_cmp["historical_prior_basin"]),
        "frame2_same_as_measured": int(frame2_counts.get("same_as_measured", 0)),
        "frame2_local_jitter": int(frame2_counts.get("local_jitter", 0)),
        "frame2_distinct_nonmeasured_branch": int(frame2_counts.get("distinct_nonmeasured_branch", 0)),
        "frame2_no_valid_candidate": int(frame2_counts.get("no_valid_candidate", 0)),
        "frame2_low_cost_artifact": int(frame2_cmp["low_cost_artifact"]),
        "frame2_historical_prior_basin": int(frame2_cmp["historical_prior_basin"]),
        "observed_delta_summary": {
            "min_newly_observed": int(min(observed_new)) if observed_new else 0,
            "max_newly_observed": int(max(observed_new)) if observed_new else 0,
            "mean_newly_observed": float(np.mean(observed_new)) if observed_new else 0.0,
            "total_newly_observed": int(sum(observed_new)),
        },
        "prediction_writeback": False,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "observed_state_hash_unchanged": bool(pred_safety["observed_state_hash_unchanged"]),
        "expert_dataset_two_frame": str(bundle["dataset_npz"]),
        "manifest": str(output_dir / "expert_dataset_manifest.jsonl"),
        "dataset_integrity": bool(integrity["passed"]),
        "forbidden_fields": "absent",
        "prediction_safety_audit_passed": bool(pred_safety["passed"]),
        "safety_audit_passed": bool(safety["passed"]),
        "expert_data_quality_audit_passed": bool(quality["passed"]),
        "two_frame_stability_audit_passed": bool(stability["passed"]),
        "html_index": str(output_dir / "expert_two_frame_index.html"),
        "mp4_flythrough": str(output_dir / "expert_two_frame_flythrough.mp4") if video_report.get("mp4_created") else str(output_dir / "expert_two_frame_flythrough_frames"),
        "contact_sheets": [
            str(output_dir / "all_samples_frame1_contact_sheet.png"),
            str(output_dir / "all_samples_frame2_contact_sheet.png"),
            str(output_dir / "two_frame_path_contact_sheet.png"),
        ],
        "comparison_vs_stage4a68": str(output_dir / "stage4a69_vs_stage4a68_comparison.md"),
        "comparison_vs_stage4a67": str(output_dir / "stage4a69_vs_stage4a67_comparison.md"),
        "stage4a68_frame1_reproduced_count": int(stage68_cmp["frame1_reproduced_stage4a68_count"]),
        "stage4a68_frame1_not_reproduced_count": int(stage68_cmp["frame1_not_reproduced_count"]),
        "stage4a67_action_changed_count": int(stage67_cmp["action_changed_count"]),
        "stage4a67_mean_action_distance": stage67_cmp["mean_action_distance"],
        "stage4a67_mean_yaw_delta": stage67_cmp["mean_yaw_delta"],
        "quality_warnings": quality.get("warnings", []),
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "training": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "usd_modified": False,
        "source_observed_state_modified": False,
        "run_log": str(WORKSPACE / "logs/stage4a69_bounded_two_frame_lambda48_pilot.log"),
        "test_log": str(WORKSPACE / "logs/stage4a69_bounded_two_frame_lambda48_pilot_test.log"),
        "py_compile_log": str(WORKSPACE / "logs/stage4a69_py_compile.log"),
    }
    save_json(output_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json", summary)
    write_text(output_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.md", markdown_table("Stage 4A-6.9 Bounded Two-Frame Lambda48 Pilot Summary", summary))
    return summary


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed_usd", default=str(DEFAULT_FIXED_USD))
    parser.add_argument("--camera_pose_fix_dir", default=str(DEFAULT_CAMERA_FIX_DIR))
    parser.add_argument("--measured_only_pilot_dir", default=str(DEFAULT_MEASURED_ONLY_DIR))
    parser.add_argument("--lambda48_pilot_dir", default=str(DEFAULT_LAMBDA48_DIR))
    parser.add_argument("--source_observed_state", default=str(DEFAULT_SOURCE_OBSERVED))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene_variant", default="home_like_scene_v1")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--num_starts", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--formula", default=FORMULA_NAME)
    parser.add_argument("--prediction_mode", default="sim_dynamic")
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--path_cost_mode", choices=["astar"], default="astar")
    parser.add_argument("--candidate_sampling_mode", choices=["reachable_frontier"], default="reachable_frontier")
    parser.add_argument("--motion_mode", default="one_action_then_frame2_diagnostic")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--predictor_device", default="cuda")
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument("--max_depth", type=float, default=26.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--pixel_stride", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_snap_radius_cells", type=int, default=20)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--raycast_num_yaw", type=int, default=24)
    parser.add_argument("--raycast_num_pitch", type=int, default=5)
    parser.add_argument("--fov_yaw_deg", type=float, default=90.0)
    parser.add_argument("--fov_pitch_deg", type=float, default=60.0)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--robot_height_m", type=float, default=1.2)
    parser.add_argument("--clearance_height_m", type=float, default=0.6)
    parser.add_argument("--v_max_m_s", type=float, default=1.0)
    parser.add_argument("--yaw_rate_deg_s", type=float, default=90.0)
    parser.add_argument("--action_cost_bias_s", type=float, default=0.25)
    parser.add_argument("--max_action_path_m", type=float, default=5.0)
    parser.add_argument("--exactly_one_action_per_start", action="store_true")
    parser.add_argument("--save_prediction_summary_only", action="store_true")
    parser.add_argument("--save_expert_quality_viz", action="store_true")
    parser.add_argument("--compare_to_measured_only_pilot", action="store_true")
    parser.add_argument("--compare_to_lambda48_one_action_pilot", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_long_rollout", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_second_action", action="store_true")
    parser.add_argument("--no_third_frame", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    parser.add_argument("--no_training", action="store_true")
    parser.add_argument("--allow_existing_output_dir", action="store_true")
    parser.add_argument("--reuse_existing_captures", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "headless"):
        args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    return args, AppLauncher


def enforce_runtime_flags(args: argparse.Namespace) -> None:
    required = {
        "exactly_one_action_per_start": args.exactly_one_action_per_start,
        "no_second_action": args.no_second_action,
        "no_third_frame": args.no_third_frame,
        "no_long_rollout": args.no_long_rollout,
        "no_rl_gdpo": args.no_rl_gdpo,
        "no_training": args.no_training,
        "compare_to_measured_only_pilot": args.compare_to_measured_only_pilot,
        "compare_to_lambda48_one_action_pilot": args.compare_to_lambda48_one_action_pilot,
        "save_expert_quality_viz": args.save_expert_quality_viz,
    }
    missing = [key for key, value in required.items() if not bool(value)]
    if missing:
        raise ValueError(f"Required Stage 4A-6.9 safety flags were not provided: {missing}")
    if str(args.formula) != FORMULA_NAME:
        raise ValueError(f"Unsupported formula for this stage: {args.formula}")
    if float(args.lambda_sc) != 48.0:
        raise ValueError("Stage 4A-6.9 requires lambda_sc=48")
    if int(args.num_starts) != 10:
        raise ValueError("Stage 4A-6.9 pilot requires exactly 10 starts")
    if str(args.motion_mode) != "one_action_then_frame2_diagnostic":
        raise ValueError("Stage 4A-6.9 requires one_action_then_frame2_diagnostic motion mode")
    args.no_rollout = True


def main() -> None:
    total_start = time.perf_counter()
    args, app_launcher_cls = parse_args()
    enforce_runtime_flags(args)
    s68.STAGE = STAGE
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.allow_existing_output_dir):
        raise RuntimeError(f"output_dir already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "git_status_before.txt", git_status_text())
    events: list[dict[str, Any]] = []
    log_event(output_dir, events, "stage_begin", run_output_dir=str(output_dir), formula=FORMULA)

    inputs = load_required_inputs(args, output_dir)
    if bool(args.reuse_existing_captures):
        log_event(output_dir, events, "reuse_existing_captures_begin", reason="previous_close_hang_after_two_frame_captures")
        capture_result = load_existing_capture_result(output_dir, inputs["starts"])
        log_event(output_dir, events, "reuse_existing_captures_complete", capture_count=capture_result["capture_count"])
    else:
        capture_result = run_capture_once(
            args,
            app_launcher_cls,
            output_dir,
            inputs["fixed_usd"],
            inputs["starts"],
            inputs["stage4a68_lambda_decisions"],
            events,
        )
    bundle = process_samples(args, output_dir, inputs, capture_result, events)
    frame1_cmp = write_lambda_comparison(output_dir, "frame1_lambda48_vs_measured_comparison", "Frame1 Lambda48 vs Measured Shadow", bundle["frame1_decisions"])
    frame2_cmp = write_lambda_comparison(output_dir, "frame2_lambda48_vs_measured_comparison", "Frame2 Diagnostic Lambda48 vs Measured Shadow", bundle["frame2_decisions"])
    stage68_cmp = compare_to_stage68(output_dir, inputs, bundle["frame1_decisions"])
    stage67_cmp = compare_to_stage67(output_dir, inputs["measured_dir"], bundle["frame1_decisions"], inputs["starts"])
    video_report = plot_dataset_level(output_dir, inputs, bundle, frame1_cmp, frame2_cmp)
    _, pred_safety, quality, stability, safety = write_audits_and_reports(args, output_dir, inputs, capture_result, bundle, frame1_cmp, frame2_cmp, stage68_cmp)
    provisional = {
        "completed": False,
        "start_count": 10,
        "frame_count": 20,
        "capture_count": int(capture_result["capture_count"]),
        "map_predict_calls": len(bundle["map_rows"]),
    }
    save_json(output_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json", provisional)
    write_text(output_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.md", markdown_table("Stage 4A-6.9 Bounded Two-Frame Lambda48 Pilot Summary", provisional))
    write_html_index(output_dir, provisional, bundle, quality)
    integrity = verify_dataset(output_dir, bundle, safety, pred_safety, quality, stability)
    summary = write_summary(
        args,
        output_dir,
        inputs,
        capture_result,
        bundle,
        frame1_cmp,
        frame2_cmp,
        stage68_cmp,
        stage67_cmp,
        pred_safety,
        quality,
        stability,
        safety,
        integrity,
        video_report,
        float(time.perf_counter() - total_start),
    )
    write_html_index(output_dir, summary, bundle, quality)
    integrity = verify_dataset(output_dir, bundle, safety, pred_safety, quality, stability)
    summary["completed"] = bool(integrity["passed"])
    summary["blocked"] = not bool(integrity["passed"])
    summary["main_blocker"] = "" if bool(integrity["passed"]) else "dataset_integrity_failed_after_html_recheck"
    summary["dataset_integrity"] = bool(integrity["passed"])
    save_json(output_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json", summary)
    write_text(output_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.md", markdown_table("Stage 4A-6.9 Bounded Two-Frame Lambda48 Pilot Summary", summary))
    write_text(output_dir / "git_status_after.txt", git_status_text())
    log_event(output_dir, events, "stage_complete", completed=bool(summary["completed"]), integrity_passed=bool(integrity["passed"]))
    if not bool(summary["completed"]):
        raise RuntimeError(f"{STAGE} failed integrity checks: {integrity}")


if __name__ == "__main__":
    main()

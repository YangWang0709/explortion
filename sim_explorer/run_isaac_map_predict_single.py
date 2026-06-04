#!/usr/bin/env python3
"""Run one Isaac depth frame through SSCNet and align it to observed_state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaac_sscnet_preprocess import (
    ALIGNMENT_CONVENTIONS,
    DEFAULT_HIGHRES_DIMS,
    DEFAULT_HIGHRES_VOXEL_SIZE,
    DEFAULT_LOCAL_VOLUME_M,
    DOMAIN_SHIFT_NOTE,
    alignment_convention_metadata,
    camera_coords_to_world,
    canonical_alignment_convention,
    local_index_grid_to_camera_coords,
    SSCNET_OUTPUT_AXIS_ORDER,
    load_json,
    preprocess_isaac_depth_for_sscnet,
    save_json,
    save_preprocessing_debug,
    world_to_global_grid,
)

try:
    from ssc_network.models import make_model
except ImportError:  # pragma: no cover - alternate PYTHONPATH layout
    from models import make_model  # type: ignore


DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)
DEFAULT_DATASET_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar"
DEFAULT_OUTPUT_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke"
NUM_CLASSES = 12
FREE_CLASS_ID = 0
UNKNOWN = np.int8(-1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_stats(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def parse_manifest_ok_rows(dataset_dir: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(dataset_dir) / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.jsonl not found: {manifest_path}")

    ok_rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                ok_rows.append(row)
    if not ok_rows:
        raise RuntimeError(f"No status=ok rows found in {manifest_path}")
    return ok_rows


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def select_input_sources(
    dataset_dir: str | Path,
    episode_index: int,
    step: int,
    depth_npy: str | Path | None = None,
    pose_json: str | Path | None = None,
    camera_info: str | Path | None = None,
    observed_state: str | Path | None = None,
    episode_summary: str | Path | None = None,
) -> dict[str, Any]:
    explicit = [depth_npy, pose_json, camera_info, observed_state]
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("Explicit input mode requires --depth_npy, --pose_json, --camera_info, and --observed_state")
        depth_path = _require_file(Path(depth_npy).resolve())  # type: ignore[arg-type]
        pose_path = _require_file(Path(pose_json).resolve())  # type: ignore[arg-type]
        camera_path = _require_file(Path(camera_info).resolve())  # type: ignore[arg-type]
        observed_path = _require_file(Path(observed_state).resolve())  # type: ignore[arg-type]
        summary_path = Path(episode_summary).resolve() if episode_summary else None
        if summary_path is not None:
            _require_file(summary_path)
        scene_metadata_path = depth_path.parent / "scene_metadata.json"
        return {
            "episode_id": depth_path.parent.name,
            "episode_dir": str(depth_path.parent),
            "episode_index": int(episode_index),
            "step": int(step),
            "depth_npy": str(depth_path),
            "pose_json": str(pose_path),
            "camera_info": str(camera_path),
            "observed_state": str(observed_path),
            "episode_summary": str(summary_path) if summary_path else "",
            "scene_metadata": str(scene_metadata_path) if scene_metadata_path.is_file() else "",
            "manifest_row": {},
        }

    ok_rows = parse_manifest_ok_rows(dataset_dir)
    if episode_index < 0 or episode_index >= len(ok_rows):
        raise IndexError(f"episode_index {episode_index} outside ok episode range [0,{len(ok_rows)})")
    row = ok_rows[int(episode_index)]
    episode_dir = Path(row["episode_dir"]).resolve()
    depth_path = _require_file(episode_dir / f"depth_{step:03d}.npy")
    pose_path = _require_file(episode_dir / f"pose_{step:03d}.json")
    camera_path = _require_file(episode_dir / "camera_info.json")
    observed_path = _require_file(episode_dir / f"observed_state_step{step:03d}.npy")
    summary_path = _require_file(episode_dir / "episode_summary.json")
    scene_metadata_path = episode_dir / "scene_metadata.json"

    return {
        "episode_id": row.get("episode_id", episode_dir.name),
        "episode_dir": str(episode_dir),
        "episode_index": int(episode_index),
        "step": int(step),
        "depth_npy": str(depth_path),
        "pose_json": str(pose_path),
        "camera_info": str(camera_path),
        "observed_state": str(observed_path),
        "episode_summary": str(summary_path),
        "scene_metadata": str(scene_metadata_path) if scene_metadata_path.is_file() else "",
        "manifest_row": row,
    }


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(path, map_location=device)


def load_sscnet_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = make_model("sscnet", num_classes=NUM_CLASSES).to(device)
    cp_states = _torch_load_checkpoint(checkpoint_path, device)
    if "state_dict" not in cp_states:
        raise KeyError(f"Checkpoint missing state_dict: {checkpoint_path}")
    model.load_state_dict(cp_states["state_dict"], strict=True)
    model.eval()
    return model


def run_sscnet_inference(
    checkpoint: str | Path,
    sscnet_depth_input: np.ndarray,
    sscnet_position: np.ndarray,
    device_name: str | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_sscnet_model(checkpoint, device)

    depth_t = torch.from_numpy(sscnet_depth_input[None, None, :, :].astype(np.float32)).to(device)
    position_t = torch.from_numpy(sscnet_position[None, :, :].astype(np.int64)).long().to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits_t = model(x_depth=depth_t, x_tsdf=None, p=position_t, x_rgb=None)
        class_prob_t = torch.softmax(logits_t, dim=1)
        confidence_t, pred_class_t = torch.max(class_prob_t, dim=1)
        free_prob_t = class_prob_t[:, FREE_CLASS_ID]
        occupied_prob_t = 1.0 - free_prob_t
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - start_time

    result = {
        "logits_shape": tuple(int(v) for v in logits_t.shape),
        "pred_class": pred_class_t.squeeze(0).detach().cpu().numpy().astype(np.uint8),
        "confidence": confidence_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "free_prob": free_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "occupied_prob": occupied_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "class_prob": class_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
        "device": str(device),
        "inference_time": float(inference_time),
        "loaded_strict": True,
        "gpu_memory_peak": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    return result


def save_local_prediction(
    output_dir: str | Path,
    inference: dict[str, Any],
    checkpoint: str | Path,
    depth_source: str | Path,
    pose_source: str | Path,
    preprocessing_notes: dict[str, Any],
    save_probs: bool = False,
) -> str:
    output_path = Path(output_dir) / "local_prediction.npz"
    payload: dict[str, Any] = {
        "pred_class": inference["pred_class"].astype(np.uint8),
        "confidence": inference["confidence"].astype(np.float32),
        "free_prob": inference["free_prob"].astype(np.float32),
        "occupied_prob": inference["occupied_prob"].astype(np.float32),
        "checkpoint": str(checkpoint),
        "depth_source": str(depth_source),
        "pose_source": str(pose_source),
        "preprocessing_notes": json.dumps(preprocessing_notes, sort_keys=True),
        "free_class_id": np.array(FREE_CLASS_ID, dtype=np.int64),
        "logits_shape": np.array(inference["logits_shape"], dtype=np.int64),
        "sscnet_output_axis_order": preprocessing_notes.get("sscnet_output_axis_order", SSCNET_OUTPUT_AXIS_ORDER),
        "alignment_convention": preprocessing_notes.get("alignment_convention", "current_default_v0"),
        "prediction_invalid_writeback": "none; local prediction is read-only",
    }
    if save_probs:
        payload["class_prob"] = inference["class_prob"].astype(np.float16)
    np.savez_compressed(output_path, **payload)
    return str(output_path)


def _pose_origin_yaw(pose: dict[str, Any]) -> tuple[np.ndarray, float]:
    if "position" not in pose:
        raise KeyError("pose JSON missing position")
    origin = np.asarray(pose["position"], dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError(f"pose position must have shape (3,), got {origin.shape}")
    if "yaw_rad" in pose:
        yaw = float(pose["yaw_rad"])
    elif "yaw_deg" in pose:
        yaw = math.radians(float(pose["yaw_deg"]))
    else:
        raise KeyError("pose JSON missing yaw_rad or yaw_deg")
    return origin, yaw


def normalize_bounds(raw: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    if raw is None:
        raw = {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]}
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def infer_map_bounds(
    episode_summary: dict[str, Any] | None,
    scene_metadata: dict[str, Any] | None,
) -> dict[str, tuple[float, float]]:
    if episode_summary and "map_bounds" in episode_summary:
        return normalize_bounds(episode_summary["map_bounds"])
    if scene_metadata and "map_bounds" in scene_metadata:
        return normalize_bounds(scene_metadata["map_bounds"])
    return normalize_bounds(None)


def infer_voxel_size(
    observed_shape: tuple[int, int, int],
    bounds: dict[str, tuple[float, float]],
    scene_metadata: dict[str, Any] | None,
) -> float:
    if scene_metadata:
        for key in ("voxel_size", "voxel_size_recommended"):
            if key in scene_metadata:
                return float(scene_metadata[key])
    return float((bounds["x"][1] - bounds["x"][0]) / float(observed_shape[0]))


def align_local_prediction_to_global(
    pred_class: np.ndarray,
    confidence: np.ndarray,
    free_prob: np.ndarray,
    occupied_prob: np.ndarray,
    observed_shape: tuple[int, int, int],
    pose: dict[str, Any],
    map_bounds: dict[str, Any],
    global_voxel_size: float,
    local_volume_m: tuple[float, float, float] = DEFAULT_LOCAL_VOLUME_M,
    local_lowres_voxel_size: float = 0.08,
    alignment_convention: str | None = "current_default_v0",
) -> dict[str, Any]:
    """Align SSCNet local output to the simulator global map."""

    convention = canonical_alignment_convention(alignment_convention)
    convention_meta = alignment_convention_metadata(convention)
    local_shape = tuple(int(v) for v in pred_class.shape)
    expected_shape = (60, 36, 60)
    if local_shape != expected_shape:
        raise ValueError(f"Expected local prediction shape {expected_shape}, got {local_shape}")
    if confidence.shape != local_shape or free_prob.shape != local_shape or occupied_prob.shape != local_shape:
        raise ValueError("Local prediction arrays must share the same shape")

    bounds = normalize_bounds(map_bounds)
    origin, yaw = _pose_origin_yaw(pose)
    x_right, y_up, z_forward = local_index_grid_to_camera_coords(
        local_shape,
        convention=convention,
        voxel_size=float(local_lowres_voxel_size),
        local_volume_m=local_volume_m,
    )
    world_x, world_y, world_z = camera_coords_to_world((x_right, y_up, z_forward), pose)

    mins = np.array([bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=np.float64)
    shape_arr = np.asarray(observed_shape, dtype=np.int64)
    gx, gy, gz = world_to_global_grid((world_x, world_y, world_z), bounds, float(global_voxel_size))
    inside = (
        (gx >= 0)
        & (gx < shape_arr[0])
        & (gy >= 0)
        & (gy < shape_arr[1])
        & (gz >= 0)
        & (gz < shape_arr[2])
    )

    global_pred_class = np.full(observed_shape, 255, dtype=np.uint8)
    global_confidence = np.zeros(observed_shape, dtype=np.float32)
    global_free_prob = np.full(observed_shape, 0.5, dtype=np.float32)
    global_occupied_prob = np.full(observed_shape, 0.5, dtype=np.float32)
    global_prediction_valid = np.zeros(observed_shape, dtype=bool)

    if np.any(inside):
        flat_global = np.ravel_multi_index((gx[inside], gy[inside], gz[inside]), observed_shape)
        conf_inside = confidence[inside]
        order = np.argsort(-conf_inside, kind="mergesort")
        flat_ordered = flat_global[order]
        _, first_positions = np.unique(flat_ordered, return_index=True)
        selected_inside_order = order[first_positions]

        selected_flat = flat_global[selected_inside_order]
        selected_idx = np.unravel_index(selected_flat, observed_shape)

        pred_inside = pred_class[inside]
        free_inside = free_prob[inside]
        occ_inside = occupied_prob[inside]

        global_pred_class[selected_idx] = pred_inside[selected_inside_order].astype(np.uint8)
        global_confidence[selected_idx] = conf_inside[selected_inside_order].astype(np.float32)
        global_free_prob[selected_idx] = free_inside[selected_inside_order].astype(np.float32)
        global_occupied_prob[selected_idx] = occ_inside[selected_inside_order].astype(np.float32)
        global_prediction_valid[selected_idx] = True

    align_stats = {
        "alignment_convention": convention,
        "alignment_convention_metadata": convention_meta,
        "local_axis_order": ",".join(convention_meta["output_axis_order"]),
        "global_axis_order": "world_x,world_y,world_z",
        "local_lowres_voxel_size": float(local_lowres_voxel_size),
        "global_voxel_size": float(global_voxel_size),
        "local_voxels_total": int(np.prod(local_shape)),
        "local_voxels_inside_global": int(np.count_nonzero(inside)),
        "inside_bounds_ratio": float(np.count_nonzero(inside) / max(1, int(np.prod(local_shape)))),
        "global_valid_prediction_count": int(np.count_nonzero(global_prediction_valid)),
        "in_front_local_ratio": float(np.count_nonzero(z_forward > 0.0) / max(1, z_forward.size)),
        "below_floor_local_count": int(np.count_nonzero(world_z < bounds["z"][0])),
        "above_ceiling_local_count": int(np.count_nonzero(world_z >= bounds["z"][1])),
        "collision_policy_for_duplicate_global_voxels": "keep highest confidence local voxel",
        "unpredicted_policy": "valid=False, confidence=0, free_prob=occupied_prob=0.5, pred_class=255",
        "map_bounds": {axis: [bounds[axis][0], bounds[axis][1]] for axis in ("x", "y", "z")},
        "camera_position": [float(v) for v in origin],
        "camera_yaw_rad": float(yaw),
    }
    return {
        "global_pred_class": global_pred_class,
        "global_confidence": global_confidence,
        "global_free_prob": global_free_prob,
        "global_occupied_prob": global_occupied_prob,
        "global_prediction_valid": global_prediction_valid,
        "align_stats": align_stats,
    }


def save_global_prediction_layer(
    output_dir: str | Path,
    aligned: dict[str, Any],
    observed_state_path: str | Path,
    local_prediction_path: str | Path,
    checkpoint: str | Path,
) -> str:
    output_path = Path(output_dir) / "global_prediction_layer.npz"
    np.savez_compressed(
        output_path,
        global_pred_class=aligned["global_pred_class"].astype(np.uint8),
        global_confidence=aligned["global_confidence"].astype(np.float32),
        global_free_prob=aligned["global_free_prob"].astype(np.float32),
        global_occupied_prob=aligned["global_occupied_prob"].astype(np.float32),
        global_prediction_valid=aligned["global_prediction_valid"].astype(bool),
        observed_state_source=str(observed_state_path),
        local_prediction_source=str(local_prediction_path),
        checkpoint=str(checkpoint),
        alignment_convention=str(aligned["align_stats"].get("alignment_convention", "current_default_v0")),
        alignment_stats_json=json.dumps(aligned["align_stats"], sort_keys=True),
        strict_no_observed_write=np.array(True, dtype=bool),
        read_only_note="prediction layer is read-only and was not written into observed_state",
    )
    return str(output_path)


def format_unique_counts(array: np.ndarray) -> str:
    values, counts = np.unique(array, return_counts=True)
    return ", ".join(f"{int(v)}:{int(c)}" for v, c in zip(values, counts))


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = select_input_sources(
        dataset_dir=args.dataset_dir,
        episode_index=int(args.episode_index),
        step=int(args.step),
        depth_npy=args.depth_npy,
        pose_json=args.pose_json,
        camera_info=args.camera_info,
        observed_state=args.observed_state,
        episode_summary=args.episode_summary,
    )
    depth_path = Path(sources["depth_npy"])
    pose_path = Path(sources["pose_json"])
    camera_info_path = Path(sources["camera_info"])
    observed_state_path = Path(sources["observed_state"])
    summary_path = Path(sources["episode_summary"]) if sources["episode_summary"] else None
    scene_metadata_path = Path(sources["scene_metadata"]) if sources["scene_metadata"] else None

    observed_hash_before = sha256_file(observed_state_path)
    depth = np.load(depth_path)
    pose = load_json(pose_path)
    camera_info = load_json(camera_info_path)
    observed_state = np.load(observed_state_path)
    episode_summary = load_json(summary_path) if summary_path and summary_path.is_file() else None
    scene_metadata = load_json(scene_metadata_path) if scene_metadata_path and scene_metadata_path.is_file() else None

    alignment_convention = canonical_alignment_convention(args.alignment_convention)
    preprocess = preprocess_isaac_depth_for_sscnet(
        depth=depth,
        camera_info=camera_info,
        pose=pose,
        alignment_convention=alignment_convention,
    )
    preprocess_paths = save_preprocessing_debug(
        output_dir=output_dir,
        preprocess_result=preprocess,
        depth_source=depth_path,
        pose_source=pose_path,
        camera_info_source=camera_info_path,
    )

    inference = run_sscnet_inference(
        checkpoint=args.checkpoint,
        sscnet_depth_input=preprocess["sscnet_depth_input"],
        sscnet_position=preprocess["sscnet_position"],
        device_name=args.device,
    )
    if inference["logits_shape"] != (1, NUM_CLASSES, 60, 36, 60):
        raise RuntimeError(f"Unexpected SSCNet logits shape: {inference['logits_shape']}")

    local_prediction_path = save_local_prediction(
        output_dir=output_dir,
        inference=inference,
        checkpoint=args.checkpoint,
        depth_source=depth_path,
        pose_source=pose_path,
        preprocessing_notes=preprocess["notes"],
        save_probs=bool(args.save_probs),
    )

    bounds = infer_map_bounds(episode_summary, scene_metadata)
    voxel_size = infer_voxel_size(tuple(int(v) for v in observed_state.shape), bounds, scene_metadata)
    aligned = align_local_prediction_to_global(
        pred_class=inference["pred_class"],
        confidence=inference["confidence"],
        free_prob=inference["free_prob"],
        occupied_prob=inference["occupied_prob"],
        observed_shape=tuple(int(v) for v in observed_state.shape),
        pose=pose,
        map_bounds=bounds,
        global_voxel_size=voxel_size,
        alignment_convention=alignment_convention,
    )
    global_prediction_path = save_global_prediction_layer(
        output_dir=output_dir,
        aligned=aligned,
        observed_state_path=observed_state_path,
        local_prediction_path=local_prediction_path,
        checkpoint=args.checkpoint,
    )
    observed_hash_after = sha256_file(observed_state_path)

    tau = float(args.tau)
    global_valid_tau = aligned["global_prediction_valid"] & (aligned["global_confidence"] >= tau)
    global_predicted_occupied = global_valid_tau & (aligned["global_occupied_prob"] >= 0.5)
    predicted_unmeasured = global_valid_tau & (observed_state == UNKNOWN)

    summary = {
        "stage": "Stage 4A-5 Isaac single-frame map_predict alignment smoke",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected_episode_id": sources["episode_id"],
        "selected_episode_index": int(sources["episode_index"]),
        "selected_step": int(sources["step"]),
        "depth_source": str(depth_path),
        "pose_source": str(pose_path),
        "camera_info_source": str(camera_info_path),
        "observed_state_source": str(observed_state_path),
        "episode_summary_source": str(summary_path) if summary_path else "",
        "scene_metadata_source": str(scene_metadata_path) if scene_metadata_path else "",
        "observed_state_shape": list(observed_state.shape),
        "local_prediction_shape": list(inference["pred_class"].shape),
        "global_prediction_shape": list(aligned["global_pred_class"].shape),
        "depth_input_shape": list(preprocess["sscnet_depth_input"].shape),
        "position_shape": list(preprocess["sscnet_position"].shape),
        "valid_position_pixels": int(np.count_nonzero(preprocess["valid_position_mask"])),
        "logits_shape": list(inference["logits_shape"]),
        "local_confidence": array_stats(inference["confidence"]),
        "local_occupied_prob": array_stats(inference["occupied_prob"]),
        "local_free_prob": array_stats(inference["free_prob"]),
        "local_pred_class_unique_counts": format_unique_counts(inference["pred_class"]),
        "global_valid_prediction_count": int(np.count_nonzero(aligned["global_prediction_valid"])),
        "global_predicted_occupied_count": int(np.count_nonzero(global_predicted_occupied)),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "tau": tau,
        "alignment_convention": alignment_convention,
        "device": inference["device"],
        "inference_time": float(inference["inference_time"]),
        "gpu_memory_peak": inference["gpu_memory_peak"],
        "preprocessing_axis_convention": preprocess["notes"]["position_axis_order"],
        "preprocessing_alignment_convention": preprocess["notes"]["alignment_convention"],
        "local_prediction_axis_convention": "pred_class/confidence/free_prob/occupied_prob arrays use "
        + ",".join(aligned["align_stats"]["alignment_convention_metadata"]["output_axis_order"]),
        "global_prediction_axis_convention": "global arrays match observed_state axis order (world_x,world_y,world_z)",
        "local_volume_convention": preprocess["notes"]["local_volume_convention"],
        "domain_shift_note": DOMAIN_SHIFT_NOTE,
        "alignment_stats": aligned["align_stats"],
        "strict_no_observed_write": observed_hash_before == observed_hash_after,
        "observed_state_sha256_before": observed_hash_before,
        "observed_state_sha256_after": observed_hash_after,
        "expert_used_prediction": False,
        "rollout_used_prediction": False,
        "expert_or_rollout_invoked": False,
        "rl_or_ppo_training": False,
        "optimizer_step": False,
        "behavior_cloning_training": False,
        "imitation_learning_training": False,
        "sscnet_training": False,
        "prediction_written_to_observed_state": False,
        "prediction_used_for_collision_or_traversability": False,
        "paths": {
            "sscnet_input_debug": preprocess_paths["sscnet_input_debug"],
            "sscnet_depth_input": preprocess_paths["sscnet_depth_input"],
            "sscnet_position": preprocess_paths["sscnet_position"],
            "valid_position_mask": preprocess_paths["valid_position_mask"],
            "local_prediction": local_prediction_path,
            "global_prediction_layer": global_prediction_path,
        },
    }

    summary_json = output_dir / "prediction_alignment_summary.json"
    save_json(summary_json, summary)

    if args.save_viz:
        from visualize_isaac_prediction_alignment import generate_visualizations

        viz_summary = generate_visualizations(output_dir=output_dir, tau=tau)
        summary["visualizations"] = viz_summary["generated_images"]
        save_json(summary_json, summary)

    if args.print_stats:
        print("Stage 4A-5 single-frame map_predict smoke complete")
        print(f"episode: {summary['selected_episode_id']}")
        print(f"step: {summary['selected_step']}")
        print(f"depth input shape: {tuple(summary['depth_input_shape'])}")
        print(f"position shape: {tuple(summary['position_shape'])}")
        print(f"valid position pixels: {summary['valid_position_pixels']}")
        print(f"logits shape: {tuple(summary['logits_shape'])}")
        print(f"local prediction shape: {tuple(summary['local_prediction_shape'])}")
        print(f"global prediction shape: {tuple(summary['global_prediction_shape'])}")
        print(f"global valid prediction count: {summary['global_valid_prediction_count']}")
        print(f"global predicted occupied count: {summary['global_predicted_occupied_count']}")
        print(f"predicted unmeasured count: {summary['predicted_unmeasured_count']}")
        print(f"alignment convention: {summary['alignment_convention']}")
        print(f"inference time: {summary['inference_time']:.4f}s")
        print(f"observed_state modified: {not summary['strict_no_observed_write']}")
        print(f"summary: {summary_json}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Isaac depth frame through SSCNet map_predict.")
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset_dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--depth_npy", type=Path, default=None)
    parser.add_argument("--pose_json", type=Path, default=None)
    parser.add_argument("--camera_info", type=Path, default=None)
    parser.add_argument("--observed_state", type=Path, default=None)
    parser.add_argument("--episode_summary", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--alignment_convention",
        choices=tuple(sorted(ALIGNMENT_CONVENTIONS.keys())),
        default="current_default_v0",
    )
    parser.add_argument("--save_probs", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--print_stats", action="store_true")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    run_single(parse_args())


if __name__ == "__main__":
    main()

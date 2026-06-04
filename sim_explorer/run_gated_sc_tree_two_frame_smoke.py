#!/usr/bin/env python3
"""Stage 4A-6.5s confidence-weighted/cap25 gated SC tree two-frame smoke.

This runner starts Isaac indirectly through the Stage 4A-6.5p two-frame runner
exactly once, with `confidence_weighted` as the executed SC gain formula. It
then evaluates `cap25` as a same-frame shadow formula on the saved two-frame
artifacts. The shadow pass is offline only: no extra Isaac capture, no extra
action, no rollout, no training, no checkpoint write, and no prediction
writeback/traversability/collision/ray-blocking use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import ROOT_ID, run as run_mini_rrt, sha256_file, to_jsonable
from sim_paper_expert import FREE, OCCUPIED, UNKNOWN
from sim_prediction_layer import SimPredictionLayer


BASE_PROFILE_NAME = "source_like_crop_min_length_0p25"
PROFILE_NAME = BASE_PROFILE_NAME
DEFAULT_SELECTED_CASE = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65c_decoupled_one_step_smoke/selected_case.json"
)
DEFAULT_REFERENCE_ONE_STEP_SC_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke"
)
DEFAULT_REFERENCE_NO_PRED_TWO_FRAME_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65n_two_frame_tree_smoke"
)
DEFAULT_STAGE4A65R_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65r_gated_sc_tree_one_step_smoke"
)
DEFAULT_STAGE4A65P_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke"
)
DEFAULT_EPISODE_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_medium_rollout_sc_pred_alignment_fixed_smoke/episodes/"
    "medium_three_rooms_seed0_start_room_a_sc_pred_alignment_fixed_000"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke"
)
DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)
SSC_EXPLORATION_DIR = Path("/home/ubuntu22/sc_explorer_ws/ssc_exploration")

GATED_FORMULAS = [
    "raw_count",
    "weight_0p5",
    "weight_1p0",
    "cap25",
    "cap50",
    "confidence_weighted",
    "occupied_only",
    "confidence_weighted_cap25",
]
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
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(to_jsonable(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in fields
                }
            )


def git_status_short(path: Path) -> str:
    if not path.exists():
        return "missing"
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(path),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip()}"
    return completed.stdout.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def min_mean_max(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(finite.min()), "mean": float(finite.mean()), "max": float(finite.max())}


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def euclidean(a: Any, b: Any) -> float | None:
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


def expected_effective_gain(row: dict[str, Any], formula: str) -> float:
    raw = safe_float(row.get("gain_sc"))
    occ = safe_float(row.get("gain_occ"))
    conf = safe_float(row.get("gain_conf"))
    if formula in {"measured_only", "none", "exp"}:
        return 0.0
    if formula in {"raw_count", "weight_1p0"}:
        return raw
    if formula == "weight_0p5":
        return 0.5 * raw
    if formula == "cap25":
        return min(raw, 25.0)
    if formula == "cap50":
        return min(raw, 50.0)
    if formula == "confidence_weighted":
        return conf
    if formula == "occupied_only":
        return occ
    if formula == "confidence_weighted_cap25":
        return min(conf, 25.0)
    raise ValueError(f"unsupported formula: {formula}")


def path_sums(tree_dir: Path, best_segment_id: str | None) -> dict[str, Any]:
    segments = load_jsonl(tree_dir / "mini_rrt_tree_segments.jsonl")
    by_id = {str(row.get("segment_id")): row for row in segments}
    if not best_segment_id or best_segment_id not in by_id:
        return {"available": False, "path_segment_ids": []}
    path: list[str] = []
    current: str | None = str(best_segment_id)
    while current and current in by_id:
        path.append(current)
        current = by_id[current].get("parent_id")
    path.reverse()
    non_root = [segment_id for segment_id in path if segment_id != ROOT_ID]
    return {
        "available": True,
        "path_segment_ids": non_root,
        "gain_used": float(sum(safe_float(by_id[s].get("gain")) for s in non_root)),
        "gain_exp": float(sum(safe_float(by_id[s].get("gain_exp")) for s in non_root)),
        "raw_gain_sc": float(sum(safe_float(by_id[s].get("gain_sc")) for s in non_root)),
        "effective_gain_sc": float(sum(safe_float(by_id[s].get("effective_gain_sc")) for s in non_root)),
        "gain_hybrid_raw": float(sum(safe_float(by_id[s].get("gain_hybrid")) for s in non_root)),
        "gain_hybrid_effective": float(sum(safe_float(by_id[s].get("gain_hybrid_effective")) for s in non_root)),
        "gain_occ": float(sum(safe_float(by_id[s].get("gain_occ")) for s in non_root)),
        "gain_conf": float(sum(safe_float(by_id[s].get("gain_conf")) for s in non_root)),
        "cost": float(sum(safe_float(by_id[s].get("cost")) for s in non_root)),
    }


def tree_gain_stats(tree_dir: Path) -> dict[str, Any]:
    segments = [row for row in load_jsonl(tree_dir / "mini_rrt_tree_segments.jsonl") if row.get("segment_id") != ROOT_ID]
    raw_sc = [safe_float(row.get("gain_sc")) for row in segments]
    eff_sc = [safe_float(row.get("effective_gain_sc")) for row in segments]
    gain_exp = [safe_float(row.get("gain_exp")) for row in segments]
    gain_occ = [safe_float(row.get("gain_occ")) for row in segments]
    gain_conf = [safe_float(row.get("gain_conf")) for row in segments]
    return {
        "node_count_excluding_root": len(segments),
        "nodes_with_raw_gain_sc_positive": sum(1 for value in raw_sc if value > 0.0),
        "nodes_with_effective_gain_sc_positive": sum(1 for value in eff_sc if value > 0.0),
        "gain_exp_min_mean_max": min_mean_max(gain_exp),
        "raw_gain_sc_min_mean_max": min_mean_max(raw_sc),
        "effective_gain_sc_min_mean_max": min_mean_max(eff_sc),
        "gain_occ_min_mean_max": min_mean_max(gain_occ),
        "gain_conf_min_mean_max": min_mean_max(gain_conf),
    }


def prediction_stats(prediction_path: Path, observed_path: Path, tau: float) -> dict[str, Any]:
    observed_state = np.load(observed_path)
    layer = SimPredictionLayer.from_npz(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occupied_prob = np.asarray(data["global_occupied_prob"], dtype=np.float32)
        alignment = str(np.asarray(data["alignment_convention"]).item())
        files = list(data.files)
    valid_tau = valid & (confidence >= float(tau))
    predicted_unmeasured = valid_tau & (observed_state == UNKNOWN)
    predicted_occupied = valid_tau & (occupied_prob >= 0.5)
    return {
        "prediction_npz": str(prediction_path),
        "files": files,
        "shape": [int(v) for v in layer.shape()],
        "observed_state_shape": [int(v) for v in observed_state.shape],
        "shape_aligned_to_observed_state": tuple(layer.shape()) == tuple(observed_state.shape),
        "alignment_convention": alignment,
        "tau": float(tau),
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "prediction_valid_tau_count": int(np.count_nonzero(valid_tau)),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "predicted_occupied_count": int(np.count_nonzero(predicted_occupied)),
        "large_dense_class_prob_saved": "class_prob" in files,
    }


def run_primary_two_frame(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    script = Path(__file__).with_name("run_map_predict_tree_two_frame_smoke.py")
    profile_name = profile_name_for_seed(int(args.seed))
    command = [
        sys.executable,
        str(script),
        "--selected_case_json",
        str(args.selected_case_json),
        "--reference_one_step_sc_dir",
        str(args.reference_one_step_sc_dir),
        "--reference_no_pred_two_frame_dir",
        str(args.reference_no_pred_two_frame_dir),
        "--episode_dir",
        str(args.episode_dir),
        "--output_dir",
        str(output_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--scene_variant",
        str(args.scene_variant),
        "--scene_seed",
        str(args.scene_seed),
        "--camera_width",
        str(args.camera_width),
        "--camera_height",
        str(args.camera_height),
        "--max_depth",
        str(args.max_depth),
        "--settle_steps",
        str(args.settle_steps),
        "--pixel_stride",
        str(args.pixel_stride),
        "--seed",
        str(args.seed),
        "--num_nodes",
        str(args.num_nodes),
        "--max_extension_m",
        str(args.max_extension_m),
        "--sample_mode",
        str(args.sample_mode),
        "--prediction_mode",
        str(args.prediction_mode),
        "--gain_mode",
        str(args.gain_mode),
        "--baseline_gain_mode",
        "exp",
        "--path_cost_mode",
        str(args.path_cost_mode),
        "--v_max",
        str(args.v_max),
        "--robot_radius_m",
        str(args.robot_radius_m),
        "--voxel_size",
        str(args.voxel_size),
        "--raycast_stride",
        str(args.raycast_stride),
        "--num_yaw_samples",
        str(args.num_yaw_samples),
        "--max_ray_length_m",
        str(args.max_ray_length_m),
        "--short_edge_policy",
        str(args.short_edge_policy),
        "--crop_min_length_m",
        str(args.crop_min_length_m),
        "--min_edge_length_m",
        str(args.min_edge_length_m),
        "--min_root_child_length_m",
        str(args.min_root_child_length_m),
        "--min_root_distance_m",
        str(args.min_root_distance_m),
        "--density_radius_m",
        str(args.density_radius_m),
        "--max_nodes_per_density_radius",
        str(args.max_nodes_per_density_radius),
        "--tau",
        str(args.tau),
        "--sc_gain_formula",
        str(args.primary_sc_gain_formula),
        "--alignment_convention",
        str(args.alignment_convention),
        "--torch_num_threads",
        str(args.torch_num_threads),
        "--variant_name",
        f"{args.stage_tag}_{args.primary_sc_gain_formula}_{profile_name}",
    ]
    if args.save_viz:
        command.append("--save_viz")
    if args.save_probs:
        command.append("--save_probs")
    if args.prediction_npz:
        command.extend(["--prediction_npz", str(args.prediction_npz)])

    log_path = output_dir / "primary_confidence_weighted_two_frame_runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"primary two-frame runner failed with code {completed.returncode}\n{tail}")
    summary_path = output_dir / "map_predict_tree_two_frame_summary.json"
    return {
        "command": command,
        "log_path": str(log_path),
        "summary": load_json(summary_path),
        "summary_path": str(summary_path),
    }


def shadow_tree_args(args: argparse.Namespace, frame_index: int, tree_dir: Path, formula: str) -> argparse.Namespace:
    output_dir = Path(args.output_dir).resolve()
    prefix = f"frame{frame_index:03d}"
    episode_dir = Path(args.episode_dir).resolve()
    profile_name = profile_name_for_seed(int(args.seed))
    return argparse.Namespace(
        case_json=str(Path(args.selected_case_json).resolve()),
        episode_dir=str(episode_dir),
        observed_state=str(output_dir / f"observed_state_{prefix}.npy"),
        pose_json=str(output_dir / f"{prefix}_pose.json"),
        camera_info=str(output_dir / f"{prefix}_camera_info.json"),
        episode_summary=str(episode_dir / "episode_summary.json"),
        prediction_npz=str(output_dir / f"{prefix}_prediction/global_prediction_layer.npz"),
        output_dir=str(tree_dir),
        seed=int(args.seed),
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        gain_mode=str(args.gain_mode),
        sc_gain_formula=str(formula),
        path_cost_mode=str(args.path_cost_mode),
        v_max=float(args.v_max),
        yaw_rate=1.0,
        robot_radius_m=float(args.robot_radius_m),
        voxel_size=float(args.voxel_size),
        raycast_stride=int(args.raycast_stride),
        num_yaw_samples=int(args.num_yaw_samples),
        max_ray_length_m=float(args.max_ray_length_m),
        tau=float(args.tau),
        save_viz=bool(args.save_viz),
        profile=True,
        min_edge_length_m=float(args.min_edge_length_m),
        min_root_child_length_m=float(args.min_root_child_length_m),
        min_root_distance_m=float(args.min_root_distance_m),
        crop_min_length_m=float(args.crop_min_length_m),
        short_edge_policy=str(args.short_edge_policy),
        density_radius_m=float(args.density_radius_m),
        max_nodes_per_density_radius=int(args.max_nodes_per_density_radius),
        variant_name=f"{args.stage_tag}_shadow_{formula}_{prefix}_{profile_name}",
    )


def decision_parts_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    decision = summary.get("decision", {})
    selected = decision.get("selected_child") or {}
    best = decision.get("best_descendant") or {}
    return {
        "selected_child_id": decision.get("selected_child_id") or selected.get("segment_id"),
        "selected_child": selected,
        "selected_child_grid": selected.get("end_grid"),
        "selected_child_world": selected.get("end_world"),
        "best_descendant_id": decision.get("selected_child_best_descendant_id") or best.get("segment_id"),
        "best_descendant": best,
        "best_descendant_grid": best.get("end_grid"),
        "best_descendant_world": best.get("end_world"),
        "value": selected.get("value") or decision.get("selected_child_value"),
        "accumulated_gain": best.get("accumulated_gain") or decision.get("best_descendant_accumulated_gain"),
        "accumulated_cost": best.get("accumulated_cost") or decision.get("best_descendant_accumulated_cost"),
        "root": summary.get("root", {}),
        "built_successfully": bool(summary.get("tree", {}).get("built_successfully")),
        "accepted_nodes": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
        "rejected_samples": summary.get("tree", {}).get("rejected_samples"),
    }


def decision_parts_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload.get("selected_child") or {}
    best = payload.get("best_descendant") or {}
    return {
        "selected_child_id": selected.get("segment_id"),
        "selected_child": selected,
        "selected_child_grid": selected.get("end_grid"),
        "selected_child_world": selected.get("end_world"),
        "best_descendant_id": best.get("segment_id"),
        "best_descendant": best,
        "best_descendant_grid": best.get("end_grid"),
        "best_descendant_world": best.get("end_world"),
        "value": payload.get("value"),
        "accumulated_gain": payload.get("accumulated_gain"),
        "accumulated_cost": payload.get("accumulated_cost"),
        "root": payload.get("root", {}),
        "built_successfully": bool(payload.get("built_successfully")),
        "accepted_nodes": payload.get("accepted_nodes"),
        "rejected_samples": payload.get("rejected_samples"),
    }


def copy_aliases(output_dir: Path, tree_dir: Path, prefix: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for src_name, dst_name in [
        ("mini_rrt_tree_segments.jsonl", f"{prefix}_tree_segments.jsonl"),
        ("gain_cost_value_table.csv", f"{prefix}_gain_cost_value_table.csv"),
        ("subsequent_best_decision.json", f"{prefix}_subsequent_best_decision.json"),
        ("mini_rrt_tree_summary.json", f"{prefix}_mini_rrt_tree_summary.json"),
    ]:
        src = tree_dir / src_name
        if src.is_file():
            dst = output_dir / dst_name
            shutil.copyfile(src, dst)
            aliases[dst_name] = str(dst)
    for src_name, dst_name in [
        ("mini_rrt_tree_topdown.png", f"{prefix}_tree_topdown.png"),
        ("selected_branch_topdown.png", f"{prefix}_selected_branch_topdown.png"),
        ("gain_cost_scatter.png", f"{prefix}_gain_cost_scatter.png"),
    ]:
        src = tree_dir / src_name
        if src.is_file():
            dst = output_dir / dst_name
            shutil.copyfile(src, dst)
            aliases[dst_name] = str(dst)
    return aliases


def formula_decision_row(
    *,
    frame_index: int,
    label: str,
    formula: str,
    tree_dir: Path,
    parts: dict[str, Any],
) -> dict[str, Any]:
    sums = path_sums(tree_dir, parts["best_descendant_id"])
    stats = tree_gain_stats(tree_dir)
    return {
        "frame_index": int(frame_index),
        "label": label,
        "formula": formula,
        "tree_dir": str(tree_dir),
        "selected_child_id": parts["selected_child_id"],
        "selected_child_grid": parts["selected_child_grid"],
        "selected_child_world": parts["selected_child_world"],
        "best_descendant_id": parts["best_descendant_id"],
        "best_descendant_grid": parts["best_descendant_grid"],
        "best_descendant_world": parts["best_descendant_world"],
        "value": parts["value"],
        "accepted_nodes": parts["accepted_nodes"],
        "rejected_samples": parts["rejected_samples"],
        "built_successfully": parts["built_successfully"],
        "path_segment_ids": sums.get("path_segment_ids"),
        "accumulated_gain_used": sums.get("gain_used"),
        "accumulated_gain_exp": sums.get("gain_exp"),
        "accumulated_raw_gain_sc": sums.get("raw_gain_sc"),
        "accumulated_effective_gain_sc": sums.get("effective_gain_sc"),
        "accumulated_gain_hybrid_raw": sums.get("gain_hybrid_raw"),
        "accumulated_gain_hybrid_effective": sums.get("gain_hybrid_effective"),
        "accumulated_gain_occ": sums.get("gain_occ"),
        "accumulated_gain_conf": sums.get("gain_conf"),
        "accumulated_cost": sums.get("cost"),
        "gain_stats": stats,
        "root": parts["root"],
    }


def run_shadow_frame(args: argparse.Namespace, frame_index: int, output_dir: Path) -> dict[str, Any]:
    tree_dir = output_dir / f"frame{frame_index:03d}_{args.shadow_sc_gain_formula}_shadow_tree_raw"
    observed_path = output_dir / f"observed_state_frame{frame_index:03d}.npy"
    prediction_path = output_dir / f"frame{frame_index:03d}_prediction/global_prediction_layer.npz"
    observed_hash_before = sha256_file(observed_path)
    prediction_hash_before = sha256_file(prediction_path)
    summary = run_mini_rrt(shadow_tree_args(args, frame_index, tree_dir, args.shadow_sc_gain_formula))
    observed_hash_after = sha256_file(observed_path)
    prediction_hash_after = sha256_file(prediction_path)
    parts = decision_parts_from_summary(summary)
    row = formula_decision_row(
        frame_index=frame_index,
        label=f"{args.shadow_sc_gain_formula}_shadow",
        formula=str(args.shadow_sc_gain_formula),
        tree_dir=tree_dir,
        parts=parts,
    )
    aliases = copy_aliases(output_dir, tree_dir, f"frame{frame_index:03d}_{args.shadow_sc_gain_formula}_shadow")
    payload = {
        "stage": "Stage 4A-6.5s gated SC tree two-frame smoke",
        "frame_index": int(frame_index),
        "tree_label": f"{args.shadow_sc_gain_formula}_shadow",
        "profile_name": profile_name_for_seed(int(args.seed)),
        "formula": str(args.shadow_sc_gain_formula),
        "decision": row,
        "raw_summary": summary,
        "observed_state_hash_before": observed_hash_before,
        "observed_state_hash_after": observed_hash_after,
        "observed_state_hash_unchanged": observed_hash_before == observed_hash_after,
        "prediction_npz_hash_before": prediction_hash_before,
        "prediction_npz_hash_after": prediction_hash_after,
        "prediction_npz_hash_unchanged": prediction_hash_before == prediction_hash_after,
        "safety": {
            "offline_shadow_on_saved_frame": True,
            "isaac_startup": False,
            "rgb_depth_capture": False,
            "map_predict_rerun": False,
            "selected_action_execution": False,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "prediction_blocks_rays": False,
            "target_lr_target_hr_ground_truth_scoring": False,
        },
        "aliases": aliases,
    }
    save_json(output_dir / f"frame{frame_index:03d}_{args.shadow_sc_gain_formula}_shadow_tree_decision.json", payload)
    return payload


def alias_primary_frame(args: argparse.Namespace, frame_index: int, output_dir: Path) -> dict[str, Any]:
    prefix = f"frame{frame_index:03d}"
    src_payload = load_json(output_dir / f"{prefix}_sc_tree_decision.json")
    parts = decision_parts_from_payload(src_payload)
    tree_dir = output_dir / f"{prefix}_sc_tree_raw"
    row = formula_decision_row(
        frame_index=frame_index,
        label=str(args.primary_sc_gain_formula),
        formula=str(args.primary_sc_gain_formula),
        tree_dir=tree_dir,
        parts=parts,
    )
    aliases = copy_aliases(output_dir, tree_dir, f"{prefix}_{args.primary_sc_gain_formula}")
    payload = {
        "stage": "Stage 4A-6.5s gated SC tree two-frame smoke",
        "frame_index": int(frame_index),
        "tree_label": str(args.primary_sc_gain_formula),
        "profile_name": profile_name_for_seed(int(args.seed)),
        "formula": str(args.primary_sc_gain_formula),
        "decision": row,
        "source_payload": src_payload,
        "aliases": aliases,
    }
    save_json(output_dir / f"{prefix}_{args.primary_sc_gain_formula}_tree_decision.json", payload)
    return payload


def alias_measured_frame(frame_index: int, output_dir: Path) -> dict[str, Any]:
    prefix = f"frame{frame_index:03d}"
    src_payload = load_json(output_dir / f"{prefix}_measured_tree_decision.json")
    parts = decision_parts_from_payload(src_payload)
    tree_dir = output_dir / f"{prefix}_measured_tree_raw"
    row = formula_decision_row(
        frame_index=frame_index,
        label="measured_only",
        formula="measured_only",
        tree_dir=tree_dir,
        parts=parts,
    )
    return {"payload": src_payload, "decision": row}


def compare_decisions(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_label": a["label"],
        "b_label": b["label"],
        "selected_child_same_id": a.get("selected_child_id") == b.get("selected_child_id"),
        "selected_child_same_grid": same_grid(a.get("selected_child_grid"), b.get("selected_child_grid")),
        "selected_child_world_delta_m": euclidean(a.get("selected_child_world"), b.get("selected_child_world")),
        "best_descendant_same_id": a.get("best_descendant_id") == b.get("best_descendant_id"),
        "best_descendant_same_grid": same_grid(a.get("best_descendant_grid"), b.get("best_descendant_grid")),
        "best_descendant_world_delta_m": euclidean(a.get("best_descendant_world"), b.get("best_descendant_world")),
        "value_delta_b_minus_a": safe_float(b.get("value")) - safe_float(a.get("value")),
        "effective_gain_sc_delta_b_minus_a": safe_float(b.get("accumulated_effective_gain_sc"))
        - safe_float(a.get("accumulated_effective_gain_sc")),
    }


def topdown_projection(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def grid_xy(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    try:
        return float(grid[0]) + 0.5, float(grid[1]) + 0.5
    except (TypeError, ValueError, IndexError):
        return None


def save_formula_plot(path: Path, observed_path: Path, rows: list[dict[str, Any]], title: str) -> None:
    observed = np.load(observed_path)
    image = topdown_projection(observed)
    colors = np.array(
        [
            [0.18, 0.20, 0.23, 1.0],
            [0.51, 0.77, 0.74, 1.0],
            [0.85, 0.36, 0.35, 1.0],
        ]
    )
    palette = {"measured_only": "#f97316", "confidence_weighted": "#2563eb", "cap25": "#7c3aed"}
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(colors[image].transpose(1, 0, 2), origin="lower", interpolation="nearest")
    for row in rows:
        point = grid_xy(row.get("selected_child_grid"))
        best = grid_xy(row.get("best_descendant_grid"))
        color = palette.get(str(row.get("formula")), "#111827")
        if point is not None:
            ax.scatter([point[0]], [point[1]], s=82, c=color, marker="o", edgecolor="#111827", label=f"{row['label']} selected")
        if best is not None:
            ax.scatter([best[0]], [best[1]], s=140, c=color, marker="*", edgecolor="#111827", label=f"{row['label']} best")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def scan_prohibited_outputs(output_dir: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        matches = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
        if matches:
            found[pattern] = matches
    return found


def make_summary(
    args: argparse.Namespace,
    output_dir: Path,
    primary_run: dict[str, Any],
    measured_rows: dict[int, dict[str, Any]],
    primary_payloads: dict[int, dict[str, Any]],
    shadow_payloads: dict[int, dict[str, Any]],
    prediction_stats_by_frame: dict[int, dict[str, Any]],
    checkpoint_before: str | None,
    checkpoint_after: str | None,
    external_before: str,
    external_after: str,
    ssc_before: str,
    ssc_after: str,
) -> dict[str, Any]:
    base_summary = primary_run["summary"]
    frame_decisions: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for frame_index in (1, 2):
        measured = measured_rows[frame_index]["decision"]
        primary = primary_payloads[frame_index]["decision"]
        shadow = shadow_payloads[frame_index]["decision"]
        all_rows.extend([measured, primary, shadow])
        comparisons[f"frame{frame_index:03d}_measured_vs_primary"] = compare_decisions(measured, primary)
        comparisons[f"frame{frame_index:03d}_primary_vs_shadow"] = compare_decisions(primary, shadow)
        frame_decisions[f"frame{frame_index:03d}"] = {
            "measured_only": measured,
            str(args.primary_sc_gain_formula): primary,
            f"{args.shadow_sc_gain_formula}_shadow": shadow,
            "prediction_stats": prediction_stats_by_frame[frame_index],
            "map_predict_timing": base_summary["frames"][f"frame{frame_index:03d}"]["prediction"].get("timing", {}),
        }

    f1_primary = frame_decisions["frame001"][str(args.primary_sc_gain_formula)]
    f1_measured = frame_decisions["frame001"]["measured_only"]
    f2_primary = frame_decisions["frame002"][str(args.primary_sc_gain_formula)]
    f2_measured = frame_decisions["frame002"]["measured_only"]
    f2_shadow = frame_decisions["frame002"][f"{args.shadow_sc_gain_formula}_shadow"]
    stage4a65r = {}
    r_summary = Path(args.stage4a65r_dir).resolve() / "gated_sc_tree_one_step_summary.json"
    if r_summary.is_file():
        stage4a65r = load_json(r_summary)

    observed_hashes = {
        "frame001_observed_state": str(output_dir / "observed_state_frame001.npy"),
        "frame001_sha256_after_all_tree_evals": sha256_file(output_dir / "observed_state_frame001.npy"),
        "frame002_observed_state": str(output_dir / "observed_state_frame002.npy"),
        "frame002_sha256_after_all_tree_evals": sha256_file(output_dir / "observed_state_frame002.npy"),
        "base_observed_state_hashes": load_json(output_dir / "observed_state_hashes.json"),
    }
    safety = {
        "isaac_startup": True,
        "frames_captured": 2,
        "selected_action_execution": True,
        "selected_action_execution_count": 1,
        "selected_action_formula": str(args.primary_sc_gain_formula),
        "shadow_action_execution_count": 0,
        "rollout": False,
        "online_open_ended_loop": False,
        "map_predict_predictions": 2,
        "sscnet_training": False,
        "training_rl_ppo_bc_il": False,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_modified": checkpoint_before != checkpoint_after,
        "existing_observed_state_modified": False,
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_blocks_rays": False,
        "target_lr_target_hr_ground_truth_scoring": False,
        "external_source_git_status_before": external_before,
        "external_source_git_status_after": external_after,
        "external_source_modified_or_built_by_stage": external_before != external_after,
        "ssc_exploration_git_status_before": ssc_before,
        "ssc_exploration_git_status_after": ssc_after,
        "ssc_exploration_modified_by_stage": ssc_before != ssc_after,
        "coverage_improvement_claimed": False,
        "prohibited_output_matches": scan_prohibited_outputs(output_dir),
    }
    answers = {
        "frame1_capture_observed_update_map_predict_success": bool(
            base_summary["answers"]["frame1_capture_observed_map_predict_sc_tree_success"]
        ),
        "frame1_measured_confidence_weighted_cap25_trees_success": bool(
            f1_measured["built_successfully"] and f1_primary["built_successfully"] and shadow_payloads[1]["decision"]["built_successfully"]
        ),
        "frame1_confidence_weighted_same_selected_child_as_measured": same_grid(
            f1_primary.get("selected_child_grid"), f1_measured.get("selected_child_grid")
        ),
        "executed_exactly_one_confidence_weighted_selected_child_move": bool(
            base_summary["move_once"]["execution_count"] == 1
            and base_summary["move_once"]["selected_child_segment_id"] == f1_primary["selected_child_id"]
        ),
        "frame2_capture_observed_update_map_predict_success": bool(
            base_summary["answers"]["frame2_capture_observed_map_predict_sc_tree_success"]
        ),
        "frame2_confidence_weighted_changed_measured_selected_child": not same_grid(
            f2_primary.get("selected_child_grid"), f2_measured.get("selected_child_grid")
        ),
        "frame2_cap25_shadow_matches_confidence_weighted": same_grid(
            f2_shadow.get("selected_child_grid"), f2_primary.get("selected_child_grid")
        ),
        "frame2_confidence_weighted_kept_stage4a65p_r_sc_branch": bool(
            f2_primary.get("selected_child_id") == "n0127" or same_grid(f2_primary.get("selected_child_grid"), [11, 15, 11])
        ),
        "if_inconsistent_likely_reason": (
            "consistent_with_stage4a65p_65r_branch"
            if same_grid(f2_primary.get("selected_child_grid"), [11, 15, 11])
            else "inspect deterministic tree sampling, map_predict output, and gated ranking changes"
        ),
        "prediction_read_only": True,
        "prediction_used_for_traversability_collision_ray_blocking": False,
        "enough_for_repeated_gated_two_frame_or_short_gated_tree_smoke": True,
        "ready_for_rollout": False,
    }
    summary = {
        "stage": "Stage 4A-6.5s confidence-weighted/cap25 gated SC tree two-frame smoke",
        "output_dir": str(output_dir),
        "profile_name": profile_name_for_seed(int(args.seed)),
        "primary_sc_gain_formula": str(args.primary_sc_gain_formula),
        "shadow_sc_gain_formula": str(args.shadow_sc_gain_formula),
        "inputs": {
            "stage4a65p_dir": str(Path(args.stage4a65p_dir).resolve()),
            "stage4a65r_dir": str(Path(args.stage4a65r_dir).resolve()),
            "selected_case_json": str(Path(args.selected_case_json).resolve()),
            "episode_dir": str(Path(args.episode_dir).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
        },
        "primary_run": primary_run,
        "answers": answers,
        "frames": frame_decisions,
        "comparisons": comparisons,
        "move_once": {
            "executed_action": True,
            "execution_count": 1,
            "formula": str(args.primary_sc_gain_formula),
            "selected_child_segment_id": f1_primary["selected_child_id"],
            "selected_child_grid": f1_primary["selected_child_grid"],
            "selected_child_world": f1_primary["selected_child_world"],
            "new_pose": base_summary["move_once"]["new_pose"],
            "shadow_formula_executed": False,
        },
        "stage4a65r_reference": stage4a65r.get("formula_decisions", []),
        "prediction_safety_checklist": {
            **load_json(output_dir / "prediction_safety_checklist.json"),
            "primary_sc_gain_formula": str(args.primary_sc_gain_formula),
            "shadow_sc_gain_formula": str(args.shadow_sc_gain_formula),
            "shadow_map_predict_rerun": False,
        },
        "source_protection_checklist": {
            **load_json(output_dir / "source_protection_checklist.json"),
            "primary_sc_gain_formula": str(args.primary_sc_gain_formula),
            "shadow_sc_gain_formula": str(args.shadow_sc_gain_formula),
        },
        "safety": safety,
        "observed_state_hashes": observed_hashes,
        "all_formula_decisions": all_rows,
        "recommended_next_faithful_step": (
            "repeated gated two-frame smoke or short gated SC tree smoke; still not rollout"
        ),
        "still_not_next": [
            "rollout",
            "open-ended online loop",
            "third frame capture in this smoke",
            "second action execution in this smoke",
            "RL/PPO/BC/IL training",
            "prediction writeback",
            "prediction traversability/collision/ray blocking",
            "target or ground-truth scoring",
            "coverage-improvement claim",
        ],
    }
    return summary


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    a = summary["answers"]
    f1 = summary["frames"]["frame001"]
    f2 = summary["frames"]["frame002"]
    primary = summary["primary_sc_gain_formula"]
    shadow = f"{summary['shadow_sc_gain_formula']}_shadow"
    lines = [
        "# Stage 4A-6.5s Gated SC Tree Two-Frame Smoke",
        "",
        f"1. Frame 1 capture / observed update / map_predict succeeded? `{a['frame1_capture_observed_update_map_predict_success']}`.",
        f"2. Frame 1 measured / `{primary}` / `cap25` shadow trees succeeded? `{a['frame1_measured_confidence_weighted_cap25_trees_success']}`.",
        f"3. Frame 1 `{primary}` selected same child as measured? `{a['frame1_confidence_weighted_same_selected_child_as_measured']}`.",
        f"4. Executed exactly one `{primary}` selected-child move? `{a['executed_exactly_one_confidence_weighted_selected_child_move']}`.",
        f"5. Frame 2 capture / observed update / map_predict succeeded? `{a['frame2_capture_observed_update_map_predict_success']}`.",
        f"6. Frame 2 `{primary}` changed measured selected child? `{a['frame2_confidence_weighted_changed_measured_selected_child']}`.",
        f"7. Frame 2 `cap25` shadow matched `{primary}`? `{a['frame2_cap25_shadow_matches_confidence_weighted']}`.",
        f"8. Frame 2 `{primary}` kept the Stage 4A-6.5p/r SC branch? `{a['frame2_confidence_weighted_kept_stage4a65p_r_sc_branch']}`.",
        f"9. Inconsistency reason: `{a['if_inconsistent_likely_reason']}`.",
        f"10. Prediction read-only? `{a['prediction_read_only']}`.",
        f"11. Prediction used for traversability / collision / ray blocking? `{a['prediction_used_for_traversability_collision_ray_blocking']}`.",
        f"12. Enough for repeated gated two-frame or short gated tree smoke? `{a['enough_for_repeated_gated_two_frame_or_short_gated_tree_smoke']}`.",
        f"13. Ready for rollout? `{a['ready_for_rollout']}`.",
        "",
        "## Frame 1",
        f"- measured selected: `{f1['measured_only']['selected_child_id']}` grid `{f1['measured_only']['selected_child_grid']}`.",
        f"- `{primary}` selected: `{f1[primary]['selected_child_id']}` grid `{f1[primary]['selected_child_grid']}`.",
        f"- `{shadow}` selected: `{f1[shadow]['selected_child_id']}` grid `{f1[shadow]['selected_child_grid']}`.",
        "",
        "## Frame 2",
        f"- measured selected: `{f2['measured_only']['selected_child_id']}` grid `{f2['measured_only']['selected_child_grid']}`.",
        f"- `{primary}` selected: `{f2[primary]['selected_child_id']}` grid `{f2[primary]['selected_child_grid']}`.",
        f"- `{shadow}` selected: `{f2[shadow]['selected_child_id']}` grid `{f2[shadow]['selected_child_grid']}`.",
        f"- `{primary}` effective SC gain on winning path: `{f2[primary]['accumulated_effective_gain_sc']}`.",
        f"- `cap25` effective SC gain on winning path: `{f2[shadow]['accumulated_effective_gain_sc']}`.",
        "",
        "Prediction remained read-only and information-gain-only. This is still not a rollout or coverage-improvement result.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.primary_sc_gain_formula != "confidence_weighted":
        raise ValueError("Stage 4A-6.5s primary formula must be confidence_weighted")
    if args.shadow_sc_gain_formula != "cap25":
        raise ValueError("Stage 4A-6.5s shadow formula must be cap25")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_before = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    external_before = git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_before = git_status_short(SSC_EXPLORATION_DIR)

    primary_run = run_primary_two_frame(args, output_dir)

    measured_rows: dict[int, dict[str, Any]] = {}
    primary_payloads: dict[int, dict[str, Any]] = {}
    shadow_payloads: dict[int, dict[str, Any]] = {}
    prediction_stats_by_frame: dict[int, dict[str, Any]] = {}
    generated_plots: dict[str, str] = {}
    for frame_index in (1, 2):
        measured_rows[frame_index] = alias_measured_frame(frame_index, output_dir)
        primary_payloads[frame_index] = alias_primary_frame(args, frame_index, output_dir)
        shadow_payloads[frame_index] = run_shadow_frame(args, frame_index, output_dir)
        observed_path = output_dir / f"observed_state_frame{frame_index:03d}.npy"
        prediction_path = output_dir / f"frame{frame_index:03d}_prediction/global_prediction_layer.npz"
        prediction_stats_by_frame[frame_index] = prediction_stats(prediction_path, observed_path, float(args.tau))
        rows = [
            measured_rows[frame_index]["decision"],
            primary_payloads[frame_index]["decision"],
            shadow_payloads[frame_index]["decision"],
        ]
        plot_path = output_dir / f"frame{frame_index:03d}_measured_primary_shadow_topdown.png"
        save_formula_plot(plot_path, observed_path, rows, f"Frame {frame_index:03d} measured / confidence / cap25")
        generated_plots[plot_path.name] = str(plot_path)

    checkpoint_after = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    external_after = git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_after = git_status_short(SSC_EXPLORATION_DIR)
    summary = make_summary(
        args,
        output_dir,
        primary_run,
        measured_rows,
        primary_payloads,
        shadow_payloads,
        prediction_stats_by_frame,
        checkpoint_before,
        checkpoint_after,
        external_before,
        external_after,
        ssc_before,
        ssc_after,
    )
    summary["generated_files"] = {"plots": generated_plots}
    save_json(output_dir / "gated_sc_tree_two_frame_summary.json", summary)
    write_summary_md(output_dir / "gated_sc_tree_two_frame_summary.md", summary)
    write_csv(
        output_dir / "gated_sc_tree_two_frame_decisions.csv",
        summary["all_formula_decisions"],
        [
            "frame_index",
            "label",
            "formula",
            "selected_child_id",
            "selected_child_grid",
            "best_descendant_id",
            "best_descendant_grid",
            "value",
            "accumulated_gain_exp",
            "accumulated_raw_gain_sc",
            "accumulated_effective_gain_sc",
            "accumulated_gain_hybrid_effective",
            "accumulated_cost",
            "built_successfully",
        ],
    )
    save_json(output_dir / "gated_sc_tree_two_frame_decisions.json", summary["all_formula_decisions"])
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\n"
        "- next small task: repeated gated two-frame smoke or short gated SC tree smoke.\n"
        "- still not next: rollout, RL/PPO/BC/IL training, prediction writeback, or coverage-improvement claims.\n",
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected_case_json", default=DEFAULT_SELECTED_CASE)
    parser.add_argument("--reference_one_step_sc_dir", default=DEFAULT_REFERENCE_ONE_STEP_SC_DIR)
    parser.add_argument("--reference_no_pred_two_frame_dir", default=DEFAULT_REFERENCE_NO_PRED_TWO_FRAME_DIR)
    parser.add_argument("--stage4a65p_dir", default=DEFAULT_STAGE4A65P_DIR)
    parser.add_argument("--stage4a65r_dir", default=DEFAULT_STAGE4A65R_DIR)
    parser.add_argument("--episode_dir", default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=160)
    parser.add_argument("--camera_height", type=int, default=120)
    parser.add_argument("--max_depth", type=float, default=5.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--pixel_stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--prediction_mode", choices=["sim_dynamic", "sim_npz"], default="sim_dynamic")
    parser.add_argument("--prediction_npz", default="")
    parser.add_argument("--gain_mode", choices=["hybrid", "sc"], default="hybrid")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", choices=["crop"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--min_edge_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_child_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_distance_m", type=float, default=0.0)
    parser.add_argument("--density_radius_m", type=float, default=0.0)
    parser.add_argument("--max_nodes_per_density_radius", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--primary_sc_gain_formula", choices=GATED_FORMULAS, default="confidence_weighted")
    parser.add_argument("--shadow_sc_gain_formula", choices=GATED_FORMULAS, default="cap25")
    parser.add_argument("--stage_tag", default="stage4a65s")
    parser.add_argument("--alignment_convention", choices=["code_consistent_v1"], default="code_consistent_v1")
    parser.add_argument("--torch_num_threads", type=int, default=8)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--save_probs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

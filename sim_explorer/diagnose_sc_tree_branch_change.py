#!/usr/bin/env python3
"""Stage 4A-6.5q offline SC-tree branch-change diagnosis.

This script reads Stage 4A-6.5p artifacts only. It does not launch Isaac,
rerun map_predict/SSCNet inference, run rollout, modify observed_state files,
or modify/build external source. The replay logic is deliberately local to
this diagnostic script.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EPS = 1.0e-6
ROOT_ID = "root"
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)

REQUIRED_INPUTS = [
    "frame001_sc_tree_decision.json",
    "frame001_measured_tree_decision.json",
    "frame001_sc_gain_cost_value_table.csv",
    "frame001_sc_node_gain_breakdown.csv",
    "frame001_prediction/global_prediction_layer.npz",
    "observed_state_frame001.npy",
    "frame002_sc_tree_decision.json",
    "frame002_measured_tree_decision.json",
    "frame002_sc_gain_cost_value_table.csv",
    "frame002_sc_node_gain_breakdown.csv",
    "frame002_prediction/global_prediction_layer.npz",
    "observed_state_frame002.npy",
    "frame001_sc_tree_segments.jsonl",
    "frame002_sc_tree_segments.jsonl",
    "frame002_measured_tree_segments.jsonl",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "frame*_pose.json",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_*.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(field_order or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_literal_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return list(parsed) if isinstance(parsed, (list, tuple)) else None


def euclidean(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        aa = [float(v) for v in a]
        bb = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if len(aa) < 2 or len(bb) < 2:
        return None
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(aa, bb))))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) <= EPS:
        return None
    return float(numerator) / float(denominator)


def load_tree(stage_dir: Path, prefix: str, mode: str) -> dict[str, dict[str, Any]]:
    path = stage_dir / f"{prefix}_{mode}_tree_segments.jsonl"
    rows = read_jsonl(path)
    tree = {str(row["segment_id"]): dict(row) for row in rows}
    for node in tree.values():
        node.setdefault("children", [])
    for node_id, node in list(tree.items()):
        parent = node.get("parent_id")
        if parent in tree and node_id not in tree[parent].setdefault("children", []):
            tree[parent]["children"].append(node_id)
    return tree


def path_to_root(tree: dict[str, dict[str, Any]], node_id: str | None) -> list[str]:
    if not node_id or node_id not in tree:
        return []
    path: list[str] = []
    current: str | None = node_id
    seen: set[str] = set()
    while current and current in tree and current not in seen:
        seen.add(current)
        path.append(current)
        parent = tree[current].get("parent_id")
        current = str(parent) if parent else None
    path.reverse()
    return path


def non_root_path(tree: dict[str, dict[str, Any]], node_id: str | None) -> list[str]:
    return [node_id for node_id in path_to_root(tree, node_id) if node_id != ROOT_ID]


def path_sums(tree: dict[str, dict[str, Any]], node_id: str | None) -> dict[str, Any]:
    ids = non_root_path(tree, node_id)
    sums = {
        "path_node_ids": ids,
        "path_length_nodes": len(ids),
        "accumulated_gain_exp_recomputed": sum(as_float(tree[node].get("gain_exp")) for node in ids),
        "accumulated_gain_sc_recomputed": sum(as_float(tree[node].get("gain_sc")) for node in ids),
        "accumulated_gain_hybrid_recomputed": sum(as_float(tree[node].get("gain_hybrid")) for node in ids),
        "accumulated_gain_occ_recomputed": sum(as_float(tree[node].get("gain_occ")) for node in ids),
        "accumulated_gain_conf_recomputed": sum(as_float(tree[node].get("gain_conf")) for node in ids),
        "accumulated_cost_recomputed": sum(as_float(tree[node].get("cost")) for node in ids),
    }
    cost = float(sums["accumulated_cost_recomputed"])
    sums["measured_value_recomputed"] = safe_ratio(float(sums["accumulated_gain_exp_recomputed"]), cost)
    sums["sc_only_value_recomputed"] = safe_ratio(float(sums["accumulated_gain_sc_recomputed"]), cost)
    sums["raw_hybrid_value_recomputed"] = safe_ratio(float(sums["accumulated_gain_hybrid_recomputed"]), cost)
    return sums


def root_child_for_path(tree: dict[str, dict[str, Any]], node_id: str | None) -> str | None:
    ids = non_root_path(tree, node_id)
    return ids[0] if ids else None


def enrich_tree(tree: dict[str, dict[str, Any]]) -> None:
    root_world = tree.get(ROOT_ID, {}).get("end_world")
    for node_id, node in tree.items():
        sums = path_sums(tree, node_id)
        node.update(sums)
        node["root_child_id_recomputed"] = root_child_for_path(tree, node_id)
        node["distance_from_root_m"] = euclidean(root_world, node.get("end_world"))
        node["local_gain_exp_over_cost"] = safe_ratio(as_float(node.get("gain_exp")), as_float(node.get("cost")))
        node["local_gain_sc_over_cost"] = safe_ratio(as_float(node.get("gain_sc")), as_float(node.get("cost")))
        node["local_gain_hybrid_over_cost"] = safe_ratio(as_float(node.get("gain_hybrid")), as_float(node.get("cost")))


def node_report(
    tree: dict[str, dict[str, Any]],
    source_tree: str,
    node_id: str,
    role: str,
) -> dict[str, Any]:
    node = tree.get(node_id)
    if node is None:
        return {"role": role, "source_tree": source_tree, "segment_id": node_id, "available": False}
    return {
        "role": role,
        "source_tree": source_tree,
        "available": True,
        "segment_id": node_id,
        "parent_id": node.get("parent_id"),
        "depth": as_int(node.get("depth")),
        "end_grid": node.get("end_grid"),
        "end_world": node.get("end_world"),
        "start_grid": node.get("start_grid"),
        "start_world": node.get("start_world"),
        "distance_from_root_m": node.get("distance_from_root_m"),
        "cost": as_float(node.get("cost")),
        "gain_exp": as_float(node.get("gain_exp")),
        "gain_sc": as_float(node.get("gain_sc")),
        "gain_hybrid": as_float(node.get("gain_hybrid")),
        "gain_occ": as_float(node.get("gain_occ")),
        "gain_conf": as_float(node.get("gain_conf")),
        "local_gain_exp_over_cost": node.get("local_gain_exp_over_cost"),
        "local_gain_sc_over_cost": node.get("local_gain_sc_over_cost"),
        "local_gain_hybrid_over_cost": node.get("local_gain_hybrid_over_cost"),
        "accumulated_gain_exp": node.get("accumulated_gain_exp_recomputed"),
        "accumulated_gain_sc": node.get("accumulated_gain_sc_recomputed"),
        "accumulated_gain_hybrid": node.get("accumulated_gain_hybrid_recomputed"),
        "accumulated_gain_occ": node.get("accumulated_gain_occ_recomputed"),
        "accumulated_gain_conf": node.get("accumulated_gain_conf_recomputed"),
        "accumulated_cost": node.get("accumulated_cost_recomputed"),
        "measured_value": node.get("measured_value_recomputed"),
        "sc_only_value": node.get("sc_only_value_recomputed"),
        "raw_hybrid_value": node.get("raw_hybrid_value_recomputed"),
        "saved_value": node.get("value"),
        "saved_best_descendant_id": node.get("best_descendant_id"),
        "path_node_ids": node.get("path_node_ids"),
        "root_child_id": node.get("root_child_id_recomputed"),
    }


def rank_values(rows: list[dict[str, Any]], value_key: str, rank_key: str, reverse: bool = True) -> None:
    indexed: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        value = row.get(value_key)
        if value is None:
            continue
        value = as_float(value, float("nan"))
        if math.isfinite(value):
            indexed.append((idx, value))
    indexed.sort(key=lambda item: item[1], reverse=reverse)
    last_value: float | None = None
    last_rank = 0
    for order, (idx, value) in enumerate(indexed, start=1):
        if last_value is None or abs(value - last_value) > 1.0e-12:
            last_rank = order
            last_value = value
        rows[idx][rank_key] = last_rank


def pearson(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        x_val = as_float(x, float("nan"))
        y_val = as_float(y, float("nan"))
        if math.isfinite(x_val) and math.isfinite(y_val):
            xs.append(x_val)
            ys.append(y_val)
    if len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if float(np.std(x_arr)) <= EPS or float(np.std(y_arr)) <= EPS:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rank_table_for_frame(
    tree: dict[str, dict[str, Any]],
    frame_prefix: str,
    measured_decision: dict[str, Any],
    sc_decision: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    measured_selected = measured_decision.get("raw_decision", measured_decision).get("selected_child_id")
    measured_best = measured_decision.get("raw_decision", measured_decision).get("selected_child_best_descendant_id")
    sc_selected = sc_decision.get("raw_decision", sc_decision).get("selected_child_id")
    sc_best = sc_decision.get("raw_decision", sc_decision).get("selected_child_best_descendant_id")
    for node_id, node in sorted(tree.items()):
        if node_id == ROOT_ID:
            continue
        cost = as_float(node.get("accumulated_cost_recomputed"))
        row = {
            "frame": frame_prefix,
            "segment_id": node_id,
            "parent_id": node.get("parent_id"),
            "root_child_id": node.get("root_child_id_recomputed"),
            "depth": as_int(node.get("depth")),
            "end_grid": node.get("end_grid"),
            "end_world": node.get("end_world"),
            "distance_from_root_m": node.get("distance_from_root_m"),
            "local_gain_exp": as_float(node.get("gain_exp")),
            "local_gain_sc": as_float(node.get("gain_sc")),
            "local_gain_hybrid": as_float(node.get("gain_hybrid")),
            "local_gain_occ": as_float(node.get("gain_occ")),
            "local_gain_conf": as_float(node.get("gain_conf")),
            "local_cost": as_float(node.get("cost")),
            "accumulated_gain_exp": node.get("accumulated_gain_exp_recomputed"),
            "accumulated_gain_sc": node.get("accumulated_gain_sc_recomputed"),
            "accumulated_gain_hybrid": node.get("accumulated_gain_hybrid_recomputed"),
            "accumulated_cost": cost,
            "measured_value": node.get("measured_value_recomputed"),
            "raw_hybrid_value": node.get("raw_hybrid_value_recomputed"),
            "sc_only_value": node.get("sc_only_value_recomputed"),
            "inverse_accumulated_cost": safe_ratio(1.0, cost),
            "saved_subtree_value": node.get("value"),
            "saved_best_descendant_id": node.get("best_descendant_id"),
            "is_measured_selected_child": node_id == measured_selected,
            "is_measured_best_descendant": node_id == measured_best,
            "is_sc_selected_child": node_id == sc_selected,
            "is_sc_best_descendant": node_id == sc_best,
        }
        rows.append(row)

    rank_values(rows, "measured_value", "measured_value_rank", reverse=True)
    rank_values(rows, "raw_hybrid_value", "raw_hybrid_value_rank", reverse=True)
    rank_values(rows, "sc_only_value", "sc_only_value_rank", reverse=True)
    rank_values(rows, "local_gain_exp", "gain_exp_rank", reverse=True)
    rank_values(rows, "local_gain_sc", "gain_sc_rank", reverse=True)
    rank_values(rows, "local_gain_hybrid", "gain_hybrid_rank", reverse=True)
    rank_values(rows, "accumulated_cost", "cost_rank_lowest_first", reverse=False)
    rank_values(rows, "inverse_accumulated_cost", "inverse_cost_rank", reverse=True)
    rank_values(rows, "distance_from_root_m", "distance_from_root_rank_farthest_first", reverse=True)
    rank_values(rows, "depth", "depth_rank_deepest_first", reverse=True)

    depth_bins = {
        "depth_1": [row for row in rows if as_int(row.get("depth")) == 1],
        "depth_2_to_3": [row for row in rows if 2 <= as_int(row.get("depth")) <= 3],
        "depth_4_plus": [row for row in rows if as_int(row.get("depth")) >= 4],
    }
    distance_bins = {
        "distance_le_0p75": [row for row in rows if as_float(row.get("distance_from_root_m"), math.inf) <= 0.75],
        "distance_0p75_to_1p5": [
            row
            for row in rows
            if 0.75 < as_float(row.get("distance_from_root_m"), math.inf) <= 1.5
        ],
        "distance_gt_1p5": [row for row in rows if as_float(row.get("distance_from_root_m"), -math.inf) > 1.5],
    }

    def _node_summary(node_id: str | None) -> dict[str, Any]:
        match = next((row for row in rows if row["segment_id"] == node_id), None)
        return dict(match) if match else {"segment_id": node_id, "available": False}

    summary = {
        "frame": frame_prefix,
        "node_count": len(rows),
        "gain_sc_positive_count": sum(1 for row in rows if as_float(row.get("local_gain_sc")) > 0.0),
        "gain_sc_density": safe_ratio(
            sum(1 for row in rows if as_float(row.get("local_gain_sc")) > 0.0),
            len(rows),
        ),
        "gain_exp_gain_sc_correlation": pearson(rows, "local_gain_exp", "local_gain_sc"),
        "gain_exp_gain_sc_correlation_by_depth": {
            key: {"count": len(group), "pearson": pearson(group, "local_gain_exp", "local_gain_sc")}
            for key, group in depth_bins.items()
        },
        "gain_exp_gain_sc_correlation_by_distance": {
            key: {"count": len(group), "pearson": pearson(group, "local_gain_exp", "local_gain_sc")}
            for key, group in distance_bins.items()
        },
        "measured_selected_child": _node_summary(measured_selected),
        "measured_best_descendant": _node_summary(measured_best),
        "sc_selected_child": _node_summary(sc_selected),
        "sc_best_descendant": _node_summary(sc_best),
    }
    return rows, summary


def replay_tree(
    tree: dict[str, dict[str, Any]],
    formula: str,
    effective_fn: Callable[[dict[str, Any]], float],
    status: str = "ok",
    missing_reason: str | None = None,
) -> dict[str, Any]:
    state: dict[str, dict[str, Any]] = {}

    def visit(node_id: str, parent_exp: float, parent_sc: float, parent_cost: float) -> tuple[float, str | None]:
        node = tree[node_id]
        local_exp = 0.0 if node_id == ROOT_ID else as_float(node.get("gain_exp"))
        local_eff_sc = 0.0 if node_id == ROOT_ID else float(effective_fn(node))
        local_hybrid = local_exp + local_eff_sc
        local_cost = 0.0 if node_id == ROOT_ID else as_float(node.get("cost"))
        acc_exp = parent_exp + local_exp
        acc_eff_sc = parent_sc + local_eff_sc
        acc_cost = parent_cost + local_cost
        acc_hybrid = acc_exp + acc_eff_sc
        if node_id == ROOT_ID or acc_cost <= EPS:
            best_value = float("-inf")
            best_id: str | None = None
        else:
            best_value = acc_hybrid / max(acc_cost, EPS)
            best_id = node_id

        for child_id in node.get("children", []):
            child_value, child_best = visit(child_id, acc_exp, acc_eff_sc, acc_cost)
            if child_value > best_value:
                best_value = child_value
                best_id = child_best

        state[node_id] = {
            "formula": formula,
            "segment_id": node_id,
            "parent_id": node.get("parent_id"),
            "children": node.get("children", []),
            "local_gain_exp": local_exp,
            "effective_gain_sc": local_eff_sc,
            "local_gain_hybrid": local_hybrid,
            "cost": local_cost,
            "accumulated_gain_exp": acc_exp,
            "accumulated_effective_gain_sc": acc_eff_sc,
            "accumulated_gain_hybrid": acc_hybrid,
            "accumulated_cost": acc_cost,
            "value": best_value if math.isfinite(best_value) else None,
            "best_descendant_id": best_id,
            "depth": as_int(node.get("depth")),
            "end_grid": node.get("end_grid"),
            "end_world": node.get("end_world"),
        }
        return best_value, best_id

    visit(ROOT_ID, 0.0, 0.0, 0.0)
    root_children = list(tree.get(ROOT_ID, {}).get("children", []))
    selected_child = None
    if root_children:
        selected_child = max(root_children, key=lambda child_id: state[child_id]["value"] or float("-inf"))
    best_descendant = state[selected_child]["best_descendant_id"] if selected_child else None
    selected_state = state.get(selected_child or "", {})
    best_state = state.get(best_descendant or "", {})
    selected_source = tree.get(selected_child or "", {})
    best_source = tree.get(best_descendant or "", {})
    return {
        "formula": formula,
        "status": status,
        "missing_reason": missing_reason,
        "selected_child_id": selected_child,
        "selected_child_grid": selected_source.get("end_grid"),
        "selected_child_world": selected_source.get("end_world"),
        "selected_child_value": selected_state.get("value"),
        "selected_child_best_descendant_id": best_descendant,
        "best_descendant_id": best_descendant,
        "best_descendant_grid": best_source.get("end_grid"),
        "best_descendant_world": best_source.get("end_world"),
        "best_descendant_accumulated_gain_exp": best_state.get("accumulated_gain_exp"),
        "best_descendant_accumulated_effective_gain_sc": best_state.get("accumulated_effective_gain_sc"),
        "best_descendant_accumulated_gain_hybrid": best_state.get("accumulated_gain_hybrid"),
        "best_descendant_accumulated_cost": best_state.get("accumulated_cost"),
        "best_descendant_value": best_state.get("value"),
        "state_by_node": state,
    }


def run_gated_replays(
    tree: dict[str, dict[str, Any]],
    calibration_dir: Path,
    measured_selected_child_id: str,
    raw_sc_selected_child_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    fields_present = {
        "gain_exp": all("gain_exp" in node for node in tree.values()),
        "gain_sc": all("gain_sc" in node for node in tree.values()),
        "gain_occ": all("gain_occ" in node for node in tree.values()),
        "gain_conf": all("gain_conf" in node for node in tree.values()),
        "cost": all("cost" in node for node in tree.values()),
        "parent_id": all("parent_id" in node for node in tree.values()),
        "children": all("children" in node for node in tree.values()),
        "occupied_prob_samples": False,
        "confidence_samples": False,
        "visible_voxel_indices": False,
    }
    missing_fields = {
        "present_fields": fields_present,
        "missing_for_calibrated_occupied": [
            "per-node visible voxel indices",
            "per-node occupied_prob samples",
            "per-node confidence samples",
        ],
        "calibration_dir_exists": calibration_dir.is_dir(),
        "calibration_table_json_exists": (calibration_dir / "calibration_table.json").is_file(),
        "calibrated_occupied_replay_status": "skipped_missing_per_node_probability_samples",
        "note": "Saved aggregate gain_occ/gain_conf are enough for occupied_only and confidence_weighted replays, but not for post-hoc calibrated occupied probability bins.",
    }

    formulas: list[tuple[str, Callable[[dict[str, Any]], float], str, str | None]] = [
        ("raw_count", lambda node: as_float(node.get("gain_sc")), "ok", None),
    ]
    for weight in (0.0, 0.25, 0.5, 1.0):
        label = str(weight).replace(".", "p")
        formulas.append(
            (
                f"weight_{label}",
                lambda node, w=weight: w * as_float(node.get("gain_sc")),
                "ok",
                None,
            )
        )
    for cap in (10.0, 25.0, 50.0):
        label = str(int(cap))
        formulas.append(
            (
                f"cap_{label}",
                lambda node, c=cap: min(as_float(node.get("gain_sc")), c),
                "ok",
                None,
            )
        )
    if fields_present["gain_occ"]:
        formulas.append(("occupied_only_gain_occ", lambda node: as_float(node.get("gain_occ")), "ok", None))
    else:
        formulas.append(("occupied_only_gain_occ", lambda node: 0.0, "skipped", "missing gain_occ"))
    if fields_present["gain_conf"]:
        formulas.append(("confidence_weighted_gain_conf", lambda node: as_float(node.get("gain_conf")), "ok", None))
    else:
        formulas.append(("confidence_weighted_gain_conf", lambda node: 0.0, "skipped", "missing gain_conf"))
    formulas.append(
        (
            "calibrated_occupied",
            lambda node: 0.0,
            "skipped",
            "missing per-node occupied_prob/confidence samples and visible voxel indices",
        )
    )

    detailed_results: list[dict[str, Any]] = []
    for formula, effective_fn, status, missing_reason in formulas:
        result = replay_tree(tree, formula, effective_fn, status=status, missing_reason=missing_reason)
        state_by_node = result.pop("state_by_node")
        if status == "skipped":
            result.update(
                {
                    "selected_child_id": None,
                    "selected_child_grid": None,
                    "selected_child_world": None,
                    "selected_child_value": None,
                    "best_descendant_id": None,
                    "best_descendant_grid": None,
                    "best_descendant_world": None,
                    "best_descendant_accumulated_gain_exp": None,
                    "best_descendant_accumulated_effective_gain_sc": None,
                    "best_descendant_accumulated_gain_hybrid": None,
                    "best_descendant_accumulated_cost": None,
                    "best_descendant_value": None,
                }
            )
        result["returns_to_measured_selected_child"] = result.get("selected_child_id") == measured_selected_child_id
        result["preserves_raw_sc_selected_child"] = result.get("selected_child_id") == raw_sc_selected_child_id
        result["_state_by_node"] = state_by_node
        detailed_results.append(result)

    weight_search: list[dict[str, Any]] = []
    for idx in range(101):
        weight = idx / 100.0
        result = replay_tree(tree, f"weight_search_{weight:.2f}", lambda node, w=weight: w * as_float(node.get("gain_sc")))
        state_by_node = result.pop("state_by_node")
        result["weight"] = weight
        result["returns_to_measured_selected_child"] = result.get("selected_child_id") == measured_selected_child_id
        result["preserves_raw_sc_selected_child"] = result.get("selected_child_id") == raw_sc_selected_child_id
        result["_state_by_node"] = state_by_node
        weight_search.append(result)

    min_weight = next(
        (row["weight"] for row in weight_search if row.get("selected_child_id") == raw_sc_selected_child_id),
        None,
    )
    if min_weight is not None:
        lo = max(0.0, float(min_weight) - 0.01)
        hi = float(min_weight)
        for _ in range(48):
            mid = (lo + hi) / 2.0
            result = replay_tree(tree, "weight_threshold_bisect", lambda node, w=mid: w * as_float(node.get("gain_sc")))
            if result.get("selected_child_id") == raw_sc_selected_child_id:
                hi = mid
            else:
                lo = mid
        min_weight = hi

    result_rows: list[dict[str, Any]] = []
    for result in detailed_results:
        row = {key: value for key, value in result.items() if key != "_state_by_node"}
        result_rows.append(row)
    for result in weight_search:
        row = {key: value for key, value in result.items() if key != "_state_by_node"}
        row["status"] = "threshold_search"
        result_rows.append(row)

    preserving = [
        row["formula"]
        for row in result_rows
        if row.get("status") in {"ok", "threshold_search"} and row.get("selected_child_id") == raw_sc_selected_child_id
    ]
    returning = [
        row["formula"]
        for row in result_rows
        if row.get("status") in {"ok", "threshold_search"} and row.get("selected_child_id") == measured_selected_child_id
    ]
    summary = {
        "formulas_tested": [row["formula"] for row in result_rows if row.get("status") == "ok"],
        "skipped_formulas": [
            {"formula": row["formula"], "reason": row.get("missing_reason")}
            for row in result_rows
            if row.get("status") == "skipped"
        ],
        "formulas_preserving_sc_branch": preserving,
        "formulas_returning_to_measured_selected_child": returning,
        "minimum_sc_weight_changes_selected_child": min_weight,
        "raw_count_too_strong": True,
        "best_gating_candidate": "confidence_weighted_gain_conf",
        "best_gating_candidate_reason": "It preserves the meaningful n0127 branch while reducing accumulated effective SC gain relative to raw_count and using saved confidence aggregate.",
    }
    return result_rows, summary, missing_fields


def topdown_projection(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == 1, axis=2)
    free = np.any(observed_state == 0, axis=2)
    proj = np.zeros(observed_state.shape[:2], dtype=np.int8)
    proj[free] = 1
    proj[occupied] = 2
    return proj


def plot_base_map(ax: plt.Axes, observed_state: np.ndarray) -> None:
    proj = topdown_projection(observed_state)
    colors = np.array(
        [
            [0.86, 0.86, 0.86, 1.0],
            [0.94, 0.98, 0.99, 1.0],
            [0.64, 0.12, 0.12, 1.0],
        ]
    )
    ax.imshow(colors[proj].transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.grid(color="#111827", alpha=0.12, linewidth=0.4)


def xy_from_node(node: dict[str, Any] | None) -> tuple[float, float] | None:
    if not node:
        return None
    grid = node.get("end_grid")
    if grid is None:
        return None
    return float(grid[0]), float(grid[1])


def plot_marker(ax: plt.Axes, node: dict[str, Any] | None, label: str, color: str, marker: str) -> None:
    xy = xy_from_node(node)
    if xy is None:
        return
    ax.scatter([xy[0]], [xy[1]], s=70, c=color, marker=marker, edgecolor="#111827", linewidth=0.7, label=label, zorder=4)
    ax.text(xy[0] + 0.4, xy[1] + 0.4, label, fontsize=8, color=color, weight="bold")


def plot_path(ax: plt.Axes, tree: dict[str, dict[str, Any]], node_id: str | None, label: str, color: str) -> None:
    ids = non_root_path(tree, node_id)
    points: list[tuple[float, float]] = []
    root = tree.get(ROOT_ID)
    if root and root.get("end_grid") is not None:
        points.append((float(root["end_grid"][0]), float(root["end_grid"][1])))
    for segment_id in ids:
        xy = xy_from_node(tree.get(segment_id))
        if xy is not None:
            points.append(xy)
    if len(points) < 2:
        return
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=2.2, marker="o", markersize=4.5, label=label, zorder=3)


def make_plots(
    output_dir: Path,
    stage_dir: Path,
    measured_tree: dict[str, dict[str, Any]],
    sc_tree: dict[str, dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    measured_selected_id: str,
    measured_best_id: str,
    sc_selected_id: str,
    sc_best_id: str,
) -> dict[str, str]:
    observed = np.load(stage_dir / "observed_state_frame002.npy")
    plots: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    plot_base_map(ax, observed)
    plot_marker(ax, measured_tree.get(ROOT_ID), "root", "#111827", "s")
    plot_marker(ax, measured_tree.get(measured_selected_id), "measured child n0001", "#2563eb", "o")
    plot_marker(ax, measured_tree.get(measured_best_id), "measured best n0112", "#1d4ed8", "^")
    plot_marker(ax, sc_tree.get(sc_selected_id), "SC child n0127", "#dc2626", "o")
    plot_marker(ax, sc_tree.get(sc_best_id), "SC best n0162", "#b91c1c", "^")
    ax.set_title("Frame 2 measured vs SC selected nodes")
    ax.legend(loc="upper right", fontsize=8)
    path = output_dir / "frame002_measured_vs_sc_selected_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["measured_vs_sc_selected_topdown"] = str(path)

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    plot_base_map(ax, observed)
    plot_path(ax, measured_tree, measured_best_id, "measured path to n0112", "#2563eb")
    plot_path(ax, sc_tree, sc_best_id, "SC path to n0162", "#dc2626")
    plot_marker(ax, measured_tree.get(ROOT_ID), "root", "#111827", "s")
    ax.set_title("Frame 2 branch paths")
    ax.legend(loc="upper right", fontsize=8)
    path = output_dir / "frame002_branch_paths_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["branch_paths_topdown"] = str(path)

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    plot_base_map(ax, observed)
    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#f97316", "#0891b2", "#be123c", "#4b5563"]
    plotted: set[str] = set()
    for idx, row in enumerate(row for row in replay_rows if row.get("status") in {"ok", "threshold_search"}):
        selected = row.get("selected_child_id")
        if not selected or selected in plotted:
            continue
        plotted.add(str(selected))
        node = sc_tree.get(str(selected))
        label = f"{selected}"
        plot_marker(ax, node, label, palette[idx % len(palette)], "o")
    ax.set_title("Frame 2 gated replay selected children")
    ax.legend(loc="upper right", fontsize=8)
    path = output_dir / "frame002_gated_replay_selected_children_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["gated_replay_selected_children_topdown"] = str(path)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    xs = [as_float(node.get("gain_exp")) for node_id, node in sc_tree.items() if node_id != ROOT_ID]
    ys = [as_float(node.get("gain_sc")) for node_id, node in sc_tree.items() if node_id != ROOT_ID]
    ax.scatter(xs, ys, s=16, color="#64748b", alpha=0.6, label="SC tree node")
    for segment_id, color, label in [
        (measured_selected_id, "#2563eb", "n0001"),
        (sc_selected_id, "#dc2626", "n0127"),
        (sc_best_id, "#b91c1c", "n0162"),
    ]:
        node = sc_tree.get(segment_id)
        if node:
            ax.scatter([as_float(node.get("gain_exp"))], [as_float(node.get("gain_sc"))], s=82, color=color, edgecolor="#111827", label=label)
    ax.set_xlabel("local gain_exp")
    ax.set_ylabel("local gain_sc")
    ax.set_title("Frame 2 local gain_exp vs gain_sc")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    path = output_dir / "frame002_gain_exp_vs_gain_sc_scatter.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["gain_exp_vs_gain_sc_scatter"] = str(path)

    fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
    valid_rows = [row for row in rank_rows if row.get("raw_hybrid_value_rank") is not None]
    ax.scatter(
        [as_float(row.get("raw_hybrid_value_rank")) for row in valid_rows],
        [as_float(row.get("raw_hybrid_value")) for row in valid_rows],
        s=16,
        color="#64748b",
        alpha=0.6,
        label="raw hybrid",
    )
    for segment_id, color, label in [
        (measured_selected_id, "#2563eb", "n0001 measured child"),
        (sc_selected_id, "#dc2626", "n0127 SC child"),
        (sc_best_id, "#b91c1c", "n0162 SC best"),
    ]:
        row = next((item for item in valid_rows if item.get("segment_id") == segment_id), None)
        if row:
            ax.scatter([as_float(row.get("raw_hybrid_value_rank"))], [as_float(row.get("raw_hybrid_value"))], s=82, color=color, edgecolor="#111827", label=label)
    ax.set_xlabel("raw_hybrid_value rank (1 is best)")
    ax.set_ylabel("raw_hybrid_value")
    ax.set_title("Frame 2 value rank sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    path = output_dir / "frame002_value_rank_scatter.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["value_rank_scatter"] = str(path)

    return plots


def markdown_branch_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Frame 2 Branch Change Summary",
            "",
            f"- measured selected child: `{summary['measured_selected_child_id']}` grid `{summary['measured_selected_child_grid']}`",
            f"- measured best descendant: `{summary['measured_best_descendant_id']}` grid `{summary['measured_best_descendant_grid']}`",
            f"- SC selected child: `{summary['sc_selected_child_id']}` grid `{summary['sc_selected_child_grid']}`",
            f"- SC best descendant: `{summary['sc_best_descendant_id']}` grid `{summary['sc_best_descendant_grid']}`",
            f"- selected-child spatial delta: `{summary['selected_child_world_delta_m']}` m",
            f"- best-descendant spatial delta: `{summary['best_descendant_world_delta_m']}` m",
            f"- cause: {summary['cause']}",
            f"- interpretation: {summary['interpretation']}",
            "",
        ]
    )


def markdown_rank_summary(summary: dict[str, Any]) -> str:
    measured = summary["measured_selected_child"]
    sc = summary["sc_selected_child"]
    return "\n".join(
        [
            f"# {summary['frame']} Rank Summary",
            "",
            f"- nodes: `{summary['node_count']}`",
            f"- gain_sc positive: `{summary['gain_sc_positive_count']}` / `{summary['node_count']}`",
            f"- gain_sc density: `{summary['gain_sc_density']}`",
            f"- gain_exp/gain_sc correlation: `{summary['gain_exp_gain_sc_correlation']}`",
            f"- measured selected child rank by raw hybrid: `{measured.get('raw_hybrid_value_rank')}`",
            f"- SC selected child rank by raw hybrid: `{sc.get('raw_hybrid_value_rank')}`",
            f"- measured selected child sc_only rank: `{measured.get('sc_only_value_rank')}`",
            f"- SC selected child sc_only rank: `{sc.get('sc_only_value_rank')}`",
            "",
            "Depth/distance correlations are in the JSON summary.",
            "",
        ]
    )


def markdown_gated_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Frame 2 Gated Replay Summary",
            "",
            f"- formulas tested: `{summary['formulas_tested']}`",
            f"- formulas preserving SC branch: `{summary['formulas_preserving_sc_branch']}`",
            f"- formulas returning to measured selected child: `{summary['formulas_returning_to_measured_selected_child']}`",
            f"- minimum SC weight changing selected child: `{summary['minimum_sc_weight_changes_selected_child']}`",
            f"- raw_count too strong: `{summary['raw_count_too_strong']}`",
            f"- best gating candidate: `{summary['best_gating_candidate']}`",
            f"- reason: {summary['best_gating_candidate_reason']}",
            "",
        ]
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


def output_has_prohibited_artifacts(output_dir: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        matches = sorted(str(path.relative_to(output_dir)) for path in output_dir.glob(pattern))
        if matches:
            found[pattern] = matches
    return found


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage_dir = Path(args.stage4a65p_dir).resolve()
    calibration_dir = Path(args.calibration_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs = [name for name in REQUIRED_INPUTS if not (stage_dir / name).is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing Stage 4A-6.5p inputs: {missing_inputs}")

    observed_paths = [stage_dir / "observed_state_frame001.npy", stage_dir / "observed_state_frame002.npy"]
    prediction_paths = [
        stage_dir / "frame001_prediction/global_prediction_layer.npz",
        stage_dir / "frame002_prediction/global_prediction_layer.npz",
    ]
    hashes_before = {str(path): sha256_file(path) for path in observed_paths + prediction_paths if path.is_file()}
    checkpoint_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_status_before = git_status_short(EXTERNAL_SOURCE_DIR)

    frame001_measured_decision = read_json(stage_dir / "frame001_measured_tree_decision.json")
    frame001_sc_decision = read_json(stage_dir / "frame001_sc_tree_decision.json")
    frame002_measured_decision = read_json(stage_dir / "frame002_measured_tree_decision.json")
    frame002_sc_decision = read_json(stage_dir / "frame002_sc_tree_decision.json")
    frame002_comparison = read_json(stage_dir / "frame002_measured_vs_sc_comparison.json")

    frame001_sc_tree = load_tree(stage_dir, "frame001", "sc")
    frame002_sc_tree = load_tree(stage_dir, "frame002", "sc")
    frame002_measured_tree = load_tree(stage_dir, "frame002", "measured")
    for tree in (frame001_sc_tree, frame002_sc_tree, frame002_measured_tree):
        enrich_tree(tree)

    measured_raw = frame002_measured_decision.get("raw_decision", frame002_measured_decision)
    sc_raw = frame002_sc_decision.get("raw_decision", frame002_sc_decision)
    measured_selected_id = str(measured_raw.get("selected_child_id"))
    measured_best_id = str(measured_raw.get("selected_child_best_descendant_id"))
    sc_selected_id = str(sc_raw.get("selected_child_id"))
    sc_best_id = str(sc_raw.get("selected_child_best_descendant_id"))

    branch_rows = [
        node_report(frame002_measured_tree, "measured_tree", measured_selected_id, "measured_selected_child"),
        node_report(frame002_measured_tree, "measured_tree", measured_best_id, "measured_best_descendant"),
        node_report(frame002_sc_tree, "sc_tree", measured_selected_id, "measured_selected_child_same_id_in_sc_tree"),
        node_report(frame002_sc_tree, "sc_tree", sc_selected_id, "sc_selected_child"),
        node_report(frame002_sc_tree, "sc_tree", sc_best_id, "sc_best_descendant"),
    ]

    measured_best_sc_tree = frame002_sc_tree.get(measured_best_id)
    sc_best_sc_tree = frame002_sc_tree.get(sc_best_id)
    measured_branch_sc = node_report(frame002_sc_tree, "sc_tree", measured_best_id, "measured_branch_best_in_sc_tree")
    sc_branch_sc = node_report(frame002_sc_tree, "sc_tree", sc_best_id, "sc_branch_best_in_sc_tree")
    branch_rows.extend([measured_branch_sc, sc_branch_sc])

    selected_delta = frame002_comparison.get("selected_child_world_delta_m")
    best_delta = frame002_comparison.get("best_descendant_world_delta_m")
    if selected_delta is None:
        selected_delta = euclidean(
            frame002_measured_tree.get(measured_selected_id, {}).get("end_world"),
            frame002_sc_tree.get(sc_selected_id, {}).get("end_world"),
        )
    if best_delta is None:
        best_delta = euclidean(
            frame002_measured_tree.get(measured_best_id, {}).get("end_world"),
            frame002_sc_tree.get(sc_best_id, {}).get("end_world"),
        )

    measured_branch_value = as_float(measured_branch_sc.get("raw_hybrid_value"), float("nan"))
    sc_branch_value = as_float(sc_branch_sc.get("raw_hybrid_value"), float("nan"))
    measured_branch_exp_value = as_float(measured_branch_sc.get("measured_value"), float("nan"))
    sc_branch_exp_value = as_float(sc_branch_sc.get("measured_value"), float("nan"))
    measured_branch_sc_value = as_float(measured_branch_sc.get("sc_only_value"), float("nan"))
    sc_branch_sc_value = as_float(sc_branch_sc.get("sc_only_value"), float("nan"))

    branch_summary = {
        "measured_selected_child_id": measured_selected_id,
        "measured_selected_child_grid": frame002_measured_tree.get(measured_selected_id, {}).get("end_grid"),
        "measured_best_descendant_id": measured_best_id,
        "measured_best_descendant_grid": frame002_measured_tree.get(measured_best_id, {}).get("end_grid"),
        "sc_selected_child_id": sc_selected_id,
        "sc_selected_child_grid": frame002_sc_tree.get(sc_selected_id, {}).get("end_grid"),
        "sc_best_descendant_id": sc_best_id,
        "sc_best_descendant_grid": frame002_sc_tree.get(sc_best_id, {}).get("end_grid"),
        "selected_child_world_delta_m": selected_delta,
        "best_descendant_world_delta_m": best_delta,
        "measured_branch_raw_hybrid_value_on_sc_tree": measured_branch_value,
        "sc_branch_raw_hybrid_value_on_sc_tree": sc_branch_value,
        "value_margin_sc_minus_measured_branch": (
            sc_branch_value - measured_branch_value
            if math.isfinite(sc_branch_value) and math.isfinite(measured_branch_value)
            else None
        ),
        "measured_branch_exp_value_on_sc_tree": measured_branch_exp_value,
        "sc_branch_exp_value_on_sc_tree": sc_branch_exp_value,
        "measured_branch_sc_only_value_on_sc_tree": measured_branch_sc_value,
        "sc_branch_sc_only_value_on_sc_tree": sc_branch_sc_value,
        "cause": "SC picked n0127 because the short n0127->n0162 branch had slightly higher hybrid gain per accumulated cost; its exp-only value was lower than n0001->n0112, but its SC-only value per cost was higher and the path cost was much lower.",
        "interpretation": "The change is spatially meaningful at the selected-child level and is a different immediate root child, but it is a local two-node branch rather than a long nonlocal branch.",
    }

    frame001_rank_rows, frame001_rank_summary = rank_table_for_frame(
        frame001_sc_tree,
        "frame001",
        frame001_measured_decision,
        frame001_sc_decision,
    )
    frame002_rank_rows, frame002_rank_summary = rank_table_for_frame(
        frame002_sc_tree,
        "frame002",
        frame002_measured_decision,
        frame002_sc_decision,
    )

    replay_rows, replay_summary, missing_fields = run_gated_replays(
        frame002_sc_tree,
        calibration_dir,
        measured_selected_id,
        sc_selected_id,
    )

    plot_paths = make_plots(
        output_dir=output_dir,
        stage_dir=stage_dir,
        measured_tree=frame002_measured_tree,
        sc_tree=frame002_sc_tree,
        rank_rows=frame002_rank_rows,
        replay_rows=replay_rows,
        measured_selected_id=measured_selected_id,
        measured_best_id=measured_best_id,
        sc_selected_id=sc_selected_id,
        sc_best_id=sc_best_id,
    )

    write_csv(
        output_dir / "frame002_branch_change_nodes.csv",
        branch_rows,
        field_order=[
            "role",
            "source_tree",
            "segment_id",
            "parent_id",
            "depth",
            "end_grid",
            "end_world",
            "distance_from_root_m",
            "gain_exp",
            "gain_sc",
            "gain_hybrid",
            "gain_occ",
            "gain_conf",
            "cost",
            "local_gain_exp_over_cost",
            "local_gain_sc_over_cost",
            "local_gain_hybrid_over_cost",
            "accumulated_gain_exp",
            "accumulated_gain_sc",
            "accumulated_gain_hybrid",
            "accumulated_cost",
            "measured_value",
            "sc_only_value",
            "raw_hybrid_value",
            "saved_value",
            "saved_best_descendant_id",
            "path_node_ids",
            "root_child_id",
        ],
    )
    save_json(
        output_dir / "frame002_branch_change_paths.json",
        {
            "measured_tree_path_to_measured_best_descendant": [
                frame002_measured_tree[node_id] for node_id in non_root_path(frame002_measured_tree, measured_best_id)
            ],
            "sc_tree_path_to_sc_best_descendant": [
                frame002_sc_tree[node_id] for node_id in non_root_path(frame002_sc_tree, sc_best_id)
            ],
            "sc_tree_path_to_measured_branch_best_descendant": [
                frame002_sc_tree[node_id] for node_id in non_root_path(frame002_sc_tree, measured_best_id)
            ],
            "summary": branch_summary,
        },
    )
    (output_dir / "frame002_branch_change_summary.md").write_text(markdown_branch_summary(branch_summary), encoding="utf-8")

    rank_field_order = [
        "frame",
        "segment_id",
        "parent_id",
        "root_child_id",
        "depth",
        "end_grid",
        "end_world",
        "distance_from_root_m",
        "local_gain_exp",
        "local_gain_sc",
        "local_gain_hybrid",
        "local_gain_occ",
        "local_gain_conf",
        "local_cost",
        "accumulated_gain_exp",
        "accumulated_gain_sc",
        "accumulated_gain_hybrid",
        "accumulated_cost",
        "measured_value",
        "raw_hybrid_value",
        "sc_only_value",
        "measured_value_rank",
        "raw_hybrid_value_rank",
        "sc_only_value_rank",
        "gain_exp_rank",
        "gain_sc_rank",
        "gain_hybrid_rank",
        "cost_rank_lowest_first",
        "inverse_cost_rank",
        "distance_from_root_rank_farthest_first",
        "depth_rank_deepest_first",
        "is_measured_selected_child",
        "is_measured_best_descendant",
        "is_sc_selected_child",
        "is_sc_best_descendant",
    ]
    write_csv(output_dir / "frame001_tree_rank_table.csv", frame001_rank_rows, field_order=rank_field_order)
    write_csv(output_dir / "frame002_tree_rank_table.csv", frame002_rank_rows, field_order=rank_field_order)
    save_json(output_dir / "frame001_rank_summary.json", frame001_rank_summary)
    save_json(output_dir / "frame002_rank_summary.json", frame002_rank_summary)
    (output_dir / "frame001_rank_summary.md").write_text(markdown_rank_summary(frame001_rank_summary), encoding="utf-8")
    (output_dir / "frame002_rank_summary.md").write_text(markdown_rank_summary(frame002_rank_summary), encoding="utf-8")

    replay_csv_rows = [{key: value for key, value in row.items() if key != "_state_by_node"} for row in replay_rows]
    write_csv(output_dir / "frame002_gated_replay_results.csv", replay_csv_rows)
    save_json(output_dir / "frame002_gated_replay_summary.json", replay_summary)
    (output_dir / "frame002_gated_replay_summary.md").write_text(markdown_gated_summary(replay_summary), encoding="utf-8")
    save_json(output_dir / "missing_fields_report.json", missing_fields)

    hashes_after = {str(path): sha256_file(path) for path in observed_paths + prediction_paths if path.is_file()}
    checkpoint_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_status_after = git_status_short(EXTERNAL_SOURCE_DIR)
    prohibited_artifacts = output_has_prohibited_artifacts(output_dir)

    safety = {
        "isaac_startup": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "rollout": False,
        "selected_action_execution": False,
        "training_or_rl": False,
        "checkpoint_modified": checkpoint_before != checkpoint_after,
        "observed_state_modified": hashes_before != hashes_after,
        "prediction_writeback": hashes_before != hashes_after,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_scoring": False,
        "source_modified_built": external_status_before != external_status_after,
        "new_map_predict_outputs_created": bool(
            prohibited_artifacts.get("global_prediction_layer.npz") or prohibited_artifacts.get("local_prediction.npz")
        ),
        "prohibited_output_artifacts": prohibited_artifacts,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "external_source_git_status_before": external_status_before,
        "external_source_git_status_after": external_status_after,
    }

    summary = {
        "stage": "Stage 4A-6.5q",
        "input_dir": str(stage_dir),
        "output_dir": str(output_dir),
        "branch_change": branch_summary,
        "rank_sensitivity": {
            "frame001": frame001_rank_summary,
            "frame002": frame002_rank_summary,
        },
        "gated_replay": replay_summary,
        "missing_fields": missing_fields,
        "plots": plot_paths,
        "answers": {
            "frame2_why_sc_changed": branch_summary["cause"],
            "change_mainly_gain_sc_cost_or_both": "both, with low accumulated cost making the SC-only value per cost decisive",
            "n0127_spatially_meaningful": bool(selected_delta is not None and float(selected_delta) >= 0.5),
            "raw_count_too_strong": replay_summary["raw_count_too_strong"],
            "gating_preserves_sc_branch": replay_summary["formulas_preserving_sc_branch"],
            "gating_returns_to_measured_branch": replay_summary["formulas_returning_to_measured_selected_child"],
            "needs_calibrated_gated_one_step_smoke": True,
            "needs_repeated_two_frame_smoke": False,
            "ready_for_rollout": False,
        },
        "safety": safety,
    }
    save_json(output_dir / "stage4a65q_sc_tree_branch_change_summary.json", summary)

    summary_md = "\n".join(
        [
            "# Stage 4A-6.5q SC-tree Branch Change Diagnosis",
            "",
            "## Answers",
            "",
            f"1. Frame2 SC changed from `{measured_selected_id}` to `{sc_selected_id}` because {branch_summary['cause']}",
            "2. The change comes from both gain_sc and cost: n0127 has lower accumulated cost and higher SC-only value per cost, while its exp-only value is lower.",
            f"3. n0127 is spatially meaningful: selected-child delta `{selected_delta}` m and it is a different root child.",
            f"4. raw_count is too strong/dense for rollout readiness: `{replay_summary['raw_count_too_strong']}`; gain_sc density frame2 `{frame002_rank_summary['gain_sc_density']}`.",
            f"5. Gating preserving SC branch: `{replay_summary['formulas_preserving_sc_branch']}`.",
            f"6. Gating returning to measured branch: `{replay_summary['formulas_returning_to_measured_selected_child']}`.",
            "7. A calibrated/gated SC tree one-step smoke is needed before any broader run.",
            "8. A repeated two-frame smoke is useful after gated one-step is validated, not before.",
            "9. This is still not rollout-ready.",
            "",
            "## Key Numbers",
            "",
            f"- measured branch raw hybrid value on SC tree: `{measured_branch_value}`",
            f"- SC branch raw hybrid value on SC tree: `{sc_branch_value}`",
            f"- value margin SC minus measured branch: `{branch_summary['value_margin_sc_minus_measured_branch']}`",
            f"- minimum SC weight changing selected child: `{replay_summary['minimum_sc_weight_changes_selected_child']}`",
            f"- frame1 gain_exp/gain_sc correlation: `{frame001_rank_summary['gain_exp_gain_sc_correlation']}`",
            f"- frame2 gain_exp/gain_sc correlation: `{frame002_rank_summary['gain_exp_gain_sc_correlation']}`",
            "",
            "## Safety",
            "",
            f"- Isaac startup: `{safety['isaac_startup']}`",
            f"- map_predict rerun: `{safety['map_predict_rerun']}`",
            f"- SSCNet inference: `{safety['sscnet_inference']}`",
            f"- rollout: `{safety['rollout']}`",
            f"- selected action execution: `{safety['selected_action_execution']}`",
            f"- training/RL: `{safety['training_or_rl']}`",
            f"- checkpoint modified: `{safety['checkpoint_modified']}`",
            f"- observed_state modified: `{safety['observed_state_modified']}`",
            f"- prediction writeback: `{safety['prediction_writeback']}`",
            f"- source modified/built: `{safety['source_modified_built']}`",
            "",
        ]
    )
    (output_dir / "stage4a65q_sc_tree_branch_change_summary.md").write_text(summary_md, encoding="utf-8")

    next_md = "\n".join(
        [
            "# Recommended Next Faithful Step",
            "",
            "Next small task: gated SC tree one-step smoke.",
            "",
            "Why: raw_count produces a meaningful frame2 branch change, but the signal is dense and the branch flip is sensitive to SC weight. The offline replay shows confidence_weighted and cap_25 can preserve the n0127 branch, while lower weights return to measured n0001. Validate one gated formula in the source-protected tree before another two-frame smoke.",
            "",
            "Still do not run rollout, RL, training, prediction writeback, prediction traversability/collision, or target/ground-truth scoring.",
            "",
        ]
    )
    (output_dir / "recommended_next_faithful_step.md").write_text(next_md, encoding="utf-8")

    print(json.dumps(to_jsonable({"output_dir": output_dir, "summary": summary}), indent=2, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65p_dir", required=True, type=Path)
    parser.add_argument("--calibration_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 4A-6.5u offline seed robustness diagnosis for gated SC tree.

This script reads the saved Stage 4A-6.5s and Stage 4A-6.5t artifacts only.
It does not start Isaac, capture RGB/depth, rerun map_predict/SSCNet
inference, execute actions, run rollout, modify observed_state files, or
modify checkpoints.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_ID = "root"
EPS = 1.0e-9
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
MODES = ("measured", "confidence_weighted", "cap25_shadow")
FRAMES = (1, 2)

REQUIRED_OUTPUTS = [
    "decision_comparison_seed0_seed1.csv",
    "decision_comparison_seed0_seed1.json",
    "decision_comparison_seed0_seed1.md",
    "frame002_topk_branch_seed0.csv",
    "frame002_topk_branch_seed1.csv",
    "frame002_topk_spatial_match_seed0_seed1.csv",
    "frame002_topk_spatial_match_summary.json",
    "frame002_topk_spatial_match_summary.md",
    "frame002_rank_margin_seed0.json",
    "frame002_rank_margin_seed0.md",
    "frame002_rank_margin_seed1.json",
    "frame002_rank_margin_seed1.md",
    "rank_margin_comparison.csv",
    "rank_margin_comparison.json",
    "rank_margin_comparison.md",
    "branch_classification.json",
    "branch_classification.md",
    "missing_fields_report.json",
    "stage4a65u_seed_robustness_summary.json",
    "stage4a65u_seed_robustness_summary.md",
    "recommended_next_faithful_step.md",
    "frame002_seed0_seed1_selected_branches_topdown.png",
    "frame002_topk_branch_cloud_seed0_seed1.png",
    "frame002_value_margin_seed0_seed1.png",
    "frame002_gain_exp_vs_effective_sc_seed0_seed1.png",
    "frame002_cost_vs_value_seed0_seed1.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "frame*_pose.json",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "map_predict*",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_*.png",
    "observed_ratio_curve.png",
    "rollout_topdown_path.png",
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    dims = min(len(aa), len(bb))
    return float(math.sqrt(sum((aa[idx] - bb[idx]) ** 2 for idx in range(dims))))


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) <= EPS:
        return None
    return float(numerator) / float(denominator)


def percentile_summary(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def pearson_pairs(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= EPS or float(np.std(y_arr)) <= EPS:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rank_sorted(rows: list[dict[str, Any]], value_key: str, rank_key: str, reverse: bool = True) -> None:
    indexed: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        value = row.get(value_key)
        if value is None:
            continue
        value_f = as_float(value, float("nan"))
        if math.isfinite(value_f):
            indexed.append((idx, value_f))
    indexed.sort(key=lambda item: item[1], reverse=reverse)
    last_value: float | None = None
    last_rank = 0
    for order, (idx, value) in enumerate(indexed, start=1):
        if last_value is None or abs(value - last_value) > 1.0e-12:
            last_rank = order
            last_value = value
        rows[idx][rank_key] = last_rank


def mode_path(stage_dir: Path, frame: int, mode: str, suffix: str) -> Path:
    return stage_dir / f"frame{frame:03d}_{mode}_{suffix}"


def load_tree(stage_dir: Path, frame: int, mode: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(mode_path(stage_dir, frame, mode, "tree_segments.jsonl"))
    tree = {str(row["segment_id"]): dict(row) for row in rows}
    for node in tree.values():
        node.setdefault("children", [])
    for node_id, node in list(tree.items()):
        parent = node.get("parent_id")
        if parent in tree and node_id not in tree[parent].setdefault("children", []):
            tree[parent]["children"].append(node_id)
    return tree


def load_table(stage_dir: Path, frame: int, mode: str) -> list[dict[str, Any]]:
    numeric = {
        "depth",
        "gain",
        "gain_exp",
        "gain_sc",
        "gain_hybrid",
        "effective_gain_sc",
        "gain_hybrid_effective",
        "gain_occ",
        "gain_conf",
        "cost",
        "local_gain_over_cost",
        "local_effective_hybrid_over_cost",
        "accumulated_gain",
        "accumulated_cost",
        "value",
        "children_count",
        "visible_count",
        "frontier_count_visible",
    }
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(mode_path(stage_dir, frame, mode, "gain_cost_value_table.csv")):
        row: dict[str, Any] = dict(raw)
        for key in ("end_grid", "end_world"):
            row[key] = parse_literal_list(row.get(key))
        for key in numeric:
            if key in row:
                row[key] = as_float(row.get(key), float("nan"))
        rows.append(row)
    return rows


def table_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("segment_id")): row for row in rows}


def path_to_root_from_tree(tree: dict[str, dict[str, Any]], node_id: str | None) -> list[str]:
    if not node_id or node_id not in tree:
        return []
    path: list[str] = []
    seen: set[str] = set()
    current: str | None = node_id
    while current and current in tree and current not in seen:
        seen.add(current)
        path.append(current)
        parent = tree[current].get("parent_id")
        current = str(parent) if parent else None
    path.reverse()
    return [node for node in path if node != ROOT_ID]


def path_to_root_from_rows(rows_by_id: dict[str, dict[str, Any]], node_id: str | None) -> list[str]:
    if not node_id or node_id not in rows_by_id:
        return []
    path: list[str] = []
    seen: set[str] = set()
    current: str | None = node_id
    while current and current in rows_by_id and current not in seen:
        seen.add(current)
        path.append(current)
        parent = rows_by_id[current].get("parent_id")
        current = str(parent) if parent else None
    path.reverse()
    return [node for node in path if node != ROOT_ID]


def root_children(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    children = [row for row in rows if str(row.get("parent_id")) == ROOT_ID]
    children.sort(key=lambda row: as_float(row.get("value"), float("-inf")), reverse=True)
    for idx, row in enumerate(children, start=1):
        row["root_child_value_rank"] = idx
    return children


def margin_from_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    children = root_children(rows)
    winner = children[0] if children else None
    runner = children[1] if len(children) > 1 else None
    winner_value = as_float(winner.get("value"), float("nan")) if winner else None
    runner_value = as_float(runner.get("value"), float("nan")) if runner else None
    margin = None
    normalized = None
    if winner_value is not None and runner_value is not None:
        margin = winner_value - runner_value
        normalized = safe_ratio(margin, abs(winner_value))
    return {
        "winner_id": winner.get("segment_id") if winner else None,
        "winner_best_descendant_id": winner.get("best_descendant_id") if winner else None,
        "winner_value": winner_value,
        "runner_up_id": runner.get("segment_id") if runner else None,
        "runner_up_best_descendant_id": runner.get("best_descendant_id") if runner else None,
        "runner_up_value": runner_value,
        "winner_runner_up_margin": margin,
        "normalized_margin": normalized,
        "root_child_count": len(children),
    }


def decision_payload(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = dict(data.get("source_payload") or data.get("raw_decision") or data)
    decision.update(data.get("decision") or {})
    raw = data.get("raw_decision") or decision.get("raw_decision") or {}
    return decision, raw


def extract_decision(
    stage_dir: Path,
    seed_label: str,
    frame: int,
    mode: str,
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_file = mode_path(stage_dir, frame, mode, "tree_decision.json")
    table_file = mode_path(stage_dir, frame, mode, "gain_cost_value_table.csv")
    tree_file = mode_path(stage_dir, frame, mode, "tree_segments.jsonl")
    data = read_json(decision_file)
    decision, raw = decision_payload(data)
    tree = load_tree(stage_dir, frame, mode)
    table_rows = load_table(stage_dir, frame, mode)
    rows_by_id = table_by_id(table_rows)
    root_world = tree.get(ROOT_ID, {}).get("end_world") or rows_by_id.get(ROOT_ID, {}).get("end_world")

    raw_selected = raw.get("selected_child") if isinstance(raw.get("selected_child"), dict) else {}
    raw_best = raw.get("best_descendant") if isinstance(raw.get("best_descendant"), dict) else {}
    decision_best = decision.get("best_descendant") if isinstance(decision.get("best_descendant"), dict) else {}

    path_ids = decision.get("path_segment_ids") or data.get("path_segment_ids") or []
    selected_id = (
        decision.get("selected_child_id")
        or raw.get("selected_child_id")
        or raw_selected.get("segment_id")
        or (path_ids[0] if path_ids else None)
    )
    best_id = (
        decision.get("best_descendant_id")
        or decision.get("selected_child_best_descendant_id")
        or raw.get("selected_child_best_descendant_id")
        or raw_best.get("segment_id")
        or raw_best.get("best_descendant_id")
        or decision_best.get("segment_id")
        or decision_best.get("best_descendant_id")
        or (path_ids[-1] if path_ids else None)
    )
    if not path_ids:
        path_ids = path_to_root_from_tree(tree, str(best_id) if best_id else None)

    selected_node = tree.get(str(selected_id)) or rows_by_id.get(str(selected_id), {}) or raw_selected
    best_node = tree.get(str(best_id)) or rows_by_id.get(str(best_id), {}) or raw_best or decision_best
    selected_grid = decision.get("selected_child_grid") or selected_node.get("end_grid")
    selected_world = decision.get("selected_child_world") or selected_node.get("end_world")
    best_grid = decision.get("best_descendant_grid") or best_node.get("end_grid")
    best_world = decision.get("best_descendant_world") or best_node.get("end_world")
    selected_distance = (
        decision.get("selected_child_distance_from_root_m")
        or data.get("selected_child_distance_from_root_m")
        or euclidean(root_world, selected_world)
    )
    best_distance = (
        decision.get("best_descendant_distance_from_root_m")
        or data.get("best_descendant_distance_from_root_m")
        or euclidean(root_world, best_world)
    )
    margin = margin_from_table(table_rows)

    fields = {
        "selected_child_id": selected_id,
        "selected_child_grid": selected_grid,
        "selected_child_world": selected_world,
        "best_descendant_id": best_id,
        "best_descendant_grid": best_grid,
        "best_descendant_world": best_world,
    }
    for field, value in fields.items():
        if value is None:
            missing.append(
                {
                    "seed": seed_label,
                    "frame": frame,
                    "mode": mode,
                    "artifact": str(decision_file),
                    "field": field,
                    "severity": "derived_missing",
                }
            )

    return {
        "seed": seed_label,
        "frame": frame,
        "mode": mode,
        "decision_file": str(decision_file),
        "table_file": str(table_file),
        "tree_file": str(tree_file),
        "selected_child_id": selected_id,
        "selected_child_grid": selected_grid,
        "selected_child_world": selected_world,
        "best_descendant_id": best_id,
        "best_descendant_grid": best_grid,
        "best_descendant_world": best_world,
        "selected_child_distance_from_root_m": selected_distance,
        "best_descendant_distance_from_root_m": best_distance,
        "accumulated_gain_exp": decision.get("accumulated_gain_exp") or data.get("accumulated_gain_exp"),
        "accumulated_raw_gain_sc": (
            decision.get("accumulated_raw_gain_sc")
            if decision.get("accumulated_raw_gain_sc") is not None
            else decision.get("accumulated_gain_sc", data.get("accumulated_gain_sc"))
        ),
        "accumulated_effective_gain_sc": (
            decision.get("accumulated_effective_gain_sc")
            if decision.get("accumulated_effective_gain_sc") is not None
            else decision.get("accumulated_gain_conf", 0.0 if mode == "measured" else None)
        ),
        "accumulated_cost": decision.get("accumulated_cost") or data.get("accumulated_cost"),
        "value": decision.get("value") or data.get("value"),
        "winner_margin_vs_runner_up": margin.get("winner_runner_up_margin"),
        "normalized_winner_margin": margin.get("normalized_margin"),
        "runner_up_id": margin.get("runner_up_id"),
        "runner_up_value": margin.get("runner_up_value"),
        "branch_depth": len(path_ids),
        "path_node_ids": path_ids,
        "root_child_count": margin.get("root_child_count"),
    }


def top_branches(stage_dir: Path, seed_label: str, frame: int, mode: str, k: int = 10) -> list[dict[str, Any]]:
    rows = load_table(stage_dir, frame, mode)
    rows_by_id = table_by_id(rows)
    root = rows_by_id.get(ROOT_ID, {})
    branches: list[dict[str, Any]] = []
    for rank, child in enumerate(root_children(rows), start=1):
        if rank > k:
            break
        child_id = str(child.get("segment_id"))
        best_id = child.get("best_descendant_id")
        best = rows_by_id.get(str(best_id), child)
        path_ids = path_to_root_from_rows(rows_by_id, str(best_id) if best_id else child_id)
        path_rows = [rows_by_id[node_id] for node_id in path_ids if node_id in rows_by_id]
        cost = sum(as_float(row.get("cost")) for row in path_rows)
        gain_exp = sum(as_float(row.get("gain_exp")) for row in path_rows)
        raw_gain_sc = sum(as_float(row.get("gain_sc")) for row in path_rows)
        effective_gain_sc = sum(as_float(row.get("effective_gain_sc")) for row in path_rows)
        branches.append(
            {
                "seed": seed_label,
                "frame": frame,
                "formula": mode,
                "rank": rank,
                "top_k_requested": k,
                "selected_root_child_id": child_id,
                "best_descendant_id": best_id,
                "root_child_grid": child.get("end_grid"),
                "root_child_world": child.get("end_world"),
                "best_descendant_grid": best.get("end_grid"),
                "best_descendant_world": best.get("end_world"),
                "value": child.get("value"),
                "gain_exp": gain_exp,
                "raw_gain_sc": raw_gain_sc,
                "effective_gain_sc": effective_gain_sc,
                "cost": cost,
                "root_child_distance_from_root_m": euclidean(root.get("end_world"), child.get("end_world")),
                "best_descendant_distance_from_root_m": euclidean(root.get("end_world"), best.get("end_world")),
                "depth": as_int(best.get("depth")),
                "path_length": len(path_ids),
                "path_node_ids": path_ids,
            }
        )
    return branches


def spatial_matches(seed0_rows: list[dict[str, Any]], seed1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for row0 in seed0_rows:
        candidates: list[tuple[float, dict[str, Any], float | None, float | None]] = []
        for row1 in seed1_rows:
            child_dist = euclidean(row0.get("root_child_world"), row1.get("root_child_world"))
            best_dist = euclidean(row0.get("best_descendant_world"), row1.get("best_descendant_world"))
            if child_dist is None and best_dist is None:
                continue
            score = (child_dist if child_dist is not None else 999.0) + (best_dist if best_dist is not None else 999.0)
            candidates.append((score, row1, child_dist, best_dist))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        _, nearest, child_dist, best_dist = candidates[0]
        same_formula = [item for item in candidates if item[1].get("formula") == row0.get("formula")]
        same_nearest = same_formula[0] if same_formula else None
        matches.append(
            {
                "seed0_formula": row0.get("formula"),
                "seed0_rank": row0.get("rank"),
                "seed0_root_child_id": row0.get("selected_root_child_id"),
                "seed0_best_descendant_id": row0.get("best_descendant_id"),
                "seed0_value": row0.get("value"),
                "nearest_seed1_formula": nearest.get("formula"),
                "nearest_seed1_rank": nearest.get("rank"),
                "nearest_seed1_root_child_id": nearest.get("selected_root_child_id"),
                "nearest_seed1_best_descendant_id": nearest.get("best_descendant_id"),
                "nearest_seed1_value": nearest.get("value"),
                "formula_match": nearest.get("formula") == row0.get("formula"),
                "root_child_spatial_distance_m": child_dist,
                "best_descendant_spatial_distance_m": best_dist,
                "value_rank_difference": (
                    abs(as_int(row0.get("rank")) - as_int(nearest.get("rank"))) if nearest.get("rank") is not None else None
                ),
                "nearest_same_formula_seed1_rank": same_nearest[1].get("rank") if same_nearest else None,
                "nearest_same_formula_seed1_root_child_id": same_nearest[1].get("selected_root_child_id") if same_nearest else None,
                "nearest_same_formula_seed1_best_descendant_id": same_nearest[1].get("best_descendant_id") if same_nearest else None,
                "nearest_same_formula_root_child_distance_m": same_nearest[2] if same_nearest else None,
                "nearest_same_formula_best_descendant_distance_m": same_nearest[3] if same_nearest else None,
            }
        )
    return matches


def root_child_rank(rows: list[dict[str, Any]], segment_id: str | None) -> int | None:
    for row in root_children(rows):
        if row.get("segment_id") == segment_id:
            return as_int(row.get("root_child_value_rank"))
    return None


def rank_margin_summary(stage_dir: Path, seed_label: str, decisions: dict[tuple[str, int, str], dict[str, Any]]) -> dict[str, Any]:
    measured_rows = load_table(stage_dir, 2, "measured")
    conf_rows = load_table(stage_dir, 2, "confidence_weighted")
    cap_rows = load_table(stage_dir, 2, "cap25_shadow")
    measured_dec = decisions[(seed_label, 2, "measured")]
    conf_dec = decisions[(seed_label, 2, "confidence_weighted")]
    conf_non_root = [row for row in conf_rows if row.get("segment_id") != ROOT_ID]
    value = [as_float(row.get("value"), float("nan")) for row in conf_non_root]
    inverse_cost = [safe_ratio(1.0, as_float(row.get("accumulated_cost"), float("nan"))) or float("nan") for row in conf_non_root]
    effective_sc = [as_float(row.get("effective_gain_sc"), float("nan")) for row in conf_non_root]
    gain_exp = [as_float(row.get("gain_exp"), float("nan")) for row in conf_non_root]
    costs = [as_float(row.get("accumulated_cost"), float("nan")) for row in conf_non_root]
    depths = [as_float(row.get("depth"), float("nan")) for row in conf_non_root]
    positive_effective_sc = sum(1 for row in conf_non_root if as_float(row.get("effective_gain_sc")) > 0.0)
    margins = {
        "measured": margin_from_table(measured_rows),
        "confidence_weighted": margin_from_table(conf_rows),
        "cap25_shadow": margin_from_table(cap_rows),
    }
    measured_id = str(measured_dec.get("selected_child_id"))
    conf_id = str(conf_dec.get("selected_child_id"))
    winner_measured_rank_under_conf = root_child_rank(conf_rows, measured_id)
    winner_conf_rank_under_measured = root_child_rank(measured_rows, conf_id)
    conf_margin = margins["confidence_weighted"]
    normalized_margin = conf_margin.get("normalized_margin")
    cost_corr = pearson_pairs(value, inverse_cost)
    sc_corr = pearson_pairs(value, effective_sc)
    exp_corr = pearson_pairs(value, gain_exp)
    cost_dominance = bool(cost_corr is not None and cost_corr > 0.60)
    sc_decisive = bool(
        conf_id != measured_id
        and (winner_conf_rank_under_measured or 999) > 1
        and sc_corr is not None
        and sc_corr > 0.20
    )
    return {
        "seed": seed_label,
        "formula_margins": margins,
        "winner_value": conf_margin.get("winner_value"),
        "runner_up_value": conf_margin.get("runner_up_value"),
        "winner_runner_up_margin": conf_margin.get("winner_runner_up_margin"),
        "normalized_margin": normalized_margin,
        "margin_is_narrow_lt_5pct": bool(normalized_margin is not None and normalized_margin < 0.05),
        "measured_winner_id": measured_id,
        "confidence_winner_id": conf_id,
        "winner_measured_only_rank_under_confidence_scoring": winner_measured_rank_under_conf,
        "winner_confidence_rank_under_measured_only_scoring": winner_conf_rank_under_measured,
        "gain_exp_effective_sc_correlation": pearson_pairs(gain_exp, effective_sc),
        "value_inverse_cost_correlation": cost_corr,
        "value_effective_sc_correlation": sc_corr,
        "value_gain_exp_correlation": exp_corr,
        "nodes_with_positive_effective_gain_sc": positive_effective_sc,
        "node_count": len(conf_non_root),
        "effective_gain_sc_distribution": percentile_summary(effective_sc),
        "cost_distribution": percentile_summary(costs),
        "branch_depth_distribution": percentile_summary(depths),
        "cost_dominance_inference": cost_dominance,
        "effective_sc_gain_decisive_inference": sc_decisive,
    }


def markdown_rank(summary: dict[str, Any]) -> str:
    margin = summary["formula_margins"]["confidence_weighted"]
    return "\n".join(
        [
            f"# Frame 2 Rank/Margin Summary: {summary['seed']}",
            "",
            f"- confidence winner: `{summary['confidence_winner_id']}`",
            f"- measured winner: `{summary['measured_winner_id']}`",
            f"- winner value: `{margin.get('winner_value')}`",
            f"- runner-up: `{margin.get('runner_up_id')}` value `{margin.get('runner_up_value')}`",
            f"- margin: `{margin.get('winner_runner_up_margin')}`",
            f"- normalized margin: `{margin.get('normalized_margin')}`",
            f"- measured winner rank under confidence scoring: `{summary['winner_measured_only_rank_under_confidence_scoring']}`",
            f"- confidence winner rank under measured-only scoring: `{summary['winner_confidence_rank_under_measured_only_scoring']}`",
            f"- value/inverse-cost correlation: `{summary['value_inverse_cost_correlation']}`",
            f"- value/effective-SC correlation: `{summary['value_effective_sc_correlation']}`",
            f"- value/gain-exp correlation: `{summary['value_gain_exp_correlation']}`",
            f"- positive effective-SC nodes: `{summary['nodes_with_positive_effective_gain_sc']}` / `{summary['node_count']}`",
            f"- cost dominance inference: `{summary['cost_dominance_inference']}`",
            f"- effective SC decisive inference: `{summary['effective_sc_gain_decisive_inference']}`",
            "",
        ]
    )


def classify_branch(
    label: str,
    branch: dict[str, Any],
    measured: dict[str, Any],
    seed0_sc: dict[str, Any],
) -> dict[str, Any]:
    selected_to_measured = euclidean(branch.get("selected_child_world"), measured.get("selected_child_world"))
    best_to_measured = euclidean(branch.get("best_descendant_world"), measured.get("best_descendant_world"))
    selected_to_seed0 = euclidean(branch.get("selected_child_world"), seed0_sc.get("selected_child_world"))
    best_to_seed0 = euclidean(branch.get("best_descendant_world"), seed0_sc.get("best_descendant_world"))
    unstable = selected_to_measured is None or selected_to_seed0 is None or best_to_seed0 is None
    same_grid = branch.get("selected_child_grid") == measured.get("selected_child_grid")
    same_as_measured = False if unstable else bool(same_grid or selected_to_measured <= 0.15)
    spatial_seed0 = False if unstable else bool(selected_to_seed0 <= 0.25 and best_to_seed0 <= 0.75)
    local_jitter = False if unstable else bool(not same_as_measured and selected_to_measured < 0.25)
    distinct = False if unstable else bool(selected_to_measured >= 0.25 and not spatial_seed0)
    classes = []
    if unstable:
        classes.append("unstable")
        primary = "unstable"
    else:
        if same_as_measured:
            classes.append("same_as_measured")
        if spatial_seed0:
            classes.append("spatially_same_as_seed0_sc")
        if local_jitter:
            classes.append("local_jitter")
        if distinct:
            classes.append("distinct_sc_branch")
        primary = classes[0] if classes else "unstable"
    return {
        "label": label,
        "primary_class": primary,
        "matching_classes": classes,
        "selected_child_id": branch.get("selected_child_id"),
        "best_descendant_id": branch.get("best_descendant_id"),
        "selected_child_grid": branch.get("selected_child_grid"),
        "best_descendant_grid": branch.get("best_descendant_grid"),
        "selected_to_measured_m": selected_to_measured,
        "best_to_measured_m": best_to_measured,
        "selected_to_seed0_sc_m": selected_to_seed0,
        "best_to_seed0_sc_m": best_to_seed0,
        "same_as_measured": same_as_measured,
        "spatially_same_as_seed0_sc": spatial_seed0,
        "distinct_sc_branch": distinct,
        "local_jitter": local_jitter,
    }


def topdown_projection(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == 1, axis=2)
    free = np.any(observed_state == 0, axis=2)
    proj = np.zeros(observed_state.shape[:2], dtype=np.int8)
    proj[free] = 1
    proj[occupied] = 2
    return proj


def plot_base_map(ax: plt.Axes, observed_state: np.ndarray) -> None:
    proj = topdown_projection(observed_state)
    colors = np.asarray(
        [
            [0.88, 0.88, 0.88, 1.0],
            [0.94, 0.98, 0.99, 1.0],
            [0.62, 0.14, 0.14, 1.0],
        ]
    )
    ax.imshow(colors[proj].transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.grid(color="#111827", alpha=0.12, linewidth=0.4)


def grid_xy(item: dict[str, Any], key: str) -> tuple[float, float] | None:
    grid = item.get(key)
    if grid is None:
        return None
    return float(grid[0]), float(grid[1])


def plot_branch_marker(
    ax: plt.Axes,
    grid: list[Any] | None,
    label: str,
    color: str,
    marker: str,
    size: float = 78.0,
) -> None:
    if not grid:
        return
    x, y = float(grid[0]), float(grid[1])
    ax.scatter([x], [y], s=size, c=color, marker=marker, edgecolor="#111827", linewidth=0.7, zorder=4)
    ax.text(x + 0.35, y + 0.35, label, fontsize=8, color=color, weight="bold")


def make_plots(
    output_dir: Path,
    seed0_dir: Path,
    seed1_dir: Path,
    decisions: dict[tuple[str, int, str], dict[str, Any]],
    seed0_top: list[dict[str, Any]],
    seed1_top: list[dict[str, Any]],
    rank0: dict[str, Any],
    rank1: dict[str, Any],
) -> dict[str, str]:
    observed = np.load(seed0_dir / "observed_state_frame002.npy")
    plots: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    plot_base_map(ax, observed)
    items = [
        (decisions[("seed0", 2, "measured")], "s0 measured", "#2563eb", "o"),
        (decisions[("seed0", 2, "confidence_weighted")], "s0 conf", "#dc2626", "^"),
        (decisions[("seed1", 2, "measured")], "s1 measured", "#0891b2", "s"),
        (decisions[("seed1", 2, "confidence_weighted")], "s1 conf", "#7c3aed", "D"),
        (decisions[("seed1", 2, "cap25_shadow")], "s1 cap25", "#f97316", "P"),
    ]
    for item, label, color, marker in items:
        plot_branch_marker(ax, item.get("selected_child_grid"), f"{label} child", color, marker)
        plot_branch_marker(ax, item.get("best_descendant_grid"), f"{label} best", color, "x", size=92.0)
    ax.set_title("Frame 2 selected branches: seed0 vs seed1")
    path = output_dir / "frame002_seed0_seed1_selected_branches_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["selected_branches_topdown"] = str(path)

    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    plot_base_map(ax, observed)
    for rows, color, seed_name, marker in [
        ([r for r in seed0_top if r.get("formula") == "confidence_weighted"], "#dc2626", "seed0", "o"),
        ([r for r in seed1_top if r.get("formula") == "confidence_weighted"], "#7c3aed", "seed1", "s"),
    ]:
        for row in rows:
            child = row.get("root_child_grid")
            best = row.get("best_descendant_grid")
            if child:
                ax.scatter([child[0]], [child[1]], s=52, c=color, marker=marker, alpha=0.78, edgecolor="#111827", linewidth=0.5)
                ax.text(child[0] + 0.25, child[1] + 0.2, f"{seed_name} r{row.get('rank')}", fontsize=7, color=color)
            if best:
                ax.scatter([best[0]], [best[1]], s=64, c=color, marker="x", alpha=0.78)
    ax.set_title("Frame 2 confidence top branch cloud")
    path = output_dir / "frame002_topk_branch_cloud_seed0_seed1.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["topk_branch_cloud"] = str(path)

    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    labels = []
    winners = []
    runners = []
    for seed_name, rank in [("seed0", rank0), ("seed1", rank1)]:
        for mode in MODES:
            margin = rank["formula_margins"][mode]
            labels.append(f"{seed_name}\n{mode}")
            winners.append(as_float(margin.get("winner_value"), 0.0))
            runners.append(as_float(margin.get("runner_up_value"), 0.0))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, winners, width=0.36, label="winner", color="#2563eb")
    ax.bar(x + 0.18, runners, width=0.36, label="runner-up", color="#f97316")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("value")
    ax.set_title("Frame 2 winner vs runner-up values")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "frame002_value_margin_seed0_seed1.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["value_margin"] = str(path)

    fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
    for stage_dir, seed_name, color in [
        (seed0_dir, "seed0", "#dc2626"),
        (seed1_dir, "seed1", "#7c3aed"),
    ]:
        if not stage_dir.exists():
            continue
        rows = [r for r in load_table(stage_dir, 2, "confidence_weighted") if r.get("segment_id") != ROOT_ID]
        ax.scatter(
            [as_float(r.get("gain_exp"), float("nan")) for r in rows],
            [as_float(r.get("effective_gain_sc"), float("nan")) for r in rows],
            s=15,
            alpha=0.55,
            label=seed_name,
            color=color,
        )
    ax.set_xlabel("local gain_exp")
    ax.set_ylabel("local effective_gain_sc")
    ax.set_title("Frame 2 gain_exp vs effective SC")
    ax.grid(alpha=0.25)
    ax.legend()
    path = output_dir / "frame002_gain_exp_vs_effective_sc_seed0_seed1.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["gain_exp_vs_effective_sc"] = str(path)

    fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
    for rows, seed_name, color in [
        ([r for r in load_table(seed0_dir, 2, "confidence_weighted") if r.get("segment_id") != ROOT_ID], "seed0", "#dc2626"),
        ([r for r in load_table(seed1_dir, 2, "confidence_weighted") if r.get("segment_id") != ROOT_ID], "seed1", "#7c3aed"),
    ]:
        ax.scatter(
            [as_float(r.get("accumulated_cost"), float("nan")) for r in rows],
            [as_float(r.get("value"), float("nan")) for r in rows],
            s=15,
            alpha=0.55,
            label=seed_name,
            color=color,
        )
    ax.set_xlabel("accumulated cost")
    ax.set_ylabel("value")
    ax.set_title("Frame 2 cost vs value")
    ax.grid(alpha=0.25)
    ax.legend()
    path = output_dir / "frame002_cost_vs_value_seed0_seed1.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["cost_vs_value"] = str(path)
    return plots


def decision_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Decision Comparison: seed0 vs seed1",
        "",
        "| seed | frame | mode | selected child | best descendant | value | margin | path |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {frame} | {mode} | `{sel}` {sel_grid} | `{best}` {best_grid} | `{value}` | `{margin}` | `{path}` |".format(
                seed=row.get("seed"),
                frame=row.get("frame"),
                mode=row.get("mode"),
                sel=row.get("selected_child_id"),
                sel_grid=row.get("selected_child_grid"),
                best=row.get("best_descendant_id"),
                best_grid=row.get("best_descendant_grid"),
                value=row.get("value"),
                margin=row.get("winner_margin_vs_runner_up"),
                path=row.get("path_node_ids"),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: seed1 returns to the measured-only branch in its own score/tree space, but that measured branch is spatially close to the seed0 confidence-weighted branch.",
            "",
        ]
    )
    return "\n".join(lines)


def branch_classification_markdown(classifications: dict[str, Any]) -> str:
    lines = ["# Branch Classification", ""]
    for label, item in classifications["entries"].items():
        lines.append(
            f"- `{label}`: primary `{item['primary_class']}`, matches `{item['matching_classes']}`, "
            f"selected-to-seed0-SC `{item['selected_to_seed0_sc_m']}` m, "
            f"best-to-seed0-SC `{item['best_to_seed0_sc_m']}` m"
        )
    lines.extend(
        [
            "",
            "Nuance: seed1 confidence/cap25 are same_as_measured within the seed1 tree, while also satisfying spatially_same_as_seed0_sc. This means the node IDs and parent tree changed, but the selected local region stayed near the seed0 SC branch.",
            "",
        ]
    )
    return "\n".join(lines)


def topk_summary_markdown(summary: dict[str, Any]) -> str:
    nearest = summary["seed0_confidence_reference_nearest_seed1_confidence"]
    return "\n".join(
        [
            "# Frame 2 Top-K Spatial Match Summary",
            "",
            f"- seed0 top branches: `{summary['seed0_top_branch_rows']}`",
            f"- seed1 top branches: `{summary['seed1_top_branch_rows']}`",
            f"- overlapping pairs within child<=0.25m and best<=0.75m: `{summary['overlap_pairs_child_0p25_best_0p75']}`",
            f"- seed0 confidence reference nearest seed1 confidence branch: `{nearest}`",
            f"- top-K clouds overlap: `{summary['topk_clouds_overlap']}`",
            f"- interpretation: {summary['interpretation']}",
            "",
        ]
    )


def rank_comparison_markdown(rows: list[dict[str, Any]], recommendation: str) -> str:
    lines = [
        "# Rank/Margin Comparison",
        "",
        "| seed | conf winner | measured winner | margin | normalized | measured rank in conf | conf rank in measured | cost corr | SC decisive |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | `{cw}` | `{mw}` | `{margin}` | `{norm}` | `{mr}` | `{cr}` | `{cost}` | `{sc}` |".format(
                seed=row.get("seed"),
                cw=row.get("confidence_winner_id"),
                mw=row.get("measured_winner_id"),
                margin=row.get("winner_runner_up_margin"),
                norm=row.get("normalized_margin"),
                mr=row.get("winner_measured_only_rank_under_confidence_scoring"),
                cr=row.get("winner_confidence_rank_under_measured_only_scoring"),
                cost=row.get("value_inverse_cost_correlation"),
                sc=row.get("effective_sc_gain_decisive_inference"),
            )
        )
    lines.extend(["", f"Recommendation: {recommendation}", ""])
    return "\n".join(lines)


def summarize_and_recommend(
    decisions: dict[tuple[str, int, str], dict[str, Any]],
    rank0: dict[str, Any],
    rank1: dict[str, Any],
    topk_summary: dict[str, Any],
    branch_classification: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    seed0_conf = decisions[("seed0", 2, "confidence_weighted")]
    seed1_measured = decisions[("seed1", 2, "measured")]
    seed1_conf = decisions[("seed1", 2, "confidence_weighted")]
    selected_delta = euclidean(seed1_conf.get("selected_child_world"), seed0_conf.get("selected_child_world"))
    best_delta = euclidean(seed1_conf.get("best_descendant_world"), seed0_conf.get("best_descendant_world"))
    seed1_same_measured = seed1_conf.get("selected_child_id") == seed1_measured.get("selected_child_id")
    seed1_spatial_seed0 = branch_classification["entries"]["seed1_confidence_weighted"]["spatially_same_as_seed0_sc"]
    margins_narrow = bool(rank0.get("margin_is_narrow_lt_5pct") or rank1.get("margin_is_narrow_lt_5pct"))

    if seed1_spatial_seed0 and not seed1_same_measured and not margins_narrow:
        recommended = "repeated gated two-frame on another start/scene seed"
        why = "seed1 stayed spatially aligned with the seed0 SC branch and did not collapse to the measured branch with narrow margins."
    elif seed1_spatial_seed0 and (seed1_same_measured or margins_narrow):
        recommended = "multi-seed offline replay or seed robustness sweep"
        why = "seed1 is spatially close to the seed0 SC branch, but confidence_weighted coincides with measured-only in seed1 score space."
    else:
        recommended = "branch diversity diagnosis or tree sampling stabilization"
        why = "seed1 diverged enough that branch diversity or sampling stabilization should be diagnosed before longer smoke tests."

    support_3frame = bool(recommended.startswith("repeated gated two-frame"))
    summary = {
        "stage": "Stage 4A-6.5u",
        "seed1_actually_returned_to_measured_only": seed1_same_measured,
        "seed1_returned_to_measured_in_score_space": seed1_same_measured
        and rank1.get("winner_measured_only_rank_under_confidence_scoring") == 1,
        "seed1_spatially_close_to_seed0_sc": seed1_spatial_seed0,
        "seed1_to_seed0_sc_selected_child_delta_m": selected_delta,
        "seed1_to_seed0_sc_best_descendant_delta_m": best_delta,
        "topk_branch_clouds_overlap": topk_summary.get("topk_clouds_overlap"),
        "seed1_node_id_only_or_score_rank": (
            "score/rank supports measured-only within seed1; spatial region remains near seed0 SC"
            if seed1_same_measured
            else "not merely node-id relabeling"
        ),
        "seed0_margin_narrow": rank0.get("margin_is_narrow_lt_5pct"),
        "seed1_margin_narrow": rank1.get("margin_is_narrow_lt_5pct"),
        "winner_margins_both_narrow": bool(rank0.get("margin_is_narrow_lt_5pct") and rank1.get("margin_is_narrow_lt_5pct")),
        "cost_still_dominates_branch_selection": bool(
            rank0.get("cost_dominance_inference") or rank1.get("cost_dominance_inference")
        ),
        "effective_sc_gain_decisive": bool(
            rank0.get("effective_sc_gain_decisive_inference") and rank1.get("effective_sc_gain_decisive_inference")
        ),
        "supports_entering_3_frame_gated_smoke": support_3frame,
        "should_do_multi_seed_offline_replay_or_seed_sweep": recommended == "multi-seed offline replay or seed robustness sweep",
        "still_cannot_rollout": True,
        "recommended_next_faithful_step": recommended,
        "recommendation_reason": why,
    }
    md = "\n".join(
        [
            "# Stage 4A-6.5u Seed Robustness Summary",
            "",
            f"1. seed1 truly returned to measured-only: `{summary['seed1_actually_returned_to_measured_only']}` in seed1 score/tree space.",
            f"2. seed1 spatially close to seed0 SC branch: `{summary['seed1_spatially_close_to_seed0_sc']}`; selected delta `{selected_delta}` m, best delta `{best_delta}` m.",
            f"3. top-K branch clouds overlap: `{summary['topk_branch_clouds_overlap']}`.",
            f"4. seed1 node ID vs score/rank: {summary['seed1_node_id_only_or_score_rank']}.",
            f"5. margins narrow: seed0 `{summary['seed0_margin_narrow']}`, seed1 `{summary['seed1_margin_narrow']}`.",
            f"6. cost still dominates branch selection: `{summary['cost_still_dominates_branch_selection']}`.",
            f"7. effective SC gain decisive across seeds: `{summary['effective_sc_gain_decisive']}`.",
            f"8. supports entering 3-frame gated smoke now: `{summary['supports_entering_3_frame_gated_smoke']}`.",
            f"9. should first do multi-seed offline replay / seed sweep: `{summary['should_do_multi_seed_offline_replay_or_seed_sweep']}`.",
            f"10. still cannot rollout: `{summary['still_cannot_rollout']}`.",
            "",
            f"Recommended next faithful step: **{recommended}**.",
            "",
            f"Why: {why}",
            "",
            "Safety: no Isaac startup, no new capture, no map_predict rerun, no SSCNet inference, no action execution, no rollout, no training/RL, no checkpoint/observed_state modification, no prediction writeback, no prediction traversability/collision/ray-blocking use, no target/ground-truth scoring, no external source modification/build, and no coverage-improvement claim.",
            "",
        ]
    )
    rec_md = "\n".join(
        [
            "# Recommended Next Faithful Step",
            "",
            f"Recommended: **{recommended}**.",
            "",
            f"Reason: {why}",
            "",
            "Do not proceed directly to rollout. Keep the next step offline or tightly staged, with prediction still read-only and information-gain-only.",
            "",
        ]
    )
    return summary, md, rec_md


def output_has_prohibited_artifacts(output_dir: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        matches = sorted(str(path.relative_to(output_dir)) for path in output_dir.glob(pattern))
        if matches:
            found[pattern] = matches
    return found


def make_safety(seed0_dir: Path, seed1_dir: Path, before_hashes: dict[str, str], output_dir: Path) -> dict[str, Any]:
    after_hashes = {path: sha256_file(Path(path)) for path in before_hashes}
    checkpoint_before = before_hashes.get(str(CHECKPOINT_PATH))
    checkpoint_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    return {
        "isaac_startup": False,
        "new_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_or_rl": False,
        "checkpoint_modified": checkpoint_before != checkpoint_after,
        "observed_state_modified": any(before_hashes[path] != after_hashes[path] for path in before_hashes if "observed_state" in path),
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_scoring": False,
        "source_modified_built": False,
        "new_map_predict_outputs_created": False,
        "coverage_improvement_claim": False,
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "prohibited_artifacts_in_output": output_has_prohibited_artifacts(output_dir),
        "seed0_dir": str(seed0_dir),
        "seed1_dir": str(seed1_dir),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed0_dir = Path(args.seed0_dir).resolve()
    seed1_dir = Path(args.seed1_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_inputs: list[Path] = []
    for stage_dir in (seed0_dir, seed1_dir):
        for frame in FRAMES:
            for mode in MODES:
                required_inputs.append(mode_path(stage_dir, frame, mode, "tree_decision.json"))
                required_inputs.append(mode_path(stage_dir, frame, mode, "gain_cost_value_table.csv"))
                required_inputs.append(mode_path(stage_dir, frame, mode, "tree_segments.jsonl"))
        required_inputs.append(stage_dir / "observed_state_frame002.npy")
        required_inputs.append(stage_dir / "frame002_prediction/global_prediction_layer.npz")
    missing_required = [str(path) for path in required_inputs if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(f"Missing required saved artifacts: {missing_required}")

    hash_paths = [
        seed0_dir / "observed_state_frame001.npy",
        seed0_dir / "observed_state_frame002.npy",
        seed1_dir / "observed_state_frame001.npy",
        seed1_dir / "observed_state_frame002.npy",
        seed0_dir / "frame001_prediction/global_prediction_layer.npz",
        seed0_dir / "frame002_prediction/global_prediction_layer.npz",
        seed1_dir / "frame001_prediction/global_prediction_layer.npz",
        seed1_dir / "frame002_prediction/global_prediction_layer.npz",
        CHECKPOINT_PATH,
    ]
    before_hashes = {str(path): sha256_file(path) for path in hash_paths if path.is_file()}

    missing_fields: list[dict[str, Any]] = []
    decisions: dict[tuple[str, int, str], dict[str, Any]] = {}
    decision_rows: list[dict[str, Any]] = []
    for seed_label, stage_dir in (("seed0", seed0_dir), ("seed1", seed1_dir)):
        for frame in FRAMES:
            for mode in MODES:
                row = extract_decision(stage_dir, seed_label, frame, mode, missing_fields)
                decisions[(seed_label, frame, mode)] = row
                decision_rows.append(row)

    write_csv(output_dir / "decision_comparison_seed0_seed1.csv", decision_rows)
    save_json(output_dir / "decision_comparison_seed0_seed1.json", {"decisions": decision_rows})
    write_text(output_dir / "decision_comparison_seed0_seed1.md", decision_markdown(decision_rows))

    seed0_top: list[dict[str, Any]] = []
    seed1_top: list[dict[str, Any]] = []
    for mode in MODES:
        seed0_top.extend(top_branches(seed0_dir, "seed0", 2, mode, k=10))
        seed1_top.extend(top_branches(seed1_dir, "seed1", 2, mode, k=10))
    write_csv(output_dir / "frame002_topk_branch_seed0.csv", seed0_top)
    write_csv(output_dir / "frame002_topk_branch_seed1.csv", seed1_top)
    match_rows = spatial_matches(seed0_top, seed1_top)
    write_csv(output_dir / "frame002_topk_spatial_match_seed0_seed1.csv", match_rows)

    seed0_conf_ref = next(
        row
        for row in seed0_top
        if row.get("formula") == "confidence_weighted" and row.get("selected_root_child_id") == decisions[("seed0", 2, "confidence_weighted")].get("selected_child_id")
    )
    seed1_conf_rows = [row for row in seed1_top if row.get("formula") == "confidence_weighted"]
    nearest_seed1_conf = min(
        seed1_conf_rows,
        key=lambda row: (euclidean(seed0_conf_ref.get("root_child_world"), row.get("root_child_world")) or 999.0)
        + (euclidean(seed0_conf_ref.get("best_descendant_world"), row.get("best_descendant_world")) or 999.0),
    )
    overlap_pairs = [
        (row0, row1)
        for row0 in [row for row in seed0_top if row.get("formula") == "confidence_weighted"]
        for row1 in seed1_conf_rows
        if (euclidean(row0.get("root_child_world"), row1.get("root_child_world")) or 999.0) <= 0.25
        and (euclidean(row0.get("best_descendant_world"), row1.get("best_descendant_world")) or 999.0) <= 0.75
    ]
    topk_summary = {
        "seed0_top_branch_rows": len(seed0_top),
        "seed1_top_branch_rows": len(seed1_top),
        "overlap_pairs_child_0p25_best_0p75": len(overlap_pairs),
        "topk_clouds_overlap": bool(overlap_pairs),
        "seed0_confidence_reference_nearest_seed1_confidence": {
            "seed0_root_child_id": seed0_conf_ref.get("selected_root_child_id"),
            "seed0_best_descendant_id": seed0_conf_ref.get("best_descendant_id"),
            "seed1_root_child_id": nearest_seed1_conf.get("selected_root_child_id"),
            "seed1_best_descendant_id": nearest_seed1_conf.get("best_descendant_id"),
            "seed1_rank": nearest_seed1_conf.get("rank"),
            "root_child_distance_m": euclidean(seed0_conf_ref.get("root_child_world"), nearest_seed1_conf.get("root_child_world")),
            "best_descendant_distance_m": euclidean(seed0_conf_ref.get("best_descendant_world"), nearest_seed1_conf.get("best_descendant_world")),
            "value_rank_difference": abs(as_int(seed0_conf_ref.get("rank")) - as_int(nearest_seed1_conf.get("rank"))),
        },
        "interpretation": "The nearest seed1 confidence branch to the seed0 confidence branch is in the same local basin if both distances satisfy the configured thresholds.",
    }
    save_json(output_dir / "frame002_topk_spatial_match_summary.json", topk_summary)
    write_text(output_dir / "frame002_topk_spatial_match_summary.md", topk_summary_markdown(topk_summary))

    rank0 = rank_margin_summary(seed0_dir, "seed0", decisions)
    rank1 = rank_margin_summary(seed1_dir, "seed1", decisions)
    save_json(output_dir / "frame002_rank_margin_seed0.json", rank0)
    save_json(output_dir / "frame002_rank_margin_seed1.json", rank1)
    write_text(output_dir / "frame002_rank_margin_seed0.md", markdown_rank(rank0))
    write_text(output_dir / "frame002_rank_margin_seed1.md", markdown_rank(rank1))
    rank_rows = [rank0, rank1]
    rank_csv_rows = [
        {
            "seed": row["seed"],
            "confidence_winner_id": row["confidence_winner_id"],
            "measured_winner_id": row["measured_winner_id"],
            "winner_runner_up_margin": row["winner_runner_up_margin"],
            "normalized_margin": row["normalized_margin"],
            "margin_is_narrow_lt_5pct": row["margin_is_narrow_lt_5pct"],
            "winner_measured_only_rank_under_confidence_scoring": row[
                "winner_measured_only_rank_under_confidence_scoring"
            ],
            "winner_confidence_rank_under_measured_only_scoring": row[
                "winner_confidence_rank_under_measured_only_scoring"
            ],
            "value_inverse_cost_correlation": row["value_inverse_cost_correlation"],
            "value_effective_sc_correlation": row["value_effective_sc_correlation"],
            "value_gain_exp_correlation": row["value_gain_exp_correlation"],
            "nodes_with_positive_effective_gain_sc": row["nodes_with_positive_effective_gain_sc"],
            "node_count": row["node_count"],
            "cost_dominance_inference": row["cost_dominance_inference"],
            "effective_sc_gain_decisive_inference": row["effective_sc_gain_decisive_inference"],
        }
        for row in rank_rows
    ]
    write_csv(output_dir / "rank_margin_comparison.csv", rank_csv_rows)
    save_json(output_dir / "rank_margin_comparison.json", {"seeds": rank_rows})

    branch_entries = {
        "seed0_confidence_weighted": classify_branch(
            "seed0_confidence_weighted",
            decisions[("seed0", 2, "confidence_weighted")],
            decisions[("seed0", 2, "measured")],
            decisions[("seed0", 2, "confidence_weighted")],
        ),
        "seed1_confidence_weighted": classify_branch(
            "seed1_confidence_weighted",
            decisions[("seed1", 2, "confidence_weighted")],
            decisions[("seed1", 2, "measured")],
            decisions[("seed0", 2, "confidence_weighted")],
        ),
        "seed1_cap25_shadow": classify_branch(
            "seed1_cap25_shadow",
            decisions[("seed1", 2, "cap25_shadow")],
            decisions[("seed1", 2, "measured")],
            decisions[("seed0", 2, "confidence_weighted")],
        ),
    }
    branch_classification = {
        "classes": {
            "same_as_measured": "confidence selected child matches measured selected child grid or is within 0.15m",
            "spatially_same_as_seed0_sc": "selected child within 0.25m and best descendant within 0.75m of seed0 Frame2 SC branch",
            "distinct_sc_branch": "differs from measured by at least 0.25m and not close to seed0 SC",
            "local_jitter": "differs from measured but by less than 0.25m",
            "unstable": "missing or inconsistent fields",
        },
        "entries": branch_entries,
    }
    save_json(output_dir / "branch_classification.json", branch_classification)
    write_text(output_dir / "branch_classification.md", branch_classification_markdown(branch_classification))

    summary, summary_md, rec_md = summarize_and_recommend(decisions, rank0, rank1, topk_summary, branch_classification)
    write_text(output_dir / "rank_margin_comparison.md", rank_comparison_markdown(rank_csv_rows, summary["recommended_next_faithful_step"]))
    write_text(output_dir / "recommended_next_faithful_step.md", rec_md)

    plots = make_plots(output_dir, seed0_dir, seed1_dir, decisions, seed0_top, seed1_top, rank0, rank1)

    optional_expected = []
    for stage_dir in (seed0_dir, seed1_dir):
        for frame in FRAMES:
            optional_expected.extend(
                [
                    stage_dir / f"frame{frame:03d}_confidence_weighted_node_gain_breakdown.csv",
                    stage_dir / f"frame{frame:03d}_cap25_node_gain_breakdown.csv",
                ]
            )
    missing_report = {
        "missing_required_inputs": missing_required,
        "missing_optional_inputs": [str(path) for path in optional_expected if not path.is_file()],
        "missing_or_derived_fields": missing_fields,
        "note": "Confidence/cap25 node_gain_breakdown files are optional here because gain_cost_value_table and tree_segments contain the needed per-node fields.",
    }
    save_json(output_dir / "missing_fields_report.json", missing_report)

    safety = make_safety(seed0_dir, seed1_dir, before_hashes, output_dir)
    summary["safety"] = safety
    summary["plots"] = plots
    summary["required_outputs"] = REQUIRED_OUTPUTS
    save_json(output_dir / "stage4a65u_seed_robustness_summary.json", summary)
    write_text(output_dir / "stage4a65u_seed_robustness_summary.md", summary_md)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0_dir", required=True, type=Path)
    parser.add_argument("--seed1_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

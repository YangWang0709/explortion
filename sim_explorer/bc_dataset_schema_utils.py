#!/usr/bin/env python3
"""Shared helpers for Stage 4A-7.0 BC dataset preparation."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
STAGE = "Stage 4A-7.0-BC-dataset-design-and-preparation"
SCHEMA_VERSION = "stage4a70_bc_candidate_set_v1"

FORBIDDEN_EXACT_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "reward",
    "policy_logits",
    "replay_buffer",
    "optimizer",
    "training_state",
    "class_prob",
}

RAW_FEATURE_GROUPS: dict[str, list[str]] = {
    "geometry_action": [
        "candidate_id",
        "candidate_rank",
        "candidate_valid",
        "grid_x",
        "grid_y",
        "grid_z",
        "world_x",
        "world_y",
        "world_z",
        "yaw",
        "delta_x",
        "delta_y",
        "delta_z",
        "distance_xy",
        "yaw_delta",
        "step_id",
        "start_variant_id",
    ],
    "measured_map_exploration": [
        "gain_exp",
        "frontier_count",
        "frontier_adjacent_count",
        "observed_ratio",
        "observed_count",
        "unknown_count",
        "free_count",
        "occupied_count",
        "newly_observed_count_if_available",
        "local_observed_density_if_available",
    ],
    "path_reachability": [
        "path_cost",
        "astar_reachable",
        "astar_path_length_m",
        "astar_num_expanded",
        "reachable_component_count",
        "reachable_frontier_adjacent_count",
        "same_cell_target",
        "repeated_target",
        "outside_bounds_target",
    ],
    "prediction_sc": [
        "source_occ_free",
        "gain_sc",
        "gain_occ",
        "gain_conf",
        "prediction_valid_count",
        "predicted_unmeasured_count",
        "predicted_occupied_count",
        "prediction_density",
        "source_occ_free_norm_per_sample",
    ],
    "uncertainty": [
        "candidate_confidence_mean",
        "candidate_confidence_min",
        "candidate_confidence_p10",
        "candidate_confidence_p50",
        "candidate_confidence_p90",
        "candidate_entropy_mean",
        "candidate_entropy_max",
        "candidate_entropy_p90",
        "candidate_margin_mean",
        "candidate_margin_min",
        "candidate_uncertain_fraction",
        "candidate_uncertain_voxel_count",
        "low_conf_count_0p7",
        "high_entropy_count_0p7",
        "low_margin_count_0p2",
        "uncertainty_composite",
        "uncertainty_composite_norm_per_sample",
    ],
    "scores": [
        "score_measured",
        "score_lambda48",
        "score_confidence_gated",
        "score_uncertainty_bonus_composite_beta8",
        "final_score_primary",
        "score_rank_primary",
        "score_rank_measured",
        "score_rank_lambda48",
        "score_rank_confidence_gated",
    ],
    "quality_flags": [
        "no_valid_candidate",
        "low_cost_artifact",
        "historical_prior_basin",
        "candidate_all_local",
        "high_uncertainty_selection",
        "low_confidence_selection",
        "low_margin_selection",
        "formula_dominated_by_uncertainty",
        "prediction_writeback",
        "uncertainty_writeback",
        "prediction_traversability_use",
        "uncertainty_traversability_use",
        "prediction_collision_use",
        "uncertainty_collision_use",
        "prediction_ray_blocking_use",
        "uncertainty_ray_blocking_use",
        "prediction_candidate_validity_use",
        "uncertainty_candidate_validity_use",
        "target_ground_truth_use",
        "future_observed_scoring_use",
    ],
}

RAW_FEATURE_NAMES = [
    name
    for group_names in RAW_FEATURE_GROUPS.values()
    for name in group_names
]

MODEL_FEATURE_NAMES = [
    "gain_exp_norm_per_sample",
    "inverse_path_cost",
    "source_occ_free_norm_per_sample",
    "candidate_confidence_mean",
    "candidate_entropy_mean",
    "candidate_margin_mean",
    "candidate_uncertain_fraction",
    "uncertainty_composite",
    "distance_xy",
    "yaw_delta",
    "astar_reachable",
    "same_cell_target",
    "repeated_target",
    "candidate_all_local",
    "low_cost_artifact",
    "historical_prior_basin",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(jsonable(value), sort_keys=True)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        text = json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(jsonable(value))
        text = text.replace("\n", " ")
        if len(text) > 1800:
            text = text[:1800] + "..."
        lines.append(f"| {key} | `{text}` |")
    return "\n".join(lines)


def markdown_rows(title: str, rows: list[dict[str, Any]], limit: int = 40) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        return "\n".join(lines + ["No rows."])
    fields = list(rows[0].keys())
    lines.extend(["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"])
    for row in rows[:limit]:
        values = []
        for field in fields:
            text = str(csv_value(row.get(field))).replace("\n", " ")
            if len(text) > 180:
                text = text[:180] + "..."
            values.append(f"`{text}`")
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > limit:
        lines.append("")
        lines.append(f"Showing {limit} of {len(rows)} rows.")
    return "\n".join(lines)


def save_report_pair(output_dir: Path, stem: str, data: Any, title: str) -> None:
    save_json(output_dir / f"{stem}.json", data)
    if isinstance(data, list):
        write_text(output_dir / f"{stem}.md", markdown_rows(title, data))
    elif isinstance(data, dict):
        write_text(output_dir / f"{stem}.md", markdown_table(title, data))
    else:
        write_text(output_dir / f"{stem}.md", f"# {title}\n\n{data}")


def sha256_file(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status_text() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return result.stdout


def git_large_artifact_policy() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    tracked = result.stdout.splitlines()
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/")
    forbidden_suffixes = (".npy", ".npz", ".png", ".mp4", ".usd", ".pth", ".tar")
    offenders = [
        path
        for path in tracked
        if path.startswith(forbidden_prefixes) or path.lower().endswith(forbidden_suffixes)
    ]
    return {
        "passed": result.returncode == 0 and not offenders,
        "offenders": offenders[:80],
        "tracked_file_count": len(tracked),
    }


def parse_literal(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        try:
            return json.loads(str(value))
        except Exception:
            return default


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_minmax(values: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    finite = np.isfinite(values)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    if not np.any(finite):
        return out
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if hi <= lo + 1.0e-12:
        return out
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def normalized_yaw_delta(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return float(abs((a - b + math.pi) % (2 * math.pi) - math.pi))


def summarize(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def rank_desc(scores: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(scores)
    ranks = np.full(scores.shape, np.nan, dtype=np.float32)
    order = np.argsort(-scores[valid])
    valid_indices = np.flatnonzero(valid)
    for rank, local_index in enumerate(order, start=1):
        ranks[valid_indices[local_index]] = float(rank)
    return ranks


def check_forbidden_names(names: list[str]) -> dict[str, Any]:
    lower = {str(name).lower() for name in names}
    hits = sorted(lower & FORBIDDEN_EXACT_KEYS)
    return {"passed": not hits, "forbidden_exact_hits": hits}


def normalization_stats(features: np.ndarray, valid_mask: np.ndarray) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    stats = {"mean": [], "std": [], "min": [], "max": []}
    for dim in range(features.shape[-1]):
        values = features[:, :, dim][valid]
        values = values[np.isfinite(values)]
        if values.size == 0:
            stats["mean"].append(0.0)
            stats["std"].append(1.0)
            stats["min"].append(0.0)
            stats["max"].append(0.0)
        else:
            stats["mean"].append(float(np.mean(values)))
            stats["std"].append(float(max(np.std(values), 1.0e-6)))
            stats["min"].append(float(np.min(values)))
            stats["max"].append(float(np.max(values)))
    return {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}

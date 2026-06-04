#!/usr/bin/env python3
"""Stage 4A-6.5a offline candidate rank sensitivity diagnosis.

This script reads existing rollout/ablation outputs only. It does not launch
Isaac, rerun rollouts, alter scoring, write predictions, or modify observed
state/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


GATED_CONFIGS = [
    "occupied_only_occ07",
    "occupied_only_occ08",
    "occupied_margin_occ06_w05",
    "confidence_weighted_conf05_cap30",
]

CORE_FEATURES = [
    "final_score",
    "gain_exp",
    "raw_gain_sc",
    "gain_sc",
    "effective_gain_sc",
    "weighted_gain_sc",
    "gain_hybrid",
    "gain_hybrid_weighted",
    "path_cost",
    "utility_exp",
    "utility_sc",
    "utility_hybrid",
    "utility_hybrid_weighted",
    "utility_effective_sc",
    "utility_hybrid_effective",
]

REQUIRED_FIELD_ALIASES: dict[str, list[str]] = {
    "expert_action": ["expert_action"],
    "candidate_ids": ["candidate_ids", "candidate_id", "top_candidate_ids", "candidate_indices"],
    "candidate_positions": ["candidate_positions_grid", "candidate_positions_world"],
    "valid_mask": ["valid_mask"],
    "final_score": ["final_score", "expert_scores"],
    "gain_exp": ["gain_exp"],
    "raw_gain_sc": ["raw_gain_sc", "gain_sc"],
    "effective_gain_sc": ["effective_gain_sc"],
    "weighted_gain_sc": ["weighted_gain_sc"],
    "gain_hybrid": ["gain_hybrid"],
    "gain_hybrid_weighted": ["gain_hybrid_weighted"],
    "path_cost": ["path_cost"],
    "utility_or_score_fields": ["utility_exp", "utility_sc", "utility_hybrid", "final_score", "expert_scores"],
    "feature_names": ["feature_names"],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _scalar(data: dict[str, Any], key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_array(values: list[Any] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def mean_or_none(values: list[Any] | np.ndarray) -> float | None:
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    return float(arr.mean())


def median_or_none(values: list[Any] | np.ndarray) -> float | None:
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def min_or_none(values: list[Any] | np.ndarray) -> float | None:
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    return float(arr.min())


def max_or_none(values: list[Any] | np.ndarray) -> float | None:
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    return float(arr.max())


def pearson_corr(x_values: list[Any] | np.ndarray, y_values: list[Any] | np.ndarray) -> float | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(values: np.ndarray, descending: bool) -> np.ndarray:
    """Return 1-based average ranks. Non-finite values get NaN."""

    arr = np.asarray(values, dtype=np.float64)
    ranks = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return ranks
    idx = np.flatnonzero(finite)
    vals = arr[idx]
    order = np.argsort(-vals if descending else vals, kind="mergesort")
    sorted_idx = idx[order]
    sorted_vals = vals[order]
    start = 0
    while start < len(sorted_idx):
        end = start + 1
        while end < len(sorted_idx) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[sorted_idx[start:end]] = avg_rank
        start = end
    return ranks


def spearman_corr(x_values: list[Any] | np.ndarray, y_values: list[Any] | np.ndarray) -> float | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return None
    rx = rank_values(x[mask], descending=False)
    ry = rank_values(y[mask], descending=False)
    return pearson_corr(rx, ry)


def rank_desc(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    arr[~valid_mask] = np.nan
    return rank_values(arr, descending=True)


def rank_asc(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    arr[~valid_mask] = np.nan
    return rank_values(arr, descending=False)


def top_indices(values: np.ndarray, valid: np.ndarray, k: int) -> list[int]:
    arr = np.asarray(values, dtype=np.float64).copy()
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    if not np.any(valid_mask):
        return []
    arr[~valid_mask] = -np.inf
    order = np.argsort(-arr, kind="mergesort")
    return [int(i) for i in order[: min(k, int(np.count_nonzero(valid_mask)))]]


def _position_key(grid: Any, world: Any) -> str:
    grid_arr = np.asarray(grid, dtype=np.float64).reshape(-1) if grid is not None else np.asarray([])
    if grid_arr.size >= 3 and np.all(np.isfinite(grid_arr[:3])):
        return "grid:" + ",".join(str(int(round(v))) for v in grid_arr[:3])
    world_arr = np.asarray(world, dtype=np.float64).reshape(-1) if world is not None else np.asarray([])
    if world_arr.size >= 3 and np.all(np.isfinite(world_arr[:3])):
        return "world:" + ",".join(f"{float(v):.4f}" for v in world_arr[:3])
    return "missing"


def _list_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return [value.item()]
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _find_candidate_ids(npz: dict[str, Any], n: int) -> tuple[np.ndarray | None, str | None]:
    for key in ("candidate_ids", "candidate_id", "top_candidate_ids", "candidate_indices"):
        if key not in npz:
            continue
        arr = np.asarray(npz[key]).reshape(-1)
        if arr.size == n:
            return arr, key
    return None, None


def _feature_lookup(feature_names: list[str], features: np.ndarray) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    for idx, name in enumerate(feature_names):
        if idx < features.shape[1]:
            lookup[str(name)] = features[:, idx].astype(np.float64, copy=False)
    return lookup


def _component_values(feature_map: dict[str, np.ndarray], expert_scores: np.ndarray, n: int) -> dict[str, np.ndarray]:
    def value(name: str, fallback: str | None = None) -> np.ndarray:
        if name in feature_map:
            return feature_map[name]
        if fallback and fallback in feature_map:
            return feature_map[fallback]
        return np.full(n, np.nan, dtype=np.float64)

    final_score = value("final_score")
    if not np.any(np.isfinite(final_score)) and expert_scores.size == n:
        final_score = expert_scores.astype(np.float64, copy=False)
    path_cost = value("path_cost")
    inverse_path_cost = np.full(n, np.nan, dtype=np.float64)
    finite_cost = np.isfinite(path_cost) & (path_cost > 1e-12)
    inverse_path_cost[finite_cost] = 1.0 / path_cost[finite_cost]
    return {
        "final_score": final_score,
        "gain_exp": value("gain_exp"),
        "gain_sc": value("gain_sc"),
        "raw_gain_sc": value("raw_gain_sc", "gain_sc"),
        "effective_gain_sc": value("effective_gain_sc"),
        "weighted_gain_sc": value("weighted_gain_sc"),
        "gain_hybrid": value("gain_hybrid"),
        "gain_hybrid_weighted": value("gain_hybrid_weighted"),
        "path_cost": path_cost,
        "inverse_path_cost": inverse_path_cost,
    }


def _has_alias(top_keys: set[str], feature_names: set[str], aliases: list[str]) -> bool:
    for alias in aliases:
        if alias in top_keys or alias in feature_names:
            return True
    return False


def _missing_fields(
    npz_keys: set[str],
    feature_names: set[str],
    candidate_ids_key: str | None,
) -> dict[str, str]:
    missing: dict[str, str] = {}
    for field, aliases in REQUIRED_FIELD_ALIASES.items():
        if field == "candidate_ids":
            if candidate_ids_key is None:
                missing[field] = (
                    "missing per-candidate ids; selected best_candidate_id is read from transitions when present"
                )
            continue
        if field == "candidate_positions":
            has_positions = "candidate_positions_grid" in npz_keys or "candidate_positions_world" in npz_keys
            if not has_positions:
                missing[field] = "missing candidate_positions_grid and candidate_positions_world"
            continue
        if not _has_alias(npz_keys, feature_names, aliases):
            missing[field] = "missing aliases: " + ",".join(aliases)
    return missing


def discover_episode_dirs(
    empty_episode_dir: Path,
    raw_sc_episode_dir: Path,
    gating_root: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    configs: dict[str, Path] = {
        "empty_baseline": empty_episode_dir,
        "fixed_raw_sc": raw_sc_episode_dir,
    }
    missing: dict[str, str] = {}
    ablation_episode_root = gating_root / "ablation" / "episodes"
    for name in GATED_CONFIGS:
        exact = ablation_episode_root / f"medium_three_rooms_seed0_start_room_a_{name}"
        if exact.exists():
            configs[name] = exact
            continue
        matches = sorted(ablation_episode_root.glob(f"*_{name}")) if ablation_episode_root.exists() else []
        if matches:
            configs[name] = matches[0]
        else:
            missing[name] = f"episode dir not found under {ablation_episode_root}"
    return configs, missing


def load_config_steps(
    config_name: str,
    episode_dir: Path,
    max_steps: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    transitions = load_jsonl(episode_dir / "transitions.jsonl")
    transition_by_step = {int(row.get("step", idx)): row for idx, row in enumerate(transitions)}
    episode_summary = load_json(episode_dir / "episode_summary.json")
    loaded: dict[int, dict[str, Any]] = {}
    for step in range(max_steps):
        path = episode_dir / f"step_{step:03d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as raw:
            npz = {key: raw[key].copy() for key in raw.files}
        feature_names = [str(x) for x in np.asarray(npz.get("feature_names", []), dtype=str).reshape(-1).tolist()]
        features = np.asarray(npz.get("candidate_features", np.empty((0, 0), dtype=np.float32)), dtype=np.float64)
        if features.ndim != 2:
            features = np.empty((0, 0), dtype=np.float64)
        n = int(features.shape[0])
        valid = np.asarray(npz.get("valid_mask", np.ones(n, dtype=bool)), dtype=bool).reshape(-1)
        if valid.size != n:
            valid = np.ones(n, dtype=bool)
        expert_scores = np.asarray(npz.get("expert_scores", np.full(n, np.nan)), dtype=np.float64).reshape(-1)
        if expert_scores.size != n:
            expert_scores = np.full(n, np.nan, dtype=np.float64)
        feature_map = _feature_lookup(feature_names, features)
        components = _component_values(feature_map, expert_scores, n)
        candidate_ids, candidate_ids_key = _find_candidate_ids(npz, n)
        transition = transition_by_step.get(step, {})
        expert_action_raw = _scalar(npz, "expert_action", transition.get("expert_action"))
        expert_action = int(expert_action_raw) if expert_action_raw is not None else None
        if expert_action is None or not (0 <= expert_action < n):
            top = top_indices(components["final_score"], valid, 1)
            selected_index = top[0] if top else None
        else:
            selected_index = expert_action
        positions_grid = np.asarray(npz.get("candidate_positions_grid", np.full((n, 3), np.nan)), dtype=np.float64)
        positions_world = np.asarray(npz.get("candidate_positions_world", np.full((n, 3), np.nan)), dtype=np.float64)
        if positions_grid.shape != (n, 3):
            positions_grid = np.full((n, 3), np.nan, dtype=np.float64)
        if positions_world.shape != (n, 3):
            positions_world = np.full((n, 3), np.nan, dtype=np.float64)
        selected_transition_grid = transition.get("selected_next_pose_grid")
        selected_transition_world = transition.get("selected_next_pose_world")
        loaded[step] = {
            "config": config_name,
            "episode_dir": episode_dir,
            "npz_path": path,
            "npz_keys": set(npz.keys()),
            "transition": transition,
            "episode_summary": episode_summary,
            "feature_names": feature_names,
            "feature_map": feature_map,
            "features": features,
            "components": components,
            "candidate_ids": candidate_ids,
            "candidate_ids_key": candidate_ids_key,
            "positions_grid": positions_grid,
            "positions_world": positions_world,
            "valid_mask": valid,
            "expert_scores": expert_scores,
            "expert_action": expert_action,
            "selected_index": selected_index,
            "selected_candidate_id": transition.get("best_candidate_id"),
            "selected_transition_grid": selected_transition_grid,
            "selected_transition_world": selected_transition_world,
            "selected_transition_yaw": transition.get("selected_next_yaw"),
        }
    return loaded, transitions, episode_summary


def build_candidate_rows(step_data: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config, steps in step_data.items():
        for step, data in sorted(steps.items()):
            components = data["components"]
            n = int(data["features"].shape[0])
            valid = data["valid_mask"]
            final_score = components["final_score"]
            ranks_final = rank_desc(final_score, valid)
            ranks_gain_exp = rank_desc(components["gain_exp"], valid)
            ranks_eff = rank_desc(components["effective_gain_sc"], valid)
            ranks_weighted = rank_desc(components["weighted_gain_sc"], valid)
            ranks_raw = rank_desc(components["raw_gain_sc"], valid)
            ranks_low_cost = rank_asc(components["path_cost"], valid)
            top1 = set(top_indices(final_score, valid, 1))
            top5 = set(top_indices(final_score, valid, 5))
            top16 = set(top_indices(final_score, valid, 16))
            selected_index = data["selected_index"]
            for idx in range(n):
                candidate_id = None
                if data["candidate_ids"] is not None:
                    candidate_id = _scalar({"candidate_id": data["candidate_ids"][idx]}, "candidate_id")
                row: dict[str, Any] = {
                    "config": config,
                    "step": int(step),
                    "episode_dir": str(data["episode_dir"]),
                    "npz_path": str(data["npz_path"]),
                    "candidate_row": int(idx),
                    "candidate_id": candidate_id,
                    "selected_candidate_id": data["selected_candidate_id"],
                    "expert_action": data["expert_action"],
                    "is_selected": bool(selected_index == idx),
                    "valid": bool(valid[idx]),
                    "rank_final_score": _as_float(ranks_final[idx]),
                    "rank_gain_exp": _as_float(ranks_gain_exp[idx]),
                    "rank_raw_gain_sc": _as_float(ranks_raw[idx]),
                    "rank_effective_gain_sc": _as_float(ranks_eff[idx]),
                    "rank_weighted_gain_sc": _as_float(ranks_weighted[idx]),
                    "rank_low_path_cost": _as_float(ranks_low_cost[idx]),
                    "top1": bool(idx in top1),
                    "top5": bool(idx in top5),
                    "top16": bool(idx in top16),
                    "candidate_grid": data["positions_grid"][idx].tolist(),
                    "candidate_world": data["positions_world"][idx].tolist(),
                    "candidate_key": _position_key(data["positions_grid"][idx], data["positions_world"][idx]),
                    "selected_transition_grid": data["selected_transition_grid"],
                    "selected_transition_world": data["selected_transition_world"],
                    "selected_transition_yaw": data["selected_transition_yaw"],
                    "feature_names": data["feature_names"],
                }
                for name in CORE_FEATURES:
                    if name in components:
                        value = components[name][idx]
                    elif name in data["feature_map"]:
                        value = data["feature_map"][name][idx]
                    else:
                        value = np.nan
                    row[name] = _as_float(value)
                for feature_name, values in data["feature_map"].items():
                    if feature_name not in row:
                        row[feature_name] = _as_float(values[idx])
                rows.append(row)
    return rows


def build_selected_summary(step_data: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config, steps in step_data.items():
        for step, data in sorted(steps.items()):
            selected_index = data["selected_index"]
            components = data["components"]
            final_score = components["final_score"]
            valid = data["valid_mask"]
            order = top_indices(final_score, valid, 16)
            runner_index = order[1] if len(order) > 1 else None
            selected_score = None
            if selected_index is not None and 0 <= selected_index < final_score.size:
                selected_score = _as_float(final_score[selected_index])
            runner_score = _as_float(final_score[runner_index]) if runner_index is not None else None
            margin = None
            relative_margin = None
            if selected_score is not None and runner_score is not None:
                margin = float(selected_score - runner_score)
                denom = max(abs(runner_score), 1e-12)
                relative_margin = float(margin / denom)
            row: dict[str, Any] = {
                "config": config,
                "step": int(step),
                "selected_row": selected_index,
                "selected_candidate_id": data["selected_candidate_id"],
                "expert_action": data["expert_action"],
                "selected_transition_grid": data["selected_transition_grid"],
                "selected_transition_world": data["selected_transition_world"],
                "selected_transition_yaw": data["selected_transition_yaw"],
                "runner_up_row": runner_index,
                "selected_score": selected_score,
                "runner_up_score": runner_score,
                "selected_vs_runner_up_margin": margin,
                "selected_vs_runner_up_relative_margin": relative_margin,
                "top16_valid_count": len(order),
            }
            for key in [
                "gain_exp",
                "raw_gain_sc",
                "gain_sc",
                "effective_gain_sc",
                "weighted_gain_sc",
                "gain_hybrid",
                "gain_hybrid_weighted",
                "path_cost",
                "inverse_path_cost",
            ]:
                vals = components[key]
                row[f"selected_{key}"] = _as_float(vals[selected_index]) if selected_index is not None else None
                row[f"runner_up_{key}"] = _as_float(vals[runner_index]) if runner_index is not None else None
            if selected_index is not None:
                row["selected_rank_low_path_cost"] = _as_float(rank_asc(components["path_cost"], valid)[selected_index])
                row["selected_rank_gain_exp"] = _as_float(rank_desc(components["gain_exp"], valid)[selected_index])
                row["selected_rank_effective_gain_sc"] = _as_float(
                    rank_desc(components["effective_gain_sc"], valid)[selected_index]
                )
                row["selected_rank_raw_gain_sc"] = _as_float(rank_desc(components["raw_gain_sc"], valid)[selected_index])
            rows.append(row)
    return rows


def build_rank_correlations(step_data: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    components = ["gain_exp", "raw_gain_sc", "effective_gain_sc", "weighted_gain_sc", "inverse_path_cost", "path_cost"]
    aggregate: dict[tuple[str, str], dict[str, list[float]]] = {}
    all_aggregate: dict[str, dict[str, list[float]]] = {}
    for config, steps in step_data.items():
        for step, data in sorted(steps.items()):
            final_score = data["components"]["final_score"]
            valid = np.asarray(data["valid_mask"], dtype=bool)
            for component in components:
                x = data["components"][component]
                mask = valid & np.isfinite(final_score) & np.isfinite(x)
                pcorr = pearson_corr(final_score[mask], x[mask])
                scorr = spearman_corr(final_score[mask], x[mask])
                rows.append(
                    {
                        "scope": "step",
                        "config": config,
                        "step": int(step),
                        "component": component,
                        "valid_count": int(np.count_nonzero(mask)),
                        "pearson_final_score": pcorr,
                        "spearman_final_rank": scorr,
                    }
                )
                key = (config, component)
                aggregate.setdefault(key, {"final": [], "component": []})
                aggregate[key]["final"].extend([float(v) for v in final_score[mask]])
                aggregate[key]["component"].extend([float(v) for v in x[mask]])
                all_aggregate.setdefault(component, {"final": [], "component": []})
                all_aggregate[component]["final"].extend([float(v) for v in final_score[mask]])
                all_aggregate[component]["component"].extend([float(v) for v in x[mask]])
    for (config, component), values in sorted(aggregate.items()):
        rows.append(
            {
                "scope": "config_aggregate",
                "config": config,
                "step": "",
                "component": component,
                "valid_count": len(values["final"]),
                "pearson_final_score": pearson_corr(values["final"], values["component"]),
                "spearman_final_rank": spearman_corr(values["final"], values["component"]),
            }
        )
    for component, values in sorted(all_aggregate.items()):
        rows.append(
            {
                "scope": "all_configs_aggregate",
                "config": "all",
                "step": "",
                "component": component,
                "valid_count": len(values["final"]),
                "pearson_final_score": pearson_corr(values["final"], values["component"]),
                "spearman_final_rank": spearman_corr(values["final"], values["component"]),
            }
        )
    return rows


def _top_keys(data: dict[str, Any], k: int) -> list[str]:
    final_score = data["components"]["final_score"]
    valid = data["valid_mask"]
    indices = top_indices(final_score, valid, k)
    return [_position_key(data["positions_grid"][idx], data["positions_world"][idx]) for idx in indices]


def build_topk_overlap(step_data: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = list(step_data.keys())
    steps = sorted({step for data in step_data.values() for step in data})
    for step in steps:
        present = [config for config in configs if step in step_data[config]]
        for i, config_a in enumerate(present):
            for config_b in present[i + 1 :]:
                a = step_data[config_a][step]
                b = step_data[config_b][step]
                row: dict[str, Any] = {
                    "step": int(step),
                    "config_a": config_a,
                    "config_b": config_b,
                    "selected_candidate_id_a": a["selected_candidate_id"],
                    "selected_candidate_id_b": b["selected_candidate_id"],
                    "selected_candidate_id_match": (
                        a["selected_candidate_id"] is not None
                        and b["selected_candidate_id"] is not None
                        and a["selected_candidate_id"] == b["selected_candidate_id"]
                    ),
                    "selected_position_key_a": _position_key(a["selected_transition_grid"], a["selected_transition_world"]),
                    "selected_position_key_b": _position_key(b["selected_transition_grid"], b["selected_transition_world"]),
                }
                row["selected_position_match"] = row["selected_position_key_a"] == row["selected_position_key_b"]
                for k in (1, 5, 16):
                    keys_a = _top_keys(a, k)
                    keys_b = _top_keys(b, k)
                    set_a = set(keys_a)
                    set_b = set(keys_b)
                    intersection = len(set_a & set_b)
                    union = len(set_a | set_b)
                    row[f"top{k}_intersection"] = int(intersection)
                    row[f"top{k}_jaccard"] = float(intersection / union) if union else None
                    row[f"top{k}_exact_order_match"] = keys_a == keys_b
                    row[f"top{k}_keys_a"] = keys_a
                    row[f"top{k}_keys_b"] = keys_b
                rows.append(row)
    return rows


def build_missing_report(
    step_data: dict[str, dict[int, dict[str, Any]]],
    missing_configs: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "missing_configs": missing_configs,
        "by_config_step": {},
        "notes": [
            "candidate_ids means per-candidate ids for all top-N rows; selected best_candidate_id is available from transitions when logged.",
            "Missing fields are reported instead of raising errors.",
        ],
    }
    for config, steps in step_data.items():
        report["by_config_step"][config] = {}
        for step, data in sorted(steps.items()):
            missing = _missing_fields(
                data["npz_keys"],
                set(data["feature_names"]),
                data["candidate_ids_key"],
            )
            report["by_config_step"][config][str(step)] = {
                "missing": missing,
                "feature_names": data["feature_names"],
                "npz_path": str(data["npz_path"]),
            }
    return report


def _config_step_rows(rows: list[dict[str, Any]], config: str | None = None) -> list[dict[str, Any]]:
    if config is None:
        return rows
    return [row for row in rows if row.get("config") == config]


def _aggregate_corr(rank_rows: list[dict[str, Any]], config: str | None, component: str) -> float | None:
    target_scope = "config_aggregate" if config else "all_configs_aggregate"
    target_config = config if config else "all"
    for row in rank_rows:
        if row["scope"] == target_scope and row["config"] == target_config and row["component"] == component:
            return row.get("pearson_final_score")
    return None


def _aggregate_spearman(rank_rows: list[dict[str, Any]], config: str | None, component: str) -> float | None:
    target_scope = "config_aggregate" if config else "all_configs_aggregate"
    target_config = config if config else "all"
    for row in rank_rows:
        if row["scope"] == target_scope and row["config"] == target_config and row["component"] == component:
            return row.get("spearman_final_rank")
    return None


def _component_best_explains(rank_rows: list[dict[str, Any]], config: str | None) -> str | None:
    candidates = ["gain_exp", "effective_gain_sc", "inverse_path_cost"]
    scored: list[tuple[float, str]] = []
    for component in candidates:
        corr = _aggregate_corr(rank_rows, config, component)
        if corr is not None and math.isfinite(float(corr)):
            scored.append((abs(float(corr)), component))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def _all_same(values: list[Any]) -> bool | None:
    clean = [json.dumps(v, sort_keys=True, default=_json_default) for v in values if v is not None]
    if not clean:
        return None
    return len(set(clean)) == 1


def summarize_findings(
    step_data: dict[str, dict[int, dict[str, Any]]],
    selected_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    missing_report: dict[str, Any],
) -> dict[str, Any]:
    loaded_configs = list(step_data.keys())
    steps_analyzed = sorted({int(step) for steps in step_data.values() for step in steps})
    gated_loaded = [cfg for cfg in GATED_CONFIGS if cfg in step_data]
    gated_id_stability: dict[str, Any] = {}
    gated_pos_stability: dict[str, Any] = {}
    for step in steps_analyzed:
        ids = [step_data[cfg][step]["selected_candidate_id"] for cfg in gated_loaded if step in step_data[cfg]]
        positions = [
            _position_key(step_data[cfg][step]["selected_transition_grid"], step_data[cfg][step]["selected_transition_world"])
            for cfg in gated_loaded
            if step in step_data[cfg]
        ]
        gated_id_stability[str(step)] = _all_same(ids)
        gated_pos_stability[str(step)] = _all_same(positions)
    gated_ids_all_identical = _all_same(list(gated_id_stability.values())) if gated_id_stability else None
    if gated_id_stability:
        gated_ids_all_identical = all(v is True for v in gated_id_stability.values())
    gated_positions_all_identical = all(v is True for v in gated_pos_stability.values()) if gated_pos_stability else None

    selected_rank_cost = [row.get("selected_rank_low_path_cost") for row in selected_rows if row.get("selected_rank_low_path_cost") is not None]
    selected_rank_gain = [row.get("selected_rank_gain_exp") for row in selected_rows if row.get("selected_rank_gain_exp") is not None]
    selected_rank_eff = [
        row.get("selected_rank_effective_gain_sc")
        for row in selected_rows
        if row.get("selected_rank_effective_gain_sc") is not None
    ]

    eff_std_rows: list[float] = []
    eff_range_rows: list[float] = []
    high_eff_lost_cases: list[dict[str, Any]] = []
    rank_changed_vs_raw: list[dict[str, Any]] = []
    for cfg in gated_loaded:
        for step, data in sorted(step_data[cfg].items()):
            eff = data["components"]["effective_gain_sc"]
            valid = data["valid_mask"] & np.isfinite(eff)
            if np.count_nonzero(valid) >= 2:
                vals = eff[valid]
                eff_std_rows.append(float(np.std(vals)))
                eff_range_rows.append(float(np.max(vals) - np.min(vals)))
            if np.count_nonzero(valid) >= 1:
                max_eff_candidates = np.flatnonzero(eff == np.nanmax(np.where(valid, eff, np.nan)))
                max_eff_idx = int(max_eff_candidates[0])
                selected_idx = data["selected_index"]
                cost_rank = rank_asc(data["components"]["path_cost"], data["valid_mask"])
                final_rank = rank_desc(data["components"]["final_score"], data["valid_mask"])
                if selected_idx is not None and max_eff_idx != selected_idx:
                    high_eff_lost_cases.append(
                        {
                            "config": cfg,
                            "step": int(step),
                            "max_effective_gain_sc_row": max_eff_idx,
                            "max_effective_gain_sc": _as_float(eff[max_eff_idx]),
                            "max_effective_gain_sc_final_rank": _as_float(final_rank[max_eff_idx]),
                            "max_effective_gain_sc_path_cost_rank": _as_float(cost_rank[max_eff_idx]),
                            "selected_row": selected_idx,
                            "selected_path_cost_rank": _as_float(cost_rank[selected_idx]),
                        }
                    )
            if "fixed_raw_sc" in step_data and step in step_data["fixed_raw_sc"]:
                raw_keys = _top_keys(step_data["fixed_raw_sc"][step], 16)
                gated_keys = _top_keys(data, 16)
                same_order = raw_keys == gated_keys
                same_top5 = set(raw_keys[:5]) == set(gated_keys[:5])
                rank_changed_vs_raw.append(
                    {
                        "config": cfg,
                        "step": int(step),
                        "top16_exact_order_match_vs_raw": same_order,
                        "top5_set_match_vs_raw": same_top5,
                    }
                )

    raw_pair_rows = [
        row
        for row in topk_rows
        if {row["config_a"], row["config_b"]} & {"fixed_raw_sc"}
        and ({row["config_a"], row["config_b"]} - {"fixed_raw_sc"}) <= set(gated_loaded)
    ]
    gated_pair_rows = [
        row for row in topk_rows if row["config_a"] in gated_loaded and row["config_b"] in gated_loaded
    ]

    margins = [row.get("selected_vs_runner_up_margin") for row in selected_rows if row.get("selected_vs_runner_up_margin") is not None]
    rel_margins = [
        row.get("selected_vs_runner_up_relative_margin")
        for row in selected_rows
        if row.get("selected_vs_runner_up_relative_margin") is not None
    ]

    inv_corr = _aggregate_corr(rank_rows, None, "inverse_path_cost")
    gain_corr = _aggregate_corr(rank_rows, None, "gain_exp")
    eff_corr = _aggregate_corr(rank_rows, None, "effective_gain_sc")
    gain_eff_pairs_final: list[float] = []
    gain_eff_pairs_eff: list[float] = []
    for cfg in gated_loaded:
        for data in step_data[cfg].values():
            gain = data["components"]["gain_exp"]
            eff = data["components"]["effective_gain_sc"]
            valid = data["valid_mask"] & np.isfinite(gain) & np.isfinite(eff)
            gain_eff_pairs_final.extend([float(v) for v in gain[valid]])
            gain_eff_pairs_eff.extend([float(v) for v in eff[valid]])
    gain_exp_vs_effective_corr = pearson_corr(gain_eff_pairs_final, gain_eff_pairs_eff)

    path_cost_dominates = False
    if inv_corr is not None and gain_corr is not None:
        path_cost_dominates = abs(float(inv_corr)) >= abs(float(gain_corr))
    selected_cost_mean = mean_or_none(selected_rank_cost)
    selected_gain_mean = mean_or_none(selected_rank_gain)
    selected_eff_mean = mean_or_none(selected_rank_eff)
    if selected_cost_mean is not None and selected_gain_mean is not None:
        path_cost_dominates = path_cost_dominates or (selected_cost_mean <= 2.0 and selected_gain_mean > selected_cost_mean)

    gain_exp_dominates = False
    if gain_corr is not None and inv_corr is not None:
        gain_exp_dominates = abs(float(gain_corr)) > abs(float(inv_corr))

    eff_has_variance = bool(mean_or_none(eff_range_rows) is not None and float(mean_or_none(eff_range_rows) or 0.0) > 1e-6)
    rank_reduced_changed = any(not row["top16_exact_order_match_vs_raw"] for row in rank_changed_vs_raw)
    high_eff_loses_due_to_cost = any(
        case.get("max_effective_gain_sc_path_cost_rank") is not None
        and case.get("selected_path_cost_rank") is not None
        and float(case["max_effective_gain_sc_path_cost_rank"]) > float(case["selected_path_cost_rank"])
        for case in high_eff_lost_cases
    )
    top1_stable = bool(raw_pair_rows and all(row["top1_jaccard"] == 1.0 for row in raw_pair_rows))
    top5_mean = mean_or_none([row["top5_jaccard"] for row in raw_pair_rows])
    top16_mean = mean_or_none([row["top16_jaccard"] for row in raw_pair_rows])

    missing_counts: dict[str, int] = {}
    for cfg_steps in missing_report.get("by_config_step", {}).values():
        for step_info in cfg_steps.values():
            for field in step_info.get("missing", {}):
                missing_counts[field] = missing_counts.get(field, 0) + 1

    diagnosis_parts: list[str] = []
    if path_cost_dominates:
        diagnosis_parts.append("final_score is most aligned with inverse path_cost and selected candidates have very low path-cost ranks")
    if gain_exp_dominates:
        diagnosis_parts.append("gain_exp also strongly tracks final_score")
    if rank_reduced_changed and top1_stable:
        diagnosis_parts.append("gating changes lower ranks but not top-1")
    if eff_has_variance:
        diagnosis_parts.append("effective_gain_sc has candidate variance, but it is not enough to overcome cost/gain_exp ordering")
    if high_eff_loses_due_to_cost:
        diagnosis_parts.append("some high effective_gain_sc candidates lose with worse path-cost rank")
    if not diagnosis_parts:
        diagnosis_parts.append("available candidate fields are insufficient for a strong component-level diagnosis")

    if path_cost_dominates:
        next_task = "offline counterfactual score analysis"
    elif "candidate_ids" in missing_counts:
        next_task = "improving candidate logging only"
    elif high_eff_loses_due_to_cost:
        next_task = "spatial visualization only"
    else:
        next_task = "offline counterfactual score analysis"

    return {
        "loaded_configs": loaded_configs,
        "gated_configs_loaded": gated_loaded,
        "steps_analyzed": steps_analyzed,
        "selected_candidate_ids_identical_across_gated_configs_by_step": gated_id_stability,
        "selected_candidate_ids_identical_across_all_gated_steps": gated_ids_all_identical,
        "selected_positions_identical_across_gated_configs_by_step": gated_pos_stability,
        "selected_positions_identical_across_all_gated_steps": gated_positions_all_identical,
        "selected_action_mostly_determined_by_path_cost": path_cost_dominates,
        "selected_action_mostly_determined_by_gain_exp": gain_exp_dominates,
        "effective_gain_sc_has_candidate_variance": eff_has_variance,
        "effective_gain_sc_std_mean": mean_or_none(eff_std_rows),
        "effective_gain_sc_range_mean": mean_or_none(eff_range_rows),
        "reducing_effective_gain_sc_changes_candidate_rank": rank_reduced_changed,
        "rank_changed_vs_raw": rank_changed_vs_raw,
        "high_effective_gain_sc_candidates_lose_due_to_path_cost": high_eff_loses_due_to_cost,
        "high_effective_gain_sc_lost_cases": high_eff_lost_cases[:20],
        "top1_stable_vs_raw_sc": top1_stable,
        "top5_jaccard_mean_vs_raw_sc": top5_mean,
        "top16_jaccard_mean_vs_raw_sc": top16_mean,
        "top1_overlap_mean_gated_pairwise": mean_or_none([row["top1_jaccard"] for row in gated_pair_rows]),
        "top5_overlap_mean_gated_pairwise": mean_or_none([row["top5_jaccard"] for row in gated_pair_rows]),
        "top16_overlap_mean_gated_pairwise": mean_or_none([row["top16_jaccard"] for row in gated_pair_rows]),
        "selected_vs_runner_up_margin_mean": mean_or_none(margins),
        "selected_vs_runner_up_margin_min": min_or_none(margins),
        "selected_vs_runner_up_margin_max": max_or_none(margins),
        "selected_vs_runner_up_relative_margin_median": median_or_none(rel_margins),
        "final_score_vs_inverse_path_cost_corr": inv_corr,
        "final_score_vs_inverse_path_cost_spearman": _aggregate_spearman(rank_rows, None, "inverse_path_cost"),
        "final_score_vs_gain_exp_corr": gain_corr,
        "final_score_vs_gain_exp_spearman": _aggregate_spearman(rank_rows, None, "gain_exp"),
        "final_score_vs_effective_gain_sc_corr": eff_corr,
        "final_score_vs_effective_gain_sc_spearman": _aggregate_spearman(rank_rows, None, "effective_gain_sc"),
        "gain_exp_vs_effective_gain_sc_corr": gain_exp_vs_effective_corr,
        "selected_rank_low_path_cost_mean": selected_cost_mean,
        "selected_rank_gain_exp_mean": selected_gain_mean,
        "selected_rank_effective_gain_sc_mean": selected_eff_mean,
        "component_best_explains_final_score_all": _component_best_explains(rank_rows, None),
        "component_best_explains_final_score_by_config": {
            cfg: _component_best_explains(rank_rows, cfg) for cfg in loaded_configs
        },
        "missing_field_counts": missing_counts,
        "main_diagnosis": "; ".join(diagnosis_parts) + ".",
        "recommended_next_small_task": next_task,
        "safety": {
            "new_isaac_rollout_started": False,
            "rl_ppo_bc_il_training": False,
            "sscnet_training": False,
            "checkpoint_modified": False,
            "observed_state_modified": False,
            "prediction_writeback": False,
            "prediction_used_for_traversability_collision_astar": False,
            "future_observations_used_for_planning": False,
            "target_ground_truth_used_for_scoring": False,
        },
    }


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, list):
            return ", ".join(str(x) for x in value)
        return str(value)

    lines = [
        "## Stage 4A-6.5a Rank Sensitivity Summary",
        f"- steps analyzed: {fmt(summary.get('steps_analyzed'))}",
        f"- configs analyzed: {fmt(summary.get('loaded_configs'))}",
        f"- selected action stability: gated ids identical = {fmt(summary.get('selected_candidate_ids_identical_across_all_gated_steps'))}; gated positions identical = {fmt(summary.get('selected_positions_identical_across_all_gated_steps'))}",
        f"- final_score vs inverse path_cost correlation: pearson {fmt(summary.get('final_score_vs_inverse_path_cost_corr'))}, spearman {fmt(summary.get('final_score_vs_inverse_path_cost_spearman'))}",
        f"- final_score vs gain_exp correlation: pearson {fmt(summary.get('final_score_vs_gain_exp_corr'))}, spearman {fmt(summary.get('final_score_vs_gain_exp_spearman'))}",
        f"- final_score vs effective_gain_sc correlation: pearson {fmt(summary.get('final_score_vs_effective_gain_sc_corr'))}, spearman {fmt(summary.get('final_score_vs_effective_gain_sc_spearman'))}",
        f"- gain_exp vs effective_gain_sc correlation: {fmt(summary.get('gain_exp_vs_effective_gain_sc_corr'))}",
        f"- top-1 / top-5 / top-16 overlap: top1 stable vs raw SC = {fmt(summary.get('top1_stable_vs_raw_sc'))}; mean top5 Jaccard vs raw SC = {fmt(summary.get('top5_jaccard_mean_vs_raw_sc'))}; mean top16 Jaccard vs raw SC = {fmt(summary.get('top16_jaccard_mean_vs_raw_sc'))}",
        f"- selected-vs-runner-up score margin: mean {fmt(summary.get('selected_vs_runner_up_margin_mean'))}, min {fmt(summary.get('selected_vs_runner_up_margin_min'))}, max {fmt(summary.get('selected_vs_runner_up_margin_max'))}",
        f"- component best explains final_score: {fmt(summary.get('component_best_explains_final_score_all'))}",
        f"- main diagnosis: {summary.get('main_diagnosis')}",
        f"- recommended next small task: {summary.get('recommended_next_small_task')}",
        "",
        "Safety:",
        "- new Isaac rollout started: no",
        "- RL/PPO/BC/IL training: no",
        "- SSCNet training: no",
        "- checkpoint modified: no",
        "- observed_state modified: no",
        "- prediction writeback: no",
        "- prediction used for traversability/collision/A*: no",
        "- future observations used for planning: no",
        "- target/ground truth used for scoring: no",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empty_episode_dir", required=True, type=Path)
    parser.add_argument("--raw_sc_episode_dir", required=True, type=Path)
    parser.add_argument("--gating_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs, missing_configs = discover_episode_dirs(
        args.empty_episode_dir.resolve(),
        args.raw_sc_episode_dir.resolve(),
        args.gating_root.resolve(),
    )

    step_data: dict[str, dict[int, dict[str, Any]]] = {}
    config_summaries: dict[str, dict[str, Any]] = {}
    transitions_by_config: dict[str, list[dict[str, Any]]] = {}
    load_errors: dict[str, str] = {}

    for config, episode_dir in configs.items():
        if not episode_dir.exists():
            load_errors[config] = f"episode dir does not exist: {episode_dir}"
            continue
        try:
            steps, transitions, episode_summary = load_config_steps(config, episode_dir, args.max_steps)
        except Exception as exc:  # noqa: BLE001 - report and continue by design.
            load_errors[config] = f"{type(exc).__name__}: {exc}"
            continue
        if not steps:
            load_errors[config] = f"no step_*.npz files found for steps 0..{args.max_steps - 1}"
            continue
        step_data[config] = steps
        transitions_by_config[config] = transitions
        config_summaries[config] = episode_summary

    if load_errors:
        missing_configs.update(load_errors)

    candidate_rows = build_candidate_rows(step_data)
    selected_rows = build_selected_summary(step_data)
    rank_rows = build_rank_correlations(step_data)
    topk_rows = build_topk_overlap(step_data)
    missing_report = build_missing_report(step_data, missing_configs)
    summary = summarize_findings(step_data, selected_rows, rank_rows, topk_rows, missing_report)
    summary["inputs"] = {
        "empty_episode_dir": str(args.empty_episode_dir.resolve()),
        "raw_sc_episode_dir": str(args.raw_sc_episode_dir.resolve()),
        "gating_root": str(args.gating_root.resolve()),
        "max_steps": int(args.max_steps),
    }
    summary["config_episode_dirs"] = {config: str(path) for config, path in configs.items()}
    summary["config_episode_summaries"] = config_summaries
    summary["load_errors"] = load_errors
    summary["validation"] = {
        "output_dir_exists": output_dir.exists(),
        "at_least_one_config_loaded": len(step_data) >= 1,
        "at_least_one_step_loaded": any(len(steps) >= 1 for steps in step_data.values()),
        "missing_fields_reported": bool(missing_report.get("by_config_step")),
    }

    candidate_csv = output_dir / "candidate_rank_table.csv"
    selected_csv = output_dir / "selected_candidate_summary.csv"
    rank_csv = output_dir / "rank_correlation_summary.csv"
    topk_csv = output_dir / "topk_overlap_summary.csv"
    missing_json = output_dir / "missing_fields_report.json"
    summary_json = output_dir / "stage4a65a_rank_sensitivity_summary.json"
    summary_md = output_dir / "stage4a65a_rank_sensitivity_summary.md"

    write_csv(candidate_csv, candidate_rows)
    write_csv(selected_csv, selected_rows)
    write_csv(rank_csv, rank_rows)
    write_csv(topk_csv, topk_rows)
    save_json(missing_json, missing_report)
    save_json(summary_json, summary)
    write_markdown_summary(summary_md, summary)

    # Minimal validation requested by Stage 4A-6.5a.
    validation = {
        "output_dir_exists": output_dir.exists(),
        "summary_json_exists": summary_json.exists(),
        "summary_md_exists": summary_md.exists(),
        "at_least_one_config_loaded": len(step_data) >= 1,
        "at_least_one_step_loaded": any(len(steps) >= 1 for steps in step_data.values()),
        "missing_fields_reported": missing_json.exists() and bool(missing_report.get("by_config_step")),
    }
    print("Stage 4A-6.5a rank sensitivity diagnosis complete")
    print(json.dumps(validation, indent=2, sort_keys=True))
    print(f"configs_loaded: {summary['loaded_configs']}")
    print(f"steps_analyzed: {summary['steps_analyzed']}")
    print(f"main_diagnosis: {summary['main_diagnosis']}")
    print(f"recommended_next_small_task: {summary['recommended_next_small_task']}")


if __name__ == "__main__":
    main()

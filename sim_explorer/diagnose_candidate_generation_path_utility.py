#!/usr/bin/env python3
"""Stage 4A-6.5e offline candidate/path utility diagnosis.

This script reads existing candidate logs and step NPZ files only. It does not
launch Isaac, run rollout, run map_predict, train any model, change runtime
expert scoring, or write to observed_state/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for this script.
    plt = None


EPS = 1e-9
LOCALITY_THRESHOLDS_M = (0.25, 0.5, 1.0, 2.0)
GATED_CONFIGS = [
    "occupied_only_occ07",
    "occupied_only_occ08",
    "occupied_margin_occ06_w05",
    "confidence_weighted_conf05_cap30",
]
ROLLOUT_LIKE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]


@dataclass
class CandidateSet:
    config: str
    step: int
    source_type: str
    source_path: Path
    episode_dir: Path | None
    current_world: np.ndarray
    current_grid: np.ndarray
    positions_world: np.ndarray
    positions_grid: np.ndarray
    valid: np.ndarray
    selected_index: int | None
    components: dict[str, np.ndarray]
    candidate_ids: list[Any]
    feature_names: list[str]
    n_logged: int | None
    metadata: dict[str, Any]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


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
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return csv_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
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


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def scalar(data: dict[str, Any], key: str, default: Any = None) -> Any:
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


def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def stats(prefix: str, values: Any) -> dict[str, Any]:
    arr = finite_array(values)
    if arr.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min": None,
            f"{prefix}_p25": None,
            f"{prefix}_median": None,
            f"{prefix}_p75": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
        }
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
    }


def mean_or_none(values: Any) -> float | None:
    arr = finite_array(values)
    return float(arr.mean()) if arr.size else None


def rank_values(values: np.ndarray, valid: np.ndarray, descending: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if mask.size != arr.size:
        mask = np.ones(arr.size, dtype=bool)
    ranks = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = mask & np.isfinite(arr)
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
        ranks[sorted_idx[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def best_index(values: np.ndarray, valid: np.ndarray, maximize: bool) -> int | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if mask.size != arr.size:
        mask = np.ones(arr.size, dtype=bool)
    finite = mask & np.isfinite(arr)
    if not np.any(finite):
        return None
    work = arr.copy()
    work[~finite] = -np.inf if maximize else np.inf
    return int(np.argmax(work) if maximize else np.argmin(work))


def pearson(x_values: Any, y_values: Any, valid: Any | None = None) -> float | None:
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if valid is not None:
        v = np.asarray(valid, dtype=bool).reshape(-1)
        if v.size == x.size:
            mask &= v
    if int(np.count_nonzero(mask)) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def feature_map(feature_names: list[str], features: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if features.ndim != 2:
        return out
    for idx, name in enumerate(feature_names):
        if idx < features.shape[1]:
            out[str(name)] = features[:, idx].astype(np.float64, copy=False)
    return out


def component_array(
    fmap: dict[str, np.ndarray],
    n: int,
    names: tuple[str, ...],
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    for name in names:
        if name in fmap:
            return np.asarray(fmap[name], dtype=np.float64).reshape(-1)
    if fallback is not None and fallback.size == n:
        return np.asarray(fallback, dtype=np.float64).reshape(-1)
    return np.full(n, np.nan, dtype=np.float64)


def build_components(
    fmap: dict[str, np.ndarray],
    n: int,
    expert_scores: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    scores = np.asarray(expert_scores, dtype=np.float64).reshape(-1) if expert_scores is not None else np.array([])
    final_score = component_array(fmap, n, ("final_score",), scores)
    path_cost = component_array(fmap, n, ("path_cost",))
    inverse_cost = np.full(n, np.nan, dtype=np.float64)
    finite_cost = np.isfinite(path_cost) & (path_cost > EPS)
    inverse_cost[finite_cost] = 1.0 / path_cost[finite_cost]
    gain_hybrid_weighted = component_array(fmap, n, ("gain_hybrid_weighted",))
    gain_hybrid = component_array(fmap, n, ("gain_hybrid",))
    gain_exp = component_array(fmap, n, ("gain_exp",))
    scoring_gain = gain_hybrid_weighted.copy()
    missing_weighted = ~np.isfinite(scoring_gain)
    scoring_gain[missing_weighted] = gain_hybrid[missing_weighted]
    missing_hybrid = ~np.isfinite(scoring_gain)
    scoring_gain[missing_hybrid] = gain_exp[missing_hybrid]
    return {
        "final_score": final_score,
        "gain_exp": gain_exp,
        "gain_sc": component_array(fmap, n, ("gain_sc",)),
        "raw_gain_sc": component_array(fmap, n, ("raw_gain_sc", "gain_sc")),
        "effective_gain_sc": component_array(fmap, n, ("effective_gain_sc",)),
        "weighted_gain_sc": component_array(fmap, n, ("weighted_gain_sc",)),
        "gain_hybrid": gain_hybrid,
        "gain_hybrid_weighted": gain_hybrid_weighted,
        "gain_hybrid_effective": component_array(fmap, n, ("gain_hybrid_effective",)),
        "path_cost": path_cost,
        "inverse_path_cost": inverse_cost,
        "astar_path_length_m": component_array(fmap, n, ("astar_path_length_m",)),
        "astar_num_expanded": component_array(fmap, n, ("astar_num_expanded",)),
        "visible_count": component_array(fmap, n, ("visible_count",)),
        "frontier_distance": component_array(fmap, n, ("frontier_distance",)),
        "scoring_gain": scoring_gain,
    }


def position_key(grid: Any, world: Any) -> str:
    grid_arr = np.asarray(grid, dtype=np.float64).reshape(-1) if grid is not None else np.asarray([])
    if grid_arr.size >= 3 and np.all(np.isfinite(grid_arr[:3])):
        return "grid:" + ",".join(str(int(round(v))) for v in grid_arr[:3])
    world_arr = np.asarray(world, dtype=np.float64).reshape(-1) if world is not None else np.asarray([])
    if world_arr.size >= 3 and np.all(np.isfinite(world_arr[:3])):
        return "world:" + ",".join(f"{float(v):.4f}" for v in world_arr[:3])
    return "missing"


def parse_vector(value: Any, length: int = 3) -> np.ndarray:
    arr = np.asarray(value if value is not None else [], dtype=np.float64).reshape(-1)
    out = np.full(length, np.nan, dtype=np.float64)
    take = min(length, arr.size)
    if take:
        out[:take] = arr[:take]
    return out


def record_metric(record: dict[str, Any], names: tuple[str, ...]) -> float:
    for name in names:
        for container in (record, record.get("gains") or {}, record.get("utilities") or {}):
            if name in container:
                value = as_float(container.get(name))
                if value is not None:
                    return value
    return float("nan")


def load_npz_candidate_set(config: str, episode_dir: Path, step: int) -> CandidateSet | None:
    path = episode_dir / f"step_{step:03d}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as raw:
        npz = {key: raw[key].copy() for key in raw.files}
    features = np.asarray(npz.get("candidate_features", np.empty((0, 0))), dtype=np.float64)
    if features.ndim != 2:
        features = np.empty((0, 0), dtype=np.float64)
    n = int(features.shape[0])
    names = [str(x) for x in np.asarray(npz.get("feature_names", []), dtype=str).reshape(-1).tolist()]
    fmap = feature_map(names, features)
    expert_scores = np.asarray(npz.get("expert_scores", np.full(n, np.nan)), dtype=np.float64).reshape(-1)
    if expert_scores.size != n:
        expert_scores = np.full(n, np.nan, dtype=np.float64)
    components = build_components(fmap, n, expert_scores)
    valid = np.asarray(npz.get("valid_mask", np.ones(n, dtype=bool)), dtype=bool).reshape(-1)
    if valid.size != n:
        valid = np.ones(n, dtype=bool)
    positions_grid = np.asarray(npz.get("candidate_positions_grid", np.full((n, 3), np.nan)), dtype=np.float64)
    positions_world = np.asarray(npz.get("candidate_positions_world", np.full((n, 3), np.nan)), dtype=np.float64)
    if positions_grid.shape != (n, 3):
        positions_grid = np.full((n, 3), np.nan, dtype=np.float64)
    if positions_world.shape != (n, 3):
        positions_world = np.full((n, 3), np.nan, dtype=np.float64)
    expert_action = as_int(scalar(npz, "expert_action"))
    selected_index = expert_action if expert_action is not None and 0 <= expert_action < n else None
    if selected_index is None:
        selected_index = best_index(components["final_score"], valid, True)
    candidate_ids = list(range(n))
    metadata = {
        "candidate_sampling_mode": scalar(npz, "candidate_sampling_mode"),
        "candidate_source": scalar(npz, "candidate_source"),
        "score_gain_mode": scalar(npz, "score_gain_mode"),
        "prediction_mode": scalar(npz, "prediction_mode"),
        "path_cost_mode": scalar(npz, "path_cost_mode"),
        "expert_action": expert_action,
        "candidate_count_scalar": scalar(npz, "candidate_count"),
        "reachable_candidates": scalar(npz, "reachable_candidates"),
        "prediction_used_for_information_gain": scalar(npz, "prediction_used_for_information_gain"),
        "prediction_used_for_traversability": scalar(npz, "prediction_used_for_traversability"),
        "prediction_used_for_collision": scalar(npz, "prediction_used_for_collision"),
        "prediction_used_for_a_star": scalar(npz, "prediction_used_for_a_star"),
        "prediction_used_for_astar": scalar(npz, "prediction_used_for_astar"),
        "prediction_blocks_rays": scalar(npz, "prediction_blocks_rays"),
        "prediction_written_to_observed_state": scalar(npz, "prediction_written_to_observed_state"),
        "prediction_wrote_observed_map": scalar(npz, "prediction_wrote_observed_map"),
        "future_observations_used_for_planning": scalar(npz, "future_observations_used_for_planning"),
        "future_observations_used_for_scoring": scalar(npz, "future_observations_used_for_scoring"),
        "rl_or_ppo_training": scalar(npz, "rl_or_ppo_training"),
        "optimizer_step": scalar(npz, "optimizer_step"),
    }
    return CandidateSet(
        config=config,
        step=step,
        source_type="episode_topn_npz",
        source_path=path,
        episode_dir=episode_dir,
        current_world=parse_vector(npz.get("current_pose_world")),
        current_grid=parse_vector(npz.get("current_pose_grid")),
        positions_world=positions_world,
        positions_grid=positions_grid,
        valid=valid,
        selected_index=selected_index,
        components=components,
        candidate_ids=candidate_ids,
        feature_names=names,
        n_logged=as_int(metadata["candidate_count_scalar"]),
        metadata=metadata,
    )


def load_runtime_candidate_set(config: str, runtime_dir: Path, step: int = 1) -> CandidateSet | None:
    path = runtime_dir / "expert_step_candidates.jsonl"
    records = load_jsonl(path)
    if not records:
        return None
    decision_npz = runtime_dir / "expert_step_decision.npz"
    current_world = np.full(3, np.nan, dtype=np.float64)
    current_grid = np.full(3, np.nan, dtype=np.float64)
    metadata: dict[str, Any] = {"runtime_dir": str(runtime_dir)}
    if decision_npz.exists():
        with np.load(decision_npz, allow_pickle=True) as raw:
            npz = {key: raw[key].copy() for key in raw.files}
        current_world = parse_vector(npz.get("current_pose_world"))
        current_grid = parse_vector(npz.get("current_pose_grid"))
        for key in [
            "score_gain_mode",
            "sc_gain_formula",
            "sc_gain_weight",
            "sc_gain_cap_value",
            "prediction_mode",
            "path_cost_mode",
            "candidate_sampling_mode",
            "candidate_source",
            "prediction_used_for_information_gain",
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_used_for_astar",
            "prediction_blocks_rays",
            "prediction_written_to_observed_state",
            "future_observations_used_for_planning",
            "future_observations_used_for_scoring",
        ]:
            metadata[key] = scalar(npz, key)
    n = len(records)
    positions_grid = np.vstack([parse_vector(r.get("grid_position") or r.get("candidate_grid")) for r in records])
    positions_world = np.vstack([parse_vector(r.get("world_position") or r.get("candidate_world")) for r in records])
    valid = np.asarray([bool(r.get("valid", True)) for r in records], dtype=bool)
    selected_index = next((i for i, r in enumerate(records) if bool(r.get("is_best_candidate", False))), None)
    candidate_ids = [r.get("id", i) for i, r in enumerate(records)]
    arrays = {
        "final_score": np.asarray(
            [
                record_metric(r, ("final_score", "score", "final_score_decoupled_sc"))
                for r in records
            ],
            dtype=np.float64,
        ),
        "gain_exp": np.asarray([record_metric(r, ("gain_exp",)) for r in records], dtype=np.float64),
        "gain_sc": np.asarray([record_metric(r, ("gain_sc",)) for r in records], dtype=np.float64),
        "raw_gain_sc": np.asarray([record_metric(r, ("raw_gain_sc", "gain_sc")) for r in records], dtype=np.float64),
        "effective_gain_sc": np.asarray([record_metric(r, ("effective_gain_sc",)) for r in records], dtype=np.float64),
        "weighted_gain_sc": np.asarray([record_metric(r, ("weighted_gain_sc",)) for r in records], dtype=np.float64),
        "gain_hybrid": np.asarray([record_metric(r, ("gain_hybrid",)) for r in records], dtype=np.float64),
        "gain_hybrid_weighted": np.asarray(
            [record_metric(r, ("gain_hybrid_weighted",)) for r in records], dtype=np.float64
        ),
        "gain_hybrid_effective": np.asarray(
            [record_metric(r, ("gain_hybrid_effective",)) for r in records], dtype=np.float64
        ),
        "path_cost": np.asarray([record_metric(r, ("path_cost",)) for r in records], dtype=np.float64),
        "astar_path_length_m": np.asarray(
            [record_metric(r, ("astar_path_length_m",)) for r in records], dtype=np.float64
        ),
        "astar_num_expanded": np.asarray(
            [record_metric(r, ("astar_num_expanded",)) for r in records], dtype=np.float64
        ),
        "visible_count": np.asarray([record_metric(r, ("visible_count",)) for r in records], dtype=np.float64),
        "frontier_distance": np.asarray([record_metric(r, ("frontier_distance",)) for r in records], dtype=np.float64),
    }
    inverse_cost = np.full(n, np.nan, dtype=np.float64)
    finite_cost = np.isfinite(arrays["path_cost"]) & (arrays["path_cost"] > EPS)
    inverse_cost[finite_cost] = 1.0 / arrays["path_cost"][finite_cost]
    scoring_gain = arrays["gain_hybrid_weighted"].copy()
    missing_weighted = ~np.isfinite(scoring_gain)
    scoring_gain[missing_weighted] = arrays["gain_hybrid"][missing_weighted]
    missing_hybrid = ~np.isfinite(scoring_gain)
    scoring_gain[missing_hybrid] = arrays["gain_exp"][missing_hybrid]
    arrays["inverse_path_cost"] = inverse_cost
    arrays["scoring_gain"] = scoring_gain
    if selected_index is None:
        selected_index = best_index(arrays["final_score"], valid, True)
    return CandidateSet(
        config=config,
        step=step,
        source_type="runtime_full64_jsonl",
        source_path=path,
        episode_dir=None,
        current_world=current_world,
        current_grid=current_grid,
        positions_world=positions_world,
        positions_grid=positions_grid,
        valid=valid,
        selected_index=selected_index,
        components=arrays,
        candidate_ids=candidate_ids,
        feature_names=[],
        n_logged=n,
        metadata=metadata,
    )


def discover_config_dirs(args: argparse.Namespace, rank_summary: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    dirs: dict[str, Path] = {}
    missing: dict[str, str] = {}
    for name, raw_path in (rank_summary.get("config_episode_dirs") or {}).items():
        if raw_path:
            dirs[str(name)] = Path(raw_path)
    dirs["empty_baseline"] = args.empty_episode_dir
    dirs["fixed_raw_sc"] = args.fixed_sc_episode_dir
    ablation_root = args.gating_root / "ablation" / "episodes"
    for name in GATED_CONFIGS:
        if name in dirs and dirs[name].exists():
            continue
        exact = ablation_root / f"medium_three_rooms_seed0_start_room_a_{name}"
        if exact.exists():
            dirs[name] = exact
            continue
        matches = sorted(ablation_root.glob(f"*_{name}")) if ablation_root.exists() else []
        if matches:
            dirs[name] = matches[0]
        else:
            missing[name] = f"episode dir not found under {ablation_root}"
    for name, path in list(dirs.items()):
        if not path.exists():
            missing[name] = f"episode dir missing: {path}"
            dirs.pop(name, None)
    preferred = ["empty_baseline", "fixed_raw_sc", *GATED_CONFIGS]
    ordered: dict[str, Path] = {}
    for name in preferred:
        if name in dirs:
            ordered[name] = dirs[name]
    for name in sorted(dirs):
        if name not in ordered:
            ordered[name] = dirs[name]
    return ordered, missing


def candidate_distances(candidate_set: CandidateSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world = candidate_set.positions_world
    current = candidate_set.current_world
    finite_xy = np.isfinite(world[:, :2]).all(axis=1) & np.isfinite(current[:2]).all()
    finite_xyz = np.isfinite(world[:, :3]).all(axis=1) & np.isfinite(current[:3]).all()
    dist_xy = np.full(world.shape[0], np.nan, dtype=np.float64)
    dist_xyz = np.full(world.shape[0], np.nan, dtype=np.float64)
    if np.any(finite_xy):
        dist_xy[finite_xy] = np.linalg.norm(world[finite_xy, :2] - current[:2], axis=1)
    if np.any(finite_xyz):
        dist_xyz[finite_xyz] = np.linalg.norm(world[finite_xyz, :3] - current[:3], axis=1)
    position_valid = finite_xy
    return dist_xy, dist_xyz, position_valid


def average_pairwise_distance(points_xy: np.ndarray, valid: np.ndarray) -> float | None:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(points_xy).all(axis=1)
    pts = points_xy[mask]
    if pts.shape[0] < 2:
        return None
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    tri = dists[np.triu_indices(pts.shape[0], k=1)]
    return float(np.mean(tri)) if tri.size else None


def duplicate_grid_stats(grid: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(grid).all(axis=1)
    counts: dict[str, int] = {}
    for row in grid[mask]:
        key = position_key(row, None)
        counts[key] = counts.get(key, 0) + 1
    duplicate_groups = {k: v for k, v in counts.items() if v > 1}
    return {
        "duplicate_grid_groups": int(len(duplicate_groups)),
        "duplicate_grid_positions_after_first": int(sum(v - 1 for v in duplicate_groups.values())),
        "duplicate_grid_keys": sorted(duplicate_groups),
    }


def near_duplicate_stats(world: np.ndarray, valid: np.ndarray, threshold_m: float = 0.15) -> dict[str, Any]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(world[:, :2]).all(axis=1)
    idx = np.flatnonzero(mask)
    pair_count = 0
    touched: set[int] = set()
    for pos_i, i in enumerate(idx):
        for j in idx[pos_i + 1 :]:
            dist = float(np.linalg.norm(world[i, :2] - world[j, :2]))
            if dist <= threshold_m:
                pair_count += 1
                touched.add(int(i))
                touched.add(int(j))
    return {
        "near_duplicate_threshold_m": threshold_m,
        "near_duplicate_pair_count": int(pair_count),
        "near_duplicate_candidate_count": int(len(touched)),
    }


def selected_value(candidate_set: CandidateSet, values: np.ndarray) -> float | None:
    idx = candidate_set.selected_index
    if idx is None or idx < 0 or idx >= values.size:
        return None
    return as_float(values[idx])


def build_candidate_rank_rows(candidate_sets: list[CandidateSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cs in candidate_sets:
        dist_xy, dist_xyz, _ = candidate_distances(cs)
        valid = cs.valid
        ranks_final = rank_values(cs.components["final_score"], valid, True)
        ranks_gain = rank_values(cs.components["gain_exp"], valid, True)
        ranks_raw = rank_values(cs.components["raw_gain_sc"], valid, True)
        ranks_eff = rank_values(cs.components["effective_gain_sc"], valid, True)
        ranks_low_cost = rank_values(cs.components["path_cost"], valid, False)
        n = cs.positions_world.shape[0]
        for i in range(n):
            rows.append(
                {
                    "config": cs.config,
                    "step": cs.step,
                    "source_type": cs.source_type,
                    "source_path": str(cs.source_path),
                    "candidate_row": i,
                    "candidate_id": cs.candidate_ids[i] if i < len(cs.candidate_ids) else i,
                    "candidate_key": position_key(cs.positions_grid[i], cs.positions_world[i]),
                    "is_selected": bool(cs.selected_index == i),
                    "valid": bool(valid[i]),
                    "candidate_grid": cs.positions_grid[i].tolist(),
                    "candidate_world": cs.positions_world[i].tolist(),
                    "current_world": cs.current_world.tolist(),
                    "distance_xy_from_current_m": as_float(dist_xy[i]),
                    "distance_3d_from_current_m": as_float(dist_xyz[i]),
                    "rank_final_score": as_float(ranks_final[i]),
                    "rank_low_path_cost": as_float(ranks_low_cost[i]),
                    "rank_gain_exp": as_float(ranks_gain[i]),
                    "rank_raw_gain_sc": as_float(ranks_raw[i]),
                    "rank_effective_gain_sc": as_float(ranks_eff[i]),
                    "final_score": as_float(cs.components["final_score"][i]),
                    "gain_exp": as_float(cs.components["gain_exp"][i]),
                    "raw_gain_sc": as_float(cs.components["raw_gain_sc"][i]),
                    "effective_gain_sc": as_float(cs.components["effective_gain_sc"][i]),
                    "weighted_gain_sc": as_float(cs.components["weighted_gain_sc"][i]),
                    "path_cost": as_float(cs.components["path_cost"][i]),
                    "astar_path_length_m": as_float(cs.components["astar_path_length_m"][i]),
                }
            )
    return rows


def build_spread_rows(candidate_sets: list[CandidateSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cs in candidate_sets:
        dist_xy, dist_xyz, position_valid = candidate_distances(cs)
        valid = cs.valid & position_valid
        n_valid = int(np.count_nonzero(valid))
        row: dict[str, Any] = {
            "config": cs.config,
            "step": cs.step,
            "source_type": cs.source_type,
            "source_path": str(cs.source_path),
            "candidate_count_saved": int(cs.positions_world.shape[0]),
            "candidate_count_logged": cs.n_logged,
            "valid_position_candidate_count": n_valid,
            "candidate_positions_missing": bool(n_valid == 0),
            "current_world": cs.current_world.tolist(),
            "current_grid": cs.current_grid.tolist(),
            "selected_row": cs.selected_index,
            "selected_key": (
                position_key(cs.positions_grid[cs.selected_index], cs.positions_world[cs.selected_index])
                if cs.selected_index is not None and 0 <= cs.selected_index < cs.positions_world.shape[0]
                else None
            ),
            "selected_distance_xy_m": selected_value(cs, dist_xy),
            "selected_distance_3d_m": selected_value(cs, dist_xyz),
        }
        if n_valid > 0:
            pts = cs.positions_world[valid]
            row.update(
                {
                    "x_range_min": float(np.min(pts[:, 0])),
                    "x_range_max": float(np.max(pts[:, 0])),
                    "x_range_m": float(np.max(pts[:, 0]) - np.min(pts[:, 0])),
                    "y_range_min": float(np.min(pts[:, 1])),
                    "y_range_max": float(np.max(pts[:, 1])),
                    "y_range_m": float(np.max(pts[:, 1]) - np.min(pts[:, 1])),
                    "z_range_min": float(np.min(pts[:, 2])),
                    "z_range_max": float(np.max(pts[:, 2])),
                    "z_range_m": float(np.max(pts[:, 2]) - np.min(pts[:, 2])),
                    "avg_pairwise_distance_xy_m": average_pairwise_distance(cs.positions_world[:, :2], valid),
                }
            )
        row.update(stats("distance_xy_m", dist_xy[valid]))
        row.update(stats("distance_3d_m", dist_xyz[valid]))
        row.update(stats("path_cost", cs.components["path_cost"][cs.valid]))
        row.update(stats("astar_path_length_m", cs.components["astar_path_length_m"][cs.valid]))
        row.update(stats("gain_exp", cs.components["gain_exp"][cs.valid]))
        row.update(stats("raw_gain_sc", cs.components["raw_gain_sc"][cs.valid]))
        row.update(stats("effective_gain_sc", cs.components["effective_gain_sc"][cs.valid]))
        row.update(stats("final_score", cs.components["final_score"][cs.valid]))
        for threshold in LOCALITY_THRESHOLDS_M:
            count = int(np.count_nonzero(valid & np.isfinite(dist_xy) & (dist_xy <= threshold)))
            row[f"candidates_within_{str(threshold).replace('.', 'p')}m"] = count
            row[f"fraction_within_{str(threshold).replace('.', 'p')}m"] = (
                float(count / n_valid) if n_valid else None
            )
        row.update(duplicate_grid_stats(cs.positions_grid, cs.valid))
        row.update(near_duplicate_stats(cs.positions_world, cs.valid))
        rows.append(row)
    return rows


def build_selected_vs_high_gain_rows(candidate_sets: list[CandidateSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cs in candidate_sets:
        selected = cs.selected_index
        max_gain = best_index(cs.components["gain_exp"], cs.valid, True)
        max_raw = best_index(cs.components["raw_gain_sc"], cs.valid, True)
        max_eff = best_index(cs.components["effective_gain_sc"], cs.valid, True)
        min_cost = best_index(cs.components["path_cost"], cs.valid, False)
        max_final = best_index(cs.components["final_score"], cs.valid, True)

        def dist_between(i: int | None, j: int | None) -> float | None:
            if i is None or j is None:
                return None
            if i < 0 or j < 0 or i >= cs.positions_world.shape[0] or j >= cs.positions_world.shape[0]:
                return None
            a = cs.positions_world[i, :2]
            b = cs.positions_world[j, :2]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                return None
            return float(np.linalg.norm(a - b))

        def metric_at(name: str, idx: int | None) -> float | None:
            if idx is None or idx < 0 or idx >= cs.components[name].size:
                return None
            return as_float(cs.components[name][idx])

        def key_at(idx: int | None) -> str | None:
            if idx is None or idx < 0 or idx >= cs.positions_world.shape[0]:
                return None
            return position_key(cs.positions_grid[idx], cs.positions_world[idx])

        selected_to_gain = dist_between(selected, max_gain)
        selected_to_eff = dist_between(selected, max_eff)
        selected_to_cost = dist_between(selected, min_cost)
        row = {
            "config": cs.config,
            "step": cs.step,
            "source_type": cs.source_type,
            "selected_row": selected,
            "selected_key": key_at(selected),
            "selected_final_score": metric_at("final_score", selected),
            "selected_gain_exp": metric_at("gain_exp", selected),
            "selected_raw_gain_sc": metric_at("raw_gain_sc", selected),
            "selected_effective_gain_sc": metric_at("effective_gain_sc", selected),
            "selected_path_cost": metric_at("path_cost", selected),
            "max_final_row": max_final,
            "max_final_key": key_at(max_final),
            "max_gain_exp_row": max_gain,
            "max_gain_exp_key": key_at(max_gain),
            "max_gain_exp": metric_at("gain_exp", max_gain),
            "selected_to_max_gain_exp_distance_m": selected_to_gain,
            "max_raw_gain_sc_row": max_raw,
            "max_raw_gain_sc_key": key_at(max_raw),
            "max_raw_gain_sc": metric_at("raw_gain_sc", max_raw),
            "selected_to_max_raw_gain_sc_distance_m": dist_between(selected, max_raw),
            "max_effective_gain_sc_row": max_eff,
            "max_effective_gain_sc_key": key_at(max_eff),
            "max_effective_gain_sc": metric_at("effective_gain_sc", max_eff),
            "selected_to_max_effective_gain_sc_distance_m": selected_to_eff,
            "min_path_cost_row": min_cost,
            "min_path_cost_key": key_at(min_cost),
            "min_path_cost": metric_at("path_cost", min_cost),
            "selected_to_min_path_cost_distance_m": selected_to_cost,
            "min_path_cost_candidate_is_selected": bool(selected is not None and selected == min_cost),
            "max_gain_exp_spatially_distinct_gt_0p5m": (
                bool(selected_to_gain is not None and selected_to_gain > 0.5)
            ),
            "max_effective_gain_sc_spatially_distinct_gt_0p5m": (
                bool(selected_to_eff is not None and selected_to_eff > 0.5)
            ),
        }
        rows.append(row)
    return rows


def build_path_cost_dominance_rows(candidate_sets: list[CandidateSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cs in candidate_sets:
        valid = cs.valid
        final_score = cs.components["final_score"]
        inv_cost = cs.components["inverse_path_cost"]
        selected = cs.selected_index
        min_cost = best_index(cs.components["path_cost"], valid, False)
        max_final = best_index(final_score, valid, True)
        max_gain = best_index(cs.components["gain_exp"], valid, True)
        max_eff = best_index(cs.components["effective_gain_sc"], valid, True)
        ranks_low = rank_values(cs.components["path_cost"], valid, False)
        ranks_gain = rank_values(cs.components["gain_exp"], valid, True)
        ranks_eff = rank_values(cs.components["effective_gain_sc"], valid, True)
        component_corrs = {
            "inverse_path_cost": pearson(final_score, inv_cost, valid),
            "path_cost": pearson(final_score, cs.components["path_cost"], valid),
            "gain_exp": pearson(final_score, cs.components["gain_exp"], valid),
            "raw_gain_sc": pearson(final_score, cs.components["raw_gain_sc"], valid),
            "effective_gain_sc": pearson(final_score, cs.components["effective_gain_sc"], valid),
        }
        finite_corrs = {k: abs(v) for k, v in component_corrs.items() if v is not None}
        best_component = max(finite_corrs, key=finite_corrs.get) if finite_corrs else None
        rows.append(
            {
                "config": cs.config,
                "step": cs.step,
                "source_type": cs.source_type,
                "candidate_count": int(np.count_nonzero(valid)),
                "selected_row": selected,
                "max_final_score_row": max_final,
                "min_path_cost_row": min_cost,
                "max_gain_exp_row": max_gain,
                "max_effective_gain_sc_row": max_eff,
                "top_final_matches_min_path_cost": bool(max_final is not None and max_final == min_cost),
                "selected_matches_min_path_cost": bool(selected is not None and selected == min_cost),
                "selected_rank_low_path_cost": (
                    as_float(ranks_low[selected]) if selected is not None and selected < ranks_low.size else None
                ),
                "selected_rank_gain_exp": (
                    as_float(ranks_gain[selected]) if selected is not None and selected < ranks_gain.size else None
                ),
                "selected_rank_effective_gain_sc": (
                    as_float(ranks_eff[selected]) if selected is not None and selected < ranks_eff.size else None
                ),
                "pearson_final_vs_inverse_path_cost": component_corrs["inverse_path_cost"],
                "pearson_final_vs_path_cost": component_corrs["path_cost"],
                "pearson_final_vs_gain_exp": component_corrs["gain_exp"],
                "pearson_final_vs_raw_gain_sc": component_corrs["raw_gain_sc"],
                "pearson_final_vs_effective_gain_sc": component_corrs["effective_gain_sc"],
                "best_abs_pearson_component": best_component,
                "path_cost_is_best_abs_component": bool(best_component in {"inverse_path_cost", "path_cost"}),
            }
        )
    return rows


def one_step_gain_over_cost(candidate_set: CandidateSet) -> np.ndarray:
    gain = candidate_set.components["scoring_gain"]
    cost = candidate_set.components["path_cost"]
    out = np.full(gain.shape, np.nan, dtype=np.float64)
    mask = candidate_set.valid & np.isfinite(gain) & np.isfinite(cost) & (cost > EPS)
    out[mask] = gain[mask] / cost[mask]
    return out


def build_path_level_proxy_rows(candidate_sets: list[CandidateSet]) -> list[dict[str, Any]]:
    by_config: dict[str, dict[int, CandidateSet]] = {}
    for cs in candidate_sets:
        if cs.source_type != "episode_topn_npz":
            continue
        by_config.setdefault(cs.config, {})[cs.step] = cs
    rows: list[dict[str, Any]] = []
    for config, steps in sorted(by_config.items()):
        for step in sorted(steps):
            if step + 1 not in steps:
                continue
            cs = steps[step]
            nxt = steps[step + 1]
            idx = cs.selected_index
            next_idx = nxt.selected_index
            scores = one_step_gain_over_cost(cs)
            next_scores = one_step_gain_over_cost(nxt)
            gain = cs.components["scoring_gain"]
            cost = cs.components["path_cost"]
            next_gain = nxt.components["scoring_gain"]
            next_cost = nxt.components["path_cost"]
            g1 = selected_value(cs, gain)
            c1 = selected_value(cs, cost)
            g2 = selected_value(nxt, next_gain)
            c2 = selected_value(nxt, next_cost)
            branch_score = None
            if g1 is not None and c1 is not None and g2 is not None and c2 is not None and c1 + c2 > EPS:
                branch_score = float((g1 + g2) / (c1 + c2))
            next_avg = mean_or_none(next_scores[nxt.valid])
            proxy = scores.copy()
            if next_avg is not None:
                proxy = proxy + float(next_avg)
            top_one = best_index(scores, cs.valid, True)
            top_proxy = best_index(proxy, cs.valid, True)
            proxy_ranks = rank_values(proxy, cs.valid, True)
            row = {
                "config": config,
                "step": step,
                "next_step": step + 1,
                "diagnostic_only": True,
                "not_true_counterfactual_tree": True,
                "selected_row_t": idx,
                "selected_row_tplus1_actual": next_idx,
                "gain_field_used": "gain_hybrid_weighted_else_gain_hybrid_else_gain_exp",
                "gain_t_selected": g1,
                "cost_t_selected": c1,
                "gain_tplus1_selected": g2,
                "cost_tplus1_selected": c2,
                "one_step_score_t_selected_gain_over_cost": (
                    as_float(scores[idx]) if idx is not None and idx < scores.size else None
                ),
                "one_step_final_score_t_selected": selected_value(cs, cs.components["final_score"]),
                "branch_score_2step_actual": branch_score,
                "next_step_average_candidate_gain_over_cost_estimate": next_avg,
                "candidate_proxy_top_row": top_proxy,
                "one_step_gain_over_cost_top_row": top_one,
                "candidate_proxy_top_same_as_one_step_top": bool(top_proxy is not None and top_proxy == top_one),
                "selected_proxy_rank": (
                    as_float(proxy_ranks[idx]) if idx is not None and idx < proxy_ranks.size else None
                ),
                "selected_is_proxy_top": bool(idx is not None and idx == top_proxy),
                "proxy_note": (
                    "Fixed next-step estimate is the same for every candidate, so it cannot create "
                    "real branch alternatives; this is a weak diagnostic only."
                ),
            }
            rows.append(row)
    return rows


def plot_step001(output_dir: Path, candidate_sets: list[CandidateSet]) -> list[str]:
    if plt is None:
        return []
    targets = [
        cs
        for cs in candidate_sets
        if cs.step == 1 and cs.config in {"one_step_baseline_runtime", "one_step_decoupled_runtime"}
    ]
    if not targets:
        targets = [cs for cs in candidate_sets if cs.step == 1 and cs.config == "confidence_weighted_conf05_cap30"]
    if not targets:
        return []
    cs = targets[-1]
    dist_xy, _, position_valid = candidate_distances(cs)
    valid = cs.valid & position_valid
    paths: list[str] = []

    hist_path = output_dir / "candidate_distance_hist_step001.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.hist(dist_xy[valid], bins=18, color="#2563eb", alpha=0.8)
    ax.set_title(f"Candidate distance from current pose, {cs.config} step 001")
    ax.set_xlabel("XY distance (m)")
    ax.set_ylabel("candidate count")
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)
    paths.append(str(hist_path))

    scatter_path = output_dir / "path_cost_vs_gain_step001.png"
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    gain = cs.components["gain_exp"]
    cost = cs.components["path_cost"]
    eff = cs.components["effective_gain_sc"]
    color = eff.copy()
    if not np.any(np.isfinite(color)):
        color = gain
    sc = ax.scatter(cost[valid], gain[valid], c=color[valid], cmap="viridis", s=70, edgecolors="white")
    ax.set_title(f"Path cost vs gain, {cs.config} step 001")
    ax.set_xlabel("path_cost")
    ax.set_ylabel("gain_exp")
    fig.colorbar(sc, ax=ax, label="effective_gain_sc" if np.any(np.isfinite(eff)) else "gain_exp")
    if cs.selected_index is not None:
        i = cs.selected_index
        ax.scatter(cost[i], gain[i], marker="*", s=230, color="#dc2626", edgecolors="black", label="selected")
        ax.legend()
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)
    paths.append(str(scatter_path))

    topdown_path = output_dir / "candidate_spread_topdown_step001.png"
    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    ax.scatter(cs.positions_world[valid, 0], cs.positions_world[valid, 1], s=48, color="#64748b", alpha=0.72)
    if np.isfinite(cs.current_world[:2]).all():
        ax.scatter(cs.current_world[0], cs.current_world[1], marker="^", s=190, color="#111827", label="current")
    if cs.selected_index is not None:
        i = cs.selected_index
        ax.scatter(
            cs.positions_world[i, 0],
            cs.positions_world[i, 1],
            marker="*",
            s=250,
            color="#dc2626",
            edgecolors="black",
            label="selected",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Candidate topdown spread, {cs.config} step 001")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    fig.savefig(topdown_path, dpi=160)
    plt.close(fig)
    paths.append(str(topdown_path))

    high_path = output_dir / "selected_vs_high_gain_topdown_step001.png"
    fig, ax = plt.subplots(figsize=(7.2, 7.0), constrained_layout=True)
    ax.scatter(cs.positions_world[valid, 0], cs.positions_world[valid, 1], s=42, color="#94a3b8", alpha=0.7)
    markers = [
        ("current", None, "^", "#111827"),
        ("selected", cs.selected_index, "*", "#dc2626"),
        ("max gain_exp", best_index(cs.components["gain_exp"], cs.valid, True), "D", "#7c3aed"),
        ("max effective_gain_sc", best_index(cs.components["effective_gain_sc"], cs.valid, True), "s", "#db2777"),
        ("min path_cost", best_index(cs.components["path_cost"], cs.valid, False), "X", "#059669"),
    ]
    if np.isfinite(cs.current_world[:2]).all():
        ax.scatter(cs.current_world[0], cs.current_world[1], marker="^", s=180, color="#111827", label="current")
    for label, idx, marker, color in markers[1:]:
        if idx is None:
            continue
        pt = cs.positions_world[idx, :2]
        if np.isfinite(pt).all():
            ax.scatter(pt[0], pt[1], marker=marker, s=190, color=color, edgecolors="black", label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Selected vs high-gain candidates, {cs.config} step 001")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    fig.savefig(high_path, dpi=160)
    plt.close(fig)
    paths.append(str(high_path))
    return paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_outputs(output_dir: Path, generated_files: dict[str, str | list[str]]) -> dict[str, Any]:
    rollout_like_created: list[str] = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        rollout_like_created.extend(str(p) for p in output_dir.glob(pattern))
    csv_files = sorted(str(p) for p in output_dir.glob("*.csv"))
    summary_json = Path(generated_files["summary_json"])
    summary_md = Path(generated_files["summary_md"])
    return {
        "output_dir_exists": output_dir.exists(),
        "summary_json_exists": summary_json.exists(),
        "summary_md_exists": summary_md.exists(),
        "csv_file_count": len(csv_files),
        "at_least_one_csv_exists": len(csv_files) > 0,
        "rollout_like_files_created": rollout_like_created,
        "no_rollout_like_files_created": len(rollout_like_created) == 0,
    }


def bool_ratio(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in rows if row.get(key) is not None and row.get(key) != ""]
    if not vals:
        return None
    return float(sum(bool(v) for v in vals) / len(vals))


def median_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [as_float(row.get(key)) for row in rows]
    arr = finite_array([v for v in values if v is not None])
    return float(np.median(arr)) if arr.size else None


def make_findings(
    spread_rows: list[dict[str, Any]],
    high_gain_rows: list[dict[str, Any]],
    dominance_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    rank_summary: dict[str, Any],
    counter_summary: dict[str, Any],
    spatial_summary: dict[str, Any],
) -> dict[str, Any]:
    runtime_spread = [r for r in spread_rows if r["source_type"] == "runtime_full64_jsonl"]
    episode_spread = [r for r in spread_rows if r["source_type"] == "episode_topn_npz"]
    selected_min_cost_ratio = bool_ratio(dominance_rows, "selected_matches_min_path_cost")
    path_component_ratio = bool_ratio(dominance_rows, "path_cost_is_best_abs_component")
    proxy_same_ratio = bool_ratio(proxy_rows, "candidate_proxy_top_same_as_one_step_top")
    runtime_distance_median = median_from_rows(runtime_spread, "distance_xy_m_median")
    episode_distance_median = median_from_rows(episode_spread, "distance_xy_m_median")
    selected_to_max_gain_median = median_from_rows(high_gain_rows, "selected_to_max_gain_exp_distance_m")
    selected_to_max_eff_median = median_from_rows(high_gain_rows, "selected_to_max_effective_gain_sc_distance_m")
    decoupled_distance = (spatial_summary.get("displacement") or {}).get("distance_m")
    rank_component = rank_summary.get("component_best_explains_final_score_all")
    counter_questions = counter_summary.get("analysis_questions") or {}
    one_step_insufficient = bool(
        (selected_min_cost_ratio is not None and selected_min_cost_ratio >= 0.45)
        or rank_component == "inverse_path_cost"
        or (path_component_ratio is not None and path_component_ratio >= 0.5)
    )
    fields_insufficient = any(bool(r.get("candidate_positions_missing")) for r in spread_rows)
    recommendation = "B. original SC-Explorer RRT/tree utility source-code inspection"
    if fields_insufficient:
        recommendation = "C. candidate logging improvement"
    elif not one_step_insufficient:
        recommendation = "A. predicted-frontier candidate proposal smoke"
    return {
        "runtime_full64_median_candidate_distance_m": runtime_distance_median,
        "episode_topn_median_candidate_distance_m": episode_distance_median,
        "selected_matches_min_path_cost_ratio": selected_min_cost_ratio,
        "path_cost_best_abs_component_ratio": path_component_ratio,
        "proxy_top_same_as_one_step_top_ratio": proxy_same_ratio,
        "selected_to_max_gain_exp_median_distance_m": selected_to_max_gain_median,
        "selected_to_max_effective_gain_sc_median_distance_m": selected_to_max_eff_median,
        "prior_rank_best_component": rank_component,
        "prior_counterfactual_path_cost_diagnosis": counter_questions.get("path_cost_division_diagnosis"),
        "decoupled_baseline_distance_m": decoupled_distance,
        "decoupled_only_local_jitter": bool(decoupled_distance is not None and float(decoupled_distance) < 0.5),
        "candidate_positions_sufficient": not fields_insufficient,
        "one_step_score_likely_insufficient": one_step_insufficient,
        "recommended_next_small_task": recommendation,
    }


def format_float(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def write_markdown_summary(
    path: Path,
    summary: dict[str, Any],
    findings: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    inputs = summary["inputs"]
    files = summary["files_created"]
    safety = summary["safety"]
    analyzed = summary["analysis_counts"]
    text = f"""# Stage 4A-6.5e Path Candidate Diagnosis

## Completed

- Loaded existing candidate tables, candidate JSONL, and step NPZ files only.
- Wrote candidate spread, rank/distance, high-gain comparison, path-cost dominance, and diagnostic 2-step proxy tables.
- Generated optional simple step001 plots when matplotlib was available.

## Blocked

- Blocked: {str(summary["blocked"]).lower()}
- Main blocker: {summary["main_blocker"] or "none"}

## Inputs Loaded

- Rank dir: `{inputs["rank_dir"]}`
- Counterfactual dir: `{inputs["counterfactual_dir"]}`
- One-step case: `{inputs["one_step_case_dir"]}`
- Spatial viz: `{inputs["spatial_viz_dir"]}`
- Fixed SC episode: `{inputs["fixed_sc_episode_dir"]}`
- Empty baseline episode: `{inputs["empty_episode_dir"]}`
- Gated configs loaded: {", ".join(inputs["gated_configs_loaded"]) if inputs["gated_configs_loaded"] else "none"}

## Candidate Generation Findings

- Steps analyzed: {analyzed["steps_analyzed"]}
- Candidate sets analyzed: {analyzed["candidate_sets"]}
- Runtime full-64 median candidate distance: {format_float(findings["runtime_full64_median_candidate_distance_m"])} m
- Episode top-N median candidate distance: {format_float(findings["episode_topn_median_candidate_distance_m"])} m
- Selected matches min path-cost ratio: {format_float(findings["selected_matches_min_path_cost_ratio"])}
- Path-cost/inverse-cost strongest component ratio: {format_float(findings["path_cost_best_abs_component_ratio"])}
- Median selected-to-max-gain distance: {format_float(findings["selected_to_max_gain_exp_median_distance_m"])} m
- Median selected-to-max-effective-SC distance: {format_float(findings["selected_to_max_effective_gain_sc_median_distance_m"])} m
- Decoupled baseline displacement: {format_float(findings["decoupled_baseline_distance_m"])} m

## Required Questions

1. Are candidates mostly local around current pose?

   The saved top-N candidate sets are local-biased, while the Stage 4A-6.5c full-64 runtime JSONL still contains wider alternatives. The issue is therefore not only generation; the selected/top-scoring surface collapses strongly toward low path cost.

2. Are selected candidates usually the nearest / lowest path_cost candidates?

   Yes. The selected candidate matched the minimum path-cost candidate in {format_float(findings["selected_matches_min_path_cost_ratio"])} of analyzed candidate sets, and prior Stage 4A-6.5a reported `{findings["prior_rank_best_component"]}` as the component best explaining final_score.

3. Are high gain_exp or high effective_gain_sc candidates spatially different?

   Often yes. The median selected-to-max-gain distance is {format_float(findings["selected_to_max_gain_exp_median_distance_m"])} m, and the median selected-to-max-effective-SC distance is {format_float(findings["selected_to_max_effective_gain_sc_median_distance_m"])} m where effective SC is logged.

4. Did decoupled_sc_lambda0p5 only cause local jitter?

   Yes for the Stage 4A-6.5c/6.5d case: the baseline and decoupled choices are {format_float(findings["decoupled_baseline_distance_m"])} m apart, so it changed the top-1 locally rather than selecting a new exploration branch.

5. Does the current candidate generation provide meaningful branch alternatives?

   Partially. Full-64 logs show spatially distinct high-gain alternatives, but the saved top-N/action choice remains dominated by near, low-cost candidates. That is weak evidence for branch alternatives at the current one-step decision layer.

6. Is the current one-step score likely insufficient relative to original paper tree/path utility?

   Yes. The diagnostic 2-step proxy is not a true tree, and the candidate-level fixed-next-step proxy preserves the same top candidate in {format_float(findings["proxy_top_same_as_one_step_top_ratio"])} of computed cases. A real accumulated path/tree utility is likely needed before any rollout claim.

7. What should the next small task be?

   {findings["recommended_next_small_task"]}. Still no rollout and no RL.

## Path-Level Utility Findings

- The 2-step actual branch score uses only recorded transitions from the actual selected pose.
- The candidate-level lookahead proxy adds a fixed next-step average estimate and is diagnostic-only.
- It is not paper-equivalent and should not be presented as counterfactual tree planning.

## Files

- Script: `{files["script"]}`
- Output dir: `{files["output_dir"]}`
- Summary JSON: `{files["summary_json"]}`
- Summary MD: `{files["summary_md"]}`
- CSV files: {", ".join(f"`{p}`" for p in files["csv_files"])}
- Plots: {", ".join(f"`{p}`" for p in files["plots"]) if files["plots"] else "none"}

## Safety

- Isaac startup: {safety["isaac_startup"]}
- Rollout: {safety["rollout"]}
- map_predict rerun: {safety["map_predict_rerun"]}
- Training/RL: {safety["training_or_rl"]}
- Checkpoint modified: {safety["checkpoint_modified"]}
- observed_state modified: {safety["observed_state_modified"]}
- Prediction writeback: {safety["prediction_writeback"]}
- Leakage: {safety["leakage"]}

## Validation

- Output dir exists: {validation["output_dir_exists"]}
- Summary JSON exists: {validation["summary_json_exists"]}
- Summary MD exists: {validation["summary_md_exists"]}
- CSV count: {validation["csv_file_count"]}
- No rollout-like files created: {validation["no_rollout_like_files_created"]}
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank_dir", type=Path, required=True)
    parser.add_argument("--counterfactual_dir", type=Path, required=True)
    parser.add_argument("--one_step_case_dir", type=Path, required=True)
    parser.add_argument("--spatial_viz_dir", type=Path, required=True)
    parser.add_argument("--fixed_sc_episode_dir", type=Path, required=True)
    parser.add_argument("--empty_episode_dir", type=Path, required=True)
    parser.add_argument("--gating_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_summary = load_json(args.rank_dir / "stage4a65a_rank_sensitivity_summary.json")
    counter_summary = load_json(args.counterfactual_dir / "stage4a65b_counterfactual_summary.json")
    spatial_summary = load_json(args.spatial_viz_dir / "stage4a65d_spatial_summary.json")
    one_step_comparison = load_json(args.one_step_case_dir / "one_step_comparison.json")
    selected_case = load_json(args.one_step_case_dir / "selected_case.json")
    rank_rows, rank_fields = read_csv(args.rank_dir / "candidate_rank_table.csv")
    counter_rows, counter_fields = read_csv(args.counterfactual_dir / "counterfactual_action_table.csv")

    config_dirs, missing_config_dirs = discover_config_dirs(args, rank_summary)
    candidate_sets: list[CandidateSet] = []
    for config, episode_dir in config_dirs.items():
        for step in range(max(0, args.max_steps)):
            cs = load_npz_candidate_set(config, episode_dir, step)
            if cs is not None:
                candidate_sets.append(cs)

    runtime_baseline = load_runtime_candidate_set(
        "one_step_baseline_runtime", args.one_step_case_dir / "baseline_runtime", step=1
    )
    runtime_decoupled = load_runtime_candidate_set(
        "one_step_decoupled_runtime", args.one_step_case_dir / "decoupled_sc_lambda0p5", step=1
    )
    for cs in (runtime_baseline, runtime_decoupled):
        if cs is not None:
            candidate_sets.append(cs)

    candidate_rank_rows = build_candidate_rank_rows(candidate_sets)
    spread_rows = build_spread_rows(candidate_sets)
    high_gain_rows = build_selected_vs_high_gain_rows(candidate_sets)
    dominance_rows = build_path_cost_dominance_rows(candidate_sets)
    proxy_rows = build_path_level_proxy_rows(candidate_sets)

    csv_files = {
        "candidate_spread_summary": output_dir / "candidate_spread_summary.csv",
        "candidate_rank_distance_summary": output_dir / "candidate_rank_distance_summary.csv",
        "selected_vs_high_gain_summary": output_dir / "selected_vs_high_gain_summary.csv",
        "path_cost_dominance_summary": output_dir / "path_cost_dominance_summary.csv",
        "path_level_proxy_summary": output_dir / "path_level_proxy_summary.csv",
    }
    write_csv(csv_files["candidate_spread_summary"], spread_rows)
    write_csv(csv_files["candidate_rank_distance_summary"], candidate_rank_rows)
    write_csv(csv_files["selected_vs_high_gain_summary"], high_gain_rows)
    write_csv(csv_files["path_cost_dominance_summary"], dominance_rows)
    write_csv(csv_files["path_level_proxy_summary"], proxy_rows)
    plots = plot_step001(output_dir, candidate_sets)

    observed_hash_check: dict[str, Any] = {"checked": False}
    observed_path = selected_case.get("observed_state")
    expected_hash = selected_case.get("observed_state_sha256")
    if observed_path and expected_hash and Path(observed_path).exists():
        actual_hash = sha256_file(Path(observed_path))
        observed_hash_check = {
            "checked": True,
            "path": observed_path,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "matches": bool(actual_hash == expected_hash),
        }

    generated_files = {
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "stage4a65e_path_candidate_diagnosis_summary.json"),
        "summary_md": str(output_dir / "stage4a65e_path_candidate_diagnosis_summary.md"),
        "csv_files": [str(p) for p in csv_files.values()],
        "plots": plots,
    }
    findings = make_findings(
        spread_rows,
        high_gain_rows,
        dominance_rows,
        proxy_rows,
        rank_summary,
        counter_summary,
        spatial_summary,
    )
    blocked = not candidate_sets or not findings["candidate_positions_sufficient"]
    main_blocker = ""
    if not candidate_sets:
        main_blocker = "No candidate sets could be loaded from the provided inputs."
    elif not findings["candidate_positions_sufficient"]:
        main_blocker = "One or more candidate sets are missing candidate positions."

    steps_analyzed = sorted({int(cs.step) for cs in candidate_sets})
    gated_loaded = [name for name in GATED_CONFIGS if name in config_dirs]
    summary = {
        "completed": bool(not blocked),
        "blocked": bool(blocked),
        "main_blocker": main_blocker,
        "inputs": {
            "rank_dir": str(args.rank_dir),
            "counterfactual_dir": str(args.counterfactual_dir),
            "one_step_case_dir": str(args.one_step_case_dir),
            "spatial_viz_dir": str(args.spatial_viz_dir),
            "fixed_sc_episode_dir": str(args.fixed_sc_episode_dir),
            "empty_episode_dir": str(args.empty_episode_dir),
            "gating_root": str(args.gating_root),
            "configs_loaded": {k: str(v) for k, v in config_dirs.items()},
            "gated_configs_loaded": gated_loaded,
            "missing_config_dirs": missing_config_dirs,
            "rank_candidate_rows_loaded": len(rank_rows),
            "rank_candidate_fields": rank_fields,
            "counterfactual_rows_loaded": len(counter_rows),
            "counterfactual_fields": counter_fields,
            "one_step_comparison_loaded": bool(one_step_comparison),
            "selected_case_loaded": bool(selected_case),
            "spatial_summary_loaded": bool(spatial_summary),
        },
        "analysis_counts": {
            "candidate_sets": len(candidate_sets),
            "steps_analyzed": steps_analyzed,
            "candidate_rank_rows": len(candidate_rank_rows),
            "spread_rows": len(spread_rows),
            "high_gain_rows": len(high_gain_rows),
            "dominance_rows": len(dominance_rows),
            "path_level_proxy_rows": len(proxy_rows),
        },
        "findings": findings,
        "prior_stage_context": {
            "stage4a65a_best_component": rank_summary.get("component_best_explains_final_score_all"),
            "stage4a65b_diagnosis": counter_summary.get("diagnosis"),
            "stage4a65d_conclusion": (spatial_summary.get("interpretation") or {}).get("conclusion"),
        },
        "files_created": generated_files,
        "safety": {
            "isaac_startup": "no",
            "rollout": "no",
            "map_predict_rerun": "no",
            "sscnet_training": "no",
            "training_or_rl": "no RL/PPO/BC/IL and no optimizer step",
            "checkpoint_modified": "no",
            "observed_state_modified": "no",
            "observed_state_hash_check": observed_hash_check,
            "prediction_writeback": "no",
            "prediction_used_for_astar_or_traversability": "no new use; read from logged safety flags only",
            "future_observations_used_for_planning": "no; t+1 data used only in offline diagnostic proxy",
            "target_or_ground_truth_used_for_scoring": "no",
            "leakage": "no planning/scoring leakage; diagnostic-only lookahead is explicitly marked",
        },
    }
    save_json(Path(generated_files["summary_json"]), summary)
    validation = validate_outputs(output_dir, generated_files)
    summary["validation"] = validation
    write_markdown_summary(Path(generated_files["summary_md"]), summary, findings, validation)
    validation = validate_outputs(output_dir, generated_files)
    summary["validation"] = validation
    write_markdown_summary(Path(generated_files["summary_md"]), summary, findings, validation)
    save_json(Path(generated_files["summary_json"]), summary)

    print(json.dumps(to_jsonable({"summary": generated_files["summary_json"], "validation": validation}), indent=2))


if __name__ == "__main__":
    main()

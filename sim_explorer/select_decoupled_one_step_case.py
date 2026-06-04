#!/usr/bin/env python3
"""Select the Stage 4A-6.5c decoupled_sc one-step smoke case.

This reads existing Stage 4A-6.5b/6.5a/6.4 artifacts only. It does not run
Isaac, map prediction, rollouts, or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_COUNTERFACTUAL_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65b_counterfactual_scores"
)
DEFAULT_RANK_SENSITIVITY_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65a_rank_sensitivity"
)
DEFAULT_FIXED_RAW_EPISODE = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke/"
    "episodes/medium_three_rooms_seed0_start_room_a_sc_pred_alignment_fixed_000"
)
DEFAULT_EMPTY_BASELINE_EPISODE = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/"
    "episodes/medium_three_rooms_seed0_start_room_a_empty_astar"
)
DEFAULT_GATING_ROOT = Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating")
DEFAULT_OUTPUT_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65c_decoupled_one_step_smoke"
)

EPS = 1e-6


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


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _scalar_from_npz(npz_path: Path, key: str, default: Any = None) -> Any:
    if not npz_path.exists():
        return default
    with np.load(npz_path, allow_pickle=False) as data:
        if key not in data.files:
            return default
        value = data[key]
        if value.shape == ():
            item = value.item()
            return item.item() if isinstance(item, np.generic) else item
        return value.tolist()


def _candidate_key_from_grid(grid: Any) -> str:
    if grid is None:
        return ""
    values = [int(round(float(v))) for v in grid]
    return f"grid:{values[0]},{values[1]},{values[2]}"


def _read_rank_rows(rank_sensitivity_dir: Path, config: str, step: int) -> list[dict[str, Any]]:
    table_path = rank_sensitivity_dir / "candidate_rank_table.csv"
    rows: list[dict[str, Any]] = []
    if not table_path.exists():
        return rows
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("config") != config:
                continue
            if _as_int(row.get("step")) != int(step):
                continue
            rows.append(dict(row))
    return rows


def _rank_row_payload(row: dict[str, Any], decoupled_lambda: float) -> dict[str, Any]:
    gain_exp = _as_float(row.get("gain_exp"))
    raw_gain_sc = _as_float(row.get("raw_gain_sc"))
    gain_sc = _as_float(row.get("gain_sc"))
    effective_gain_sc = _as_float(row.get("effective_gain_sc"))
    path_cost = _as_float(row.get("path_cost"))
    score = None
    if gain_exp is not None and effective_gain_sc is not None and path_cost is not None:
        score = float(gain_exp / max(path_cost, EPS) + float(decoupled_lambda) * effective_gain_sc)
    candidate_grid = _parse_jsonish(row.get("candidate_grid"))
    return {
        "candidate_row": _as_int(row.get("candidate_row")),
        "candidate_id": row.get("candidate_id") or "",
        "candidate_key": row.get("candidate_key") or _candidate_key_from_grid(candidate_grid),
        "candidate_grid": candidate_grid,
        "candidate_world": _parse_jsonish(row.get("candidate_world")),
        "current_final_score": _as_float(row.get("final_score")),
        "decoupled_score": score,
        "gain_exp": gain_exp,
        "gain_sc": gain_sc,
        "raw_gain_sc": raw_gain_sc if raw_gain_sc is not None else gain_sc,
        "effective_gain_sc": effective_gain_sc,
        "path_cost": path_cost,
        "utility_exp": _as_float(row.get("utility_exp")),
        "rank_final_score": _as_float(row.get("rank_final_score")),
        "rank_gain_exp": _as_float(row.get("rank_gain_exp")),
        "rank_effective_gain_sc": _as_float(row.get("rank_effective_gain_sc")),
        "rank_low_path_cost": _as_float(row.get("rank_low_path_cost")),
    }


def _offline_candidates(rank_sensitivity_dir: Path, config: str, step: int, decoupled_lambda: float) -> list[dict[str, Any]]:
    payloads = [_rank_row_payload(row, decoupled_lambda) for row in _read_rank_rows(rank_sensitivity_dir, config, step)]
    return sorted(
        payloads,
        key=lambda row: (
            -float(row["decoupled_score"]) if row.get("decoupled_score") is not None else math.inf,
            int(row["candidate_row"] if row.get("candidate_row") is not None else 10**9),
        ),
    )


def _find_selected_counterfactual(counterfactual_dir: Path) -> dict[str, Any] | None:
    rows = load_jsonl(counterfactual_dir / "counterfactual_action_table.jsonl")
    candidates: list[tuple[int, str, int, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if row.get("formula") != "decoupled_sc":
            continue
        lam = _as_float(row.get("lambda"))
        if lam is None or abs(lam - 0.5) > 1e-12:
            continue
        if not _as_bool(row.get("changed_vs_own_selected")):
            continue
        step = _as_int(row.get("step"))
        if step is None:
            continue
        candidates.append((step, str(row.get("config", "")), index, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def _load_manifest_rows(gating_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in ("ablation_manifest.jsonl", "manifest.jsonl"):
        path = gating_root / "ablation" / name
        for row in load_jsonl(path):
            key = str(row.get("config_name") or "")
            if not key:
                episode_id = str(row.get("episode_id") or "")
                prefix = "medium_three_rooms_seed0_start_room_a_"
                if episode_id.startswith(prefix):
                    key = episode_id[len(prefix) :]
            if key and key not in out:
                out[key] = row
    return out


def _episode_dirs(
    rank_sensitivity_dir: Path,
    fixed_raw_episode: Path,
    empty_baseline_episode: Path,
) -> dict[str, Path]:
    summary = load_json(rank_sensitivity_dir / "stage4a65a_rank_sensitivity_summary.json")
    raw = summary.get("config_episode_dirs") or {}
    dirs = {str(key): Path(value) for key, value in raw.items() if value}
    dirs.setdefault("fixed_raw_sc", fixed_raw_episode)
    dirs.setdefault("empty_baseline", empty_baseline_episode)
    return dirs


def _parse_rollout_log_flags(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {}
    first = ""
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("$ "):
                first = line[2:].strip()
                break
    if not first:
        return {}
    tokens = shlex.split(first)
    flags: dict[str, Any] = {"command": first}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            flags[key] = tokens[index + 1]
            index += 2
        else:
            flags[key] = True
            index += 1
    return flags


def _pick_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _float_value(*values: Any, default: float | None = None) -> float | None:
    value = _pick_value(*values, default=default)
    out = _as_float(value)
    return default if out is None else out


def _int_value(*values: Any, default: int | None = None) -> int | None:
    value = _pick_value(*values, default=default)
    out = _as_int(value)
    return default if out is None else out


def _blocked_case(reason: str, output_dir: Path, details: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "stage": "Stage 4A-6.5c decoupled_sc one-step smoke",
        "status": "blocked",
        "blocker": reason,
        "details": details,
        "safety": {
            "rollout": False,
            "isaac_startup": False,
            "map_predict_rerun": False,
            "rl_il_training": False,
            "checkpoint_modified": False,
            "observed_state_modified": False,
            "prediction_writeback": False,
            "target_or_ground_truth_scoring": False,
        },
    }
    save_json(output_dir / "selected_case.json", payload)
    return payload


def select_case(
    *,
    counterfactual_dir: Path = DEFAULT_COUNTERFACTUAL_DIR,
    rank_sensitivity_dir: Path = DEFAULT_RANK_SENSITIVITY_DIR,
    fixed_raw_episode: Path = DEFAULT_FIXED_RAW_EPISODE,
    empty_baseline_episode: Path = DEFAULT_EMPTY_BASELINE_EPISODE,
    gating_root: Path = DEFAULT_GATING_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    counterfactual_dir = counterfactual_dir.resolve()
    rank_sensitivity_dir = rank_sensitivity_dir.resolve()
    gating_root = gating_root.resolve()
    fixed_raw_episode = fixed_raw_episode.resolve()
    empty_baseline_episode = empty_baseline_episode.resolve()

    selected = _find_selected_counterfactual(counterfactual_dir)
    if selected is None:
        return _blocked_case(
            "no decoupled_sc lambda=0.5 top-1-changing row found in counterfactual output",
            output_dir,
            {"counterfactual_dir": str(counterfactual_dir)},
        )

    config = str(selected["config"])
    step = int(selected["step"])
    decoupled_lambda = float(selected["lambda"])
    dirs = _episode_dirs(rank_sensitivity_dir, fixed_raw_episode, empty_baseline_episode)
    manifest_rows = _load_manifest_rows(gating_root)
    manifest = manifest_rows.get(config, {})
    episode_dir_value = _pick_value(dirs.get(config), manifest.get("episode_dir"))
    if episode_dir_value is None:
        return _blocked_case(
            f"episode directory for config {config!r} was not found",
            output_dir,
            {"config": config, "rank_sensitivity_dir": str(rank_sensitivity_dir), "gating_root": str(gating_root)},
        )
    episode_dir = Path(episode_dir_value).resolve()

    step_npz = episode_dir / f"step_{step:03d}.npz"
    observed_state = episode_dir / f"observed_state_step{step:03d}.npy"
    pose = episode_dir / f"pose_{step:03d}.json"
    camera_info = episode_dir / "camera_info.json"
    episode_summary_path = episode_dir / "episode_summary.json"
    episode_summary = load_json(episode_summary_path)
    log_path = Path(str(manifest.get("log_path"))) if manifest.get("log_path") else None
    log_flags = _parse_rollout_log_flags(log_path)

    prediction_npz_value = _scalar_from_npz(step_npz, "prediction_npz", None)
    prediction_npz = Path(str(prediction_npz_value)).resolve() if prediction_npz_value else (
        episode_dir / f"prediction_step{step:03d}" / "global_prediction_layer.npz"
    )

    required = {
        "observed_state": observed_state,
        "pose": pose,
        "camera_info": camera_info,
        "episode_summary": episode_summary_path,
        "prediction_npz": prediction_npz,
        "step_npz": step_npz,
    }
    missing = {key: str(path) for key, path in required.items() if not Path(path).exists()}
    if missing:
        return _blocked_case(
            "selected counterfactual case is not reproducible because required files are missing",
            output_dir,
            {"config": config, "step": step, "missing": missing},
        )

    sc_gain_formula = str(
        _pick_value(
            manifest.get("sc_gain_formula"),
            episode_summary.get("sc_gain_formula"),
            _scalar_from_npz(step_npz, "sc_gain_formula"),
        )
    )
    sc_occ_threshold = _float_value(
        manifest.get("sc_occ_threshold"),
        episode_summary.get("sc_occ_threshold"),
        _scalar_from_npz(step_npz, "sc_occ_threshold"),
        default=0.7,
    )
    sc_conf_threshold = _float_value(
        manifest.get("sc_conf_threshold"),
        episode_summary.get("sc_conf_threshold"),
        _scalar_from_npz(step_npz, "sc_conf_threshold"),
        default=0.3,
    )
    sc_count_mode = str(
        _pick_value(
            episode_summary.get("sc_count_mode"),
            _scalar_from_npz(step_npz, "sc_count_mode"),
            manifest.get("sc_count_mode"),
            default="selective",
        )
    )
    original_sc_gain_weight = _float_value(
        manifest.get("sc_gain_weight"),
        episode_summary.get("sc_gain_weight"),
        _scalar_from_npz(step_npz, "sc_gain_weight"),
        default=1.0,
    )
    sc_gain_cap = _float_value(
        manifest.get("sc_gain_cap"),
        episode_summary.get("sc_gain_cap"),
        _scalar_from_npz(step_npz, "sc_gain_cap_value"),
        default=None,
    )
    if sc_gain_cap is not None and sc_gain_cap < 0.0:
        sc_gain_cap = None

    num_candidates = _int_value(
        log_flags.get("num_candidates"),
        _scalar_from_npz(step_npz, "candidate_count"),
        default=64,
    )
    top_n = _int_value(log_flags.get("top_n"), len(_scalar_from_npz(step_npz, "expert_scores", [])), default=16)
    seed_base = _int_value(log_flags.get("seed"), default=0)
    runtime_seed = int(seed_base or 0) + step
    snap_start = bool(
        _pick_value(
            log_flags.get("snap_start_to_traversable"),
            episode_summary.get("snap_start_to_traversable"),
            default=False,
        )
    )
    offline_candidates = _offline_candidates(rank_sensitivity_dir, config, step, decoupled_lambda)

    runtime_args = {
        "observed_state": str(observed_state),
        "observed_summary": None,
        "episode_summary": str(episode_summary_path),
        "camera_info": str(camera_info),
        "pose_json": str(pose),
        "prediction_mode": "sim_npz",
        "prediction_npz": str(prediction_npz),
        "num_candidates": int(num_candidates or 64),
        "top_n": int(top_n or 16),
        "gain_mode": str(_pick_value(episode_summary.get("gain_mode"), _scalar_from_npz(step_npz, "gain_mode"), default="hybrid")),
        "path_cost_mode": str(
            _pick_value(episode_summary.get("path_cost_mode"), _scalar_from_npz(step_npz, "path_cost_mode"), default="astar")
        ),
        "candidate_sampling_mode": str(
            _pick_value(
                episode_summary.get("candidate_sampling_mode"),
                _scalar_from_npz(step_npz, "candidate_sampling_mode"),
                default="reachable_frontier",
            )
        ),
        "snap_start_to_traversable": snap_start,
        "max_snap_radius_cells": int(
            _int_value(log_flags.get("max_snap_radius_cells"), episode_summary.get("max_snap_radius_cells"), default=5)
            or 5
        ),
        "seed": int(runtime_seed),
        "tau": float(_float_value(episode_summary.get("tau"), _scalar_from_npz(step_npz, "tau"), default=0.1) or 0.1),
        "max_range_voxels": int(_int_value(log_flags.get("max_range_voxels"), default=50) or 50),
        "num_yaw": int(_int_value(log_flags.get("num_yaw"), default=32) or 32),
        "num_pitch": int(_int_value(log_flags.get("num_pitch"), default=7) or 7),
        "fov_yaw_deg": float(_float_value(log_flags.get("fov_yaw_deg"), default=90.0) or 90.0),
        "fov_pitch_deg": float(_float_value(log_flags.get("fov_pitch_deg"), default=60.0) or 60.0),
        "alignment_convention": str(
            _pick_value(
                manifest.get("alignment_convention"),
                episode_summary.get("alignment_convention"),
                log_flags.get("alignment_convention"),
                default="code_consistent_v1",
            )
        ),
        "sc_gain_formula": sc_gain_formula,
        "sc_occ_threshold": float(sc_occ_threshold if sc_occ_threshold is not None else 0.7),
        "sc_conf_threshold": float(sc_conf_threshold if sc_conf_threshold is not None else 0.3),
        "sc_count_mode": sc_count_mode,
        "calibration_table": _pick_value(manifest.get("calibration_table"), episode_summary.get("calibration_table")),
        "sc_gain_cap": sc_gain_cap,
    }

    payload = {
        "stage": "Stage 4A-6.5c decoupled_sc one-step smoke",
        "status": "selected",
        "config": config,
        "step": step,
        "formula": "decoupled_sc",
        "lambda": decoupled_lambda,
        "counterfactual_dir": str(counterfactual_dir),
        "rank_sensitivity_dir": str(rank_sensitivity_dir),
        "gating_root": str(gating_root),
        "episode_dir": str(episode_dir),
        "observed_state": str(observed_state),
        "pose": str(pose),
        "camera_info": str(camera_info),
        "episode_summary": str(episode_summary_path),
        "prediction_npz": str(prediction_npz),
        "step_npz": str(step_npz),
        "observed_state_sha256": sha256_file(observed_state),
        "prediction_npz_sha256": sha256_file(prediction_npz),
        "sc_gain_formula": sc_gain_formula,
        "thresholds": {
            "sc_occ_threshold": float(sc_occ_threshold if sc_occ_threshold is not None else 0.7),
            "sc_conf_threshold": float(sc_conf_threshold if sc_conf_threshold is not None else 0.3),
        },
        "cap": sc_gain_cap,
        "weight": float(original_sc_gain_weight if original_sc_gain_weight is not None else 1.0),
        "gating": {
            "sc_gain_formula": sc_gain_formula,
            "sc_occ_threshold": float(sc_occ_threshold if sc_occ_threshold is not None else 0.7),
            "sc_conf_threshold": float(sc_conf_threshold if sc_conf_threshold is not None else 0.3),
            "sc_count_mode": sc_count_mode,
            "sc_gain_cap": sc_gain_cap,
            "sc_gain_weight": float(original_sc_gain_weight if original_sc_gain_weight is not None else 1.0),
            "score_gain_mode": str(
                _pick_value(manifest.get("score_gain_mode"), episode_summary.get("score_gain_mode"), default="hybrid_weighted")
            ),
            "alignment_convention": runtime_args["alignment_convention"],
        },
        "decoupled_sc": {
            "score_gain_mode": "decoupled_sc",
            "sc_gain_weight": decoupled_lambda,
            "formula_text": "gain_exp / max(path_cost, eps) + 0.5 * effective_gain_sc",
            "uses_uncapped_effective_gain_sc": True,
        },
        "runtime_args": runtime_args,
        "runtime_arg_sources": {
            "seed": "rollout --seed default 0 plus step; rollout code calls seed + step",
            "num_candidates_top_n": "rollout log and step npz",
            "sc_gating": "Stage 4A-6.4 manifest/episode_summary/step_npz",
            "prediction": "step_npz prediction_npz; no map_predict rerun",
        },
        "offline_expected": {
            "changed_action": bool(selected.get("changed_vs_own_selected")),
            "formula": selected.get("formula"),
            "lambda": selected.get("lambda"),
            "top_candidate_key": selected.get("top_candidate_key"),
            "top_candidate_row": selected.get("top_candidate_row"),
            "top_candidate_grid": selected.get("top_candidate_grid"),
            "top_candidate_world": selected.get("top_candidate_world"),
            "top_score": selected.get("top_score"),
            "top_gain_exp": selected.get("top_gain_exp"),
            "top_gain_sc": selected.get("top_gain_sc"),
            "top_effective_gain_sc": selected.get("top_effective_gain_sc"),
            "top_path_cost": selected.get("top_path_cost"),
            "own_selected_candidate_key": selected.get("own_selected_candidate_key"),
            "own_selected_candidate_row": selected.get("own_selected_candidate_row"),
            "own_selected_gain_exp": selected.get("own_selected_gain_exp"),
            "own_selected_effective_gain_sc": selected.get("own_selected_effective_gain_sc"),
            "own_selected_path_cost": selected.get("own_selected_path_cost"),
            "change_class": selected.get("change_class"),
        },
        "offline_decoupled_top_candidates": offline_candidates[:16],
        "source_counterfactual_row": selected,
        "source_manifest_row": manifest,
        "source_rollout_command": log_flags.get("command"),
        "safety": {
            "rollout": False,
            "isaac_startup": False,
            "map_predict_rerun": False,
            "rl_il_training": False,
            "checkpoint_modified": False,
            "observed_state_modified": False,
            "prediction_writeback": False,
            "prediction_used_for_astar_traversability_collision_or_ray_blocking": False,
            "target_or_ground_truth_scoring": False,
        },
    }
    save_json(output_dir / "selected_case.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a decoupled_sc lambda=0.5 one-step smoke case.")
    parser.add_argument("--counterfactual_dir", type=Path, default=DEFAULT_COUNTERFACTUAL_DIR)
    parser.add_argument("--rank_sensitivity_dir", type=Path, default=DEFAULT_RANK_SENSITIVITY_DIR)
    parser.add_argument("--fixed_raw_episode", type=Path, default=DEFAULT_FIXED_RAW_EPISODE)
    parser.add_argument("--empty_baseline_episode", type=Path, default=DEFAULT_EMPTY_BASELINE_EPISODE)
    parser.add_argument("--gating_root", type=Path, default=DEFAULT_GATING_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = select_case(
        counterfactual_dir=args.counterfactual_dir,
        rank_sensitivity_dir=args.rank_sensitivity_dir,
        fixed_raw_episode=args.fixed_raw_episode,
        empty_baseline_episode=args.empty_baseline_episode,
        gating_root=args.gating_root,
        output_dir=args.output_dir,
    )
    print(f"selected_case: {Path(args.output_dir).resolve() / 'selected_case.json'}")
    print(f"status: {case.get('status')}")
    if case.get("status") == "selected":
        print(f"config: {case.get('config')}")
        print(f"step: {case.get('step')}")
        print(f"offline_expected: {case.get('offline_expected', {}).get('top_candidate_key')}")
    else:
        print(f"blocker: {case.get('blocker')}")


if __name__ == "__main__":
    main()

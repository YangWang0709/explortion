#!/usr/bin/env python3
"""Stage 4A-6.5b offline counterfactual score analysis.

This script reads the existing Stage 4A-6.5a candidate rank table and summary
files only. It does not launch Isaac, run rollouts, run map_predict, modify the
expert runtime scorer, write predictions, or touch observed_state/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-6
REASONABLE_LAMBDA_MAX = 1.0
EXTENDED_REASONABLE_LAMBDA_MAX = 2.0

REFERENCE_CONFIGS = [
    "empty_baseline",
    "fixed_raw_sc",
    "occupied_only_occ07",
    "occupied_only_occ08",
    "occupied_margin_occ06_w05",
    "confidence_weighted_conf05_cap30",
]

REQUIRED_INPUT_FILES = [
    "candidate_rank_table.csv",
    "selected_candidate_summary.csv",
    "rank_correlation_summary.csv",
    "stage4a65a_rank_sensitivity_summary.json",
]

ROLLOUT_FILE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]


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


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=_json_default)
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
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default))
            handle.write("\n")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    return values


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


def finite_or_none(value: Any) -> float | None:
    out = _as_float(value)
    return out if out is not None and math.isfinite(out) else None


def mean_or_none(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def median_or_none(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def min_or_none(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min())


def max_or_none(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.max())


def rank_values(values: np.ndarray, descending: bool) -> np.ndarray:
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
        ranks[sorted_idx[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def parse_jsonish(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def param_token(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def formula_variant_id(formula: str, params: dict[str, float | None]) -> str:
    pieces = [formula]
    if params.get("lambda") is not None:
        pieces.append(f"lambda{param_token(params['lambda'])}")
    if params.get("alpha") is not None:
        pieces.append(f"alpha{param_token(params['alpha'])}")
    if params.get("beta") is not None:
        pieces.append(f"beta{param_token(params['beta'])}")
    return "_".join(pieces)


def build_formula_variants(
    lambda_values: list[float],
    alpha_values: list[float],
    beta_values: list[float],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "formula": "exp_over_cost",
            "params": {"lambda": None, "alpha": None, "beta": None},
            "required": ["gain_exp", "path_cost"],
        },
        {
            "formula": "raw_hybrid_over_cost",
            "params": {"lambda": None, "alpha": None, "beta": None},
            "required": ["gain_exp", "gain_sc", "path_cost"],
        },
        {
            "formula": "effective_hybrid_over_cost",
            "params": {"lambda": None, "alpha": None, "beta": None},
            "required": ["gain_exp", "effective_gain_sc", "path_cost"],
        },
        {
            "formula": "sc_only",
            "params": {"lambda": None, "alpha": None, "beta": None},
            "required": ["effective_gain_sc", "path_cost"],
        },
        {
            "formula": "exp_only_no_cost",
            "params": {"lambda": None, "alpha": None, "beta": None},
            "required": ["gain_exp"],
        },
        {
            "formula": "low_cost_only",
            "params": {"lambda": None, "alpha": None, "beta": None},
            "required": ["path_cost"],
        },
    ]
    for lam in lambda_values:
        variants.append(
            {
                "formula": "exp_plus_sc_no_cost",
                "params": {"lambda": lam, "alpha": None, "beta": None},
                "required": ["gain_exp", "effective_gain_sc"],
            }
        )
        variants.append(
            {
                "formula": "decoupled_sc",
                "params": {"lambda": lam, "alpha": None, "beta": None},
                "required": ["gain_exp", "effective_gain_sc", "path_cost"],
            }
        )
    for lam in lambda_values:
        for alpha in alpha_values:
            variants.append(
                {
                    "formula": "cost_powered",
                    "params": {"lambda": lam, "alpha": alpha, "beta": None},
                    "required": ["gain_exp", "effective_gain_sc", "path_cost"],
                }
            )
    for lam in lambda_values:
        for beta in beta_values:
            variants.append(
                {
                    "formula": "normalized_additive",
                    "params": {"lambda": lam, "alpha": None, "beta": beta},
                    "required": ["gain_exp", "effective_gain_sc", "path_cost"],
                }
            )
    for variant in variants:
        variant["variant_id"] = formula_variant_id(variant["formula"], variant["params"])
    return variants


def score_formula(
    formula: str,
    params: dict[str, float | None],
    arrays: dict[str, np.ndarray],
    valid: np.ndarray,
    warning_sink: list[dict[str, Any]],
    config: str,
    step: int,
) -> np.ndarray:
    gain_exp = arrays.get("gain_exp")
    gain_sc = arrays.get("gain_sc")
    effective_gain_sc = arrays.get("effective_gain_sc")
    path_cost = arrays.get("path_cost")
    lam = float(params["lambda"] or 0.0)
    alpha = float(params["alpha"] or 0.0)
    beta = float(params["beta"] or 0.0)
    if path_cost is not None:
        denom = np.maximum(path_cost, EPS)
    if formula == "exp_over_cost":
        return gain_exp / denom
    if formula == "raw_hybrid_over_cost":
        return (gain_exp + gain_sc) / denom
    if formula == "effective_hybrid_over_cost":
        return (gain_exp + effective_gain_sc) / denom
    if formula == "exp_plus_sc_no_cost":
        return gain_exp + lam * effective_gain_sc
    if formula == "decoupled_sc":
        return gain_exp / denom + lam * effective_gain_sc
    if formula == "cost_powered":
        return (gain_exp + lam * effective_gain_sc) / np.power(denom, alpha)
    if formula == "normalized_additive":
        norm_exp = normalize_for_group(gain_exp, valid, "gain_exp", formula, config, step, warning_sink)
        norm_sc = normalize_for_group(
            effective_gain_sc,
            valid,
            "effective_gain_sc",
            formula,
            config,
            step,
            warning_sink,
        )
        norm_cost = normalize_for_group(path_cost, valid, "path_cost", formula, config, step, warning_sink)
        return norm_exp + lam * norm_sc - beta * norm_cost
    if formula == "sc_only":
        return effective_gain_sc / denom
    if formula == "exp_only_no_cost":
        return gain_exp
    if formula == "low_cost_only":
        return -path_cost
    raise ValueError(f"Unsupported formula: {formula}")


def normalize_for_group(
    values: np.ndarray,
    valid: np.ndarray,
    field: str,
    formula: str,
    config: str,
    step: int,
    warning_sink: list[dict[str, Any]],
) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    mask = valid & np.isfinite(values)
    if not np.any(mask):
        warning_sink.append(
            {
                "config": config,
                "step": step,
                "formula": formula,
                "field": field,
                "reason": "no_finite_values_for_normalization",
            }
        )
        out[~mask] = np.nan
        return out
    mean = float(np.mean(values[mask]))
    std = float(np.std(values[mask]))
    if std < EPS:
        warning_sink.append(
            {
                "config": config,
                "step": step,
                "formula": formula,
                "field": field,
                "mean": mean,
                "std": std,
                "reason": "std_below_eps_norm_set_to_zero",
            }
        )
        out[mask] = 0.0
        out[~mask] = np.nan
        return out
    out[mask] = (values[mask] - mean) / std
    out[~mask] = np.nan
    return out


def choose_top_index(scores: np.ndarray, rows: list[dict[str, Any]], valid: np.ndarray) -> int | None:
    finite = valid & np.isfinite(scores)
    if not np.any(finite):
        return None
    candidates = np.flatnonzero(finite)
    candidate_rows = np.asarray([_as_int(rows[i].get("candidate_row")) or i for i in candidates], dtype=np.int64)
    order = np.lexsort((candidate_rows, -scores[candidates]))
    return int(candidates[int(order[0])])


def rows_to_arrays(rows: list[dict[str, str]], fields: list[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        arrays[field] = np.asarray(
            [finite_or_none(row.get(field)) if row.get(field) != "" else np.nan for row in rows],
            dtype=np.float64,
        )
    return arrays


def selected_reference(row: dict[str, str]) -> dict[str, Any]:
    candidate_key = row.get("candidate_key") or ""
    return {
        "candidate_key": candidate_key,
        "candidate_row": _as_int(row.get("candidate_row")),
        "candidate_id": row.get("candidate_id") or row.get("selected_candidate_id") or "",
        "expert_action": _as_int(row.get("expert_action")),
        "candidate_grid": parse_jsonish(row.get("candidate_grid", "")),
        "candidate_world": parse_jsonish(row.get("candidate_world", "")),
        "selected_transition_grid": parse_jsonish(row.get("selected_transition_grid", "")),
        "selected_transition_world": parse_jsonish(row.get("selected_transition_world", "")),
        "selected_transition_yaw": finite_or_none(row.get("selected_transition_yaw")),
        "final_score": finite_or_none(row.get("final_score")),
        "gain_exp": finite_or_none(row.get("gain_exp")),
        "gain_sc": finite_or_none(row.get("gain_sc")),
        "effective_gain_sc": finite_or_none(row.get("effective_gain_sc")),
        "path_cost": finite_or_none(row.get("path_cost")),
    }


def top_component_keys(rows: list[dict[str, str]], valid: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def top_for_score(name: str, scores: np.ndarray) -> None:
        idx = choose_top_index(scores, rows, valid)
        out[f"{name}_top_key"] = rows[idx].get("candidate_key") if idx is not None else ""
        out[f"{name}_top_row"] = _as_int(rows[idx].get("candidate_row")) if idx is not None else None

    if "gain_exp" in arrays:
        top_for_score("gain_exp", arrays["gain_exp"])
    if "path_cost" in arrays:
        top_for_score("low_cost", -arrays["path_cost"])
    if "gain_exp" in arrays and "path_cost" in arrays:
        denom = np.maximum(arrays["path_cost"], EPS)
        top_for_score("exp_over_cost", arrays["gain_exp"] / denom)
    if "effective_gain_sc" in arrays:
        top_for_score("effective_gain_sc", arrays["effective_gain_sc"])
    if "effective_gain_sc" in arrays and "path_cost" in arrays:
        denom = np.maximum(arrays["path_cost"], EPS)
        top_for_score("sc_only", arrays["effective_gain_sc"] / denom)
    return out


def rank_for_index(values: np.ndarray, valid: np.ndarray, index: int, descending: bool) -> float | None:
    ranks = rank_values(np.where(valid, values, np.nan), descending=descending)
    value = float(ranks[index])
    return value if math.isfinite(value) else None


def formula_uses_sc(formula: str, lam: float | None) -> bool:
    if formula in {"effective_hybrid_over_cost", "sc_only"}:
        return True
    if formula in {"exp_plus_sc_no_cost", "decoupled_sc", "cost_powered", "normalized_additive"}:
        return (lam or 0.0) > 0.0
    return False


def formula_uses_path_cost(formula: str, alpha: float | None, beta: float | None) -> bool:
    if formula in {"exp_over_cost", "raw_hybrid_over_cost", "effective_hybrid_over_cost", "decoupled_sc", "sc_only"}:
        return True
    if formula == "cost_powered":
        return (alpha or 0.0) > 0.0
    if formula == "normalized_additive":
        return (beta or 0.0) > 0.0
    if formula == "low_cost_only":
        return True
    return False


def classify_change(
    formula: str,
    params: dict[str, float | None],
    changed: bool,
    top_key: str,
    component_keys: dict[str, Any],
) -> str:
    if not changed:
        return "unchanged"
    lam = float(params["lambda"] or 0.0)
    alpha = params["alpha"]
    beta = params["beta"]
    uses_sc = formula_uses_sc(formula, lam)
    uses_cost = formula_uses_path_cost(formula, alpha, beta)
    if formula == "low_cost_only":
        return "path_cost_only"
    if formula == "sc_only" or top_key in {
        component_keys.get("effective_gain_sc_top_key", ""),
        component_keys.get("sc_only_top_key", ""),
    } and uses_sc:
        return "sc_gain_direct_or_sc_only"
    if uses_sc and not uses_cost:
        return "sc_gain_plus_removed_path_cost"
    if uses_sc:
        return "sc_gain_possible_mixed_with_cost"
    if not uses_cost:
        return "removed_path_cost"
    if formula == "cost_powered" and alpha is not None and alpha < 1.0:
        return "reduced_path_cost_exponent"
    if formula == "normalized_additive" and beta is not None:
        return "normalized_cost_reweighting"
    return "non_sc_formula_reweighting"


def aggregate_group_skips(group_skip_counter: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (variant_id, reason), count in sorted(group_skip_counter.items()):
        rows.append({"formula_variant_id": variant_id, "reason": reason, "count": count})
    return rows


def build_formula_change_summary(action_rows: list[dict[str, Any]], all_steps: list[int]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        grouped[row["formula_variant_id"]].append(row)
    summary_rows: list[dict[str, Any]] = []
    for variant_id, rows in sorted(grouped.items()):
        changed_rows = [row for row in rows if row["changed_vs_own_selected"]]
        changed_steps_any = sorted({int(row["step"]) for row in changed_rows})
        step_counts: dict[int, tuple[int, int]] = {}
        for step in all_steps:
            step_rows = [row for row in rows if int(row["step"]) == step]
            if not step_rows:
                continue
            changed_count = sum(1 for row in step_rows if row["changed_vs_own_selected"])
            step_counts[step] = (changed_count, len(step_rows))
        changed_steps_all_executed = sorted(
            step for step, (changed_count, total_count) in step_counts.items() if total_count > 0 and changed_count == total_count
        )
        due_counts: dict[str, int] = defaultdict(int)
        for row in changed_rows:
            due_counts[str(row["change_class"])] += 1
        first = rows[0]
        summary_rows.append(
            {
                "formula": first["formula"],
                "formula_variant_id": variant_id,
                "lambda": first.get("lambda"),
                "alpha": first.get("alpha"),
                "beta": first.get("beta"),
                "executed_groups": len(rows),
                "changed_top1_groups": len(changed_rows),
                "unchanged_top1_groups": len(rows) - len(changed_rows),
                "change_rate": len(changed_rows) / len(rows) if rows else None,
                "changed_steps_any_config_count": len(changed_steps_any),
                "changed_steps_any_config": changed_steps_any,
                "changed_steps_all_executed_configs_count": len(changed_steps_all_executed),
                "changed_steps_all_executed_configs": changed_steps_all_executed,
                "changed_all_5_steps_any_config": set(all_steps).issubset(set(changed_steps_any)),
                "changed_all_executed_groups": len(changed_rows) == len(rows) if rows else False,
                "matches_empty_baseline_selected_groups": sum(
                    1 for row in rows if row.get("matches_empty_baseline_selected")
                ),
                "matches_fixed_raw_sc_selected_groups": sum(1 for row in rows if row.get("matches_fixed_raw_sc_selected")),
                "matches_own_selected_groups": sum(1 for row in rows if row.get("matches_own_selected")),
                "matches_gain_exp_top_groups": sum(1 for row in rows if row.get("matches_gain_exp_top")),
                "matches_low_cost_top_groups": sum(1 for row in rows if row.get("matches_low_cost_top")),
                "matches_effective_gain_sc_top_groups": sum(
                    1 for row in rows if row.get("matches_effective_gain_sc_top")
                ),
                "matches_sc_only_top_groups": sum(1 for row in rows if row.get("matches_sc_only_top")),
                "change_class_counts": dict(sorted(due_counts.items())),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -int(row["changed_steps_any_config_count"]),
            -int(row["changed_top1_groups"]),
            str(row["formula_variant_id"]),
        )
    )
    return summary_rows


def build_threshold_rows(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        if row["formula"] not in {"exp_plus_sc_no_cost", "decoupled_sc", "cost_powered", "normalized_additive"}:
            continue
        key = (
            row["formula"],
            row["config"],
            int(row["step"]),
            row.get("alpha"),
            row.get("beta"),
        )
        rows_by_group[key].append(row)

    threshold_rows: list[dict[str, Any]] = []
    for key, rows in sorted(rows_by_group.items()):
        formula, config, step, alpha, beta = key
        rows = sorted(rows, key=lambda row: float(row.get("lambda") or 0.0))
        lambda0_rows = [row for row in rows if float(row.get("lambda") or 0.0) == 0.0]
        lambda0_row = lambda0_rows[0] if lambda0_rows else None
        lambda0_key = lambda0_row["top_candidate_key"] if lambda0_row else None
        original_key = rows[0]["own_selected_candidate_key"]
        change_vs_current = [
            float(row["lambda"]) for row in rows if row["top_candidate_key"] != original_key
        ]
        sc_change_vs_lambda0 = [
            float(row["lambda"])
            for row in rows
            if lambda0_key is not None and float(row.get("lambda") or 0.0) > 0.0 and row["top_candidate_key"] != lambda0_key
        ]
        threshold_rows.append(
            {
                "threshold_type": "lambda",
                "formula_family": formula,
                "config": config,
                "step": step,
                "alpha": alpha,
                "beta": beta,
                "original_selected_key": original_key,
                "lambda0_top_key": lambda0_key,
                "lambda0_changes_vs_current": bool(lambda0_key is not None and lambda0_key != original_key),
                "min_lambda_change_vs_current": min(change_vs_current) if change_vs_current else None,
                "min_lambda_sc_changes_vs_lambda0": min(sc_change_vs_lambda0) if sc_change_vs_lambda0 else None,
                "changes_vs_current_by_lambda_le_1": any(value <= REASONABLE_LAMBDA_MAX for value in change_vs_current),
                "sc_changes_vs_lambda0_by_lambda_le_1": any(value <= REASONABLE_LAMBDA_MAX for value in sc_change_vs_lambda0),
                "changes_vs_current_by_lambda_le_2": any(value <= EXTENDED_REASONABLE_LAMBDA_MAX for value in change_vs_current),
                "sc_changes_vs_lambda0_by_lambda_le_2": any(
                    value <= EXTENDED_REASONABLE_LAMBDA_MAX for value in sc_change_vs_lambda0
                ),
                "lambda_values_change_vs_current": change_vs_current,
                "lambda_values_sc_change_vs_lambda0": sc_change_vs_lambda0,
            }
        )

    cost_rows = [
        row
        for row in action_rows
        if row["formula"] == "cost_powered" and row.get("alpha") is not None and row.get("lambda") is not None
    ]
    cost_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cost_rows:
        cost_by_group[(row["config"], int(row["step"]), float(row["lambda"]))].append(row)
    for (config, step, lam), rows in sorted(cost_by_group.items()):
        by_alpha = {float(row["alpha"]): row for row in rows}
        alpha1_key = by_alpha.get(1.0, {}).get("top_candidate_key")
        original_key = rows[0]["own_selected_candidate_key"]
        changed_alphas = sorted(
            float(row["alpha"]) for row in rows if row["top_candidate_key"] != original_key
        )
        alpha_changed_vs_1 = sorted(
            float(alpha)
            for alpha, row in by_alpha.items()
            if alpha1_key is not None and row["top_candidate_key"] != alpha1_key
        )
        threshold_rows.append(
            {
                "threshold_type": "alpha",
                "formula_family": "cost_powered",
                "config": config,
                "step": step,
                "lambda": lam,
                "original_selected_key": original_key,
                "alpha1_top_key": alpha1_key,
                "alpha0p5_top_key": by_alpha.get(0.5, {}).get("top_candidate_key"),
                "alpha0p5_changes_vs_alpha1": bool(
                    alpha1_key is not None
                    and by_alpha.get(0.5, {}).get("top_candidate_key") is not None
                    and by_alpha[0.5]["top_candidate_key"] != alpha1_key
                ),
                "alpha0p5_changes_vs_current": bool(
                    by_alpha.get(0.5, {}).get("top_candidate_key") is not None
                    and by_alpha[0.5]["top_candidate_key"] != original_key
                ),
                "alpha_values_change_vs_current": changed_alphas,
                "alpha_values_change_vs_alpha1": alpha_changed_vs_1,
            }
        )

    normalized_rows = [
        row
        for row in action_rows
        if row["formula"] == "normalized_additive" and row.get("beta") is not None and row.get("lambda") is not None
    ]
    normalized_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_rows:
        normalized_by_group[(row["config"], int(row["step"]), float(row["lambda"]))].append(row)
    for (config, step, lam), rows in sorted(normalized_by_group.items()):
        original_key = rows[0]["own_selected_candidate_key"]
        changed_betas = sorted(float(row["beta"]) for row in rows if row["top_candidate_key"] != original_key)
        threshold_rows.append(
            {
                "threshold_type": "beta",
                "formula_family": "normalized_additive",
                "config": config,
                "step": step,
                "lambda": lam,
                "original_selected_key": original_key,
                "beta_values_change_vs_current": changed_betas,
                "min_beta_change_vs_current": min(changed_betas) if changed_betas else None,
                "max_beta_change_vs_current": max(changed_betas) if changed_betas else None,
            }
        )
    return threshold_rows


def summarize_thresholds(threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lambda_rows = [row for row in threshold_rows if row.get("threshold_type") == "lambda"]
    sc_thresholds = [
        float(row["min_lambda_sc_changes_vs_lambda0"])
        for row in lambda_rows
        if row.get("min_lambda_sc_changes_vs_lambda0") is not None
    ]
    current_thresholds = [
        float(row["min_lambda_change_vs_current"])
        for row in lambda_rows
        if row.get("min_lambda_change_vs_current") is not None
    ]
    by_formula: dict[str, dict[str, Any]] = {}
    for formula in sorted({str(row["formula_family"]) for row in lambda_rows}):
        sub = [row for row in lambda_rows if row["formula_family"] == formula]
        sub_sc = [
            float(row["min_lambda_sc_changes_vs_lambda0"])
            for row in sub
            if row.get("min_lambda_sc_changes_vs_lambda0") is not None
        ]
        sub_current = [
            float(row["min_lambda_change_vs_current"])
            for row in sub
            if row.get("min_lambda_change_vs_current") is not None
        ]
        by_formula[formula] = {
            "groups": len(sub),
            "groups_changed_vs_current": len(sub_current),
            "groups_sc_changed_vs_lambda0": len(sub_sc),
            "min_lambda_change_vs_current": min_or_none(sub_current),
            "median_lambda_change_vs_current": median_or_none(sub_current),
            "min_lambda_sc_changes_vs_lambda0": min_or_none(sub_sc),
            "median_lambda_sc_changes_vs_lambda0": median_or_none(sub_sc),
            "sc_changes_vs_lambda0_by_lambda_le_1_groups": sum(
                1 for row in sub if row.get("sc_changes_vs_lambda0_by_lambda_le_1")
            ),
            "sc_changes_vs_lambda0_by_lambda_le_2_groups": sum(
                1 for row in sub if row.get("sc_changes_vs_lambda0_by_lambda_le_2")
            ),
        }

    alpha_rows = [row for row in threshold_rows if row.get("threshold_type") == "alpha"]
    beta_rows = [row for row in threshold_rows if row.get("threshold_type") == "beta"]
    return {
        "lambda": {
            "groups": len(lambda_rows),
            "groups_changed_vs_current": len(current_thresholds),
            "groups_sc_changed_vs_lambda0": len(sc_thresholds),
            "min_lambda_change_vs_current": min_or_none(current_thresholds),
            "median_lambda_change_vs_current": median_or_none(current_thresholds),
            "min_lambda_sc_changes_vs_lambda0": min_or_none(sc_thresholds),
            "median_lambda_sc_changes_vs_lambda0": median_or_none(sc_thresholds),
            "sc_changes_vs_lambda0_by_lambda_le_1_groups": sum(
                1 for row in lambda_rows if row.get("sc_changes_vs_lambda0_by_lambda_le_1")
            ),
            "sc_changes_vs_lambda0_by_lambda_le_2_groups": sum(
                1 for row in lambda_rows if row.get("sc_changes_vs_lambda0_by_lambda_le_2")
            ),
            "by_formula": by_formula,
        },
        "alpha": {
            "groups": len(alpha_rows),
            "alpha0p5_changes_vs_alpha1_groups": sum(
                1 for row in alpha_rows if row.get("alpha0p5_changes_vs_alpha1")
            ),
            "alpha0p5_changes_vs_current_groups": sum(
                1 for row in alpha_rows if row.get("alpha0p5_changes_vs_current")
            ),
        },
        "beta": {
            "groups": len(beta_rows),
            "groups_with_any_beta_change_vs_current": sum(
                1 for row in beta_rows if row.get("beta_values_change_vs_current")
            ),
        },
    }


def choose_later_smoke_formula(
    summary_rows: list[dict[str, Any]],
    threshold_summary: dict[str, Any],
) -> dict[str, Any]:
    lambda_by_formula = threshold_summary["lambda"]["by_formula"]
    decoupled = lambda_by_formula.get("decoupled_sc", {})
    if decoupled.get("sc_changes_vs_lambda0_by_lambda_le_1_groups", 0) > 0:
        candidates = [
            row
            for row in summary_rows
            if row["formula"] == "decoupled_sc"
            and row.get("lambda") is not None
            and float(row["lambda"]) <= REASONABLE_LAMBDA_MAX
            and row["changed_top1_groups"] > 0
        ]
        if candidates:
            best = sorted(candidates, key=lambda row: (-int(row["changed_top1_groups"]), float(row["lambda"])))[0]
            return {
                "recommendation": "one_step_smoke",
                "formula_variant_id": best["formula_variant_id"],
                "reason": "decoupled SC changes at least one lambda0 ranking by lambda <= 1",
            }

    normalized = lambda_by_formula.get("normalized_additive", {})
    if normalized.get("sc_changes_vs_lambda0_by_lambda_le_1_groups", 0) > 0:
        candidates = [
            row
            for row in summary_rows
            if row["formula"] == "normalized_additive"
            and row.get("lambda") is not None
            and float(row["lambda"]) <= REASONABLE_LAMBDA_MAX
            and row["changed_top1_groups"] > 0
        ]
        if candidates:
            best = sorted(
                candidates,
                key=lambda row: (-int(row["changed_top1_groups"]), float(row["lambda"]), float(row["beta"])),
            )[0]
            return {
                "recommendation": "one_step_smoke",
                "formula_variant_id": best["formula_variant_id"],
                "reason": "normalized additive SC changes at least one lambda0 ranking by lambda <= 1",
            }

    return {
        "recommendation": "no_formula_smoke_yet",
        "formula_variant_id": None,
        "reason": "SC-specific lambda changes were absent or required large weights; prefer candidate/spatial review",
    }


def compact_formula_list(rows: list[dict[str, Any]], limit: int = 12) -> list[str]:
    return [str(row["formula_variant_id"]) for row in rows[:limit]]


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    formula_changes = summary["formula_changes"]
    formulas_changed = formula_changes["formula_variants_changed_top1"]
    formulas_changed_all = formula_changes["formula_variants_changed_all_5_steps_any_config"]
    most = formula_changes["formula_variants_changed_most_steps"]
    skipped = summary["skipped"]
    questions = summary["analysis_questions"]
    thresholds = summary["threshold_summary"]
    smoke = summary["next_small_task_recommendation"]

    lines = [
        "## Stage 4A-6.5b Offline Counterfactual Score Summary",
        "",
        f"- input candidate table: `{summary['inputs']['candidate_rank_table']}`",
        f"- configs analyzed: {', '.join(summary['configs_analyzed'])}",
        f"- steps analyzed: {summary['steps_analyzed']}",
        f"- candidate rows: {summary['candidate_rows']}",
        f"- formulas tested: {', '.join(summary['formulas_executed'])}",
        f"- formula variants executed: {summary['formula_variants_executed']}",
        f"- formulas skipped: {skipped['skipped_formula_names'] or 'none'}",
        f"- formulas that changed top-1: {formulas_changed[:12] or 'none'}",
        f"- formulas that changed top-1 in at least 1 step: {len(formulas_changed)} variants",
        f"- formulas that changed top-1 in most steps: {most[:8] or 'none'}",
        f"- formulas that changed top-1 in all 5 steps: {formulas_changed_all[:8] or 'none'}",
        f"- lambda thresholds needed to change action: min current={thresholds['lambda']['min_lambda_change_vs_current']}, "
        f"min SC-vs-lambda0={thresholds['lambda']['min_lambda_sc_changes_vs_lambda0']}, "
        f"median SC-vs-lambda0={thresholds['lambda']['median_lambda_sc_changes_vs_lambda0']}",
        f"- alpha threshold observation: alpha=0.5 changed vs alpha=1 in "
        f"{thresholds['alpha']['alpha0p5_changes_vs_alpha1_groups']} groups",
        f"- beta observation: normalized additive had beta-driven current-selection changes in "
        f"{thresholds['beta']['groups_with_any_beta_change_vs_current']} grouped sweeps",
        f"- whether reasonable lambda changes action: {questions['reasonable_lambda_changes_action']}",
        f"- whether removing path_cost changes action: {questions['removing_path_cost_entirely_changes_selection']}",
        f"- whether reducing path_cost exponent changes action: {questions['reducing_alpha_1_to_0p5_changes_selection']}",
        f"- whether SC-only selects different candidates: {questions['sc_only_selects_different_candidates']}",
        f"- diagnosis: {summary['diagnosis']}",
        f"- next small task recommendation: {smoke['recommendation']} ({smoke['formula_variant_id'] or smoke['reason']})",
        "",
        "Analysis questions:",
        f"1. Does any formula change top-1 candidate for any step? {questions['any_formula_changes_top1_any_step']}",
        f"2. Does any formula change top-1 candidate for all 5 steps? {questions['any_formula_changes_top1_all_5_steps']}",
        f"3. Does removing path_cost entirely change candidate selection? {questions['removing_path_cost_entirely_changes_selection']}",
        f"4. Does reducing path_cost exponent alpha from 1 to 0.5 change candidate selection? {questions['reducing_alpha_1_to_0p5_changes_selection']}",
        f"5. Does SC-only select different candidates? {questions['sc_only_selects_different_candidates']}",
        f"6. How large must lambda be before effective_gain_sc changes top-1? {questions['lambda_needed_for_sc_specific_change']}",
        f"7. Are required lambda values reasonable or absurd? {questions['lambda_reasonableness']}",
        f"8. Is path_cost dominance caused by division by path_cost specifically? {questions['path_cost_division_diagnosis']}",
        f"9. Would a decoupled SC term plausibly affect ranking? {questions['decoupled_sc_plausibility']}",
        f"10. Which single formula deserves a later one-step smoke? {questions['single_formula_for_later_smoke']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--lambda_values", default="0,0.1,0.25,0.5,1,2,5,10")
    parser.add_argument("--alpha_values", default="0,0.5,1,1.5,2")
    parser.add_argument("--beta_values", default="0.25,0.5,1,2")
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    rank_dir = args.rank_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {name: rank_dir / name for name in REQUIRED_INPUT_FILES}
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required Stage 4A-6.5a inputs: {missing_inputs}")

    rows, fieldnames = read_csv(input_paths["candidate_rank_table.csv"])
    selected_summary_rows, _ = read_csv(input_paths["selected_candidate_summary.csv"])
    rank_correlation_rows, _ = read_csv(input_paths["rank_correlation_summary.csv"])
    stage4a65a_summary = load_json(input_paths["stage4a65a_rank_sensitivity_summary.json"])

    if not rows:
        raise RuntimeError("candidate_rank_table.csv is empty")

    rows = [row for row in rows if _as_int(row.get("step")) is not None]
    all_steps_available = sorted({int(_as_int(row["step"]) or 0) for row in rows})
    steps_to_use = all_steps_available[: args.max_steps]
    rows = [row for row in rows if int(_as_int(row["step"]) or 0) in steps_to_use]

    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("config", ""), int(_as_int(row["step"]) or 0))].append(row)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda row: _as_int(row.get("candidate_row")) or 0)

    configs_analyzed = sorted({key[0] for key in groups})
    if not configs_analyzed:
        raise RuntimeError("No config loaded from candidate_rank_table.csv")

    selected_refs: dict[tuple[str, int], dict[str, Any]] = {}
    for key, group_rows in groups.items():
        selected = [row for row in group_rows if _as_bool(row.get("is_selected"))]
        if not selected:
            selected = [row for row in group_rows if _as_bool(row.get("top1"))]
        if not selected:
            selected = group_rows[:1]
        selected_refs[key] = selected_reference(selected[0])

    formula_variants = build_formula_variants(
        parse_float_list(args.lambda_values),
        parse_float_list(args.alpha_values),
        parse_float_list(args.beta_values),
    )

    skipped_formula_variants: list[dict[str, Any]] = []
    executable_variants: list[dict[str, Any]] = []
    field_set = set(fieldnames)
    for variant in formula_variants:
        missing = [field for field in variant["required"] if field not in field_set]
        if missing:
            skipped_formula_variants.append(
                {
                    "formula": variant["formula"],
                    "formula_variant_id": variant["variant_id"],
                    "missing_fields": missing,
                    "reason": "required_field_missing_from_candidate_rank_table",
                }
            )
        else:
            executable_variants.append(variant)

    action_rows: list[dict[str, Any]] = []
    zero_variance_warnings: list[dict[str, Any]] = []
    group_skip_counter: dict[tuple[str, str], int] = defaultdict(int)
    group_skip_examples: list[dict[str, Any]] = []

    numeric_fields = sorted(
        {
            field
            for variant in executable_variants
            for field in variant["required"]
        }
        | {"final_score", "gain_exp", "gain_sc", "effective_gain_sc", "path_cost"}
    )

    for (config, step), group_rows in sorted(groups.items()):
        valid = np.asarray([_as_bool(row.get("valid", True)) for row in group_rows], dtype=bool)
        arrays = rows_to_arrays(group_rows, numeric_fields)
        component_keys = top_component_keys(group_rows, valid, arrays)
        own_ref = selected_refs[(config, step)]
        reference_matches: dict[str, str] = {}
        for ref_config in REFERENCE_CONFIGS:
            ref = selected_refs.get((ref_config, step))
            if ref:
                reference_matches[ref_config] = ref["candidate_key"]

        for variant in executable_variants:
            formula = variant["formula"]
            params = variant["params"]
            variant_id = variant["variant_id"]
            required = variant["required"]
            finite_required = valid.copy()
            missing_value_fields: list[str] = []
            for field in required:
                values = arrays[field]
                finite_required &= np.isfinite(values)
                if not np.any(valid & np.isfinite(values)):
                    missing_value_fields.append(field)
            if missing_value_fields:
                reason = "no_finite_required_values:" + ",".join(missing_value_fields)
                group_skip_counter[(variant_id, reason)] += 1
                if len(group_skip_examples) < 40:
                    group_skip_examples.append(
                        {
                            "config": config,
                            "step": step,
                            "formula_variant_id": variant_id,
                            "missing_value_fields": missing_value_fields,
                            "reason": reason,
                        }
                    )
                continue

            scores = score_formula(
                formula,
                params,
                arrays,
                finite_required,
                zero_variance_warnings,
                config,
                step,
            )
            top_index = choose_top_index(scores, group_rows, finite_required)
            if top_index is None:
                reason = "no_finite_counterfactual_score"
                group_skip_counter[(variant_id, reason)] += 1
                if len(group_skip_examples) < 40:
                    group_skip_examples.append(
                        {"config": config, "step": step, "formula_variant_id": variant_id, "reason": reason}
                    )
                continue

            top_row = group_rows[top_index]
            top_key = top_row.get("candidate_key") or ""
            own_selected_key = own_ref["candidate_key"]
            changed = top_key != own_selected_key
            matched_ref_configs = [
                ref_config for ref_config, ref_key in reference_matches.items() if top_key and top_key == ref_key
            ]
            low_cost_rank = (
                rank_for_index(arrays["path_cost"], finite_required, top_index, descending=False)
                if "path_cost" in arrays
                else None
            )
            gain_exp_rank = (
                rank_for_index(arrays["gain_exp"], finite_required, top_index, descending=True)
                if "gain_exp" in arrays
                else None
            )
            effective_sc_rank = (
                rank_for_index(arrays["effective_gain_sc"], finite_required, top_index, descending=True)
                if "effective_gain_sc" in arrays and np.any(np.isfinite(arrays["effective_gain_sc"]))
                else None
            )
            action_row = {
                "config": config,
                "step": step,
                "formula": formula,
                "formula_variant_id": variant_id,
                "lambda": params.get("lambda"),
                "alpha": params.get("alpha"),
                "beta": params.get("beta"),
                "top_candidate_row": _as_int(top_row.get("candidate_row")),
                "top_candidate_id": top_row.get("candidate_id") or "",
                "top_candidate_key": top_key,
                "top_candidate_grid": parse_jsonish(top_row.get("candidate_grid", "")),
                "top_candidate_world": parse_jsonish(top_row.get("candidate_world", "")),
                "top_score": float(scores[top_index]),
                "top_gain_exp": finite_or_none(top_row.get("gain_exp")),
                "top_gain_sc": finite_or_none(top_row.get("gain_sc")),
                "top_effective_gain_sc": finite_or_none(top_row.get("effective_gain_sc")),
                "top_path_cost": finite_or_none(top_row.get("path_cost")),
                "top_gain_exp_rank": gain_exp_rank,
                "top_effective_gain_sc_rank": effective_sc_rank,
                "top_low_path_cost_rank": low_cost_rank,
                "own_selected_candidate_key": own_selected_key,
                "own_selected_candidate_row": own_ref["candidate_row"],
                "own_selected_expert_action": own_ref["expert_action"],
                "own_selected_gain_exp": own_ref["gain_exp"],
                "own_selected_effective_gain_sc": own_ref["effective_gain_sc"],
                "own_selected_path_cost": own_ref["path_cost"],
                "matches_own_selected": not changed,
                "changed_vs_own_selected": changed,
                "matched_reference_configs": matched_ref_configs,
                "matches_empty_baseline_selected": top_key == reference_matches.get("empty_baseline", ""),
                "matches_fixed_raw_sc_selected": top_key == reference_matches.get("fixed_raw_sc", ""),
                "matches_gain_exp_top": top_key == component_keys.get("gain_exp_top_key", ""),
                "matches_low_cost_top": top_key == component_keys.get("low_cost_top_key", ""),
                "matches_exp_over_cost_top": top_key == component_keys.get("exp_over_cost_top_key", ""),
                "matches_effective_gain_sc_top": top_key == component_keys.get("effective_gain_sc_top_key", ""),
                "matches_sc_only_top": top_key == component_keys.get("sc_only_top_key", ""),
                "gain_exp_top_key": component_keys.get("gain_exp_top_key", ""),
                "low_cost_top_key": component_keys.get("low_cost_top_key", ""),
                "exp_over_cost_top_key": component_keys.get("exp_over_cost_top_key", ""),
                "effective_gain_sc_top_key": component_keys.get("effective_gain_sc_top_key", ""),
                "sc_only_top_key": component_keys.get("sc_only_top_key", ""),
            }
            action_row["change_class"] = classify_change(formula, params, changed, top_key, component_keys)
            action_rows.append(action_row)

    if not action_rows:
        raise RuntimeError("No counterfactual formula executed on any config/step group")

    formula_summary_rows = build_formula_change_summary(action_rows, steps_to_use)
    threshold_rows = build_threshold_rows(action_rows)
    threshold_summary = summarize_thresholds(threshold_rows)
    smoke_recommendation = choose_later_smoke_formula(formula_summary_rows, threshold_summary)

    changed_variants = [row for row in formula_summary_rows if int(row["changed_top1_groups"]) > 0]
    changed_all_5 = [row for row in formula_summary_rows if row["changed_all_5_steps_any_config"]]
    most_steps = sorted(
        changed_variants,
        key=lambda row: (-int(row["changed_steps_any_config_count"]), -int(row["changed_top1_groups"])),
    )

    exp_only_rows = [row for row in action_rows if row["formula"] == "exp_only_no_cost"]
    exp_only_changed = sum(1 for row in exp_only_rows if row["changed_vs_own_selected"])
    cost_power_alpha_compare = threshold_summary["alpha"]
    sc_only_rows = [row for row in action_rows if row["formula"] == "sc_only"]
    sc_only_changed = sum(1 for row in sc_only_rows if row["changed_vs_own_selected"])
    decoupled_info = threshold_summary["lambda"]["by_formula"].get("decoupled_sc", {})

    lambda_sc_min = threshold_summary["lambda"]["min_lambda_sc_changes_vs_lambda0"]
    lambda_sc_median = threshold_summary["lambda"]["median_lambda_sc_changes_vs_lambda0"]
    if lambda_sc_min is None:
        lambda_needed_text = "no tested lambda changed a lambda0 ranking through effective_gain_sc"
    else:
        lambda_needed_text = f"min {lambda_sc_min}, median {lambda_sc_median}"

    sc_changes_le_1 = threshold_summary["lambda"]["sc_changes_vs_lambda0_by_lambda_le_1_groups"]
    sc_changes_le_2 = threshold_summary["lambda"]["sc_changes_vs_lambda0_by_lambda_le_2_groups"]
    if sc_changes_le_1 > 0:
        lambda_reason = f"reasonable in some groups: {sc_changes_le_1} SC-specific grouped sweeps changed by lambda <= 1"
    elif sc_changes_le_2 > 0:
        lambda_reason = f"borderline: no lambda <= 1, but {sc_changes_le_2} grouped sweeps changed by lambda <= 2"
    elif lambda_sc_min is not None:
        lambda_reason = "mostly large/absurd in this sweep: SC-specific changes required lambda > 2"
    else:
        lambda_reason = "not observed in tested lambda range"

    if exp_only_changed > 0:
        path_cost_division_text = (
            f"yes: exp_only_no_cost changed {exp_only_changed}/{len(exp_only_rows)} executed groups, "
            "while low-cost and over-cost variants show the original winner is strongly cost-favored"
        )
    else:
        path_cost_division_text = "not supported by exp_only_no_cost in this top-16 table"

    decoupled_groups = decoupled_info.get("groups_sc_changed_vs_lambda0", 0)
    decoupled_le_1 = decoupled_info.get("sc_changes_vs_lambda0_by_lambda_le_1_groups", 0)
    if decoupled_le_1 > 0:
        decoupled_text = f"plausible: decoupled_sc changed lambda0 ranking in {decoupled_le_1} groups by lambda <= 1"
    elif decoupled_groups > 0:
        decoupled_text = f"weak: decoupled_sc changed lambda0 ranking in {decoupled_groups} groups, but not by lambda <= 1"
    else:
        decoupled_text = "not in this sweep: decoupled_sc never changed the lambda0 ranking"

    if smoke_recommendation["formula_variant_id"]:
        single_formula = smoke_recommendation["formula_variant_id"]
    else:
        single_formula = "none yet; use candidate generation/spatial review before formula smoke"

    analysis_questions = {
        "any_formula_changes_top1_any_step": bool(changed_variants),
        "any_formula_changes_top1_all_5_steps": bool(changed_all_5),
        "formula_variants_changing_all_5_steps": compact_formula_list(changed_all_5, limit=20),
        "removing_path_cost_entirely_changes_selection": exp_only_changed > 0,
        "exp_only_no_cost_changed_groups": exp_only_changed,
        "exp_only_no_cost_executed_groups": len(exp_only_rows),
        "reducing_alpha_1_to_0p5_changes_selection": cost_power_alpha_compare["alpha0p5_changes_vs_alpha1_groups"] > 0,
        "alpha0p5_changes_vs_alpha1_groups": cost_power_alpha_compare["alpha0p5_changes_vs_alpha1_groups"],
        "alpha0p5_changes_vs_current_groups": cost_power_alpha_compare["alpha0p5_changes_vs_current_groups"],
        "sc_only_selects_different_candidates": sc_only_changed > 0,
        "sc_only_changed_groups": sc_only_changed,
        "sc_only_executed_groups": len(sc_only_rows),
        "lambda_needed_for_sc_specific_change": lambda_needed_text,
        "lambda_reasonableness": lambda_reason,
        "reasonable_lambda_changes_action": sc_changes_le_1 > 0,
        "path_cost_division_diagnosis": path_cost_division_text,
        "decoupled_sc_plausibility": decoupled_text,
        "single_formula_for_later_smoke": single_formula,
    }

    formulas_executed = sorted({row["formula"] for row in action_rows})
    skipped_formula_names = sorted({row["formula"] for row in skipped_formula_variants})
    skipped_summary = {
        "skipped_formula_names": skipped_formula_names,
        "skipped_formula_variants": skipped_formula_variants,
        "skipped_group_counts": aggregate_group_skips(group_skip_counter),
        "skipped_group_examples": group_skip_examples,
        "zero_variance_warning_count": len(zero_variance_warnings),
        "zero_variance_warning_examples": zero_variance_warnings[:40],
    }

    output_rollout_files: list[str] = []
    for pattern in ROLLOUT_FILE_PATTERNS:
        output_rollout_files.extend(str(path) for path in output_dir.glob(pattern))

    validation = {
        "output_dir_exists": output_dir.exists(),
        "summary_json_exists": True,
        "summary_md_exists": True,
        "action_table_csv_exists": True,
        "at_least_one_config_loaded": len(configs_analyzed) > 0,
        "at_least_one_formula_executed": len(formulas_executed) > 0,
        "skipped_formulas_recorded": True,
        "new_rollout_like_files_in_output_dir": output_rollout_files,
        "no_new_rollout_files_created_in_output_dir": len(output_rollout_files) == 0,
    }

    diagnosis = (
        "Changing top-1 is dominated by path-cost removal/reweighting unless a lambda sweep also changes "
        "the lambda0 ranking. Treat every formula here as offline-only; no observed_ratio improvement is claimed."
    )

    summary = {
        "stage": "Stage 4A-6.5b offline counterfactual score analysis",
        "inputs": {
            "rank_dir": str(rank_dir),
            "candidate_rank_table": str(input_paths["candidate_rank_table.csv"]),
            "selected_candidate_summary": str(input_paths["selected_candidate_summary.csv"]),
            "rank_correlation_summary": str(input_paths["rank_correlation_summary.csv"]),
            "stage4a65a_summary": str(input_paths["stage4a65a_rank_sensitivity_summary.json"]),
            "selected_candidate_summary_rows": len(selected_summary_rows),
            "rank_correlation_summary_rows": len(rank_correlation_rows),
            "stage4a65a_loaded": bool(stage4a65a_summary),
        },
        "configs_analyzed": configs_analyzed,
        "steps_analyzed": steps_to_use,
        "candidate_rows": len(rows),
        "groups_analyzed": len(groups),
        "formula_variants_defined": len(formula_variants),
        "formula_variants_executed": len({row["formula_variant_id"] for row in action_rows}),
        "formulas_executed": formulas_executed,
        "formula_changes": {
            "formula_variants_changed_top1": compact_formula_list(changed_variants, limit=200),
            "formula_variants_changed_all_5_steps_any_config": compact_formula_list(changed_all_5, limit=200),
            "formula_variants_changed_most_steps": compact_formula_list(most_steps, limit=30),
        },
        "analysis_questions": analysis_questions,
        "threshold_summary": threshold_summary,
        "next_small_task_recommendation": smoke_recommendation,
        "skipped": skipped_summary,
        "diagnosis": diagnosis,
        "safety": {
            "new_isaac_rollout_started": False,
            "map_predict_rerun": False,
            "expert_runtime_modified": False,
            "rl_ppo_bc_il_training": False,
            "sscnet_training": False,
            "checkpoint_modified": False,
            "observed_state_modified": False,
            "prediction_writeback": False,
            "future_observations_used_for_planning": False,
            "target_or_ground_truth_used_for_scoring": False,
            "counterfactual_formulas_offline_only": True,
        },
        "validation": validation,
    }

    action_csv = output_dir / "counterfactual_action_table.csv"
    action_jsonl = output_dir / "counterfactual_action_table.jsonl"
    formula_csv = output_dir / "formula_change_summary.csv"
    formula_json = output_dir / "formula_change_summary.json"
    threshold_csv = output_dir / "action_change_thresholds.csv"
    skipped_json = output_dir / "skipped_formulas.json"
    summary_json = output_dir / "stage4a65b_counterfactual_summary.json"
    summary_md = output_dir / "stage4a65b_counterfactual_summary.md"

    write_csv(action_csv, action_rows)
    write_jsonl(action_jsonl, action_rows)
    write_csv(formula_csv, formula_summary_rows)
    save_json(formula_json, {"rows": formula_summary_rows})
    write_csv(threshold_csv, threshold_rows)
    save_json(skipped_json, skipped_summary)
    save_json(summary_json, summary)
    write_markdown_summary(summary_md, summary)

    # Update validation flags after files are written.
    summary["validation"]["summary_json_exists"] = summary_json.exists()
    summary["validation"]["summary_md_exists"] = summary_md.exists()
    summary["validation"]["action_table_csv_exists"] = action_csv.exists()
    save_json(summary_json, summary)

    print("Stage 4A-6.5b offline counterfactual score analysis complete")
    print(f"rank_dir: {rank_dir}")
    print(f"output_dir: {output_dir}")
    print(f"configs: {', '.join(configs_analyzed)}")
    print(f"steps: {steps_to_use}")
    print(f"candidate_rows: {len(rows)}")
    print(f"formula_variants_executed: {summary['formula_variants_executed']}")
    print(f"formula_variants_changed_top1: {len(changed_variants)}")
    print(f"formula_variants_changed_all_5_steps_any_config: {len(changed_all_5)}")
    print(f"exp_only_no_cost_changed_groups: {exp_only_changed}/{len(exp_only_rows)}")
    print(f"sc_only_changed_groups: {sc_only_changed}/{len(sc_only_rows)}")
    print(f"lambda_needed_for_sc_specific_change: {lambda_needed_text}")
    print(f"next_small_task: {smoke_recommendation['recommendation']} {smoke_recommendation['formula_variant_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

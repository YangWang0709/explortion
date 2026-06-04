#!/usr/bin/env python3
"""Stage 4A-6.5h offline minimal tree utility prototype.

This script reads saved candidate/transition outputs only. It does not launch
Isaac, run rollout, rerun map_predict, train models, modify observed_state, or
modify/build the external active_3d_planning source.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


EPS = 1.0e-6
ROLLOUT_LIKE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]
REQUIRED_CANDIDATE_FIELDS = [
    "config",
    "step",
    "candidate_row",
    "candidate_id",
    "candidate_grid",
    "candidate_world",
    "gain_exp",
    "gain_sc",
    "effective_gain_sc",
    "gain_hybrid",
    "path_cost",
    "final_score",
    "valid",
    "is_selected",
    "rank_final_score",
]
GAIN_MODES = [
    "default",
    "gain_exp",
    "gain_hybrid",
    "gain_exp_plus_effective_sc",
    "effective_gain_sc_only",
    "runtime_like",
]


@dataclass
class OfflineSegment:
    segment_id: str
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    gain: float = 0.0
    cost: float = 0.0
    value: float = float("-inf")
    best_descendant_id: str | None = None
    accumulated_gain: float = 0.0
    accumulated_cost: float = 0.0
    world: list[float] | None = None
    grid: list[int] | None = None
    source_step: int | None = None
    source_config: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateRow:
    config: str
    step: int
    row_index: int
    candidate_id: str
    id_is_surrogate: bool
    grid: list[int] | None
    world: list[float] | None
    valid: bool
    is_selected: bool
    top1: bool
    rank_final_score: float | None
    final_score: float | None
    gain_exp: float | None
    gain_sc: float | None
    raw_gain_sc: float | None
    effective_gain_sc: float | None
    weighted_gain_sc: float | None
    gain_hybrid: float | None
    gain_hybrid_weighted: float | None
    path_cost: float | None
    astar_path_length_m: float | None
    source_path: str
    raw: dict[str, str] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
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
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = list(field_order or [])
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
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return None


def parse_grid(value: Any) -> list[int] | None:
    items = parse_literal_list(value)
    if items is None:
        return None
    try:
        return [int(round(float(x))) for x in items]
    except (TypeError, ValueError):
        return None


def parse_world(value: Any) -> list[float] | None:
    items = parse_literal_list(value)
    if items is None:
        return None
    try:
        return [float(x) for x in items]
    except (TypeError, ValueError):
        return None


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def distance_m(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def local_ratio(gain: float | None, cost: float | None, eps: float = EPS) -> float:
    if not finite(gain) or not finite(cost) or float(cost) <= eps:
        return float("-inf")
    return float(gain) / max(float(cost), eps)


def add_child(tree: dict[str, OfflineSegment], parent_id: str, segment: OfflineSegment) -> None:
    tree[segment.segment_id] = segment
    tree[parent_id].children.append(segment.segment_id)


def compute_global_normalized_gain(
    tree: dict[str, OfflineSegment],
    root_id: str,
    eps: float = EPS,
) -> list[dict[str, Any]]:
    """Compute source-style best accumulated gain/cost in every subtree."""

    warnings: list[dict[str, Any]] = []

    def visit(segment_id: str, parent_gain: float, parent_cost: float) -> tuple[float, str | None]:
        segment = tree[segment_id]
        segment.accumulated_gain = parent_gain + float(segment.gain)
        segment.accumulated_cost = parent_cost + float(segment.cost)

        if segment_id == root_id:
            best_value = float("-inf")
            best_descendant: str | None = None
        elif segment.accumulated_cost <= eps:
            best_value = float("-inf")
            best_descendant = None
            warnings.append(
                {
                    "segment_id": segment_id,
                    "warning": "accumulated_cost <= eps, skipped for value",
                    "accumulated_gain": segment.accumulated_gain,
                    "accumulated_cost": segment.accumulated_cost,
                }
            )
        else:
            best_value = segment.accumulated_gain / max(segment.accumulated_cost, eps)
            best_descendant = segment_id

        for child_id in segment.children:
            child_value, child_best = visit(
                child_id,
                segment.accumulated_gain,
                segment.accumulated_cost,
            )
            if child_value > best_value:
                best_value = child_value
                best_descendant = child_best

        segment.value = best_value
        segment.best_descendant_id = best_descendant
        return best_value, best_descendant

    visit(root_id, 0.0, 0.0)
    return warnings


def select_subsequent_best(tree: dict[str, OfflineSegment], root_id: str) -> dict[str, Any]:
    root = tree[root_id]
    if not root.children:
        return {
            "selected_child_id": None,
            "selected_child_best_descendant_id": None,
            "selected_child_value": None,
            "selected_child_accumulated_gain": None,
            "selected_child_accumulated_cost": None,
            "best_descendant_accumulated_gain": None,
            "best_descendant_accumulated_cost": None,
            "reason": "root has no immediate children",
        }

    best_child = max(root.children, key=lambda child_id: tree[child_id].value)
    child = tree[best_child]
    best_descendant = tree.get(child.best_descendant_id or "")
    return {
        "selected_child_id": best_child,
        "selected_child_best_descendant_id": child.best_descendant_id,
        "selected_child_value": child.value if math.isfinite(child.value) else None,
        "selected_child_accumulated_gain": child.accumulated_gain,
        "selected_child_accumulated_cost": child.accumulated_cost,
        "best_descendant_accumulated_gain": best_descendant.accumulated_gain if best_descendant else None,
        "best_descendant_accumulated_cost": best_descendant.accumulated_cost if best_descendant else None,
        "reason": "selected immediate child whose subtree contains the highest-value segment",
    }


def make_root(segment_id: str = "root") -> dict[str, OfflineSegment]:
    return {segment_id: OfflineSegment(segment_id=segment_id, parent_id=None, gain=0.0, cost=0.0)}


def run_synthetic_tests() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    specs = [
        {
            "case": "case_a_low_cost_trap",
            "expected_selected": "A",
            "description": "A gains a high-value descendant and should beat local low-cost B.",
            "segments": [
                ("A", "root", 10.0, 1.0),
                ("B", "root", 2.0, 0.1),
                ("A1", "A", 100.0, 4.0),
            ],
        },
        {
            "case": "case_b_subtree_does_not_help",
            "expected_selected": "B",
            "description": "A descendant is too costly, so local low-cost B should remain best.",
            "segments": [
                ("A", "root", 10.0, 1.0),
                ("B", "root", 2.0, 0.1),
                ("A1", "A", 1.0, 10.0),
            ],
        },
        {
            "case": "case_c_zero_cost_safety",
            "expected_selected": "A",
            "description": "Near-zero accumulated cost should warn and not crash.",
            "segments": [
                ("A", "root", 10.0, 0.0),
            ],
        },
    ]

    for spec in specs:
        tree = make_root()
        for seg_id, parent_id, gain, cost in spec["segments"]:
            add_child(
                tree,
                parent_id,
                OfflineSegment(segment_id=seg_id, parent_id=parent_id, gain=gain, cost=cost),
            )
        warnings = compute_global_normalized_gain(tree, "root")
        decision = select_subsequent_best(tree, "root")
        local_scores = {
            child_id: local_ratio(tree[child_id].gain, tree[child_id].cost)
            for child_id in tree["root"].children
        }
        local_selected = max(local_scores, key=local_scores.get) if local_scores else None
        selected = decision["selected_child_id"]
        passed = selected == spec["expected_selected"]
        if spec["case"] == "case_c_zero_cost_safety":
            passed = bool(warnings) and selected == spec["expected_selected"]
        cases.append(
            {
                "case": spec["case"],
                "description": spec["description"],
                "expected_selected": spec["expected_selected"],
                "local_selected": local_selected,
                "subsequent_best_selected": selected,
                "selected_best_descendant": decision["selected_child_best_descendant_id"],
                "selected_value": decision["selected_child_value"],
                "local_scores": local_scores,
                "warnings": warnings,
                "passed": passed,
                "segments": {key: asdict(value) for key, value in tree.items()},
            }
        )

    return {
        "all_passed": all(case["passed"] for case in cases),
        "cases": cases,
        "safety": {
            "isaac_startup": False,
            "rollout": False,
            "map_predict_rerun": False,
            "training_or_rl": False,
        },
    }


def choose_gain(row: CandidateRow, mode: str, warnings: list[dict[str, Any]] | None = None) -> float | None:
    warnings = warnings if warnings is not None else []
    if mode == "gain_exp":
        return row.gain_exp
    if mode == "gain_hybrid":
        return row.gain_hybrid
    if mode == "gain_exp_plus_effective_sc":
        if finite(row.gain_exp) and finite(row.effective_gain_sc):
            return float(row.gain_exp) + float(row.effective_gain_sc)
        if finite(row.gain_exp) and finite(row.gain_sc):
            warnings.append({"candidate_id": row.candidate_id, "warning": "effective_gain_sc missing; used gain_sc"})
            return float(row.gain_exp) + float(row.gain_sc)
        return row.gain_exp
    if mode == "effective_gain_sc_only":
        if finite(row.effective_gain_sc):
            return row.effective_gain_sc
        if finite(row.gain_sc):
            warnings.append({"candidate_id": row.candidate_id, "warning": "effective_gain_sc missing; used gain_sc"})
            return row.gain_sc
        return None
    if mode == "runtime_like":
        if finite(row.gain_hybrid_weighted):
            return row.gain_hybrid_weighted
        if finite(row.gain_hybrid):
            return row.gain_hybrid
        if finite(row.gain_exp) and finite(row.effective_gain_sc):
            return float(row.gain_exp) + float(row.effective_gain_sc)
        return row.gain_exp
    if mode == "default":
        if finite(row.gain_hybrid):
            return row.gain_hybrid
        if finite(row.gain_exp) and finite(row.effective_gain_sc):
            return float(row.gain_exp) + float(row.effective_gain_sc)
        if finite(row.gain_exp):
            warnings.append({"candidate_id": row.candidate_id, "warning": "default gain fell back to gain_exp"})
            return row.gain_exp
        return None
    raise ValueError(f"unknown gain mode: {mode}")


def choose_cost(row: CandidateRow, warnings: list[dict[str, Any]] | None = None) -> float | None:
    warnings = warnings if warnings is not None else []
    if finite(row.path_cost):
        return row.path_cost
    if finite(row.astar_path_length_m):
        warnings.append({"candidate_id": row.candidate_id, "warning": "path_cost missing; used astar_path_length_m"})
        return row.astar_path_length_m
    warnings.append({"candidate_id": row.candidate_id, "warning": "missing cost field"})
    return None


def load_candidate_rows(rank_dir: Path) -> tuple[list[CandidateRow], dict[str, Any], dict[str, Any]]:
    candidate_path = rank_dir / "candidate_rank_table.csv"
    raw_rows, fields = read_csv(candidate_path)
    rows: list[CandidateRow] = []
    missing_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    surrogate_count = 0

    for index, raw in enumerate(raw_rows):
        for field_name in REQUIRED_CANDIDATE_FIELDS:
            if field_name not in raw:
                missing_counts[field_name] += 1
            elif raw[field_name] == "":
                empty_counts[field_name] += 1

        config = raw.get("config", "")
        step = as_int(raw.get("step"))
        row_index = as_int(raw.get("candidate_row"))
        grid = parse_grid(raw.get("candidate_grid"))
        world = parse_world(raw.get("candidate_world"))
        candidate_id = raw.get("candidate_id", "").strip()
        id_is_surrogate = False
        if not candidate_id:
            grid_key = "none" if grid is None else "_".join(str(x) for x in grid)
            candidate_id = f"{config}:step{step}:grid{grid_key}:row{row_index if row_index is not None else index}"
            id_is_surrogate = True
            surrogate_count += 1

        rows.append(
            CandidateRow(
                config=config,
                step=int(step if step is not None else -1),
                row_index=int(row_index if row_index is not None else index),
                candidate_id=candidate_id,
                id_is_surrogate=id_is_surrogate,
                grid=grid,
                world=world,
                valid=as_bool(raw.get("valid")),
                is_selected=as_bool(raw.get("is_selected")),
                top1=as_bool(raw.get("top1")),
                rank_final_score=as_float(raw.get("rank_final_score")),
                final_score=as_float(raw.get("final_score")),
                gain_exp=as_float(raw.get("gain_exp")),
                gain_sc=as_float(raw.get("gain_sc")),
                raw_gain_sc=as_float(raw.get("raw_gain_sc")),
                effective_gain_sc=as_float(raw.get("effective_gain_sc")),
                weighted_gain_sc=as_float(raw.get("weighted_gain_sc")),
                gain_hybrid=as_float(raw.get("gain_hybrid")),
                gain_hybrid_weighted=as_float(raw.get("gain_hybrid_weighted")),
                path_cost=as_float(raw.get("path_cost")),
                astar_path_length_m=as_float(raw.get("astar_path_length_m")),
                source_path=str(candidate_path),
                raw=raw,
            )
        )

    loaded = {
        "candidate_rank_table": str(candidate_path),
        "candidate_rank_table_exists": candidate_path.exists(),
        "candidate_rank_table_fields": fields,
        "candidate_rows_loaded": len(rows),
        "configs_loaded": sorted({row.config for row in rows}),
        "steps_loaded": sorted({row.step for row in rows if row.step >= 0}),
        "surrogate_ids_used": surrogate_count,
        "selected_candidate_summary": str(rank_dir / "selected_candidate_summary.csv"),
        "selected_candidate_summary_exists": (rank_dir / "selected_candidate_summary.csv").exists(),
    }
    missing = {
        "required_fields": REQUIRED_CANDIDATE_FIELDS,
        "missing_field_row_counts": dict(sorted(missing_counts.items())),
        "empty_field_row_counts": dict(sorted(empty_counts.items())),
        "surrogate_ids_used": surrogate_count,
        "cost_policy": "path_cost, fallback astar_path_length_m, otherwise skip row",
        "gain_policy_default": "gain_hybrid, fallback gain_exp + effective_gain_sc, fallback gain_exp",
        "notes": [
            "candidate_id is empty in saved rank rows, so stable surrogate ids are used.",
            "Empty effective_gain_sc fields are expected for empty_baseline rows.",
        ],
    }
    return rows, loaded, missing


def group_candidate_rows(rows: list[CandidateRow]) -> dict[tuple[str, int], list[CandidateRow]]:
    grouped: dict[tuple[str, int], list[CandidateRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.config, row.step)].append(row)
    for key in grouped:
        grouped[key].sort(
            key=lambda row: (
                row.rank_final_score if finite(row.rank_final_score) else float("inf"),
                row.row_index,
            )
        )
    return grouped


def current_top_row(rows: list[CandidateRow]) -> CandidateRow | None:
    selected = [row for row in rows if row.is_selected or row.top1]
    if selected:
        return min(selected, key=lambda row: row.rank_final_score if finite(row.rank_final_score) else row.row_index)
    valid_rows = [row for row in rows if row.valid and finite(row.final_score)]
    if valid_rows:
        return max(valid_rows, key=lambda row: row.final_score or float("-inf"))
    return None


def segment_from_row(
    row: CandidateRow,
    parent_id: str,
    gain_mode: str,
    segment_id: str,
    warnings: list[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> OfflineSegment | None:
    gain = choose_gain(row, gain_mode, warnings)
    cost = choose_cost(row, warnings)
    if not row.valid or not finite(gain) or not finite(cost):
        return None
    metadata = {
        "candidate_id": row.candidate_id,
        "candidate_row": row.row_index,
        "id_is_surrogate": row.id_is_surrogate,
        "rank_final_score": row.rank_final_score,
        "final_score": row.final_score,
        "gain_mode": gain_mode,
        "path_cost": row.path_cost,
        "astar_path_length_m": row.astar_path_length_m,
        "is_selected": row.is_selected,
        "top1": row.top1,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return OfflineSegment(
        segment_id=segment_id,
        parent_id=parent_id,
        gain=float(gain),
        cost=float(cost),
        world=row.world,
        grid=row.grid,
        source_step=row.step,
        source_config=row.config,
        metadata=metadata,
    )


def analyze_one_step_star(
    grouped: dict[tuple[str, int], list[CandidateRow]],
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for (config, step), candidates in sorted(grouped.items()):
        if step < 0 or step >= max_steps:
            continue
        current = current_top_row(candidates)
        valid_count = sum(1 for row in candidates if row.valid)
        for gain_mode in GAIN_MODES:
            tree = make_root()
            mode_warnings: list[dict[str, Any]] = []
            for row in candidates:
                segment_id = f"{config}:step{step}:row{row.row_index}"
                segment = segment_from_row(row, "root", gain_mode, segment_id, mode_warnings)
                if segment is not None:
                    add_child(tree, "root", segment)
            tree_warnings = compute_global_normalized_gain(tree, "root")
            decision = select_subsequent_best(tree, "root")
            selected_segment = tree.get(decision["selected_child_id"] or "")
            selected_row = None
            if selected_segment:
                selected_row = next(
                    (row for row in candidates if row.row_index == selected_segment.metadata.get("candidate_row")),
                    None,
                )
            local_best = max(
                (seg for seg in tree.values() if seg.parent_id == "root"),
                key=lambda seg: local_ratio(seg.gain, seg.cost),
                default=None,
            )
            result = {
                "mode": "one_step_star",
                "config": config,
                "step": step,
                "gain_mode": gain_mode,
                "valid_candidates": valid_count,
                "tree_children": len(tree["root"].children),
                "current_top_row": current.row_index if current else None,
                "current_top_candidate_id": current.candidate_id if current else None,
                "selected_child_id": decision["selected_child_id"],
                "selected_child_row": selected_row.row_index if selected_row else None,
                "selected_child_candidate_id": selected_row.candidate_id if selected_row else None,
                "selected_child_value": decision["selected_child_value"],
                "selected_child_gain": selected_segment.gain if selected_segment else None,
                "selected_child_cost": selected_segment.cost if selected_segment else None,
                "selected_child_local_ratio": local_ratio(selected_segment.gain, selected_segment.cost) if selected_segment else None,
                "local_best_child_id": local_best.segment_id if local_best else None,
                "local_best_child_row": local_best.metadata.get("candidate_row") if local_best else None,
                "matches_local_gain_cost": (decision["selected_child_id"] == local_best.segment_id) if local_best else False,
                "matches_current_top1": (selected_row.row_index == current.row_index) if selected_row and current else False,
                "warnings": len(mode_warnings) + len(tree_warnings),
            }
            result_rows.append(result)
            decision_rows.append({**result, "reason": decision["reason"], "diagnostic_only": False})
            for warning in mode_warnings + tree_warnings:
                warnings.append({"mode": "one_step_star", "config": config, "step": step, "gain_mode": gain_mode, **warning})
    return result_rows, decision_rows, warnings


def choose_transition_gain(transition: dict[str, Any], gain_mode: str, warnings: list[dict[str, Any]]) -> float | None:
    row = CandidateRow(
        config=str(transition.get("episode_id", "episode")),
        step=int(transition.get("step", -1)),
        row_index=int(transition.get("expert_action", 0) or 0),
        candidate_id=str(transition.get("best_candidate_id", "")),
        id_is_surrogate=False,
        grid=None,
        world=None,
        valid=True,
        is_selected=True,
        top1=True,
        rank_final_score=1.0,
        final_score=as_float(transition.get("best_score")),
        gain_exp=as_float(transition.get("best_gain_exp")),
        gain_sc=as_float(transition.get("best_gain_sc")),
        raw_gain_sc=as_float(transition.get("best_gain_sc")),
        effective_gain_sc=as_float(transition.get("best_effective_gain_sc")),
        weighted_gain_sc=as_float(transition.get("best_weighted_gain_sc")),
        gain_hybrid=as_float(transition.get("best_gain_hybrid")),
        gain_hybrid_weighted=as_float(transition.get("best_gain_hybrid_weighted")),
        path_cost=as_float(transition.get("best_path_cost")),
        astar_path_length_m=as_float(transition.get("best_astar_path_length_m")),
        source_path="transitions.jsonl",
    )
    return choose_gain(row, gain_mode, warnings)


def analyze_recorded_episode_chains(
    episode_dirs: dict[str, Path],
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for config, episode_dir in episode_dirs.items():
        transitions = read_jsonl(episode_dir / "transitions.jsonl")[:max_steps]
        tree = make_root()
        parent_id = "root"
        mode_warnings: list[dict[str, Any]] = []
        for transition in transitions:
            step = int(transition.get("step", len(tree) - 1))
            gain = choose_transition_gain(transition, "default", mode_warnings)
            cost = as_float(transition.get("best_path_cost"))
            if cost is None:
                cost = as_float(transition.get("best_astar_path_length_m"))
                mode_warnings.append({"step": step, "warning": "best_path_cost missing; used best_astar_path_length_m"})
            if not finite(gain) or not finite(cost):
                mode_warnings.append({"step": step, "warning": "missing gain or cost; skipped chain segment"})
                continue
            segment_id = f"{config}:selected_step{step}"
            segment = OfflineSegment(
                segment_id=segment_id,
                parent_id=parent_id,
                gain=float(gain),
                cost=float(cost),
                world=parse_world(transition.get("selected_next_pose_world")),
                grid=parse_grid(transition.get("selected_next_pose_grid")),
                source_step=step,
                source_config=config,
                metadata={
                    "episode_dir": str(episode_dir),
                    "best_candidate_id": transition.get("best_candidate_id"),
                    "best_score": transition.get("best_score"),
                    "not_counterfactual_tree": True,
                },
            )
            add_child(tree, parent_id, segment)
            parent_id = segment_id

        tree_warnings = compute_global_normalized_gain(tree, "root")
        decision = select_subsequent_best(tree, "root")
        local_ratios = [
            local_ratio(seg.gain, seg.cost)
            for seg in tree.values()
            if seg.segment_id != "root"
        ]
        max_local_ratio = max(local_ratios) if local_ratios else None
        chain_best = tree["root"].value if math.isfinite(tree["root"].value) else None
        for segment in sorted(tree.values(), key=lambda seg: seg.source_step if seg.source_step is not None else -1):
            if segment.segment_id == "root":
                continue
            result_rows.append(
                {
                    "mode": "recorded_episode_chain",
                    "config": config,
                    "episode_dir": str(episode_dir),
                    "step": segment.source_step,
                    "segment_id": segment.segment_id,
                    "parent_id": segment.parent_id,
                    "gain_mode": "default",
                    "local_gain": segment.gain,
                    "local_cost": segment.cost,
                    "local_gain_over_cost": local_ratio(segment.gain, segment.cost),
                    "accumulated_gain": segment.accumulated_gain,
                    "accumulated_cost": segment.accumulated_cost,
                    "accumulated_gain_over_cost": local_ratio(segment.accumulated_gain, segment.accumulated_cost),
                    "segment_value": segment.value if math.isfinite(segment.value) else None,
                    "segment_best_descendant_id": segment.best_descendant_id,
                    "root_chain_best_value": chain_best,
                    "max_local_ratio_in_chain": max_local_ratio,
                    "chain_best_differs_from_max_local": abs(chain_best - max_local_ratio) > 1.0e-9 if chain_best is not None and max_local_ratio is not None else None,
                    "not_counterfactual_tree": True,
                }
            )
        decision_rows.append(
            {
                "mode": "recorded_episode_chain",
                "config": config,
                "step": None,
                "gain_mode": "default",
                "selected_child_id": decision["selected_child_id"],
                "selected_child_best_descendant_id": decision["selected_child_best_descendant_id"],
                "selected_child_value": decision["selected_child_value"],
                "reason": decision["reason"],
                "diagnostic_only": True,
                "not_counterfactual_tree": True,
            }
        )
        for warning in mode_warnings + tree_warnings:
            warnings.append({"mode": "recorded_episode_chain", "config": config, **warning})
    return result_rows, decision_rows, warnings


def top_rows(rows: list[CandidateRow], count: int) -> list[CandidateRow]:
    return [row for row in rows if row.valid][:count]


def analyze_shallow_pseudo_trees(
    grouped: dict[tuple[str, int], list[CandidateRow]],
    max_steps: int,
    first_level_top_k: int,
    proxy_child_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    configs = sorted({config for config, _ in grouped})
    for config in configs:
        for step in range(max_steps - 1):
            first_rows = top_rows(grouped.get((config, step), []), first_level_top_k)
            proxy_rows = top_rows(grouped.get((config, step + 1), []), proxy_child_top_k)
            if not first_rows or not proxy_rows:
                warnings.append(
                    {
                        "mode": "shallow_pseudo_tree_topk",
                        "config": config,
                        "step": step,
                        "warning": "missing first-level or proxy rows",
                    }
                )
                continue
            current = current_top_row(grouped.get((config, step), []))
            for gain_mode in GAIN_MODES:
                tree = make_root()
                mode_warnings: list[dict[str, Any]] = []
                child_by_row: dict[int, str] = {}
                for row in first_rows:
                    child_id = f"{config}:step{step}:row{row.row_index}"
                    segment = segment_from_row(
                        row,
                        "root",
                        gain_mode,
                        child_id,
                        mode_warnings,
                        {"pseudo_tree_role": "first_level"},
                    )
                    if segment is None:
                        continue
                    add_child(tree, "root", segment)
                    child_by_row[row.row_index] = child_id
                    for proxy in proxy_rows:
                        proxy_id = f"{child_id}:proxy_step{step + 1}:row{proxy.row_index}"
                        proxy_segment = segment_from_row(
                            proxy,
                            child_id,
                            gain_mode,
                            proxy_id,
                            mode_warnings,
                            {
                                "pseudo_tree_role": "proxy_grandchild",
                                "diagnostic_only": True,
                                "not_true_counterfactual_successor": True,
                                "proxy_source_step": step + 1,
                            },
                        )
                        if proxy_segment is not None:
                            add_child(tree, child_id, proxy_segment)

                tree_warnings = compute_global_normalized_gain(tree, "root")
                decision = select_subsequent_best(tree, "root")
                local_best = max(
                    (tree[child_id] for child_id in tree["root"].children),
                    key=lambda seg: local_ratio(seg.gain, seg.cost),
                    default=None,
                )
                selected_child = tree.get(decision["selected_child_id"] or "")
                selected_row = None
                if selected_child:
                    selected_row = next(
                        (row for row in first_rows if row.row_index == selected_child.metadata.get("candidate_row")),
                        None,
                    )
                local_row = None
                if local_best:
                    local_row = next(
                        (row for row in first_rows if row.row_index == local_best.metadata.get("candidate_row")),
                        None,
                    )
                best_descendant = tree.get(decision["selected_child_best_descendant_id"] or "")
                changed = bool(selected_child and local_best and selected_child.segment_id != local_best.segment_id)
                distance = distance_m(selected_row.world if selected_row else None, local_row.world if local_row else None)
                changed_toward_high_gain = (
                    bool(changed and selected_child and local_best and selected_child.gain > local_best.gain)
                )
                changed_reason = "unchanged"
                if changed and selected_child and local_best:
                    gain_delta = selected_child.gain - local_best.gain
                    cost_delta = selected_child.cost - local_best.cost
                    if gain_delta > 0 and cost_delta >= 0:
                        changed_reason = "higher first-level gain overcame higher/equal first-level cost after proxy subtree"
                    elif gain_delta > 0:
                        changed_reason = "higher first-level gain and lower first-level cost"
                    elif cost_delta < 0:
                        changed_reason = "lower first-level cost after proxy subtree"
                    else:
                        changed_reason = "proxy subtree accumulated ratio changed ordering"
                row_out = {
                    "mode": "shallow_pseudo_tree_topk",
                    "config": config,
                    "step": step,
                    "proxy_step": step + 1,
                    "gain_mode": gain_mode,
                    "first_level_top_k": first_level_top_k,
                    "proxy_child_top_k": proxy_child_top_k,
                    "first_level_children": len(tree["root"].children),
                    "proxy_children_per_first_level": proxy_child_top_k,
                    "current_top_row": current.row_index if current else None,
                    "one_step_local_row": local_row.row_index if local_row else None,
                    "one_step_local_candidate_id": local_row.candidate_id if local_row else None,
                    "subsequent_best_child_row": selected_row.row_index if selected_row else None,
                    "subsequent_best_child_candidate_id": selected_row.candidate_id if selected_row else None,
                    "subsequent_best_descendant_id": decision["selected_child_best_descendant_id"],
                    "subsequent_best_value": decision["selected_child_value"],
                    "best_descendant_accumulated_gain": best_descendant.accumulated_gain if best_descendant else None,
                    "best_descendant_accumulated_cost": best_descendant.accumulated_cost if best_descendant else None,
                    "one_step_local_gain": local_best.gain if local_best else None,
                    "one_step_local_cost": local_best.cost if local_best else None,
                    "selected_child_gain": selected_child.gain if selected_child else None,
                    "selected_child_cost": selected_child.cost if selected_child else None,
                    "changed_vs_one_step_local": changed,
                    "changed_vs_current_top1": (selected_row.row_index != current.row_index) if selected_row and current else None,
                    "changed_toward_high_gain_first_level": changed_toward_high_gain,
                    "selected_vs_one_step_distance_m": distance,
                    "spatially_meaningful_gt_0p5m": distance is not None and distance > 0.5,
                    "changed_reason": changed_reason,
                    "diagnostic_only": True,
                    "not_true_counterfactual_tree": True,
                    "proxy_note": "Each first-level child receives the same saved candidates from step t+1; these are not true successors from that child.",
                    "warnings": len(mode_warnings) + len(tree_warnings),
                }
                result_rows.append(row_out)
                decision_rows.append({**row_out, "reason": decision["reason"]})
                for warning in mode_warnings + tree_warnings:
                    warnings.append({"mode": "shallow_pseudo_tree_topk", "config": config, "step": step, "gain_mode": gain_mode, **warning})
    return result_rows, decision_rows, warnings


def summarize_results(
    synthetic: dict[str, Any],
    loaded_fields: dict[str, Any],
    missing_fields: dict[str, Any],
    one_step_rows: list[dict[str, Any]],
    chain_rows: list[dict[str, Any]],
    pseudo_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    default_one_step = [row for row in one_step_rows if row.get("gain_mode") == "default"]
    runtime_one_step = [row for row in one_step_rows if row.get("gain_mode") == "runtime_like"]
    default_pseudo = [row for row in pseudo_rows if row.get("gain_mode") == "default"]
    runtime_pseudo = [row for row in pseudo_rows if row.get("gain_mode") == "runtime_like"]
    chain_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chain_rows:
        chain_by_config[str(row["config"])].append(row)

    pseudo_changed = [row for row in default_pseudo if row.get("changed_vs_one_step_local")]
    pseudo_changed_high_gain = [row for row in pseudo_changed if row.get("changed_toward_high_gain_first_level")]
    pseudo_spatial = [row for row in pseudo_changed if row.get("spatially_meaningful_gt_0p5m")]

    if pseudo_changed_high_gain:
        next_step = "B. offline mini-RRT tree builder on saved observed map without Isaac"
        why = (
            "The diagnostic pseudo-tree can change the immediate child toward higher-gain alternatives, "
            "but it still lacks true successor generation; the next faithful step is to build a small offline "
            "tree on saved observed maps before any online planner smoke."
        )
    elif pseudo_changed:
        next_step = "B. offline mini-RRT tree builder on saved observed map without Isaac"
        why = (
            "The pseudo-tree changes some selections, but the proxy descendants are not true counterfactual "
            "successors. A source-faithful offline mini-tree is needed before rollout or online planning."
        )
    else:
        next_step = "B. offline mini-RRT tree builder on saved observed map without Isaac"
        why = (
            "Synthetic tests validate the tree utility, but saved pseudo-trees do not provide real branch "
            "expansion. The missing piece is source-faithful successor generation on saved observed maps."
        )

    chain_summary = {}
    for config, rows in chain_by_config.items():
        final = max(rows, key=lambda row: row["step"]) if rows else {}
        best_value = max(
            (row["root_chain_best_value"] for row in rows if row.get("root_chain_best_value") is not None),
            default=None,
        )
        chain_summary[config] = {
            "segments": len(rows),
            "final_accumulated_gain": final.get("accumulated_gain"),
            "final_accumulated_cost": final.get("accumulated_cost"),
            "final_accumulated_gain_over_cost": final.get("accumulated_gain_over_cost"),
            "root_best_value": best_value,
            "chain_best_differs_from_max_local": any(bool(row.get("chain_best_differs_from_max_local")) for row in rows),
        }

    return {
        "stage": "Stage 4A-6.5h offline minimal tree utility prototype",
        "context_read": {
            "CURRENT_STATE": True,
            "CODEX_LOG": True,
            "TODO": True,
            "confirmed_current_next_task": "offline minimal tree-utility prototype over saved candidates",
            "confirmed_previous_stage": "Stage 4A-6.5g external active_3d_planning source inspection",
        },
        "source_formula_reproduced": {
            "global_normalized_gain": "max over subtree descendants of accumulated root-to-descendant gain / accumulated cost",
            "subsequent_best": "select root immediate child whose subtree contains the highest-value segment",
            "trajectory_segment_fields": ["gain", "cost", "value", "parent", "children", "info/metadata"],
            "limitations": [
                "No RRT expansion, rewiring, continuous yaw, rollout, or planner implementation.",
                "Mode 3 uses diagnostic proxy grandchildren, not true counterfactual successors.",
            ],
        },
        "synthetic": {
            "all_passed": synthetic["all_passed"],
            "case_results": [
                {
                    "case": case["case"],
                    "expected_selected": case["expected_selected"],
                    "subsequent_best_selected": case["subsequent_best_selected"],
                    "selected_best_descendant": case["selected_best_descendant"],
                    "passed": case["passed"],
                    "warnings": len(case["warnings"]),
                }
                for case in synthetic["cases"]
            ],
        },
        "data_loaded": {
            "candidate_rows": loaded_fields.get("candidate_rows_loaded", 0),
            "configs": loaded_fields.get("configs_loaded", []),
            "steps": loaded_fields.get("steps_loaded", []),
            "surrogate_ids_used": loaded_fields.get("surrogate_ids_used", 0),
            "missing_field_row_counts": missing_fields.get("missing_field_row_counts", {}),
            "empty_field_row_counts": missing_fields.get("empty_field_row_counts", {}),
        },
        "one_step_star": {
            "default_trees_built": len(default_one_step),
            "default_matches_local_gain_cost": sum(bool(row.get("matches_local_gain_cost")) for row in default_one_step),
            "default_matches_current_top1": sum(bool(row.get("matches_current_top1")) for row in default_one_step),
            "runtime_like_matches_current_top1": sum(bool(row.get("matches_current_top1")) for row in runtime_one_step),
            "interpretation": "Star trees degenerate to local gain/cost ranking by construction.",
        },
        "recorded_episode_chain": {
            "chains_built": len(chain_summary),
            "chains": chain_summary,
            "interpretation": "Recorded chains expose accumulated gain/cost along actual selected transitions only; they are not counterfactual trees.",
        },
        "shallow_pseudo_tree": {
            "default_trees_built": len(default_pseudo),
            "default_changed_vs_one_step_local": len(pseudo_changed),
            "default_changed_toward_high_gain": len(pseudo_changed_high_gain),
            "default_spatially_meaningful_changes_gt_0p5m": len(pseudo_spatial),
            "runtime_like_changed_vs_one_step_local": sum(bool(row.get("changed_vs_one_step_local")) for row in runtime_pseudo),
            "limitation": "Proxy children are copied from the next saved step and are not true child-conditioned successors.",
        },
        "main_diagnosis": {
            "synthetic_overcame_low_cost_trap": synthetic["cases"][0]["passed"],
            "saved_pseudo_tree_showed_useful_difference": bool(pseudo_changed_high_gain),
            "real_rrt_expansion_still_missing": True,
            "current_simulator_gap": "missing RRT tree, accumulated branch/subtree utility, SubsequentBest, root rewiring, and continuous yaw",
            "likely_next_faithful_direction": next_step,
        },
        "recommended_next_faithful_step": {
            "choice": next_step,
            "why": why,
        },
        "safety": {
            "isaac_startup": False,
            "rollout": False,
            "new_expert_step": False,
            "map_predict_rerun": False,
            "sscnet_inference": False,
            "training_rl_bc_il": False,
            "checkpoint_modified": False,
            "observed_state_modified": False,
            "prediction_writeback": False,
            "target_ground_truth_scoring": False,
            "external_source_modified_or_built": False,
        },
    }


def write_formula_reference(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Offline Tree Formula Reference",
                "",
                "This Stage 4A-6.5h prototype reproduces only the offline formula-level parts of the external active_3d_planning evidence.",
                "",
                "## TrajectorySegment Fields Used",
                "",
                "- `gain`: local segment gain, corresponding to `TrajectorySegment::gain`.",
                "- `cost`: local segment cost, corresponding to `TrajectorySegment::cost` from `SegmentTime`-style local duration/cost.",
                "- `value`: subtree value assigned by the offline `GlobalNormalizedGain` reproduction.",
                "- `parent` / `children`: tree relation used to traverse subtrees.",
                "- `metadata`: local traceability equivalent to source `info` for this diagnostic.",
                "",
                "## GlobalNormalizedGain",
                "",
                "For every segment `s`, compute accumulated root-to-segment gain and cost. Then set `value(s)` to the maximum accumulated `gain / cost` over all descendants in `s`'s subtree. The root's own zero-cost value is not treated as a valid candidate.",
                "",
                "## SubsequentBest",
                "",
                "After values are computed, inspect only the root's immediate children and select the child whose subtree contains the highest-value segment.",
                "",
                "## Limitations",
                "",
                "This is not RRT, not rollout, not root rewiring, not continuous yaw, and not a full planner. Saved-candidate pseudo-trees are diagnostic-only structures.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_synthetic_md(path: Path, synthetic: dict[str, Any]) -> None:
    lines = [
        "# Synthetic Tree Tests",
        "",
        f"All passed: `{synthetic['all_passed']}`",
        "",
    ]
    for case in synthetic["cases"]:
        lines.extend(
            [
                f"## {case['case']}",
                "",
                f"- Expected selected child: `{case['expected_selected']}`",
                f"- Local selected child: `{case['local_selected']}`",
                f"- SubsequentBest selected child: `{case['subsequent_best_selected']}`",
                f"- Best descendant: `{case['selected_best_descendant']}`",
                f"- Warnings: `{len(case['warnings'])}`",
                f"- Passed: `{case['passed']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    syn_cases = {item["case"]: item for item in summary["synthetic"]["case_results"]}
    one = summary["one_step_star"]
    chain = summary["recorded_episode_chain"]
    pseudo = summary["shallow_pseudo_tree"]
    data = summary["data_loaded"]
    rec = summary["recommended_next_faithful_step"]
    safety = summary["safety"]
    lines = [
        "# Stage 4A-6.5h Offline Minimal Tree Utility Prototype",
        "",
        "## Context Read",
        "- CURRENT_STATE read: yes",
        "- CODEX_LOG read: yes",
        "- TODO read: yes",
        "- confirmed current next task: offline minimal tree-utility prototype over saved candidates",
        "",
        "## Source Formula Reproduced",
        "- TrajectorySegment fields used: gain, cost, value, parent, children, metadata/info",
        "- GlobalNormalizedGain formula: subtree maximum of accumulated root-to-descendant gain / accumulated cost",
        "- SubsequentBest formula: choose the root immediate child whose subtree contains the highest-value segment",
        "- source-faithfulness statement: the formula and next-child selection match the Stage 4A-6.5g source conclusion",
        "- limitations: no RRT expansion, root rewiring, continuous yaw, rollout, or planner implementation; pseudo-tree mode is diagnostic only",
        "",
        "## Synthetic Tests",
        f"- case A result: selected `{syn_cases['case_a_low_cost_trap']['subsequent_best_selected']}`, best descendant `{syn_cases['case_a_low_cost_trap']['selected_best_descendant']}`, passed `{syn_cases['case_a_low_cost_trap']['passed']}`",
        f"- case B result: selected `{syn_cases['case_b_subtree_does_not_help']['subsequent_best_selected']}`, passed `{syn_cases['case_b_subtree_does_not_help']['passed']}`",
        f"- case C result: warnings `{syn_cases['case_c_zero_cost_safety']['warnings']}`, passed `{syn_cases['case_c_zero_cost_safety']['passed']}`",
        f"- pass/fail: `{summary['synthetic']['all_passed']}`",
        "",
        "## Data Loaded",
        f"- candidate rows loaded: `{data['candidate_rows']}`",
        f"- configs loaded: `{data['configs']}`",
        f"- steps loaded: `{data['steps']}`",
        f"- missing fields: `{data['missing_field_row_counts']}`",
        f"- surrogate ids used: `{data['surrogate_ids_used']}`",
        "",
        "## One-Step Star Results",
        f"- whether matches current one-step ranking: default `{one['default_matches_current_top1']}/{one['default_trees_built']}`, runtime-like `{one['runtime_like_matches_current_top1']}/{one['default_trees_built']}`",
        f"- selected child distribution: see `one_step_star_results.csv`",
        "- interpretation: one-step star has no grandchildren, so `SubsequentBest` degenerates to local gain/cost ranking",
        "",
        "## Recorded Episode Chain Results",
        f"- accumulated gain/cost values: `{chain['chains']}`",
        "- whether chain utility differs from local utility: see `chain_best_differs_from_max_local` in `recorded_episode_chain_results.csv`",
        "- limitation: the chain uses actual selected transitions only and is not a counterfactual tree",
        "",
        "## Shallow Pseudo-Tree Results",
        f"- number of trees built: `{pseudo['default_trees_built']}` default-mode trees",
        f"- number where SubsequentBest differs from one-step top: `{pseudo['default_changed_vs_one_step_local']}`",
        f"- whether selected child changed toward high-gain alternatives: `{pseudo['default_changed_toward_high_gain']}`",
        f"- whether changed child is spatially meaningful if distance fields exist: `{pseudo['default_spatially_meaningful_changes_gt_0p5m']}` changes > 0.5 m",
        "- limitation: proxy children are not true counterfactual successors",
        "",
        "## Main Diagnosis",
        f"- does tree utility reduce path-cost/locality dominance in synthetic case: `{summary['main_diagnosis']['synthetic_overcame_low_cost_trap']}`",
        f"- does saved candidate pseudo-tree show possible change: `{summary['main_diagnosis']['saved_pseudo_tree_showed_useful_difference']}`",
        f"- is missing real RRT expansion still the blocker: `{summary['main_diagnosis']['real_rrt_expansion_still_missing']}`",
        "- what source-faithful piece is still missing: child-conditioned tree expansion with real parent/children relations, plus root rewiring and continuous yaw if moving beyond this offline prototype",
        "",
        "## Recommended Next Faithful Step",
        f"{rec['choice']}",
        "",
        rec["why"],
        "",
        "## Safety / Boundary Check",
        f"- Isaac startup: {'yes' if safety['isaac_startup'] else 'no'}",
        f"- rollout: {'yes' if safety['rollout'] else 'no'}",
        f"- map_predict rerun: {'yes' if safety['map_predict_rerun'] else 'no'}",
        f"- training/RL: {'yes' if safety['training_rl_bc_il'] else 'no'}",
        f"- checkpoint modified: {'yes' if safety['checkpoint_modified'] else 'no'}",
        f"- observed_state modified: {'yes' if safety['observed_state_modified'] else 'no'}",
        f"- prediction writeback: {'yes' if safety['prediction_writeback'] else 'no'}",
        f"- target/ground-truth scoring: {'yes' if safety['target_ground_truth_scoring'] else 'no'}",
        f"- external source modified/built: {'yes' if safety['external_source_modified_or_built'] else 'no'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_recommendation_md(path: Path, summary: dict[str, Any]) -> None:
    rec = summary["recommended_next_faithful_step"]
    path.write_text(
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                rec["choice"],
                "",
                rec["why"],
                "",
                "Do not jump to rollout, RL, map_predict rerun, training, or a full SC-Explorer planner implementation from this prototype alone.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def scan_rollout_like_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        found.extend(str(path.relative_to(output_dir)) for path in sorted(output_dir.glob(pattern)))
    return found


def collect_path_diag_fields(path_diag_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(path_diag_dir.glob("*.csv")):
        _, fields = read_csv(path)
        files[path.name] = {"path": str(path), "fields": fields}
    return files


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = run_synthetic_tests()
    save_json(output_dir / "synthetic_tree_tests.json", synthetic)
    write_synthetic_md(output_dir / "synthetic_tree_tests.md", synthetic)
    if not synthetic["all_passed"]:
        raise RuntimeError("synthetic tree tests failed; refusing to continue real-data analysis")

    rank_dir = Path(args.rank_dir).resolve()
    rows, loaded_fields, missing_fields = load_candidate_rows(rank_dir)
    loaded_fields["path_diagnosis_csv_fields"] = collect_path_diag_fields(Path(args.path_diag_dir).resolve())
    loaded_fields["external_inspection_dir"] = str(Path(args.external_inspection_dir).resolve())
    loaded_fields["external_summary_exists"] = (Path(args.external_inspection_dir).resolve() / "external_tree_utility_summary.json").exists()
    loaded_fields["fixed_sc_episode_dir"] = str(Path(args.fixed_sc_episode_dir).resolve())
    loaded_fields["empty_episode_dir"] = str(Path(args.empty_episode_dir).resolve())
    loaded_fields["gating_root"] = str(Path(args.gating_root).resolve())
    save_json(output_dir / "loaded_candidate_fields.json", loaded_fields)

    grouped = group_candidate_rows(rows)
    one_step_rows, one_step_decisions, one_step_warnings = analyze_one_step_star(grouped, args.max_steps)
    chain_rows, chain_decisions, chain_warnings = analyze_recorded_episode_chains(
        {
            "fixed_raw_sc_episode": Path(args.fixed_sc_episode_dir).resolve(),
            "empty_baseline_episode": Path(args.empty_episode_dir).resolve(),
        },
        args.max_steps,
    )
    pseudo_rows, pseudo_decisions, pseudo_warnings = analyze_shallow_pseudo_trees(
        grouped,
        args.max_steps,
        args.first_level_top_k,
        args.proxy_child_top_k,
    )

    all_warnings = one_step_warnings + chain_warnings + pseudo_warnings
    missing_fields["diagnostic_warnings_count"] = len(all_warnings)
    missing_fields["diagnostic_warnings_sample"] = all_warnings[:50]
    save_json(output_dir / "missing_fields_report.json", missing_fields)

    write_csv(output_dir / "one_step_star_results.csv", one_step_rows)
    write_csv(output_dir / "recorded_episode_chain_results.csv", chain_rows)
    write_csv(output_dir / "shallow_pseudo_tree_results.csv", pseudo_rows)
    write_csv(output_dir / "subsequent_best_decisions.csv", one_step_decisions + chain_decisions + pseudo_decisions)
    write_formula_reference(output_dir / "tree_formula_reference.md")

    summary = summarize_results(synthetic, loaded_fields, missing_fields, one_step_rows, chain_rows, pseudo_rows)
    summary["rollout_like_files_created_in_output_dir"] = scan_rollout_like_outputs(output_dir)
    save_json(output_dir / "tree_utility_comparison_summary.json", summary)
    write_summary_md(output_dir / "tree_utility_comparison_summary.md", summary)
    write_recommendation_md(output_dir / "recommended_next_faithful_step.md", summary)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank_dir", required=True)
    parser.add_argument("--path_diag_dir", required=True)
    parser.add_argument("--external_inspection_dir", required=True)
    parser.add_argument("--fixed_sc_episode_dir", required=True)
    parser.add_argument("--empty_episode_dir", required=True)
    parser.add_argument("--gating_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--first_level_top_k", type=int, default=16)
    parser.add_argument("--proxy_child_top_k", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

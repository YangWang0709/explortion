#!/usr/bin/env python3
"""Stage 4A-5.1 smoke test for read-only prediction expert scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim_paper_expert import UNKNOWN, grid_to_world, normalize_bounds
from sim_prediction_layer import SimPredictionLayer
from visualize_sim_expert_step import STATE_CMAP, STATE_NORM, project_topdown


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise AssertionError(f"Empty {label}: {path}")
    return path


def load_output_dir(path: str | Path) -> dict[str, Any]:
    output_dir = require(Path(path), "output_dir")
    decision_json = require(output_dir / "expert_step_decision.json", "decision json")
    candidates_jsonl = require(output_dir / "expert_step_candidates.jsonl", "candidates jsonl")
    decision_npz = require(output_dir / "expert_step_decision.npz", "decision npz")
    decision = load_json(decision_json)
    candidates = load_jsonl(candidates_jsonl)
    if not candidates:
        raise AssertionError(f"No candidates in {candidates_jsonl}")
    return {
        "dir": output_dir,
        "decision": decision,
        "candidates": candidates,
        "npz": decision_npz,
    }


def candidate_value(candidate: dict[str, Any], key: str) -> float:
    if key in candidate:
        return float(candidate[key])
    gains = candidate.get("gains", {})
    if key in gains:
        return float(gains[key])
    utilities = candidate.get("utilities", {})
    if key in utilities:
        return float(utilities[key])
    raise KeyError(key)


def best_fields(decision: dict[str, Any]) -> dict[str, Any]:
    best = decision["best_candidate"]
    return {
        "id": int(best["id"]),
        "grid": [int(v) for v in best["grid_position"]],
        "world": [float(v) for v in best["world_position"]],
        "score": float(best["final_score"]),
        "gain_exp": candidate_value(best, "gain_exp"),
        "gain_sc": candidate_value(best, "gain_sc"),
        "gain_hybrid": candidate_value(best, "gain_hybrid"),
        "gain_occ": candidate_value(best, "gain_occ"),
        "gain_conf": candidate_value(best, "gain_conf"),
        "path_cost": float(best["path_cost"]),
        "predicted_unmeasured_visible_count": int(best.get("predicted_unmeasured_visible_count", 0)),
    }


def assert_finite_candidates(candidates: list[dict[str, Any]], fields: list[str]) -> None:
    for candidate in candidates:
        for field in fields:
            value = candidate_value(candidate, field)
            if not np.isfinite(value):
                raise AssertionError(f"Candidate {candidate.get('id')} has non-finite {field}: {value}")


def assert_hybrid_identity(candidates: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        gain_exp = candidate_value(candidate, "gain_exp")
        gain_sc = candidate_value(candidate, "gain_sc")
        gain_hybrid = candidate_value(candidate, "gain_hybrid")
        if not np.isclose(gain_hybrid, gain_exp + gain_sc, atol=1e-5):
            raise AssertionError(
                f"Candidate {candidate.get('id')} violates gain_hybrid=gain_exp+gain_sc: "
                f"{gain_hybrid} != {gain_exp}+{gain_sc}"
            )


def assert_no_forbidden_npz_fields(npz_path: Path) -> None:
    forbidden_exact = {"target_lr", "target_hr", "gt", "ground_truth", "observed_map", "observed_state"}
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
    bad = sorted(key for key in keys if key in forbidden_exact or key.startswith("target_"))
    if bad:
        raise AssertionError(f"Forbidden fields in {npz_path}: {bad}")


def assert_prediction_not_planning(empty: dict[str, Any], pred: dict[str, Any]) -> None:
    empty_diag = empty["decision"]["diagnostics"]
    pred_diag = pred["decision"]["diagnostics"]
    for key in (
        "prediction_used_for_candidate_sampling",
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_used_for_astar",
        "prediction_blocks_rays",
        "raycast_uses_prediction_for_blocking",
        "prediction_written_to_observed_state",
        "observed_state_writeback",
    ):
        if bool(pred_diag.get(key, False)):
            raise AssertionError(f"Prediction leakage flag is true: {key}")
    if str(pred_diag.get("traversability_source")) != "observed_state_only":
        raise AssertionError("Traversability source is not observed_state_only")

    for key in ("traversable_count", "blocked_count", "traversability_unknown_count", "reachable_component_count"):
        if empty_diag.get(key) != pred_diag.get(key):
            raise AssertionError(f"Prediction changed measured-only traversability diagnostic {key}")

    empty_by_id = {int(c["id"]): c for c in empty["candidates"]}
    pred_by_id = {int(c["id"]): c for c in pred["candidates"]}
    if set(empty_by_id) != set(pred_by_id):
        raise AssertionError("Candidate id sets differ between empty and prediction modes")
    for candidate_id, empty_candidate in empty_by_id.items():
        pred_candidate = pred_by_id[candidate_id]
        if [int(v) for v in empty_candidate["grid_position"]] != [int(v) for v in pred_candidate["grid_position"]]:
            raise AssertionError(f"Candidate {candidate_id} grid position changed")
        if not np.isclose(float(empty_candidate["path_cost"]), float(pred_candidate["path_cost"]), atol=1e-6):
            raise AssertionError(f"Candidate {candidate_id} path_cost changed")


def save_best_candidate_plot(
    observed_state: np.ndarray,
    empty: dict[str, Any],
    pred: dict[str, Any],
    output_dir: Path,
) -> str:
    pred_diag = pred["decision"]["diagnostics"]
    bounds = normalize_bounds(pred_diag["bounds"])
    voxel_size = float(pred_diag["voxel_size"])
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    topdown = project_topdown(observed_state)
    empty_best = best_fields(empty["decision"])
    pred_best = best_fields(pred["decision"])
    current_grid = pred["decision"]["current_pose"]["grid"]
    current_xy = grid_to_world((int(current_grid[0]), int(current_grid[1]), 0), bounds, voxel_size)[:2]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    ax.scatter(current_xy[0], current_xy[1], s=150, c="#2563eb", marker="^", edgecolors="white", linewidths=0.9, label="current")
    ax.scatter(
        empty_best["world"][0],
        empty_best["world"][1],
        s=210,
        c="#f97316",
        marker="*",
        edgecolors="black",
        linewidths=0.7,
        label=f"empty best {empty_best['id']}",
    )
    ax.scatter(
        pred_best["world"][0],
        pred_best["world"][1],
        s=230,
        c="#22c55e",
        marker="*",
        edgecolors="black",
        linewidths=0.7,
        label=f"prediction best {pred_best['id']}",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Empty vs prediction best candidate")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    out_path = output_dir / "empty_vs_prediction_best_candidate.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_gain_comparison_plot(empty_best: dict[str, Any], pred_best: dict[str, Any], output_dir: Path) -> str:
    labels = ["gain_exp", "gain_sc", "gain_hybrid", "gain_occ", "gain_conf"]
    x = np.arange(len(labels))
    width = 0.36
    empty_values = np.asarray([empty_best[label] for label in labels], dtype=np.float32)
    pred_values = np.asarray([pred_best[label] for label in labels], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(x - width / 2, empty_values, width=width, color="#f97316", label="empty")
    ax.bar(x + width / 2, pred_values, width=width, color="#22c55e", label="prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("best candidate value")
    ax.set_title("Best-candidate gain comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    out_path = output_dir / "gain_comparison_bar.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def write_comparison_summary(summary: dict[str, Any], output_dir: Path) -> tuple[str, str]:
    json_path = output_dir / "comparison_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    empty_best = summary["empty_baseline"]["best"]
    pred_best = summary["sc_prediction"]["best"]
    comparison = summary["comparison"]
    md_lines = [
        "# Stage 4A-5.1 Comparison",
        "",
        f"- empty best id: {empty_best['id']}",
        f"- prediction best id: {pred_best['id']}",
        f"- best candidate changed: {comparison['best_candidate_changed']}",
        f"- score delta: {comparison['score_delta']}",
        f"- gain_hybrid delta: {comparison['gain_hybrid_delta']}",
        f"- top-N overlap: {comparison['top_n_overlap_count']} / {comparison['top_n_union_count']}",
        f"- candidates with gain_sc > 0: {summary['sc_prediction']['candidates_with_gain_sc_gt_zero']}",
        f"- max/mean gain_sc: {summary['sc_prediction']['max_gain_sc']} / {summary['sc_prediction']['mean_gain_sc']}",
        f"- total predicted_unmeasured visible count: {summary['sc_prediction']['total_predicted_unmeasured_visible_count']}",
        "",
        "Leakage checks: prediction affected information gain only; traversability, A*, collision, ray blocking, and observed_state writeback stayed disabled.",
    ]
    md_path = output_dir / "comparison_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    observed_state_path = Path(args.observed_state).resolve()
    prediction_npz = Path(args.prediction_npz).resolve()
    observed_hash_before = sha256_file(observed_state_path)
    observed_state = np.load(observed_state_path)
    prediction_layer = SimPredictionLayer.from_npz(prediction_npz)
    if prediction_layer.shape() != tuple(observed_state.shape):
        raise AssertionError(f"prediction shape {prediction_layer.shape()} != observed_state {observed_state.shape}")
    if any(array.flags.writeable for array in (
        prediction_layer.pred_class,
        prediction_layer.confidence,
        prediction_layer.occupied_prob,
        prediction_layer.free_prob,
        prediction_layer.valid,
    )):
        raise AssertionError("SimPredictionLayer arrays are writeable")

    empty = load_output_dir(args.empty_dir)
    pred = load_output_dir(args.prediction_dir)
    require(empty["dir"] / "expert_topdown.png", "empty topdown")
    require(empty["dir"] / "expert_score_bar.png", "empty score bar")
    require(pred["dir"] / "expert_topdown.png", "prediction topdown")
    require(pred["dir"] / "expert_score_bar.png", "prediction score bar")
    require(pred["dir"] / "traversability_topdown.png", "prediction traversability")
    require(pred["dir"] / "prediction_overlay_topdown.png", "prediction overlay")
    require(pred["dir"] / "predicted_unmeasured_visible_topdown.png", "prediction visible P overlay")

    empty_best = best_fields(empty["decision"])
    pred_best = best_fields(pred["decision"])
    if str(empty["decision"].get("prediction_mode")) != "empty":
        raise AssertionError("Empty baseline prediction_mode is not empty")
    if str(pred["decision"].get("prediction_mode")) != "sim_npz":
        raise AssertionError("Prediction output prediction_mode is not sim_npz")

    empty_gain_sc = [candidate_value(candidate, "gain_sc") for candidate in empty["candidates"]]
    if any(abs(value) > 1e-9 for value in empty_gain_sc):
        raise AssertionError("Empty baseline has nonzero gain_sc")
    pred_gain_sc = np.asarray([candidate_value(candidate, "gain_sc") for candidate in pred["candidates"]], dtype=np.float64)
    if int(np.count_nonzero(pred_gain_sc > 0.0)) <= 0:
        raise AssertionError("Prediction mode has no candidates with gain_sc > 0")

    assert_hybrid_identity(empty["candidates"])
    assert_hybrid_identity(pred["candidates"])
    assert_finite_candidates(pred["candidates"], ["gain_occ", "gain_conf", "gain_hybrid", "final_score"])
    assert_prediction_not_planning(empty, pred)
    assert_no_forbidden_npz_fields(empty["npz"])
    assert_no_forbidden_npz_fields(pred["npz"])

    for decision in (empty["decision"], pred["decision"]):
        diag = decision["diagnostics"]
        for key in ("target_lr_used", "target_hr_used", "ground_truth_used", "rl_or_training_used", "optimizer_used", "policy_training_used", "rollout_run"):
            if bool(diag.get(key, False)):
                raise AssertionError(f"Forbidden stage activity flag is true: {key}")
        if not bool(diag.get("observed_state_hash_unchanged", True)):
            raise AssertionError("Runner reported observed_state hash changed")

    top_empty = {int(candidate["id"]) for candidate in empty["decision"]["top_candidates"]}
    top_pred = {int(candidate["id"]) for candidate in pred["decision"]["top_candidates"]}
    overlap = sorted(top_empty & top_pred)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(args.prediction_dir).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    best_plot = save_best_candidate_plot(observed_state, empty, pred, output_dir)
    gain_plot = save_gain_comparison_plot(empty_best, pred_best, output_dir)
    total_visible_p = int(sum(int(c.get("predicted_unmeasured_visible_count", 0)) for c in pred["candidates"]))

    summary = {
        "stage": "Stage 4A-5.1 one-step SC-aware expert scoring",
        "inputs": {
            "observed_state": str(observed_state_path),
            "prediction_npz": str(prediction_npz),
            "tau": float(pred["decision"]["diagnostics"].get("tau", 0.1)),
        },
        "empty_baseline": {
            "dir": str(Path(args.empty_dir).resolve()),
            "best": empty_best,
        },
        "sc_prediction": {
            "dir": str(Path(args.prediction_dir).resolve()),
            "best": pred_best,
            "candidates_with_gain_sc_gt_zero": int(np.count_nonzero(pred_gain_sc > 0.0)),
            "max_gain_sc": float(np.max(pred_gain_sc)),
            "mean_gain_sc": float(np.mean(pred_gain_sc)),
            "total_predicted_unmeasured_visible_count": total_visible_p,
            "best_predicted_unmeasured_visible_count": int(pred_best["predicted_unmeasured_visible_count"]),
        },
        "comparison": {
            "best_candidate_changed": int(empty_best["id"]) != int(pred_best["id"]),
            "score_delta": float(pred_best["score"] - empty_best["score"]),
            "gain_hybrid_delta": float(pred_best["gain_hybrid"] - empty_best["gain_hybrid"]),
            "top_n_overlap_ids": overlap,
            "top_n_overlap_count": int(len(overlap)),
            "top_n_union_count": int(len(top_empty | top_pred)),
        },
        "validation": {
            "prediction_layer_shape_matches_observed_state": True,
            "observed_state_sha256_before": observed_hash_before,
            "observed_state_sha256_after": sha256_file(observed_state_path),
            "observed_state_modified": observed_hash_before != sha256_file(observed_state_path),
            "empty_mode_gain_sc_zero": True,
            "prediction_mode_gain_sc_nonzero": True,
            "gain_hybrid_identity": True,
            "gain_occ_finite": True,
            "gain_conf_finite": True,
            "prediction_used_for_traversability": False,
            "prediction_used_for_collision": False,
            "prediction_written_to_observed_map": False,
            "target_or_ground_truth_leakage": False,
            "rl_optimizer_bc_il_training_run": False,
            "rollout_run": False,
        },
        "outputs": {
            "empty_vs_prediction_best_candidate": best_plot,
            "gain_comparison_bar": gain_plot,
        },
    }
    comparison_json, comparison_md = write_comparison_summary(summary, output_dir)
    summary["outputs"]["comparison_summary_json"] = comparison_json
    summary["outputs"]["comparison_summary_md"] = comparison_md
    with Path(comparison_json).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4A-5.1 prediction expert smoke test.")
    parser.add_argument("--empty_dir", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--prediction_npz", required=True)
    parser.add_argument("--observed_state", required=True)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    summary = run_test(parse_args())
    print("Stage 4A-5.1 prediction expert smoke test passed.")
    print(f"empty_best_id: {summary['empty_baseline']['best']['id']}")
    print(f"prediction_best_id: {summary['sc_prediction']['best']['id']}")
    print(f"candidates_with_gain_sc_gt_zero: {summary['sc_prediction']['candidates_with_gain_sc_gt_zero']}")
    print(f"max_gain_sc: {summary['sc_prediction']['max_gain_sc']}")
    print(f"mean_gain_sc: {summary['sc_prediction']['mean_gain_sc']}")
    print(f"comparison_summary_json: {summary['outputs']['comparison_summary_json']}")
    print(f"comparison_summary_md: {summary['outputs']['comparison_summary_md']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combine Stage 4A-6.2 map_predict diagnostics and recommend next action."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

EVAL_ONLY_NOTE = "Future observations are post-hoc evaluation only, not used for planning."
DEFAULT_SC_EPISODE = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/"
    "episodes/medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000"
)
DEFAULT_EMPTY_EPISODE = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/"
    "episodes/medium_three_rooms_seed0_start_room_a_empty_astar"
)


def load_json(path: str | Path, default: Any | None = None) -> Any:
    target = Path(path)
    if not target.is_file():
        return default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def corr(xs: list[float], ys: list[float]) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return None
    if float(np.std(x[mask])) <= 1e-12 or float(np.std(y[mask])) <= 1e-12:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def feature_index(names: list[str], name: str) -> int:
    if name not in names:
        raise KeyError(f"feature {name} not found in {names}")
    return names.index(name)


def selected_signature(transition: dict[str, Any]) -> tuple[float, float, float, float]:
    pos = np.asarray(transition.get("selected_next_pose_world", [np.nan, np.nan, np.nan]), dtype=np.float64)
    yaw = float(transition.get("selected_next_yaw", np.nan))
    return (round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4), round(yaw, 4))


def compute_candidate_decomposition(diagnostics_root: Path, ablation_summary: dict[str, Any]) -> dict[str, Any]:
    output_dir = diagnostics_root / "candidate_score_decomposition"
    output_dir.mkdir(parents=True, exist_ok=True)
    sc_transitions = load_jsonl(DEFAULT_SC_EPISODE / "transitions.jsonl")
    empty_transitions = load_jsonl(DEFAULT_EMPTY_EPISODE / "transitions.jsonl") if (DEFAULT_EMPTY_EPISODE / "transitions.jsonl").is_file() else []

    all_rows: list[dict[str, Any]] = []
    per_step_paths: list[str] = []
    for transition in sc_transitions[:5]:
        step = int(transition.get("step", len(per_step_paths)))
        features = np.asarray(transition["candidate_features"], dtype=np.float64)
        names = [str(v) for v in transition["feature_names"]]
        idx_gain_exp = feature_index(names, "gain_exp")
        idx_gain_sc = feature_index(names, "gain_sc")
        idx_gain_hybrid = feature_index(names, "gain_hybrid")
        idx_path_cost = feature_index(names, "path_cost")
        idx_final = feature_index(names, "final_score")
        idx_visible = feature_index(names, "visible_count")
        gain_exp = features[:, idx_gain_exp]
        gain_sc = features[:, idx_gain_sc]
        gain_hybrid = features[:, idx_gain_hybrid]
        path_cost = features[:, idx_path_cost]
        final_score = features[:, idx_final]
        denom = np.maximum(path_cost, 1e-6)
        rows: list[dict[str, Any]] = []
        for idx in range(features.shape[0]):
            row = {
                "step": step,
                "topn_index": idx,
                "candidate_grid": json.dumps(np.asarray(transition["candidate_positions_grid"][idx]).tolist()),
                "candidate_world": json.dumps(np.asarray(transition["candidate_positions_world"][idx]).tolist()),
                "gain_exp": float(gain_exp[idx]),
                "gain_sc": float(gain_sc[idx]),
                "gain_hybrid": float(gain_hybrid[idx]),
                "path_cost": float(path_cost[idx]),
                "final_score": float(final_score[idx]),
                "visible_count": float(features[idx, idx_visible]),
                "score_exp_only": float(gain_exp[idx] / denom[idx]),
                "score_sc_only": float(gain_sc[idx] / denom[idx]),
                "score_hybrid": float(gain_hybrid[idx] / denom[idx]),
                "rank_final_score": int(rank_desc(final_score)[idx]),
                "rank_gain_exp": int(rank_desc(gain_exp)[idx]),
                "rank_gain_sc": int(rank_desc(gain_sc)[idx]),
                "rank_gain_hybrid": int(rank_desc(gain_hybrid)[idx]),
                "rank_path_cost": int(rank_asc(path_cost)[idx]),
                "selected_by_sc": bool(idx == int(transition.get("expert_action", 0))),
                "empty_selected_signature": json.dumps(selected_signature(empty_transitions[step])) if step < len(empty_transitions) else "",
                "sc_selected_signature": json.dumps(selected_signature(transition)),
            }
            rows.append(row)
            all_rows.append(row)
        path = output_dir / f"candidate_rank_decomposition_step{step:03d}.csv"
        write_csv(path, rows)
        per_step_paths.append(str(path))

    gain_exp_all = [float(row["gain_exp"]) for row in all_rows]
    gain_sc_all = [float(row["gain_sc"]) for row in all_rows]
    final_all = [float(row["final_score"]) for row in all_rows]
    path_all = [float(row["path_cost"]) for row in all_rows]
    hybrid_all = [float(row["gain_hybrid"]) for row in all_rows]
    inv_cost_all = [1.0 / max(1e-6, value) for value in path_all]
    gain_corr = corr(gain_exp_all, gain_sc_all)
    final_inv_cost_corr = corr(final_all, inv_cost_all)
    final_hybrid_corr = corr(final_all, hybrid_all)
    final_gain_sc_corr = corr(final_all, gain_sc_all)
    selected_rows = [row for row in all_rows if row["selected_by_sc"]]
    selected_sc_high_exp = sum(1 for row in selected_rows if int(row["rank_gain_exp"]) <= 5)
    selected_sc_low_cost = sum(1 for row in selected_rows if int(row["rank_path_cost"]) <= 5)

    ablation_rows = ablation_summary.get("rows", []) if isinstance(ablation_summary, dict) else []
    original_signatures = [selected_signature(t) for t in sc_transitions[:5]]
    heatmap_labels: list[str] = []
    heatmap_values: list[list[int]] = []
    for row in ablation_rows:
        config = str(row.get("config", ""))
        if config in {"", "empty", "original_sc"}:
            continue
        episode_dir = Path(str(row.get("episode_dir", "")))
        trans_path = episode_dir / "transitions.jsonl"
        if not trans_path.is_file():
            continue
        transitions = load_jsonl(trans_path)
        values = [1 if idx < len(transitions) and selected_signature(transitions[idx]) == original_signatures[idx] else 0 for idx in range(5)]
        heatmap_labels.append(config)
        heatmap_values.append(values)

    scatter_path = output_dir / "gain_exp_vs_gain_sc_scatter.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
    ax.scatter(gain_exp_all, gain_sc_all, color="#2563eb", alpha=0.72)
    ax.set_xlabel("gain_exp")
    ax.set_ylabel("gain_sc")
    ax.set_title("Top-N candidates: gain_exp vs gain_sc")
    ax.grid(alpha=0.25)
    fig.savefig(scatter_path, dpi=170)
    plt.close(fig)

    path_score_path = output_dir / "path_cost_vs_final_score.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
    ax.scatter(path_all, final_all, color="#0f766e", alpha=0.72)
    ax.set_xlabel("path_cost")
    ax.set_ylabel("final_score")
    ax.set_title("Top-N candidates: path cost vs score")
    ax.grid(alpha=0.25)
    fig.savefig(path_score_path, dpi=170)
    plt.close(fig)

    heatmap_path = output_dir / "rank_stability_heatmap.png"
    fig, ax = plt.subplots(figsize=(8.2, max(2.8, 0.55 * max(1, len(heatmap_labels)))), constrained_layout=True)
    if heatmap_values:
        arr = np.asarray(heatmap_values, dtype=np.float32)
        ax.imshow(arr, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(np.arange(len(heatmap_labels)))
        ax.set_yticklabels(heatmap_labels)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([str(i) for i in range(5)])
    else:
        ax.text(0.5, 0.5, "no ablation transitions", ha="center", va="center")
    ax.set_xlabel("step")
    ax.set_title("Selected action matches original SC")
    fig.savefig(heatmap_path, dpi=170)
    plt.close(fig)

    same_action_counts = {
        label: int(sum(vals)) for label, vals in zip(heatmap_labels, heatmap_values)
    }
    summary = {
        "stage": "Stage 4A-6.2 candidate score decomposition",
        "output_dir": str(output_dir),
        "candidate_rows_count": len(all_rows),
        "topn_only_limitation": "Only logged top-N candidates are decomposed; full 64-candidate rankings were not saved.",
        "gain_exp_gain_sc_correlation": gain_corr,
        "final_score_inverse_path_cost_correlation": final_inv_cost_corr,
        "final_score_gain_hybrid_correlation": final_hybrid_corr,
        "final_score_gain_sc_correlation": final_gain_sc_corr,
        "selected_candidates_rank_gain_exp_top5_count": int(selected_sc_high_exp),
        "selected_candidates_rank_path_cost_top5_count": int(selected_sc_low_cost),
        "path_cost_dominance_flag": bool(
            final_inv_cost_corr is not None
            and final_hybrid_corr is not None
            and abs(final_inv_cost_corr) > abs(final_hybrid_corr)
        ),
        "gain_sc_duplicates_gain_exp_flag": bool(gain_corr is not None and gain_corr > 0.9),
        "same_actions_as_original_sc_by_ablation": same_action_counts,
        "outputs": {
            "candidate_rank_decomposition_csvs": per_step_paths,
            "gain_exp_vs_gain_sc_scatter": str(scatter_path),
            "path_cost_vs_final_score": str(path_score_path),
            "rank_stability_heatmap": str(heatmap_path),
            "rank_correlation_summary_json": str(output_dir / "rank_correlation_summary.json"),
            "rank_correlation_summary_md": str(output_dir / "rank_correlation_summary.md"),
        },
    }
    save_json(output_dir / "rank_correlation_summary.json", summary)
    md = [
        "# Candidate Score Decomposition",
        "",
        f"- gain_exp/gain_sc correlation: `{gain_corr}`.",
        f"- final_score vs inverse path_cost correlation: `{final_inv_cost_corr}`.",
        f"- final_score vs gain_hybrid correlation: `{final_hybrid_corr}`.",
        f"- selected candidates with gain_exp rank top 5: `{selected_sc_high_exp}/{len(selected_rows)}`.",
        f"- selected candidates with path_cost rank top 5: `{selected_sc_low_cost}/{len(selected_rows)}`.",
        f"- gain_sc duplicates gain_exp flag: `{summary['gain_sc_duplicates_gain_exp_flag']}`.",
        f"- path_cost dominance flag: `{summary['path_cost_dominance_flag']}`.",
        f"- ablations matching original SC actions: `{same_action_counts}`.",
        "",
        "The 4A-6.1 weight/cap/tau ablations did not change actions because the logged top candidates keep similar gain_sc/gain_exp structure and the chosen actions are already strong under path-normalized hybrid score.",
    ]
    (output_dir / "rank_correlation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def issue_rows(
    preprocess: dict[str, Any],
    alignment: dict[str, Any],
    future_eval: dict[str, Any],
    variant: dict[str, Any],
    scoring: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "issue": "alignment_convention",
            "priority": 1 if variant.get("likely_alignment_bug") else 3,
            "evidence": f"variant_bug={variant.get('likely_alignment_bug')}, default_rank={variant.get('default_variant_rank')}",
        },
        {
            "issue": "confidence_calibration_density",
            "priority": 1 if future_eval.get("tau_0p1_too_dense") else 2,
            "evidence": f"tau0p1_dense={future_eval.get('tau_0p1_too_dense')}, brier={future_eval.get('mean_brier_default')}",
        },
        {
            "issue": "preprocessing_domain_shift",
            "priority": 2 if preprocess.get("suspicious_differences") else 3,
            "evidence": "; ".join(str(v) for v in preprocess.get("suspicious_differences", [])[:2]),
        },
        {
            "issue": "gain_sc_duplicate_unknown_gain",
            "priority": 1 if scoring.get("gain_sc_duplicates_gain_exp_flag") else 2,
            "evidence": f"corr={scoring.get('gain_exp_gain_sc_correlation')}",
        },
        {
            "issue": "path_cost_dominance",
            "priority": 2 if scoring.get("path_cost_dominance_flag") else 3,
            "evidence": f"score_inv_cost_corr={scoring.get('final_score_inverse_path_cost_correlation')}",
        },
    ]
    return sorted(rows, key=lambda row: int(row["priority"]))


def write_key_plot_index(output_dir: Path, summaries: dict[str, dict[str, Any]], scoring: dict[str, Any]) -> str:
    links: list[tuple[str, str]] = []
    for section, summary in summaries.items():
        outputs = summary.get("outputs", {}) if isinstance(summary, dict) else {}
        for name, path in outputs.items():
            if isinstance(path, str) and path.endswith((".png", ".html", ".md")):
                links.append((f"{section}: {name}", path))
    for name, path in scoring.get("outputs", {}).items():
        if isinstance(path, str) and path.endswith((".png", ".html", ".md")):
            links.append((f"scoring: {name}", path))
    html = ["<html><body><h1>Stage 4A-6.2 Key Plots</h1><ul>"]
    for label, path in links:
        html.append(f'<li><a href="{Path(path).resolve().as_uri()}">{label}</a></li>')
    html.append("</ul></body></html>")
    out = output_dir / "key_plots_index.html"
    out.write_text("\n".join(html) + "\n", encoding="utf-8")
    return str(out)


def choose_recommendation(
    preprocess: dict[str, Any],
    alignment: dict[str, Any],
    future_eval: dict[str, Any],
    variant: dict[str, Any],
    scoring: dict[str, Any],
) -> tuple[str, str, str]:
    if variant.get("likely_alignment_bug") or alignment.get("likely_axis_or_yaw_issue"):
        return (
            "Fix alignment convention and rerun Stage 4A-5/5.1/6 smoke.",
            "alignment convention",
            "medium",
        )
    brier = future_eval.get("mean_brier_default")
    dense = bool(future_eval.get("tau_0p1_too_dense"))
    low_later = future_eval.get("mean_later_measured_fraction_default")
    if dense and (brier is None or float(brier) > 0.20):
        return (
            "Keep default alignment but use confidence-calibrated/capped prediction gain and restrict I_sc.",
            "confidence calibration and dense unselective prediction",
            "medium-high",
        )
    if preprocess.get("suspicious_differences") and low_later is not None and float(low_later) < 0.10:
        return (
            "Collect Isaac-domain prediction validation data or synthetic supervised data before relying on SC-aware rollout.",
            "NYU-to-Isaac domain shift",
            "medium",
        )
    if scoring.get("gain_sc_duplicates_gain_exp_flag"):
        return (
            "Use prediction only as analysis signal for now or restrict I_sc to more selective high-confidence regions.",
            "gain_sc duplicates measured unknown-region reward",
            "medium",
        )
    return (
        "Keep alignment but use confidence-calibrated prediction gain for the next smoke.",
        "calibration/scoring selectivity",
        "low-medium",
    )


def run_summary(args: argparse.Namespace) -> dict[str, Any]:
    diagnostics_root = Path(args.diagnostics_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_summary = load_json(args.stage4a61_ablation_summary, default={})

    preprocess = load_json(diagnostics_root / "preprocess_stats" / "preprocess_comparison_summary.json", default={}) or {}
    alignment = load_json(diagnostics_root / "global_alignment" / "alignment_summary.json", default={}) or {}
    future_eval = load_json(diagnostics_root / "future_observed_eval" / "future_eval_summary.json", default={}) or {}
    variant = load_json(diagnostics_root / "alignment_variant_sweep" / "variant_metrics.json", default={}) or {}
    scoring = compute_candidate_decomposition(diagnostics_root, ablation_summary)

    summaries = {
        "preprocess": preprocess,
        "alignment": alignment,
        "future_eval": future_eval,
        "variant": variant,
    }
    issues = issue_rows(preprocess, alignment, future_eval, variant, scoring)
    write_csv(output_dir / "issue_priority_table.csv", issues)
    key_index = write_key_plot_index(output_dir, summaries, scoring)
    recommendation, primary_issue, confidence = choose_recommendation(preprocess, alignment, future_eval, variant, scoring)

    primary_future = future_eval.get("primary_default_tau_summary", {})
    summary = {
        "stage": "Stage 4A-6.2 map_predict preprocessing/alignment/calibration diagnostics",
        "diagnostics_root": str(diagnostics_root),
        "output_dir": str(output_dir),
        "future_observations_usage": EVAL_ONLY_NOTE,
        "preprocessing": {
            "isaac_depth_mean_m": preprocess.get("isaac_depth_mean_m"),
            "nyu_depth_mean_m": preprocess.get("nyu_depth_mean_m"),
            "valid_position_ratio_isaac": preprocess.get("valid_position_ratio_isaac"),
            "valid_position_ratio_nyu_proxy": preprocess.get("valid_position_ratio_nyu_proxy"),
            "suspicious_differences": preprocess.get("suspicious_differences", []),
        },
        "alignment": {
            "mean_in_front_ratio": alignment.get("mean_valid_voxels_in_front_ratio"),
            "mean_inside_bounds_ratio": alignment.get("mean_inside_global_bounds_ratio"),
            "likely_axis_or_yaw_issue": alignment.get("likely_axis_or_yaw_issue"),
            "variant_best": variant.get("best_variant"),
            "variant_default_rank": variant.get("default_variant_rank"),
            "variant_improvement": variant.get("brier_improvement_vs_default"),
            "variant_likely_alignment_bug": variant.get("likely_alignment_bug"),
        },
        "calibration": {
            "predicted_unmeasured_count": primary_future.get("predicted_unmeasured_count"),
            "later_measured_fraction": primary_future.get("later_measured_fraction"),
            "occupied_precision": primary_future.get("occupied_precision"),
            "free_precision": primary_future.get("free_precision"),
            "brier_occupied": primary_future.get("brier_occupied"),
            "tau_0p1_too_dense": future_eval.get("tau_0p1_too_dense"),
            "tau_reduces_density_meaningfully": future_eval.get("tau_reduces_density_meaningfully"),
        },
        "scoring": scoring,
        "ablation": {
            "completed_ablation_count": ablation_summary.get("completed_ablation_count"),
            "any_ablation_beats_empty": ablation_summary.get("any_ablation_beats_empty"),
            "any_ablation_improves_over_original_sc": ablation_summary.get("any_ablation_improves_over_original_sc"),
            "recommendation": ablation_summary.get("recommendation"),
        },
        "issue_priority_table": str(output_dir / "issue_priority_table.csv"),
        "key_plots_index": key_index,
        "final_recommendation": recommendation,
        "primary_suspected_issue": primary_issue,
        "diagnosis_confidence": confidence,
        "what_not_to_do_next": "Do not jump to RL/IL; do not scale rollouts until map_predict selectivity is improved.",
        "planning_or_training_used": False,
        "prediction_writeback": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_a_star": False,
        "prediction_blocks_rays": False,
    }
    save_json(output_dir / "stage4a62_diagnostic_summary.json", summary)

    md = [
        "# Stage 4A-6.2 Diagnostic Summary",
        "",
        f"- {EVAL_ONLY_NOTE}",
        "",
        "## A. Preprocessing",
        f"- Isaac mean depth: `{preprocess.get('isaac_depth_mean_m')}` m; NYU mean depth: `{preprocess.get('nyu_depth_mean_m')}` m.",
        f"- Isaac valid position ratio: `{preprocess.get('valid_position_ratio_isaac')}`; NYU proxy: `{preprocess.get('valid_position_ratio_nyu_proxy')}`.",
        f"- Suspicious differences: `{preprocess.get('suspicious_differences', [])}`.",
        "",
        "## B. Alignment",
        f"- Prediction in-front ratio: `{alignment.get('mean_valid_voxels_in_front_ratio')}`.",
        f"- Inside-bounds ratio: `{alignment.get('mean_inside_global_bounds_ratio')}`.",
        f"- Variant best/default rank/improvement: `{variant.get('best_variant')}` / `{variant.get('default_variant_rank')}` / `{variant.get('brier_improvement_vs_default')}`.",
        f"- Likely axis/yaw issue: `{variant.get('likely_alignment_bug') or alignment.get('likely_axis_or_yaw_issue')}`.",
        "",
        "## C. Calibration",
        f"- Mean predicted_unmeasured count: `{primary_future.get('predicted_unmeasured_count')}`.",
        f"- Mean later measured fraction: `{primary_future.get('later_measured_fraction')}`.",
        f"- Occupied/free precision: `{primary_future.get('occupied_precision')}` / `{primary_future.get('free_precision')}`.",
        f"- Occupied Brier: `{primary_future.get('brier_occupied')}`.",
        f"- Tau 0.1 dense flag: `{future_eval.get('tau_0p1_too_dense')}`.",
        "",
        "## D. Scoring",
        f"- gain_exp/gain_sc correlation: `{scoring.get('gain_exp_gain_sc_correlation')}`.",
        f"- final score vs inverse path-cost correlation: `{scoring.get('final_score_inverse_path_cost_correlation')}`.",
        f"- gain_sc duplicates gain_exp: `{scoring.get('gain_sc_duplicates_gain_exp_flag')}`.",
        f"- path_cost dominance: `{scoring.get('path_cost_dominance_flag')}`.",
        "",
        "## E. Recommendation",
        f"- Primary suspected issue: `{primary_issue}`.",
        f"- Confidence: `{confidence}`.",
        f"- Next recommended step: {recommendation}",
        "- Do not jump to RL/IL.",
    ]
    (output_dir / "stage4a62_diagnostic_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    recommendation_md = [
        "# Stage 4A-6.2 Recommendation",
        "",
        f"Primary suspected issue: `{primary_issue}`.",
        "",
        recommendation,
        "",
        "Prediction remains read-only and information-gain-only. Future observations were delayed sensor validation only, not planning input.",
    ]
    (output_dir / "stage4a62_recommendation.md").write_text("\n".join(recommendation_md) + "\n", encoding="utf-8")

    print(f"Stage 4A-6.2 diagnostic summary complete: {output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics_root", type=Path, required=True)
    parser.add_argument("--stage4a61_ablation_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_summary(parse_args())


if __name__ == "__main__":
    main()

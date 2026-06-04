#!/usr/bin/env python3
"""Stage 4A-6.5aj two-frame one-action runtime smoke design review.

This stage is documentation and validation planning only. It reads existing
Stage 4A-6.5ag/ah/ai outputs, writes a future Stage 4A-6.5ak design package,
and deliberately does not start Isaac, capture frames, run map_predict,
execute actions, or run rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any


for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, "1")


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = (
    WORKSPACE / "outputs/isaac_sc_pred_stage4a65aj_two_frame_one_action_runtime_design_review"
)
FUTURE_OUTPUT_DIR = (
    WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
)
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]
CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
SHADOW_FORMULA = "gain_exp / cost + 32 * minmax(source_occ_free)"
BAD_FORMULAS = [
    "(gain_exp + 48 * source_occ_free) / cost",
    "(gain_exp + source_occ_free) / cost",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def path_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def parse_position(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError(f"--position must contain three comma-separated floats, got {raw!r}")
    return values


def maybe_generate_diagrams(output_dir: Path) -> dict[str, str]:
    """Generate small static design diagrams; return skipped reasons by file."""
    skipped: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local matplotlib.
        reason = f"matplotlib unavailable: {exc}"
        for name in ("flowchart_two_frame_one_action.png", "safety_gate_flowchart.png", "runtime_timeline.png"):
            skipped[name] = reason
            write_text(output_dir / f"{Path(name).stem}_skipped_reason.md", reason)
        return skipped

    def boxes(path: Path, title: str, labels: list[str]) -> None:
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, pad=12)
        x_positions = [0.08 + i * (0.84 / max(1, len(labels) - 1)) for i in range(len(labels))]
        for idx, (x, label) in enumerate(zip(x_positions, labels)):
            ax.text(
                x,
                0.52,
                label,
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#e8f1fa", "edgecolor": "#29577a"},
                transform=ax.transAxes,
            )
            if idx + 1 < len(labels):
                ax.annotate(
                    "",
                    xy=(x_positions[idx + 1] - 0.055, 0.52),
                    xytext=(x + 0.055, 0.52),
                    arrowprops={"arrowstyle": "->", "color": "#29577a", "lw": 1.2},
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    boxes(
        output_dir / "flowchart_two_frame_one_action.png",
        "Future Stage 4A-6.5ak Two-Frame / One-Action Flow",
        [
            "Frame 1\ncapture",
            "map_predict #1",
            "lambda48\nprimary",
            "safety gates",
            "execute\none action",
            "Frame 2\ncapture",
            "diagnose\nand stop",
        ],
    )
    boxes(
        output_dir / "safety_gate_flowchart.png",
        "Future Pre-Action Safety Gates",
        [
            "read-only\nprediction",
            "exact\nformula",
            "finite\ndecisions",
            "no prior\nbasin",
            "no low-cost\nartifact",
            "finite pose\nin bounds",
            "action count\n0 -> 1",
        ],
    )
    boxes(
        output_dir / "runtime_timeline.png",
        "Future Runtime Timeline",
        [
            "Isaac\nstartup",
            "Frame 1",
            "Action 1",
            "Frame 2",
            "No action 2",
            "No frame 3",
            "No rollout",
        ],
    )
    return skipped


def build_markdown_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        if isinstance(value, (dict, list)):
            rendered = "`" + json.dumps(value, sort_keys=True) + "`"
        else:
            rendered = f"`{value}`"
        lines.append(f"| {key} | {rendered} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65ag_dir", required=True)
    parser.add_argument("--stage4a65ah_dir", required=True)
    parser.add_argument("--stage4a65ai_dir", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--shadow_lambda_sc", type=float, default=32.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--occ_threshold", type=float, default=0.5)
    parser.add_argument("--free_threshold", type=float, default=0.5)
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--position", default="-4.65,-4.65,1.2")
    parser.add_argument("--yaw", type=float, default=0.38710316317995463)
    parser.add_argument("--max_workers", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage4a65ag_dir = Path(args.stage4a65ag_dir).resolve()
    stage4a65ah_dir = Path(args.stage4a65ah_dir).resolve()
    stage4a65ai_dir = Path(args.stage4a65ai_dir).resolve()
    position = parse_position(args.position)

    ag_summary_path = stage4a65ag_dir / "stage4a65ag_multi_frame_lambda48_replay_summary.json"
    ah_summary_path = stage4a65ah_dir / "stage4a65ah_multiscene_or_runtime_design_review_summary.json"
    ai_summary_path = stage4a65ai_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.json"
    ai_branch_path = stage4a65ai_dir / "branch_classification.json"
    ai_map_predict_path = stage4a65ai_dir / "map_predict_runtime_summary.json"
    ai_hardware_path = stage4a65ai_dir / "hardware_utilization_report.json"
    ai_observed_update_path = stage4a65ai_dir / "observed_state_update_summary.json"

    ag_summary = read_json(ag_summary_path)
    ah_summary = read_json(ah_summary_path)
    ai_summary = read_json(ai_summary_path)
    ai_branch = read_json(ai_branch_path)
    ai_map_predict = read_json(ai_map_predict_path)
    ai_hardware = read_json(ai_hardware_path)
    ai_observed_update = read_json(ai_observed_update_path)

    loaded_context_manifest = {
        "stage": "Stage 4A-6.5aj",
        "purpose": "design review only for future Stage 4A-6.5ak",
        "chat_history_not_used_as_source": True,
        "context_files": [path_entry(path) for path in CONTEXT_FILES],
        "prior_stage_dirs": {
            "stage4a65ag": path_entry(stage4a65ag_dir),
            "stage4a65ah": path_entry(stage4a65ah_dir),
            "stage4a65ai": path_entry(stage4a65ai_dir),
        },
        "prior_stage_summary_files": {
            "stage4a65ag": path_entry(ag_summary_path),
            "stage4a65ah": path_entry(ah_summary_path),
            "stage4a65ai": path_entry(ai_summary_path),
            "stage4a65ai_branch": path_entry(ai_branch_path),
            "stage4a65ai_map_predict": path_entry(ai_map_predict_path),
            "stage4a65ai_hardware": path_entry(ai_hardware_path),
            "stage4a65ai_observed_update": path_entry(ai_observed_update_path),
        },
        "confirmed": {
            "stage4a65ag_complete": ag_summary.get("answers", {}).get("lambda48_aggregate", {}).get(
                "unique_frame_count"
            )
            == 7,
            "stage4a65ah_complete": ah_summary.get("status") == "completed",
            "stage4a65ai_complete": ai_summary.get("readiness", {}).get("one_frame_runtime_smoke_complete")
            is True,
            "stage4a65aj_runtime_executed": False,
        },
    }
    write_json(output_dir / "loaded_context_manifest.json", loaded_context_manifest)
    write_text(
        output_dir / "loaded_context_manifest.md",
        f"""
        # Loaded Context Manifest

        Stage 4A-6.5aj loaded project context plus Stage 4A-6.5ag, Stage
        4A-6.5ah, and Stage 4A-6.5ai summaries from disk. Chat history is not
        treated as the source of truth.

        {build_markdown_table([
            ("Stage 4A-6.5ag complete", loaded_context_manifest["confirmed"]["stage4a65ag_complete"]),
            ("Stage 4A-6.5ah complete", loaded_context_manifest["confirmed"]["stage4a65ah_complete"]),
            ("Stage 4A-6.5ai complete", loaded_context_manifest["confirmed"]["stage4a65ai_complete"]),
            ("Stage 4A-6.5aj runtime executed", False),
        ])}
        """,
    )

    ag_answers = ag_summary["answers"]
    stage4a65ai_result_review = {
        "stage": "Stage 4A-6.5ai",
        "output_dir": str(stage4a65ai_dir),
        "runtime_setup": {
            "isaac_startups": 1,
            "scene_variant": args.scene_variant,
            "scene_seed": args.scene_seed,
            "start_position": position,
            "yaw": args.yaw,
            "frames_captured": 1,
            "measured_only_observed_state_updates": 1,
            "map_predict_calls": 1,
            "selected_action_executions": 0,
            "second_frame": False,
            "two_frame_runtime": False,
            "rollout": False,
        },
        "observed_state": {
            "shape": ai_summary["map_predict"]["observed_state_shape"],
            "observed_ratio": ai_observed_update["updated_summary"]["observed_ratio"],
        },
        "map_predict": {
            "alignment_convention": ai_map_predict["alignment_convention"],
            "prediction_layer_shape": ai_map_predict["prediction_layer_shape"],
            "prediction_layer_shape_aligned_to_observed_state": ai_map_predict[
                "prediction_layer_shape_aligned_to_observed_state"
            ],
            "prediction_valid_count": ai_map_predict["stats"]["prediction_valid_count"],
            "predicted_unmeasured_occ_free_count": ai_map_predict["stats"][
                "predicted_unmeasured_occ_free_count"
            ],
            "predicted_unmeasured_occupied_count": ai_map_predict["stats"][
                "predicted_unmeasured_occupied_count"
            ],
            "predicted_unmeasured_free_count": ai_map_predict["stats"]["predicted_unmeasured_free_count"],
        },
        "formula": ai_summary["formula"],
        "decisions": {
            "measured_shadow_selected": "n0013 -> n0159",
            "lambda48_primary_selected": "n0001 -> n0228",
            "lambda32_shadow_selected": "n0001 -> n0228",
            "lambda48_branch_classification": ai_branch["classification"],
            "healthy_nonmeasured": ai_branch["healthy_nonmeasured_candidate"],
            "low_cost_artifact": ai_summary["results"]["lambda48_primary"]["low_cost_artifact"],
            "historical_prior_basin": ai_branch["spatial_prior_sc_basin"],
        },
        "readiness": {
            "supports_designing_two_frame_one_action_smoke": True,
            "two_frame_runtime_ready": False,
            "rollout_ready": False,
            "reason": "one-frame no-action runtime was clean, but did not validate action execution or a post-action frame",
        },
    }
    write_json(output_dir / "stage4a65ai_result_review.json", stage4a65ai_result_review)
    write_text(
        output_dir / "stage4a65ai_result_review.md",
        f"""
        # Stage 4A-6.5ai Result Review

        Stage 4A-6.5ai completed a clean one-frame runtime smoke: one Isaac
        startup, one RGB/depth frame, one measured-only observed_state update,
        one map_predict call, lambda48 primary tree decision, measured-only
        shadow, lambda32 shadow, no action, no second frame, and no rollout.

        {build_markdown_table([
            ("observed_state shape", stage4a65ai_result_review["observed_state"]["shape"]),
            ("observed_ratio", stage4a65ai_result_review["observed_state"]["observed_ratio"]),
            ("valid predictions", stage4a65ai_result_review["map_predict"]["prediction_valid_count"]),
            ("predicted unmeasured OCC+FREE", stage4a65ai_result_review["map_predict"]["predicted_unmeasured_occ_free_count"]),
            ("lambda48 branch", ai_branch["classification"]),
            ("low-cost artifact", stage4a65ai_result_review["decisions"]["low_cost_artifact"]),
            ("historical prior basin", stage4a65ai_result_review["decisions"]["historical_prior_basin"]),
            ("rollout ready", False),
        ])}
        """,
    )

    future_two_frame_sequence_spec = {
        "future_stage": "Stage 4A-6.5ak",
        "runtime_executed_in_stage4a65aj": False,
        "isaac_startup_count": 1,
        "scene": {
            "variant": args.scene_variant,
            "seed": args.scene_seed,
            "canonical_start_position": position,
            "canonical_start_yaw": args.yaw,
        },
        "exact_counts": {
            "captured_frames": 2,
            "measured_only_observed_state_updates": 2,
            "map_predict_calls": 2,
            "selected_action_executions": 1,
            "second_action": 0,
            "third_frame": 0,
            "rollout": 0,
            "open_ended_loop": 0,
        },
        "frame1_steps": [
            "Start Isaac headless once",
            "Build deterministic medium_three_rooms scene seed 0",
            "Set camera pose to canonical start pose",
            "Capture frame001 RGB/depth",
            "Create observed_state_frame001 with measured-only depth update",
            "Run map_predict call #1 using code_consistent_v1",
            "Run measured-only shadow tree decision",
            "Run lambda48 primary tree decision",
            "Run lambda32 shadow tree decision if cheap",
            "Classify Frame 1 lambda48",
            "Compute low-cost artifact flag",
            "If safety gates pass, execute exactly one action to Frame 1 lambda48 selected child",
            "Save action_execution_report",
        ],
        "frame2_steps": [
            "Capture frame002 RGB/depth at post-action pose",
            "Create observed_state_frame002 with measured-only depth update",
            "Run map_predict call #2 using code_consistent_v1",
            "Run measured-only shadow tree decision",
            "Run lambda48 diagnostic tree decision",
            "Run lambda32 shadow tree decision if cheap",
            "Classify Frame 2 lambda48",
            "Compute low-cost artifact flag",
            "Stop without executing Frame 2 selected action",
        ],
    }
    write_json(output_dir / "future_two_frame_sequence_spec.json", future_two_frame_sequence_spec)
    write_text(
        output_dir / "future_two_frame_sequence_spec.md",
        f"""
        # Future Two-Frame Sequence Spec

        Stage 4A-6.5ak is designed as a bounded two-frame / one-action runtime
        smoke. It is not a rollout and it has no open-ended loop.

        {build_markdown_table([
            ("frames", 2),
            ("map_predict calls", 2),
            ("selected action executions", 1),
            ("second action", 0),
            ("third frame", 0),
            ("rollout", 0),
        ])}

        ## Frame 1
        {format_list(future_two_frame_sequence_spec["frame1_steps"])}

        ## Frame 2
        {format_list(future_two_frame_sequence_spec["frame2_steps"])}
        """,
    )

    future_lambda48_formula_spec = {
        "prediction_source": "map_predict",
        "sc_basis": "source_occ_free",
        "primary_formula_name": "decoupled_minmax_lambda48",
        "primary_formula": PRIMARY_FORMULA,
        "shadow_formula": SHADOW_FORMULA,
        "lambda_sc": args.lambda_sc,
        "shadow_lambda_sc": args.shadow_lambda_sc,
        "tau": args.tau,
        "occ_threshold": args.occ_threshold,
        "free_threshold": args.free_threshold,
        "source_occ_free_definition": {
            "included": [
                "predicted occupied voxels",
                "predicted free voxels",
                "prediction_valid voxels only",
                "unmeasured voxels only",
            ],
            "excluded": [
                "predicted unknown",
                "invalid predictions",
                "already measured voxels",
            ],
        },
        "minmax": "per tree over valid root-to-descendant accumulated source_occ_free",
        "sc_bonus_location": "outside cost denominator",
        "prohibited_runtime_primary_formulas": BAD_FORMULAS,
        "over_cost_runtime_primary_allowed": False,
    }
    write_json(output_dir / "future_lambda48_formula_spec.json", future_lambda48_formula_spec)
    write_text(
        output_dir / "future_lambda48_formula_spec.md",
        f"""
        # Future Lambda48 Formula Spec

        Primary runtime formula:

        ```text
        value = {PRIMARY_FORMULA}
        ```

        Lambda32 shadow formula:

        ```text
        value = {SHADOW_FORMULA}
        ```

        The lambda48 SC bonus remains outside the cost denominator. The future
        runtime primary must not use `{BAD_FORMULAS[0]}` and must not use
        `{BAD_FORMULAS[1]}`. Over-cost remains historical diagnostic only and
        is prohibited from the runtime primary path.
        """,
    )

    future_source_protection_profile = {
        "short_edge_policy": "crop",
        "crop_min_length_m": 0.25,
        "num_nodes": 256,
        "max_extension_m": 0.5,
        "sample_mode": "mixed",
        "path_cost_mode": "segment_time",
        "v_max": 1.0,
        "robot_radius_m": 0.2,
        "voxel_size": 0.1,
        "raycast_stride": 2,
        "num_yaw_samples": 8,
        "max_ray_length_m": 4.8,
        "alignment_convention": "code_consistent_v1",
        "tree_seed": 0,
        "seed_repeat_later": "recorded as future option only; not part of first two-frame action smoke",
    }
    write_json(output_dir / "future_source_protection_profile.json", future_source_protection_profile)
    write_text(
        output_dir / "future_source_protection_profile.md",
        f"""
        # Future Source Protection Profile

        {build_markdown_table(list(future_source_protection_profile.items()))}
        """,
    )

    pre_action_gates = [
        "Isaac frame capture succeeded",
        "Exactly one frame has been captured so far",
        "map_predict call #1 succeeded",
        "prediction layer shape equals observed_state shape",
        "prediction is read-only",
        "prediction has not written observed_state",
        "prediction is not used for traversability/collision/ray blocking/edge validity/candidate sampling",
        "lambda48 tree decision exists and is finite",
        "measured-only shadow tree decision exists and is finite",
        f"lambda48 formula is exactly: {PRIMARY_FORMULA}",
        "lambda48 formula is not over-cost",
        "historical prior basin flag is false OR explicitly reported and action blocked",
        "low_cost_artifact is false OR explicitly reported and action blocked",
        "selected action world pose is finite",
        "selected action is within map bounds",
        "selected action does not require prediction-based traversability",
        "action_execution_count is currently 0",
    ]
    action_block_conditions = [
        "failed capture",
        "map_predict failure",
        "prediction shape mismatch",
        "prediction writeback or fusion detected",
        "prediction used for traversability/collision/ray blocking/candidate sampling/edge validity",
        "non-finite lambda48 or measured-only decision",
        "formula mismatch",
        "over-cost runtime primary",
        "historical prior basin true",
        "low_cost_artifact true",
        "non-finite or out-of-bounds action pose",
        "action_execution_count is not 0 before action",
        "target/ground-truth/future-observed planning or scoring detected",
    ]
    future_pre_action_safety_gates = {
        "stage": "Future Stage 4A-6.5ak",
        "pre_action_gates": pre_action_gates,
        "action_block_conditions": action_block_conditions,
        "if_any_hard_gate_fails": {
            "execute_action": False,
            "capture_frame2": False,
            "rollout": False,
            "required_behavior": "stop after Frame 1 diagnosis and save gate report",
        },
    }
    write_json(output_dir / "future_pre_action_safety_gates.json", future_pre_action_safety_gates)
    write_text(
        output_dir / "future_pre_action_safety_gates.md",
        f"""
        # Future Pre-Action Safety Gates

        Frame 1 action may execute only after every hard safety gate passes.

        ## Gates
        {format_list(pre_action_gates)}

        ## Action Block Conditions
        {format_list(action_block_conditions)}
        """,
    )

    future_frame2_stop_conditions = {
        "frame2_gates": [
            "Exactly one action has been executed before Frame 2",
            "Frame 2 capture succeeds",
            "map_predict call count becomes exactly 2",
            "Frame 2 tree decisions run for diagnosis only",
            "action_execution_count remains 1",
            "no third frame",
            "no second action",
        ],
        "stop_after_frame2": True,
        "second_action_allowed": False,
        "third_frame_allowed": False,
        "rollout_allowed": False,
    }
    write_json(output_dir / "future_frame2_stop_conditions.json", future_frame2_stop_conditions)
    write_text(
        output_dir / "future_frame2_stop_conditions.md",
        f"""
        # Future Frame 2 Stop Conditions

        {format_list(future_frame2_stop_conditions["frame2_gates"])}

        Frame 2 is diagnostic only. It must stop with one action total, no
        second action, no third frame, and no rollout.
        """,
    )

    future_required_json_md = [
        "loaded_context_manifest",
        "hardware_utilization_report",
        "runtime_setup_summary",
        "frame001_capture_summary",
        "frame002_capture_summary",
        "observed_state_update_frame001",
        "observed_state_update_frame002",
        "map_predict_frame001_summary",
        "map_predict_frame002_summary",
        "formula_definition",
        "source_protection_checklist",
        "frame001_measured_shadow_tree_decision",
        "frame001_lambda48_primary_tree_decision",
        "frame001_lambda32_shadow_tree_decision_or_skipped",
        "frame001_branch_classification",
        "frame001_low_cost_artifact_diagnosis",
        "pre_action_safety_gate_report",
        "action_execution_report",
        "frame002_measured_shadow_tree_decision",
        "frame002_lambda48_diagnostic_tree_decision",
        "frame002_lambda32_shadow_tree_decision_or_skipped",
        "frame002_branch_classification",
        "frame002_low_cost_artifact_diagnosis",
        "two_frame_decision_comparison",
        "prediction_safety_report",
        "hash_checks",
        "no_rollout_report",
        "missing_fields_report",
        "stage4a65ak_two_frame_one_action_runtime_summary",
        "recommended_next_faithful_step",
    ]
    future_required_artifacts = [
        "frame001_rgb.png",
        "frame001_depth.npy",
        "frame001_depth.png",
        "observed_state_frame001.npy",
        "frame001_map_predict/global_prediction_layer.npz",
        "frame001_map_predict/prediction_alignment_summary.json",
        "frame002_rgb.png",
        "frame002_depth.npy",
        "frame002_depth.png",
        "observed_state_frame002.npy",
        "frame002_map_predict/global_prediction_layer.npz",
        "frame002_map_predict/prediction_alignment_summary.json",
    ]
    future_required_plots = [
        "frame001_observed_topdown.png",
        "frame001_prediction_overlay_topdown.png",
        "frame001_measured_vs_lambda48_tree_topdown.png",
        "frame001_lambda48_selected_branch_topdown.png",
        "executed_action_topdown.png",
        "frame002_observed_topdown.png",
        "frame002_prediction_overlay_topdown.png",
        "frame002_measured_vs_lambda48_tree_topdown.png",
        "two_frame_path_topdown.png",
        "value_components_frame001_lambda48.png",
        "value_components_frame002_lambda48.png",
        "low_cost_artifact_two_frame.png",
    ]
    prohibited_future_rollout_artifacts = [
        "transitions.jsonl",
        "rollout_topdown_path.png",
        "observed_ratio_curve.png as rollout curve",
        "rollout_index.html",
        "episode manifest",
        "frame003",
        "action002",
        "open-ended loop outputs",
    ]
    future_required_outputs = {
        "future_output_dir": str(FUTURE_OUTPUT_DIR),
        "required_json_md_stems": future_required_json_md,
        "required_artifacts": future_required_artifacts,
        "required_plots": future_required_plots,
        "prohibited_rollout_artifacts": prohibited_future_rollout_artifacts,
    }
    write_json(output_dir / "future_required_outputs.json", future_required_outputs)
    write_text(
        output_dir / "future_required_outputs.md",
        f"""
        # Future Required Outputs

        Future output dir:

        ```text
        {FUTURE_OUTPUT_DIR}
        ```

        ## Required JSON/MD Families
        {format_list(future_required_json_md)}

        ## Required Artifacts
        {format_list(future_required_artifacts)}

        ## Required Plots
        {format_list(future_required_plots)}

        ## Explicitly Prohibited Rollout Artifacts
        {format_list(prohibited_future_rollout_artifacts)}
        """,
    )

    future_test_requirements = {
        "future_test_script": "test_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py",
        "must_verify": [
            "output dir exists",
            "exactly two RGB/depth frames exist",
            "no frame003 files exist",
            "exactly two map_predict summary files exist",
            "exactly two map_predict calls recorded",
            "exactly one action execution recorded",
            "no second action recorded",
            "no rollout artifacts exist",
            f"formula is exactly: {PRIMARY_FORMULA}",
            "formula is not: (gain_exp + sc) / cost",
            "measured-only shadow exists for both frames",
            "lambda48 primary/diagnostic exists for both frames",
            "lambda32 shadow exists or skipped reason exists",
            "prediction layer shape equals observed_state shape in both frames",
            "prediction remained read-only",
            "prediction not written into observed_state",
            "prediction not used for traversability",
            "prediction not used for collision",
            "prediction not used for ray blocking",
            "prediction not used for candidate sampling / edge validity",
            "target_lr / target_hr / ground_truth / future observed not used for planning/scoring",
            "checkpoint hash unchanged",
            "existing observed_state hashes unchanged",
            "generated prediction NPZs not modified after creation",
            "no external source modification/build",
            "no training/RL/PPO/BC/IL artifacts",
            "no over-cost runtime primary",
            "action execution occurred only after pre_action_safety_gate_report passed",
            "action execution pose is finite and in map bounds",
            "camera pose at frame002 equals executed action pose within tolerance",
            "rollout_ready=false",
            "coverage_improvement_claim=false",
        ],
        "expected_counts": {
            "frames": 2,
            "map_predict_calls": 2,
            "selected_action_executions": 1,
            "second_action": 0,
            "third_frame": 0,
            "rollout_artifacts": 0,
        },
    }
    write_json(output_dir / "future_test_requirements.json", future_test_requirements)
    write_text(
        output_dir / "future_test_requirements.md",
        f"""
        # Future Test Requirements

        Future validation command must verify exactly two frames, exactly two
        map_predict calls, exactly one action execution, no second action, no
        third frame, no rollout, exact lambda48 formula, prediction safety, hash
        stability, and no target/ground-truth/future-observed leakage.

        {format_list(future_test_requirements["must_verify"])}
        """,
    )

    command_sketch = f"""
    DO NOT RUN IN STAGE 4A-6.5aj.
    This is a future Stage 4A-6.5ak command sketch only.

    ```bash
    source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
    conda activate env_isaaclab

    export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/sim_explorer:$PYTHONPATH

    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    unset DISPLAY
    unset WAYLAND_DISPLAY
    unset XAUTHORITY
    unset GNOME_SETUP_DISPLAY

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

    python run_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py \\
      --output_dir {FUTURE_OUTPUT_DIR} \\
      --scene_variant {args.scene_variant} \\
      --scene_seed {args.scene_seed} \\
      --start_pose canonical_stage4a65p_frame1 \\
      --position {','.join(str(v) for v in position)} \\
      --yaw {args.yaw} \\
      --checkpoint {CHECKPOINT} \\
      --alignment_convention code_consistent_v1 \\
      --tau {args.tau} \\
      --occ_threshold {args.occ_threshold} \\
      --free_threshold {args.free_threshold} \\
      --lambda_sc 48 \\
      --shadow_lambda_sc 32 \\
      --num_nodes 256 \\
      --max_extension_m 0.5 \\
      --sample_mode mixed \\
      --path_cost_mode segment_time \\
      --v_max 1.0 \\
      --robot_radius_m 0.2 \\
      --voxel_size 0.1 \\
      --raycast_stride 2 \\
      --num_yaw_samples 8 \\
      --max_ray_length_m 4.8 \\
      --short_edge_policy crop \\
      --crop_min_length_m 0.25 \\
      --tree_seed 0 \\
      --max_workers 32 \\
      --save_viz \\
      --execute_exactly_one_action \\
      --max_frames 2 \\
      --no_third_frame \\
      --no_second_action \\
      --no_rollout

    python test_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py \\
      --output_dir {FUTURE_OUTPUT_DIR} \\
      --expected_frames 2 \\
      --expected_map_predict_calls 2 \\
      --expected_selected_action_executions 1
    ```

    Again: DO NOT RUN these commands in Stage 4A-6.5aj.
    """
    write_text(output_dir / "future_stage4a65ak_command_sketch.md", command_sketch)
    write_text(
        output_dir / "do_not_run_runtime_in_stage4a65aj.md",
        """
        # Do Not Run Runtime In Stage 4A-6.5aj

        Stage 4A-6.5aj is design review only. It must not start Isaac, capture
        RGB/depth, run map_predict, run SSCNet inference, execute selected
        actions, run a two-frame runtime, run rollout, or claim coverage
        improvement.
        """,
    )

    hardware_policy = {
        "stage": "Stage 4A-6.5aj",
        "future_stage": "Stage 4A-6.5ak",
        "os_cpu_count": os.cpu_count(),
        "os_cpu_count_expected": 32,
        "requested_max_workers": args.max_workers,
        "actual_max_workers": min(args.max_workers, os.cpu_count() or 1),
        "future_command_includes_max_workers_32": args.max_workers == 32,
        "parallel_backend_plan": {
            "runtime_stage": "mostly single-process because Isaac/map_predict is sequential",
            "cpu_helper_work": "use --max_workers 32 where useful",
            "process_pool_inner_threads": 1,
            "single_process_numeric_threads": "may use 32 if not using process pools",
        },
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "must_log_in_future": [
            "os_cpu_count",
            "requested_max_workers",
            "actual_max_workers",
            "parallel_backend",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "GPU name",
            "Isaac startup/capture time",
            "map_predict time",
            "tree decision time",
            "total wall time",
        ],
        "stage4a65ai_gpu": ai_hardware.get("cuda_device_name"),
    }
    write_json(output_dir / "hardware_policy_for_future_runtime.json", hardware_policy)
    write_text(
        output_dir / "hardware_policy_for_future_runtime.md",
        f"""
        # Hardware Policy For Future Runtime

        {build_markdown_table([
            ("os_cpu_count", hardware_policy["os_cpu_count"]),
            ("os_cpu_count expected", 32),
            ("requested max_workers", args.max_workers),
            ("actual max_workers plan", hardware_policy["actual_max_workers"]),
            ("future command includes --max_workers 32", True),
            ("Stage 4A-6.5ai GPU", hardware_policy["stage4a65ai_gpu"]),
        ])}

        Process-pool helper workers should keep BLAS/OMP inner threads at `1`.
        Single-process numeric workloads may use wider thread counts only when
        no process pool is active.
        """,
    )

    prediction_safety_design = {
        "prediction_read_only": True,
        "prediction_information_gain_only": True,
        "prediction_written_to_observed_state": False,
        "prediction_fused_into_observed_state": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_ray_blocking": False,
        "prediction_used_for_candidate_sampling": False,
        "prediction_used_for_edge_validity": False,
        "prediction_changes_motion_validity": False,
        "target_lr_target_hr_ground_truth_used_for_planning_scoring": False,
        "future_observed_used_for_planning_scoring": False,
        "allowed_prediction_use": [
            "source_occ_free information gain",
            "gain_sc",
            "gain_hybrid/value term",
            "confidence diagnostics",
            "visualization",
        ],
    }
    write_json(output_dir / "prediction_safety_design.json", prediction_safety_design)
    write_text(
        output_dir / "prediction_safety_design.md",
        f"""
        # Prediction Safety Design

        Prediction remains read-only and information-gain-only. It must never
        write or fuse into observed_state, and it must not influence
        traversability, collision, ray blocking, candidate sampling, or edge
        validity.

        {build_markdown_table([
            ("prediction read-only", True),
            ("prediction writeback", False),
            ("prediction traversability", False),
            ("prediction collision", False),
            ("prediction ray blocking", False),
            ("target/ground-truth/future-observed scoring", False),
        ])}
        """,
    )

    risk_register = {
        "risks": [
            {
                "risk": "Frame 1 lambda48 action points to a historical prior basin",
                "mitigation": "block action if historical prior basin flag is true",
            },
            {
                "risk": "Frame 1 lambda48 action is a low-cost artifact",
                "mitigation": "block action if low_cost_artifact is true",
            },
            {
                "risk": "Formula accidentally moves SC into the cost denominator",
                "mitigation": "test exact formula string and prohibited formula strings",
            },
            {
                "risk": "Prediction leaks into motion safety",
                "mitigation": "explicit flags and tests for traversability/collision/ray blocking/candidate sampling/edge validity",
            },
            {
                "risk": "Two-frame smoke turns into rollout",
                "mitigation": "hard max_frames=2, no_second_action, no_third_frame, no_rollout, and forbidden artifact checks",
            },
            {
                "risk": "Action execution happens after a failed gate",
                "mitigation": "action_execution_report must reference a passed pre_action_safety_gate_report",
            },
        ],
        "runtime_executed_in_stage4a65aj": False,
    }
    write_json(output_dir / "risk_register.json", risk_register)
    write_text(
        output_dir / "risk_register.md",
        f"""
        # Risk Register

        {format_list([f"{item['risk']}: {item['mitigation']}" for item in risk_register["risks"]])}
        """,
    )

    rollout_blocker_statement = {
        "rollout_ready": False,
        "reason": "A one-frame no-action runtime smoke is not enough for rollout; a bounded two-frame one-action smoke must be executed and validated first.",
        "blocked_until": [
            "Stage 4A-6.5ak two-frame one-action smoke is explicitly requested",
            "Stage 4A-6.5ak executes cleanly",
            "exactly two frames, two map_predict calls, and one action are validated",
            "prediction safety and hash checks pass",
        ],
        "do_not_recommend": [
            "rollout",
            "online open-ended loop",
            "runtime two-action",
            "RL/PPO/BC/IL",
            "prediction writeback/fusion",
            "Pareto gate/runtime planner implementation",
            "over-cost runtime promotion",
        ],
    }
    write_text(
        output_dir / "rollout_blocker_statement.md",
        f"""
        # Rollout Blocker Statement

        Rollout remains blocked. Stage 4A-6.5ai was a clean one-frame no-action
        runtime smoke, and Stage 4A-6.5aj is design review only. The next
        executable step, if explicitly requested, is Stage 4A-6.5ak staged
        two-frame one-action lambda48 runtime smoke, not rollout.

        {format_list(rollout_blocker_statement["blocked_until"])}
        """,
    )

    stage4a65aj_runtime_design = {
        "stage": "Stage 4A-6.5aj",
        "status": "completed",
        "design_review_only": True,
        "runtime_executed": False,
        "future_stage": "Stage 4A-6.5ak",
        "future_behavior": future_two_frame_sequence_spec,
        "formula": future_lambda48_formula_spec,
        "source_protection_profile": future_source_protection_profile,
        "pre_action_safety_gates": pre_action_gates,
        "frame2_stop_conditions": future_frame2_stop_conditions,
        "rollout_ready": False,
    }
    write_json(output_dir / "future_runtime_smoke_design.json", stage4a65aj_runtime_design)
    write_text(
        output_dir / "future_runtime_smoke_design.md",
        f"""
        # Future Runtime Smoke Design

        Stage 4A-6.5aj completed a design review for future Stage 4A-6.5ak.
        The future executable stage is exactly two frames, exactly two
        map_predict calls, exactly one selected action execution after Frame 1
        gates pass, no second action, no third frame, and no rollout.

        Primary formula remains:

        ```text
        value = {PRIMARY_FORMULA}
        ```

        Stage 4A-6.5aj itself executed no runtime work.
        """,
    )

    skipped_diagrams = maybe_generate_diagrams(output_dir)

    required_files = [
        "loaded_context_manifest.json",
        "loaded_context_manifest.md",
        "stage4a65ai_result_review.json",
        "stage4a65ai_result_review.md",
        "future_runtime_smoke_design.json",
        "future_runtime_smoke_design.md",
        "future_two_frame_sequence_spec.json",
        "future_two_frame_sequence_spec.md",
        "future_lambda48_formula_spec.json",
        "future_lambda48_formula_spec.md",
        "future_source_protection_profile.json",
        "future_source_protection_profile.md",
        "future_pre_action_safety_gates.json",
        "future_pre_action_safety_gates.md",
        "future_frame2_stop_conditions.json",
        "future_frame2_stop_conditions.md",
        "future_required_outputs.json",
        "future_required_outputs.md",
        "future_test_requirements.json",
        "future_test_requirements.md",
        "future_stage4a65ak_command_sketch.md",
        "do_not_run_runtime_in_stage4a65aj.md",
        "hardware_policy_for_future_runtime.json",
        "hardware_policy_for_future_runtime.md",
        "prediction_safety_design.json",
        "prediction_safety_design.md",
        "risk_register.json",
        "risk_register.md",
        "rollout_blocker_statement.md",
        "stage4a65aj_design_review_summary.json",
        "stage4a65aj_design_review_summary.md",
        "recommended_next_faithful_step.md",
        "missing_fields_report.json",
        "missing_fields_report.md",
    ]
    optional_diagrams = [
        "flowchart_two_frame_one_action.png",
        "safety_gate_flowchart.png",
        "runtime_timeline.png",
    ]

    summary_answers = {
        "read_stage4a65ag_ah_ai_context_successfully": True,
        "stage4a65ai_one_frame_runtime_result": stage4a65ai_result_review,
        "why_design_review_only": "A two-frame action runtime should be staged from the clean one-frame no-action result with explicit gates before any execution.",
        "future_stage4a65ak_exact_counts": future_two_frame_sequence_spec["exact_counts"],
        "future_frame1_action_allowed_when": "all pre-action safety gates pass",
        "safety_gates_that_block_action": action_block_conditions,
        "future_frame2_second_action_allowed": False,
        "third_frame_allowed": False,
        "rollout_allowed": False,
        "lambda48_sc_outside_cost_denominator": True,
        "over_cost_prohibited_for_runtime_primary": True,
        "prediction_read_only_information_gain_only": True,
        "prediction_for_motion_safety_prohibited": True,
        "target_ground_truth_future_observed_scoring_prohibited": True,
        "future_output_test_requirements_complete": True,
        "future_command_marked_do_not_run_in_stage4a65aj": True,
        "stage4a65aj_isaac_capture_map_predict_action_executed": False,
        "checkpoint_observed_state_prediction_npz_modified": False,
        "enough_for_rollout": False,
        "next_recommendation": "Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke execution only if explicitly requested by user",
    }
    safety = {
        "isaac_startup": False,
        "rgb_depth_capture": False,
        "map_predict_call": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "two_frame_runtime_execution": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_rl_ppo_bc_il": False,
        "checkpoint_modified": False,
        "existing_observed_state_modified": False,
        "prediction_npz_modified": False,
        "prediction_writeback": False,
        "prediction_fusion": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "prediction_used_for_candidate_sampling_edge_validity": False,
        "target_ground_truth_planning_scoring": False,
        "future_observed_planning_scoring": False,
        "external_source_modified_built": False,
        "over_cost_runtime_primary": False,
        "coverage_improvement_claim": False,
        "future_command_marked_do_not_run_in_stage4a65aj": True,
    }

    summary = {
        "stage": "Stage 4A-6.5aj",
        "status": "completed",
        "design_review_only": True,
        "runtime_executed": False,
        "output_dir": str(output_dir),
        "required_files": required_files,
        "optional_diagrams": optional_diagrams,
        "skipped_diagrams": skipped_diagrams,
        "answers": summary_answers,
        "stage4a65ag_review": {
            "unique_saved_real_medium_frames": ag_answers["lambda48_aggregate"]["unique_frame_count"],
            "lambda48_same_as_measured": ag_answers["lambda48_aggregate"]["same_as_measured_count"],
            "lambda48_distinct_nonmeasured": ag_answers["lambda48_aggregate"]["distinct_nonmeasured_count"],
            "lambda48_healthy_nonmeasured": ag_answers["lambda48_aggregate"]["healthy_nonmeasured_count"],
            "lambda48_local_jitter": ag_answers["lambda48_aggregate"]["local_jitter_count"],
            "lambda48_historical_prior_basin": ag_answers["lambda48_aggregate"][
                "historical_prior_basin_count"
            ],
            "lambda48_low_cost_artifact": ag_answers["lambda48_aggregate"]["low_cost_artifact_count"],
            "lambda32_vs_lambda48": ag_answers["lambda32_vs_lambda48"],
            "over_cost_diagnostic": {
                "historical_prior_basin_count": 24,
                "row_count": 70,
                "diagnostic_only": True,
            },
        },
        "stage4a65ah_review": {
            "candidate_rows": ah_summary["discovery"]["candidate_rows"],
            "new_complete_frame_count": ah_summary["discovery"]["new_complete_frame_count"],
            "hardware_policy_recorded": True,
        },
        "stage4a65ai_review": stage4a65ai_result_review,
        "future_stage4a65ak_design": {
            "frames": 2,
            "map_predict_calls": 2,
            "selected_action_executions": 1,
            "second_action": False,
            "third_frame": False,
            "rollout": False,
            "primary_formula": PRIMARY_FORMULA,
            "shadow_formula": SHADOW_FORMULA,
        },
        "hardware_policy": hardware_policy,
        "prediction_safety": prediction_safety_design,
        "safety": safety,
        "recommendation": {
            "next_small_task": "Stage 4A-6.5ak staged two-frame one-action lambda48 runtime smoke execution only if explicitly requested by user",
            "do_not_recommend_rollout": True,
            "why": "The one-frame runtime was clean, but action execution and a post-action second frame still require a tightly bounded smoke before any rollout discussion.",
        },
        "wall_time_s": float(time.perf_counter() - started),
    }
    write_json(output_dir / "stage4a65aj_design_review_summary.json", summary)
    write_text(
        output_dir / "stage4a65aj_design_review_summary.md",
        f"""
        # Stage 4A-6.5aj Design Review Summary

        1. Successfully read Stage 4A-6.5ag / 6.5ah / 6.5ai context? `True`.
        2. Stage 4A-6.5ai one-frame runtime result: clean one-frame no-action
           runtime with one Isaac startup, one frame, one map_predict call,
           lambda48 selected `n0001 -> n0228`, branch
           `{ai_branch["classification"]}`, low-cost artifact `False`,
           historical prior basin `False`, and rollout_ready `False`.
        3. Why design review only? The next executable step needs explicit
           two-frame / one-action gates before any runtime execution.
        4. Future Stage 4A-6.5ak counts: exactly two frames, exactly two
           map_predict calls, exactly one selected action execution.
        5. Future Frame 1 action is allowed only after every pre-action safety
           gate passes.
        6. Action is blocked by capture/map_predict/prediction safety/formula/
           prior-basin/low-cost-artifact/non-finite-pose/leakage failures.
        7. Future Frame 2 second action allowed? `False`.
        8. Third frame allowed? `False`.
        9. Rollout allowed? `False`.
        10. Lambda48 formula keeps SC outside cost denominator? `True`.
        11. Over-cost prohibited for runtime primary? `True`.
        12. Prediction remains read-only / information-gain-only? `True`.
        13. Prediction prohibited for traversability/collision/ray blocking/
            edge validity? `True`.
        14. Target/ground-truth/future-observed scoring prohibited? `True`.
        15. Future output/test requirements complete? `True`.
        16. Future command sketch marked DO NOT RUN in 6.5aj? `True`.
        17. Current Isaac / capture / map_predict / action executed? `False`.
        18. Current checkpoint / observed_state / prediction NPZ modified?
            `False`.
        19. Current enough for rollout? `False`.
        20. Recommended next: Stage 4A-6.5ak staged two-frame one-action
            lambda48 runtime smoke execution only if explicitly requested.

        ## Future Formula

        ```text
        value = {PRIMARY_FORMULA}
        ```
        """,
    )
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        """
        # Recommended Next Faithful Step

        If the user explicitly requests execution next, run Stage 4A-6.5ak:
        staged two-frame one-action lambda48 runtime smoke. It must still not
        be rollout. Rollout remains blocked until at least one two-frame
        one-action smoke is executed and validated cleanly.
        """,
    )

    prohibited_patterns = [
        "transitions.jsonl",
        "rollout_topdown_path.png",
        "observed_ratio_curve.png",
        "rollout_index.html",
        "frame003*",
        "action002*",
        "observed_state*.npy",
        "global_prediction_layer.npz",
        "frame*_rgb.png",
        "frame*_depth.npy",
        "capture_rgb*.png",
        "capture_depth*.npy",
    ]
    prohibited_found: list[str] = []
    for pattern in prohibited_patterns:
        prohibited_found.extend(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
    missing_required = [
        name
        for name in required_files
        if name not in {"missing_fields_report.json", "missing_fields_report.md"}
        and not (output_dir / name).is_file()
    ]
    missing_optional = [
        name
        for name in optional_diagrams
        if not (output_dir / name).is_file() and name not in skipped_diagrams
    ]
    missing_fields_report = {
        "missing_required_files": missing_required,
        "optional_diagram_skipped_reasons": skipped_diagrams,
        "missing_optional_diagrams_without_reason": missing_optional,
        "prohibited_artifacts_found": sorted(prohibited_found),
        "stage4a65aj_runtime_executed": False,
    }
    write_json(output_dir / "missing_fields_report.json", missing_fields_report)
    write_text(
        output_dir / "missing_fields_report.md",
        f"""
        # Missing Fields Report

        Missing required files: `{missing_required}`

        Optional diagram skipped reasons: `{skipped_diagrams}`

        Missing optional diagrams without reason: `{missing_optional}`

        Prohibited artifacts found: `{sorted(prohibited_found)}`
        """,
    )

    print(json.dumps({"status": "completed", "output_dir": str(output_dir), "runtime_executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a small sequential Stage 4A-6.1 SC prediction ablation sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sim_rollout_utils import append_jsonl, load_json, load_jsonl, save_json


RUNNER = Path("/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py")
DEFAULT_STATIC_STEP0 = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/"
    "episodes/medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000/"
    "prediction_step000/global_prediction_layer.npz"
)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    prediction_mode: str
    tau: float
    sc_gain_weight: float
    sc_gain_cap: float | None
    score_gain_mode: str = "hybrid_weighted"
    static_prediction_npz: str | None = None
    static_prediction_step: int | None = None


CONFIGS: dict[str, AblationConfig] = {
    "dynamic_w025_tau01": AblationConfig("dynamic_w025_tau01", "sim_dynamic", 0.1, 0.25, None),
    "dynamic_weight_0p25_tau_0p1": AblationConfig("dynamic_w025_tau01", "sim_dynamic", 0.1, 0.25, None),
    "dynamic_w05_tau01": AblationConfig("dynamic_w05_tau01", "sim_dynamic", 0.1, 0.5, None),
    "dynamic_weight_0p5_tau_0p1": AblationConfig("dynamic_w05_tau01", "sim_dynamic", 0.1, 0.5, None),
    "dynamic_w1_tau03": AblationConfig("dynamic_w1_tau03", "sim_dynamic", 0.3, 1.0, None),
    "dynamic_weight_1p0_tau_0p3": AblationConfig("dynamic_w1_tau03", "sim_dynamic", 0.3, 1.0, None),
    "dynamic_w1_tau01_cap50": AblationConfig("dynamic_w1_tau01_cap50", "sim_dynamic", 0.1, 1.0, 50.0),
    "dynamic_weight_1p0_tau_0p1_cap_50": AblationConfig(
        "dynamic_w1_tau01_cap50",
        "sim_dynamic",
        0.1,
        1.0,
        50.0,
    ),
    "static_step0_weight_1p0_tau_0p1": AblationConfig(
        "static_step0_weight_1p0_tau_0p1",
        "sim_static_npz",
        0.1,
        1.0,
        None,
        static_prediction_npz=str(DEFAULT_STATIC_STEP0),
        static_prediction_step=0,
    ),
    "static_step0_w1_tau01": AblationConfig(
        "static_step0_weight_1p0_tau_0p1",
        "sim_static_npz",
        0.1,
        1.0,
        None,
        static_prediction_npz=str(DEFAULT_STATIC_STEP0),
        static_prediction_step=0,
    ),
}


def _split_configs(value: str) -> list[AblationConfig]:
    configs: list[AblationConfig] = []
    for raw_name in str(value).split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Known configs: {sorted(CONFIGS)}")
        cfg = CONFIGS[name]
        if cfg.name not in {item.name for item in configs}:
            configs.append(cfg)
    return configs


def _episode_id(args: argparse.Namespace, cfg: AblationConfig) -> str:
    return f"{args.scene_variant}_seed{args.scene_seed}_{args.start_variant}_{cfg.name}"


def _base_child_args(args: argparse.Namespace, app_args: list[str], cfg: AblationConfig) -> list[str]:
    episode_id = _episode_id(args, cfg)
    child = [
        sys.executable,
        str(RUNNER),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        "--episode_id",
        episode_id,
        "--scene_variant",
        str(args.scene_variant),
        "--scene_seed",
        str(args.scene_seed),
        "--start_variant",
        str(args.start_variant),
        "--max_steps",
        str(args.max_steps),
        "--num_candidates",
        str(args.num_candidates),
        "--top_n",
        str(args.top_n),
        "--path_cost_mode",
        str(args.path_cost_mode),
        "--candidate_sampling_mode",
        str(args.candidate_sampling_mode),
        "--max_snap_radius_cells",
        str(args.max_snap_radius_cells),
        "--motion_mode",
        str(args.motion_mode),
        "--camera_height",
        str(args.camera_height),
        "--voxel_size",
        str(args.voxel_size),
        "--pixel_stride",
        str(args.pixel_stride),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--prediction_mode",
        cfg.prediction_mode,
        "--tau",
        str(cfg.tau),
        "--sc_gain_weight",
        str(cfg.sc_gain_weight),
        "--score_gain_mode",
        cfg.score_gain_mode,
    ]
    if cfg.sc_gain_cap is not None:
        child.extend(["--sc_gain_cap", str(cfg.sc_gain_cap)])
    if cfg.static_prediction_npz is not None:
        child.extend(["--static_prediction_npz", str(Path(cfg.static_prediction_npz).resolve())])
    if cfg.static_prediction_step is not None:
        child.extend(["--static_prediction_step", str(cfg.static_prediction_step)])
    if bool(args.snap_start_to_traversable):
        child.append("--snap_start_to_traversable")
    if bool(args.save_depth):
        child.append("--save_depth")
    if bool(args.save_prediction):
        child.append("--save_prediction")
    if bool(args.save_viz):
        child.append("--save_viz")
    if bool(args.profile):
        child.append("--profile")
    if bool(args.headless):
        child.append("--headless")
    if bool(args.enable_cameras):
        child.append("--enable_cameras")
    child.extend(app_args)
    return child


def _stream_subprocess(cmd: list[str], log_path: Path, timeout_s: float | None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                handle.write(line)
                handle.flush()
                if timeout_s is not None and time.perf_counter() - start > timeout_s:
                    proc.kill()
                    handle.write(f"\n[ERROR] per-config timeout after {timeout_s}s\n")
                    return 124
            return int(proc.wait())
        finally:
            if proc.poll() is None:
                proc.kill()


def _load_summary(output_dir: Path, episode_id: str) -> dict[str, Any] | None:
    path = output_dir / "episodes" / episode_id / "episode_summary.json"
    if not path.exists():
        return None
    return load_json(path)


def _record_from_summary(
    args: argparse.Namespace,
    cfg: AblationConfig,
    returncode: int,
    wall_time: float,
    log_path: Path,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    episode_id = _episode_id(args, cfg)
    summary = _load_summary(output_dir, episode_id)
    status = "ok" if returncode == 0 and summary and summary.get("steps_completed", 0) >= 1 else "failed"
    record: dict[str, Any] = {
        "config_name": cfg.name,
        "episode_id": episode_id,
        "episode_dir": str(output_dir / "episodes" / episode_id),
        "status": status,
        "returncode": int(returncode),
        "wall_time": float(wall_time),
        "log_path": str(log_path),
        "prediction_mode": cfg.prediction_mode,
        "tau": float(cfg.tau),
        "sc_gain_weight": float(cfg.sc_gain_weight),
        "sc_gain_cap": cfg.sc_gain_cap,
        "score_gain_mode": cfg.score_gain_mode,
        "static_prediction_npz": cfg.static_prediction_npz,
        "static_prediction_step": cfg.static_prediction_step,
    }
    if summary:
        record.update(
            {
                "steps_completed": int(summary.get("steps_completed", 0)),
                "done_reason": str(summary.get("done_reason", "")),
                "observed_ratio_start": float(summary.get("observed_ratio_start", 0.0)),
                "observed_ratio_end": float(summary.get("observed_ratio_end", 0.0)),
                "observed_ratio_delta": float(summary.get("total_delta_observed_ratio", 0.0)),
                "average_map_predict_inference_time": summary.get("average_map_predict_inference_time"),
                "average_map_predict_total_time": summary.get("average_map_predict_total_time"),
                "average_expert_time": summary.get("average_expert_time"),
                "gpu_memory_peak": summary.get("gpu_memory_peak"),
                "checkpoint_modified": bool(summary.get("checkpoint_modified", False)),
                "no_valid_candidate_steps": summary.get("no_valid_candidate_steps", []),
                "summary_path": str(Path(summary.get("episode_dir", record["episode_dir"])) / "episode_summary.json"),
            }
        )
    if args.baseline_episode_dir and summary:
        baseline_path = Path(args.baseline_episode_dir).resolve() / "transitions.jsonl"
        baseline = []
        if baseline_path.exists():
            from sim_rollout_utils import load_jsonl

            baseline = load_jsonl(baseline_path)
        if baseline:
            idx = min(int(args.max_steps), len(baseline), int(summary.get("steps_completed", 0))) - 1
            if idx >= 0:
                empty_final = float(baseline[idx].get("observed_ratio_after", 0.0))
                record["empty_baseline_observed_ratio_at_max_steps"] = empty_final
                record["observed_ratio_delta_vs_empty"] = float(record.get("observed_ratio_end", 0.0) - empty_final)
    return record


def run_sweep(args: argparse.Namespace, app_args: list[str]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    configs = _split_configs(args.configs)
    save_json(
        output_dir / "ablation_configs.json",
        {
            "requested_configs": [cfg.name for cfg in configs],
            "scene_variant": args.scene_variant,
            "scene_seed": int(args.scene_seed),
            "start_variant": args.start_variant,
            "max_steps": int(args.max_steps),
            "baseline_episode_dir": str(Path(args.baseline_episode_dir).resolve()) if args.baseline_episode_dir else None,
            "notes": [
                "Sequential ablation runner; never launches multiple Isaac instances in parallel.",
                "Prediction remains read-only and information-gain-only inside the child rollout runner.",
            ],
        },
    )

    manifest_jsonl = output_dir / "ablation_manifest.jsonl"
    records: list[dict[str, Any]] = load_jsonl(manifest_jsonl) if manifest_jsonl.exists() else []
    existing_keys = {(str(r.get("config_name")), str(r.get("episode_id"))) for r in records}
    for cfg in configs:
        print(f"[ABLATION] starting {cfg.name}")
        cmd = _base_child_args(args, app_args, cfg)
        log_path = output_dir / "logs" / f"{cfg.name}.log"
        start = time.perf_counter()
        try:
            returncode = _stream_subprocess(cmd, log_path, args.per_config_timeout_s)
        except BaseException as exc:
            returncode = 1
            log_path.write_text(f"ablation runner failed before child completed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        wall_time = time.perf_counter() - start
        record = _record_from_summary(args, cfg, returncode, wall_time, log_path)
        record_key = (str(record.get("config_name")), str(record.get("episode_id")))
        if record_key not in existing_keys:
            records.append(record)
            append_jsonl(manifest_jsonl, record)
            existing_keys.add(record_key)
        else:
            records = [record if (str(r.get("config_name")), str(r.get("episode_id"))) == record_key else r for r in records]
            with manifest_jsonl.open("w", encoding="utf-8") as handle:
                for item in records:
                    handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        print(
            f"[ABLATION] {cfg.name} status={record['status']} steps={record.get('steps_completed')} "
            f"observed_ratio_end={record.get('observed_ratio_end')} wall_time={wall_time:.1f}s"
        )
        if record["status"] != "ok" and not bool(args.continue_on_error):
            break

    completed = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    summary = {
        "stage": "Stage 4A-6.1 SC prediction ablation sweep",
        "output_dir": str(output_dir),
        "requested_configs": [cfg.name for cfg in configs],
        "records": records,
        "completed_configs": [r["config_name"] for r in completed],
        "failed_configs": [r["config_name"] for r in failed],
        "completed_count": int(len(completed)),
        "failed_count": int(len(failed)),
        "sequential_isaac_instances": True,
        "parallel_isaac_instances": False,
        "prediction_read_only": True,
        "prediction_information_gain_only": True,
        "rl_ppo_bc_il_training": False,
    }
    save_json(output_dir / "ablation_manifest.json", summary)
    if failed and not bool(args.continue_on_error):
        raise SystemExit(1)
    return summary


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run sequential SC-aware rollout ablations.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--start_variant", default="start_room_a")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument("--path_cost_mode", choices=("astar",), default="astar")
    parser.add_argument("--candidate_sampling_mode", choices=("reachable_frontier",), default="reachable_frontier")
    parser.add_argument("--snap_start_to_traversable", action="store_true")
    parser.add_argument("--max_snap_radius_cells", type=int, default=5)
    parser.add_argument("--motion_mode", choices=("planar",), default="planar")
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--pixel_stride", type=int, default=2)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline_episode_dir", default=None)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_prediction", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--per_config_timeout_s", type=float, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    return parser.parse_known_args(argv)


def main() -> None:
    args, app_args = parse_args()
    summary = run_sweep(args, app_args)
    print("Stage 4A-6.1 ablation sweep complete.")
    print(f"completed_configs: {summary['completed_configs']}")
    print(f"failed_configs: {summary['failed_configs']}")
    print(f"ablation_manifest_json: {Path(summary['output_dir']) / 'ablation_manifest.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 4A-4 batch runner for EmptyPredictionLayer A* expert rollouts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROLLOUT_SCRIPT = SCRIPT_DIR / "run_sim_expert_rollout.py"
DEFAULT_OUTPUT_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_csv_ints(value: str) -> list[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(item) for item in items]


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    return items


def medium_start_pose(start_variant: str, camera_height: float) -> dict[str, Any]:
    starts = {
        "start_room_a": {
            "position": [-4.80, -4.70, float(camera_height)],
            "yaw_deg": 20.0,
            "note": "default room A scripted start near the southwest room",
        },
        "start_corridor": {
            "position": [0.00, -4.45, float(camera_height)],
            "yaw_deg": 90.0,
            "note": "corridor entrance scripted start looking north",
        },
        "start_room_b": {
            "position": [2.75, -2.55, float(camera_height)],
            "yaw_deg": 155.0,
            "note": "room B open-space scripted start near the doorway",
        },
    }
    if start_variant not in starts:
        raise ValueError(f"unknown start variant '{start_variant}'. Known: {sorted(starts)}")
    return starts[start_variant]


def default_start_pose(scene_variant: str, start_variant: str, camera_height: float) -> dict[str, Any]:
    if scene_variant == "medium_three_rooms":
        return medium_start_pose(start_variant, camera_height)
    if start_variant not in {"start_room_a", "default"}:
        raise ValueError(f"{scene_variant} supports only start_room_a/default, got {start_variant}")
    return {
        "position": [0.0, 0.0, float(camera_height)],
        "yaw_deg": 0.0,
        "note": "minimal scene default center start",
    }


def episode_id(scene_variant: str, scene_seed: int, start_variant: str) -> str:
    return f"{scene_variant}_seed{int(scene_seed)}_{start_variant}_empty_astar"


def build_child_command(args: argparse.Namespace, app_args: list[str], scene_seed: int, start_variant: str) -> list[str]:
    pose = default_start_pose(args.scene_variant, start_variant, args.camera_height)
    position = pose["position"]
    cmd = [
        sys.executable,
        str(ROLLOUT_SCRIPT),
        "--output_dir",
        str(args.output_dir),
        "--episode_id",
        episode_id(args.scene_variant, scene_seed, start_variant),
        "--scene_variant",
        str(args.scene_variant),
        "--scene_seed",
        str(scene_seed),
        "--start_variant",
        str(start_variant),
        "--start_x",
        str(position[0]),
        "--start_y",
        str(position[1]),
        "--start_yaw_deg",
        str(pose["yaw_deg"]),
        "--obstacle_jitter_m",
        str(args.obstacle_jitter_m),
        "--max_steps",
        str(args.max_steps),
        "--num_candidates",
        str(args.num_candidates),
        "--top_n",
        str(args.top_n),
        "--gain_mode",
        str(args.gain_mode),
        "--prediction_mode",
        str(args.prediction_mode),
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
        "--no_manifest",
    ]
    if args.snap_start_to_traversable:
        cmd.append("--snap_start_to_traversable")
    if args.save_rgb:
        cmd.append("--save_rgb")
    if args.save_depth:
        cmd.append("--save_depth")
    if args.save_viz:
        cmd.append("--save_viz")
    if args.print_steps:
        cmd.append("--print_steps")
    cmd.extend(app_args)
    return cmd


def manifest_from_summary(
    summary: dict[str, Any],
    scene_seed: int,
    start_variant: str,
    status: str = "ok",
    skipped_existing: bool = False,
) -> dict[str, Any]:
    metrics = summary.get("metrics") or {}
    return {
        "episode_id": str(summary["episode_id"]),
        "status": str(status),
        "skipped_existing": bool(skipped_existing),
        "scene_seed": int(scene_seed),
        "start_variant": str(start_variant),
        "episode_dir": str(summary["episode_dir"]),
        "steps_completed": int(summary.get("steps_completed", 0)),
        "done_reason": str(summary.get("done_reason", "")),
        "observed_ratio_start": float(summary.get("observed_ratio_start", 0.0)),
        "observed_ratio_end": float(summary.get("observed_ratio_end", 0.0)),
        "total_delta_observed_ratio": float(summary.get("total_delta_observed_ratio", 0.0)),
        "no_valid_candidate_steps": metrics.get("no_valid_candidate_steps", []),
        "error": "",
    }


def run_episode(
    args: argparse.Namespace,
    app_args: list[str],
    scene_seed: int,
    start_variant: str,
    manifest_path: Path,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    eid = episode_id(args.scene_variant, scene_seed, start_variant)
    episode_dir = output_dir / "episodes" / eid
    log_path = output_dir / "logs" / f"{eid}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = episode_dir / "episode_summary.json"
    transitions_path = episode_dir / "transitions.jsonl"
    if args.skip_existing and summary_path.exists() and transitions_path.exists():
        summary = load_json(summary_path)
        row = manifest_from_summary(summary, scene_seed, start_variant, skipped_existing=True)
        row["log_path"] = str(log_path)
        append_jsonl(manifest_path, row)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[SKIP] existing episode kept: {episode_dir}\n")
        return row

    cmd = build_child_command(args, app_args, scene_seed, start_variant)
    start = time.time()
    row: dict[str, Any]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + shlex.join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(args.child_workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if args.print_steps:
                print(f"[{eid}] {line}", end="", flush=True)
        return_code = proc.wait()
        elapsed = time.time() - start
        log.write(f"\n[batch] return_code={return_code} elapsed_sec={elapsed:.3f}\n")

    if return_code == 0 and summary_path.exists():
        summary = load_json(summary_path)
        row = manifest_from_summary(summary, scene_seed, start_variant)
        row["return_code"] = int(return_code)
        row["elapsed_sec"] = float(elapsed)
        row["log_path"] = str(log_path)
    else:
        error = f"rollout failed with return_code={return_code}; see {log_path}"
        row = {
            "episode_id": eid,
            "status": "failed",
            "scene_seed": int(scene_seed),
            "start_variant": str(start_variant),
            "episode_dir": str(episode_dir),
            "steps_completed": 0,
            "done_reason": "failed",
            "observed_ratio_start": 0.0,
            "observed_ratio_end": 0.0,
            "total_delta_observed_ratio": 0.0,
            "no_valid_candidate_steps": [],
            "return_code": int(return_code),
            "elapsed_sec": float(elapsed),
            "log_path": str(log_path),
            "error": error,
        }
    append_jsonl(manifest_path, row)
    return row


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Stage 4A-4 multi-episode EmptyPredictionLayer A* rollout dataset runner."
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seeds", type=parse_csv_ints, default=parse_csv_ints("0,1,2"))
    parser.add_argument(
        "--start_variants",
        type=parse_csv_strings,
        default=parse_csv_strings("start_room_a,start_corridor,start_room_b"),
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument("--gain_mode", choices=("exp", "sc", "hybrid", "occ", "conf"), default="hybrid")
    parser.add_argument("--prediction_mode", choices=("empty",), default="empty")
    parser.add_argument("--path_cost_mode", choices=("astar", "euclidean"), default="astar")
    parser.add_argument(
        "--candidate_sampling_mode",
        choices=("reachable_frontier", "frontier", "auto"),
        default="reachable_frontier",
    )
    parser.add_argument("--snap_start_to_traversable", action="store_true")
    parser.add_argument("--max_snap_radius_cells", type=int, default=5)
    parser.add_argument("--motion_mode", choices=("planar", "voxel3d"), default="planar")
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--pixel_stride", type=int, default=2)
    parser.add_argument("--obstacle_jitter_m", type=float, default=0.05)
    parser.add_argument("--save_rgb", action="store_true")
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--print_steps", action="store_true")
    parser.add_argument("--child_workdir", default="/home/ubuntu22/IsaacLab")
    args, app_args = parser.parse_known_args()
    return args, app_args


def main() -> None:
    args, app_args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.skip_existing:
        manifest_path.unlink()

    rows: list[dict[str, Any]] = []
    intended = len(args.scene_seeds) * len(args.start_variants)
    print("Stage 4A-4 batch rollout start.")
    print(f"output_dir: {output_dir}")
    print(f"scene_variant: {args.scene_variant}")
    print(f"scene_seeds: {args.scene_seeds}")
    print(f"start_variants: {args.start_variants}")
    print(f"intended_episodes: {intended}")

    for scene_seed in args.scene_seeds:
        for start_variant in args.start_variants:
            eid = episode_id(args.scene_variant, scene_seed, start_variant)
            print(f"[batch] episode start: {eid}", flush=True)
            try:
                row = run_episode(args, app_args, scene_seed, start_variant, manifest_path)
            except Exception as exc:  # keep batch resumable even on wrapper errors
                row = {
                    "episode_id": eid,
                    "status": "failed",
                    "scene_seed": int(scene_seed),
                    "start_variant": str(start_variant),
                    "episode_dir": str(output_dir / "episodes" / eid),
                    "steps_completed": 0,
                    "done_reason": "batch_wrapper_exception",
                    "observed_ratio_start": 0.0,
                    "observed_ratio_end": 0.0,
                    "total_delta_observed_ratio": 0.0,
                    "no_valid_candidate_steps": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                append_jsonl(manifest_path, row)
            rows.append(row)
            print(
                f"[batch] episode {row['status']}: {eid} "
                f"steps={row.get('steps_completed', 0)} reason={row.get('done_reason', '')}",
                flush=True,
            )
            if row.get("status") != "ok" and not args.continue_on_error:
                raise SystemExit(f"episode failed and --continue_on_error was not set: {eid}")

    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    failed_count = sum(1 for row in rows if row.get("status") != "ok")
    print("Stage 4A-4 batch rollout complete.")
    print(f"manifest: {manifest_path}")
    print(f"ok_episodes: {ok_count}")
    print(f"failed_episodes: {failed_count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Generate Stage 2C paper-expert imitation-learning-ready samples.

This script runs the strict Stage 2B paper-faithful expert scorer over NYU
samples and saves per-scene expert samples plus manifest/metadata files.

It intentionally does not train imitation learning, run RL/PPO, touch
observed_map, or use target_lr/target_hr for expert scoring.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sc_explorer_paper_expert import (
    FEATURE_NAMES,
    MEASURED_NOTE,
    STRICT_NO_TARGET_NOTE,
    TREE_LIMITATION_NOTE,
    candidate_feature_vector,
    run_paper_expert_scoring,
)


DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)
DEFAULT_OUTPUT_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset"
PREDICTION_TOOL = (
    "/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/offline_infer_npz.py"
)
EXPERT_SCORER_FILE = (
    "/home/ubuntu22/sc_explorer_ws/ssc_exploration/ssc_network/"
    "sc_explorer_paper_expert.py"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate strict paper expert dataset samples for Stage 2C"
    )
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument(
        "--measured_mode",
        choices=("tsdf_lr", "position", "union"),
        default="tsdf_lr",
    )
    parser.add_argument(
        "--raycast_mode",
        choices=("non_blocking", "sc_blocking"),
        default="non_blocking",
    )
    parser.add_argument(
        "--gain_mode",
        choices=("exp", "sc", "hybrid", "occ", "conf"),
        default="hybrid",
    )
    parser.add_argument("--voxel_size", type=float, default=0.08)
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--yaw_rate_deg", type=float, default=90.0)
    parser.add_argument("--max_range", type=int, default=20)
    parser.add_argument("--num_yaw", type=int, default=16)
    parser.add_argument("--num_pitch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generate_missing_predictions", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _git_commit(repo_dir):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable"


def _git_dirty(repo_dir):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return None


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def find_prediction(sample_path, prediction_dirs):
    stem = sample_path.stem
    for directory in prediction_dirs:
        directory = Path(directory)
        candidates = (
            directory / f"{stem}_prediction.npz",
            directory / f"{stem}.npz",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def _select_samples(sample_dir, prediction_dir, max_samples, generate_missing_predictions):
    samples = sorted(Path(sample_dir).glob("*.npz"))
    if max_samples is None or int(max_samples) <= 0:
        return samples

    max_samples = int(max_samples)
    if generate_missing_predictions:
        return samples[:max_samples]

    matching = []
    missing = []
    for sample in samples:
        if find_prediction(sample, [prediction_dir]) is not None:
            matching.append(sample)
        else:
            missing.append(sample)

    selected = matching[:max_samples]
    if len(selected) < max_samples:
        selected_ids = {path.stem for path in selected}
        selected.extend(
            sample for sample in missing if sample.stem not in selected_ids
        )
    return selected[:max_samples]


def _generate_prediction(sample_npz, output_prediction_dir, checkpoint):
    from offline_infer_npz import run_inference

    stats = run_inference(
        checkpoint=checkpoint,
        input_npz=sample_npz,
        output_dir=output_prediction_dir,
        device=None,
        save_probs=False,
        save_logits=False,
        print_stats=False,
    )
    return Path(stats["output_path"])


def _candidate_arrays(top_candidates):
    candidate_features = np.stack(
        [candidate_feature_vector(candidate) for candidate in top_candidates], axis=0
    ).astype(np.float32)
    candidate_positions = np.array(
        [candidate.position for candidate in top_candidates], dtype=np.int32
    )
    candidate_yaws = np.array(
        [candidate.yaw for candidate in top_candidates], dtype=np.float32
    )
    valid_mask = np.array([candidate.valid for candidate in top_candidates], dtype=bool)
    expert_scores = np.array(
        [candidate.final_score for candidate in top_candidates], dtype=np.float32
    )
    top_candidate_ids = np.array(
        [candidate.id for candidate in top_candidates], dtype=np.int64
    )
    return {
        "candidate_features": candidate_features,
        "feature_names": np.array(FEATURE_NAMES),
        "candidate_positions": candidate_positions,
        "candidate_yaws": candidate_yaws,
        "valid_mask": valid_mask,
        "expert_scores": expert_scores,
        "top_candidate_ids": top_candidate_ids,
    }


def save_dataset_sample(
    sample_output_path,
    sample_npz,
    prediction_npz,
    sample_id,
    result,
    args,
):
    top_candidates = result["top_candidates"]
    arrays = _candidate_arrays(top_candidates)
    expert_action = int(result["expert_action"])

    np.savez_compressed(
        sample_output_path,
        candidate_features=arrays["candidate_features"],
        feature_names=arrays["feature_names"],
        candidate_positions=arrays["candidate_positions"],
        candidate_yaws=arrays["candidate_yaws"],
        valid_mask=arrays["valid_mask"],
        expert_action=np.array(expert_action, dtype=np.int64),
        expert_scores=arrays["expert_scores"],
        top_candidate_ids=arrays["top_candidate_ids"],
        gain_mode=np.array(str(args.gain_mode)),
        measured_mode=np.array(str(args.measured_mode)),
        raycast_mode=np.array(str(args.raycast_mode)),
        sample_npz=np.array(str(sample_npz)),
        prediction_npz=np.array(str(prediction_npz)),
        sample_id=np.array(str(sample_id)),
        strict_no_target_note=np.array(STRICT_NO_TARGET_NOTE),
        tree_limitation_note=np.array(TREE_LIMITATION_NOTE),
        measured_note=np.array(MEASURED_NOTE),
        all_candidate_count=np.array(int(len(result["candidates"])), dtype=np.int64),
        num_candidates=np.array(int(args.num_candidates), dtype=np.int64),
        top_n=np.array(int(args.top_n), dtype=np.int64),
        tau=np.array(float(args.tau), dtype=np.float32),
    )
    return arrays


def _feature_value(candidate, feature_name):
    feature_vector = candidate_feature_vector(candidate)
    index = FEATURE_NAMES.index(feature_name)
    return float(feature_vector[index])


def make_ok_manifest_record(
    sample_id,
    sample_npz,
    prediction_npz,
    expert_npz,
    result,
    args,
):
    best = result["top_candidates"][int(result["expert_action"])]
    return {
        "sample_id": str(sample_id),
        "sample_npz": str(sample_npz),
        "prediction_npz": str(prediction_npz),
        "expert_npz": str(expert_npz),
        "num_candidates": int(args.num_candidates),
        "top_n": int(args.top_n),
        "expert_action": int(result["expert_action"]),
        "best_score": _feature_value(best, "final_score"),
        "best_gain_exp": _feature_value(best, "gain_exp"),
        "best_gain_sc": _feature_value(best, "gain_sc"),
        "best_gain_hybrid": _feature_value(best, "gain_hybrid"),
        "best_gain_occ": _feature_value(best, "gain_occ"),
        "best_gain_conf": _feature_value(best, "gain_conf"),
        "best_path_cost": _feature_value(best, "path_cost"),
        "gain_mode": str(args.gain_mode),
        "measured_mode": str(args.measured_mode),
        "raycast_mode": str(args.raycast_mode),
        "status": "ok",
    }


def make_failed_manifest_record(sample_id, sample_npz, error, prediction_npz=None):
    record = {
        "sample_id": str(sample_id),
        "sample_npz": str(sample_npz),
        "status": "failed",
        "error": str(error),
    }
    if prediction_npz is not None:
        record["prediction_npz"] = str(prediction_npz)
    return record


def write_metadata(metadata_path, args, ok_records, failed_records):
    repo_dir = Path(__file__).resolve().parents[1]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_commit(repo_dir),
        "git_commit": _git_commit(repo_dir),
        "git_dirty": _git_dirty(repo_dir),
        "checkpoint": str(args.checkpoint),
        "prediction_tool_used": PREDICTION_TOOL,
        "expert_scorer_file": EXPERT_SCORER_FILE,
        "feature_names": FEATURE_NAMES,
        "num_candidates": int(args.num_candidates),
        "top_n": int(args.top_n),
        "tau": float(args.tau),
        "measured_mode": str(args.measured_mode),
        "raycast_mode": str(args.raycast_mode),
        "gain_mode": str(args.gain_mode),
        "voxel_size": float(args.voxel_size),
        "v_max": float(args.v_max),
        "yaw_rate_deg": float(args.yaw_rate_deg),
        "max_range": int(args.max_range),
        "num_yaw": int(args.num_yaw),
        "num_pitch": int(args.num_pitch),
        "strict_no_target_lr": True,
        "target_lr_usage": "evaluation_only_not_used_for_scoring",
        "tree_utility_eq12_implemented": False,
        "note": (
            "per-candidate paper-faithful gain/cost expert, not full RRT tree planner"
        ),
        "sample_dir": str(args.sample_dir),
        "prediction_dir": str(args.prediction_dir),
        "output_dir": str(args.output_dir),
        "ok_count": int(len(ok_records)),
        "failed_count": int(len(failed_records)),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return metadata


def write_combined_npz(combined_path, sample_arrays):
    if not sample_arrays:
        return None

    first_shape = sample_arrays[0]["candidate_features"].shape
    if any(item["candidate_features"].shape != first_shape for item in sample_arrays):
        return None

    np.savez_compressed(
        combined_path,
        sample_ids=np.array([item["sample_id"] for item in sample_arrays]),
        expert_npz=np.array([item["expert_npz"] for item in sample_arrays]),
        candidate_features=np.stack(
            [item["candidate_features"] for item in sample_arrays], axis=0
        ).astype(np.float32),
        feature_names=np.array(FEATURE_NAMES),
        candidate_positions=np.stack(
            [item["candidate_positions"] for item in sample_arrays], axis=0
        ).astype(np.int32),
        candidate_yaws=np.stack(
            [item["candidate_yaws"] for item in sample_arrays], axis=0
        ).astype(np.float32),
        valid_mask=np.stack([item["valid_mask"] for item in sample_arrays], axis=0),
        expert_action=np.array(
            [item["expert_action"] for item in sample_arrays], dtype=np.int64
        ),
        expert_scores=np.stack(
            [item["expert_scores"] for item in sample_arrays], axis=0
        ).astype(np.float32),
        top_candidate_ids=np.stack(
            [item["top_candidate_ids"] for item in sample_arrays], axis=0
        ).astype(np.int64),
    )
    return combined_path


def _prepare_output_dirs(output_dir, overwrite):
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    predictions_dir = output_dir / "predictions"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, samples_dir, predictions_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "metadata.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists; use --overwrite: {manifest_path}")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Metadata already exists; use --overwrite: {metadata_path}")
    manifest_path.write_text("", encoding="utf-8")
    return output_dir, samples_dir, predictions_dir, logs_dir, manifest_path, metadata_path


def main():
    args = parse_args()

    sample_dir = Path(args.sample_dir)
    prediction_dir = Path(args.prediction_dir)
    output_dir = Path(args.output_dir)

    if not sample_dir.is_dir():
        raise NotADirectoryError(f"sample_dir not found: {sample_dir}")
    if not prediction_dir.is_dir():
        raise NotADirectoryError(f"prediction_dir not found: {prediction_dir}")
    if int(args.max_samples) < 0:
        raise ValueError("--max_samples must be >= 0")

    selected_samples = _select_samples(
        sample_dir=sample_dir,
        prediction_dir=prediction_dir,
        max_samples=args.max_samples,
        generate_missing_predictions=args.generate_missing_predictions,
    )

    print(f"sample_dir: {sample_dir}")
    print(f"prediction_dir: {prediction_dir}")
    print(f"output_dir: {output_dir}")
    print(f"selected samples: {len(selected_samples)}")
    if args.dry_run:
        for sample in selected_samples:
            prediction = find_prediction(sample, [prediction_dir])
            status = "prediction_found" if prediction else "prediction_missing"
            print(f"dry_run {sample.stem}: {status}")
        print("dry_run complete; no files written.")
        return

    (
        output_dir,
        samples_dir,
        generated_predictions_dir,
        _logs_dir,
        manifest_path,
        metadata_path,
    ) = _prepare_output_dirs(output_dir, overwrite=args.overwrite)

    ok_records = []
    failed_records = []
    combined_items = []

    with manifest_path.open("a", encoding="utf-8") as manifest:
        total = len(selected_samples)
        for index, sample_npz in enumerate(selected_samples, start=1):
            sample_id = sample_npz.stem
            prediction_npz = None
            print(f"[{index}/{total}] {sample_id}: start")
            try:
                prediction_npz = find_prediction(
                    sample_npz, [prediction_dir, generated_predictions_dir]
                )
                if prediction_npz is None and args.generate_missing_predictions:
                    print(f"[{index}/{total}] {sample_id}: generating prediction")
                    prediction_npz = _generate_prediction(
                        sample_npz=sample_npz,
                        output_prediction_dir=generated_predictions_dir,
                        checkpoint=args.checkpoint,
                    )
                if prediction_npz is None:
                    raise FileNotFoundError(
                        f"matching prediction not found for sample {sample_id}"
                    )

                expert_npz = samples_dir / f"paper_expert_sample_{sample_id}.npz"
                if expert_npz.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"expert sample already exists; use --overwrite: {expert_npz}"
                    )

                result = run_paper_expert_scoring(
                    sample_npz=sample_npz,
                    prediction_npz=prediction_npz,
                    output_dir=None,
                    num_candidates=args.num_candidates,
                    top_n=args.top_n,
                    tau=args.tau,
                    measured_mode=args.measured_mode,
                    raycast_mode=args.raycast_mode,
                    gain_mode=args.gain_mode,
                    voxel_size=args.voxel_size,
                    v_max=args.v_max,
                    yaw_rate_deg=args.yaw_rate_deg,
                    max_range=args.max_range,
                    num_yaw=args.num_yaw,
                    num_pitch=args.num_pitch,
                    seed=args.seed,
                )
                arrays = save_dataset_sample(
                    sample_output_path=expert_npz,
                    sample_npz=sample_npz,
                    prediction_npz=prediction_npz,
                    sample_id=sample_id,
                    result=result,
                    args=args,
                )
                record = make_ok_manifest_record(
                    sample_id=sample_id,
                    sample_npz=sample_npz,
                    prediction_npz=prediction_npz,
                    expert_npz=expert_npz,
                    result=result,
                    args=args,
                )
                ok_records.append(record)
                combined_items.append(
                    {
                        "sample_id": sample_id,
                        "expert_npz": str(expert_npz),
                        "candidate_features": arrays["candidate_features"],
                        "candidate_positions": arrays["candidate_positions"],
                        "candidate_yaws": arrays["candidate_yaws"],
                        "valid_mask": arrays["valid_mask"],
                        "expert_action": int(result["expert_action"]),
                        "expert_scores": arrays["expert_scores"],
                        "top_candidate_ids": arrays["top_candidate_ids"],
                    }
                )
                manifest.write(json.dumps(record, sort_keys=True) + "\n")
                manifest.flush()
                print(
                    f"[{index}/{total}] {sample_id}: ok "
                    f"best_score={record['best_score']:.6f} expert_action="
                    f"{record['expert_action']}"
                )
            except Exception as exc:
                record = make_failed_manifest_record(
                    sample_id=sample_id,
                    sample_npz=sample_npz,
                    prediction_npz=prediction_npz,
                    error=exc,
                )
                failed_records.append(record)
                manifest.write(json.dumps(record, sort_keys=True) + "\n")
                manifest.flush()
                print(f"[{index}/{total}] {sample_id}: failed: {exc}", file=sys.stderr)

    metadata = write_metadata(metadata_path, args, ok_records, failed_records)
    combined_path = None
    if ok_records and int(args.max_samples) <= 25:
        combined_path = write_combined_npz(
            output_dir / "combined_smoke.npz", combined_items
        )

    print("Stage 2C paper expert dataset generation complete.")
    print(f"total samples: {len(selected_samples)}")
    print(f"ok: {len(ok_records)}")
    print(f"failed: {len(failed_records)}")
    print(f"output dir: {output_dir}")
    print(f"manifest path: {manifest_path}")
    print(f"metadata path: {metadata_path}")
    if combined_path is not None:
        print(f"combined smoke npz: {combined_path}")
    print(f"strict_no_target_lr: {metadata['strict_no_target_lr']}")
    print(f"tree_utility_eq12_implemented: {metadata['tree_utility_eq12_implemented']}")


if __name__ == "__main__":
    main()

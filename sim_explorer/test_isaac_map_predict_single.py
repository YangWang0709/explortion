#!/usr/bin/env python3
"""Smoke-test Stage 4A-5 single-frame Isaac map_predict artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim_prediction_layer import SimPredictionLayer


DEFAULT_OUTPUT_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke"
UNKNOWN = np.int8(-1)
FORBIDDEN_NPZ_KEYS = {"target_lr", "target_hr", "ground_truth", "gt"}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_prob_array(name: str, array: np.ndarray) -> None:
    require(np.all(np.isfinite(array)), f"{name} contains non-finite values")
    require(float(array.min()) >= 0.0, f"{name} min < 0")
    require(float(array.max()) <= 1.0, f"{name} max > 1")


def assert_no_forbidden_npz_keys(path: Path) -> None:
    with np.load(path) as data:
        keys = set(data.files)
    forbidden = sorted(keys & FORBIDDEN_NPZ_KEYS)
    require(not forbidden, f"{path} contains forbidden keys: {forbidden}")


def run_test(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "prediction_alignment_summary.json"
    debug_path = output_dir / "sscnet_input_debug.npz"
    local_path = output_dir / "local_prediction.npz"
    global_path = output_dir / "global_prediction_layer.npz"
    required_paths = [summary_path, debug_path, local_path, global_path]
    required_paths.extend(
        output_dir / name
        for name in (
            "isaac_depth_input.png",
            "local_prediction_slices.png",
            "global_prediction_topdown.png",
            "observed_vs_prediction_topdown.png",
            "prediction_not_measured_topdown.png",
        )
    )
    for path in required_paths:
        require(path.is_file(), f"Missing expected artifact: {path}")
        require(path.stat().st_size > 0, f"Artifact is empty: {path}")

    summary = load_json(summary_path)
    depth_source = Path(summary["depth_source"])
    pose_source = Path(summary["pose_source"])
    observed_source = Path(summary["observed_state_source"])
    for path in (depth_source, pose_source, observed_source):
        require(path.is_file(), f"Selected input does not exist: {path}")
    require(summary["selected_step"] >= 0, "selected step must be non-negative")

    observed_hash_current = sha256_file(observed_source)
    require(summary["observed_state_sha256_before"] == summary["observed_state_sha256_after"], "observed_state hash changed during runner")
    require(observed_hash_current == summary["observed_state_sha256_before"], "observed_state hash differs from recorded hash")

    observed_state = np.load(observed_source)
    require(tuple(summary["observed_state_shape"]) == observed_state.shape, "summary observed_state shape mismatch")

    with np.load(debug_path) as debug:
        depth_input = np.array(debug["sscnet_depth_input"])
        position = np.array(debug["sscnet_position"])
        valid_position_mask = np.array(debug["valid_position_mask"])
    require(depth_input.shape == (480, 640), f"SSCNet depth input shape mismatch: {depth_input.shape}")
    require(position.shape == (480, 640), f"SSCNet position shape mismatch: {position.shape}")
    require(valid_position_mask.shape == (480, 640), "valid_position_mask shape mismatch")
    require(position.dtype.kind in ("i", "u"), f"position dtype must be integer, got {position.dtype}")
    require(int(position.min()) >= 0, "position min < 0")
    require(int(position.max()) < 240 * 144 * 240, "position index exceeds high-res SSC volume")
    require(int(np.count_nonzero(valid_position_mask)) == int(summary["valid_position_pixels"]), "valid position count mismatch")
    require(int(np.count_nonzero(valid_position_mask)) > 0, "valid position pixels must be > 0")

    with np.load(local_path) as local:
        pred_class = np.array(local["pred_class"])
        confidence = np.array(local["confidence"])
        free_prob = np.array(local["free_prob"])
        occupied_prob = np.array(local["occupied_prob"])
        logits_shape = tuple(int(v) for v in np.array(local["logits_shape"]).tolist())
    require(logits_shape == (1, 12, 60, 36, 60), f"logits shape mismatch: {logits_shape}")
    require(tuple(summary["logits_shape"]) == logits_shape, "summary logits shape mismatch")
    require(pred_class.shape == (60, 36, 60), f"local pred_class shape mismatch: {pred_class.shape}")
    require(confidence.shape == pred_class.shape, "local confidence shape mismatch")
    require(free_prob.shape == pred_class.shape, "local free_prob shape mismatch")
    require(occupied_prob.shape == pred_class.shape, "local occupied_prob shape mismatch")
    assert_prob_array("local confidence", confidence)
    assert_prob_array("local free_prob", free_prob)
    assert_prob_array("local occupied_prob", occupied_prob)

    with np.load(global_path) as global_layer:
        global_pred_class = np.array(global_layer["global_pred_class"])
        global_confidence = np.array(global_layer["global_confidence"])
        global_free_prob = np.array(global_layer["global_free_prob"])
        global_occupied_prob = np.array(global_layer["global_occupied_prob"])
        global_valid = np.array(global_layer["global_prediction_valid"])
    require(global_pred_class.shape == observed_state.shape, "global pred_class shape must match observed_state")
    require(global_confidence.shape == observed_state.shape, "global confidence shape must match observed_state")
    require(global_free_prob.shape == observed_state.shape, "global free_prob shape must match observed_state")
    require(global_occupied_prob.shape == observed_state.shape, "global occupied_prob shape must match observed_state")
    require(global_valid.shape == observed_state.shape, "global valid shape must match observed_state")
    assert_prob_array("global confidence", global_confidence)
    assert_prob_array("global free_prob", global_free_prob)
    assert_prob_array("global occupied_prob", global_occupied_prob)
    valid_count = int(np.count_nonzero(global_valid))
    require(valid_count > 0, "global valid prediction count must be > 0")
    require(valid_count == int(summary["global_valid_prediction_count"]), "global valid count mismatch")

    layer = SimPredictionLayer.from_npz(global_path)
    require(layer.shape() == observed_state.shape, "SimPredictionLayer shape mismatch")
    valid_indices = np.argwhere(global_valid)
    idx = tuple(int(v) for v in valid_indices[0])
    require(np.isfinite(layer.get_confidence(idx)), "SimPredictionLayer confidence not finite")
    require(0.0 <= layer.get_occupied_prob(idx) <= 1.0, "SimPredictionLayer occupied_prob outside [0,1]")
    require(0.0 <= layer.get_free_prob(idx) <= 1.0, "SimPredictionLayer free_prob outside [0,1]")
    _ = layer.is_predicted(idx, tau=float(summary["tau"]))
    _ = layer.is_predicted_occupied(idx, tau=float(summary["tau"]))
    _ = layer.is_predicted_free(idx, tau=float(summary["tau"]))
    _ = layer.get_prediction_gain(idx, tau=float(summary["tau"]))

    valid_tau = global_valid & (global_confidence >= float(summary["tau"]))
    predicted_unmeasured = valid_tau & (observed_state == UNKNOWN)
    require(
        int(np.count_nonzero(predicted_unmeasured)) == int(summary["predicted_unmeasured_count"]),
        "predicted_unmeasured count mismatch",
    )

    for path in (debug_path, local_path, global_path):
        assert_no_forbidden_npz_keys(path)

    require(summary["strict_no_observed_write"] is True, "strict_no_observed_write must be true")
    require(summary["prediction_written_to_observed_state"] is False, "prediction writeback must be false")
    require(summary["prediction_used_for_collision_or_traversability"] is False, "prediction must not be used for traversability")
    require(summary["rl_or_ppo_training"] is False, "RL/PPO training flag must be false")
    require(summary["optimizer_step"] is False, "optimizer_step flag must be false")
    require(summary["behavior_cloning_training"] is False, "BC training flag must be false")
    require(summary["imitation_learning_training"] is False, "IL training flag must be false")
    require(summary["sscnet_training"] is False, "SSCNet training flag must be false")
    require(summary["expert_used_prediction"] is False, "expert_used_prediction must be false")
    require(summary["rollout_used_prediction"] is False, "rollout_used_prediction must be false")
    require(summary["expert_or_rollout_invoked"] is False, "expert/rollout must not be invoked")

    result = {
        "output_dir": str(output_dir),
        "episode": summary["selected_episode_id"],
        "step": int(summary["selected_step"]),
        "observed_state_modified": False,
        "depth_input_shape": list(depth_input.shape),
        "position_shape": list(position.shape),
        "logits_shape": list(logits_shape),
        "local_prediction_shape": list(pred_class.shape),
        "global_prediction_shape": list(global_pred_class.shape),
        "valid_global_prediction_count": valid_count,
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "sim_prediction_layer_api": "ok",
        "leakage_checks": "ok",
        "rl_or_optimizer_run": False,
        "expert_or_rollout_used_prediction": False,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-5 single-frame map_predict smoke outputs.")
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_test(args.output_dir.resolve())
    print("Stage 4A-5 single-frame map_predict smoke test passed")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


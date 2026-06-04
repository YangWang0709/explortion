#!/usr/bin/env python3
"""Smoke test for Stage 4A-2 simulator observed-map expert scorer."""

from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path

import numpy as np

from run_sim_expert_step import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, run_expert_step
from sim_paper_expert import (
    FEATURE_NAMES,
    FREE,
    OCCUPIED,
    UNKNOWN,
    detect_frontier_adjacent_free_voxels,
    detect_frontier_voxels,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: str | Path) -> None:
    path = Path(path)
    assert path.exists(), f"missing output file: {path}"
    assert path.stat().st_size > 0, f"empty output file: {path}"


def main() -> None:
    observed_state_path = DEFAULT_INPUT_DIR / "observed_state_step2.npy"
    before_hash = sha256_file(observed_state_path)
    before_state = np.load(observed_state_path)

    assert before_state.shape == (80, 80, 30), before_state.shape
    values = set(int(v) for v in np.unique(before_state))
    assert int(UNKNOWN) in values, values
    assert int(FREE) in values, values
    assert int(OCCUPIED) in values, values

    frontier_voxels = detect_frontier_voxels(before_state)
    adjacent_free = detect_frontier_adjacent_free_voxels(before_state)
    assert len(frontier_voxels) > 0, "frontier_count must be > 0"
    assert len(adjacent_free) > 0, "frontier_adjacent_free_count must be > 0"

    args = Namespace(
        observed_state=str(observed_state_path),
        observed_summary=str(DEFAULT_INPUT_DIR / "observed_summary.json"),
        camera_info=str(DEFAULT_INPUT_DIR / "camera_info.json"),
        pose_json=str(DEFAULT_INPUT_DIR / "pose_002.json"),
        output_dir=str(DEFAULT_OUTPUT_DIR),
        num_candidates=64,
        top_n=16,
        gain_mode="hybrid",
        prediction_mode="empty",
        path_cost_mode="euclidean",
        candidate_sampling_mode="auto",
        snap_start_to_traversable=False,
        max_snap_radius_cells=5,
        seed=0,
        max_range_voxels=50,
        num_yaw=32,
        num_pitch=7,
        fov_yaw_deg=90.0,
        fov_pitch_deg=60.0,
        save_viz=True,
    )
    result = run_expert_step(args)

    assert len(result["all_candidates"]) > 0, "candidate_count must be > 0"
    for candidate in result["all_candidates"]:
        assert before_state[candidate.grid_position] == FREE, f"candidate not in FREE voxel: {candidate}"
    assert any(candidate.visible_count > 0 for candidate in result["all_candidates"]), "raycast returned no visible voxels"

    for candidate in result["all_candidates"]:
        assert candidate.gain_sc == 0.0, candidate
        assert candidate.gain_occ == 0.0, candidate
        assert candidate.gain_conf == 0.0, candidate
        assert candidate.gain_hybrid == candidate.gain_exp, candidate
        assert np.isfinite(candidate.final_score), candidate

    expert_action = int(result["expert_action"])
    assert 0 <= expert_action < len(result["top_candidates"]), expert_action

    paths = result["output_paths"]
    for key in ("npz", "json", "jsonl", "topdown", "score_bar"):
        require_file(paths[key])

    with np.load(paths["npz"], allow_pickle=False) as data:
        required_fields = {
            "candidate_features",
            "feature_names",
            "candidate_positions_grid",
            "candidate_positions_world",
            "candidate_yaws",
            "valid_mask",
            "expert_action",
            "expert_scores",
            "current_pose_world",
            "current_pose_grid",
            "observed_state_path",
            "prediction_mode",
            "gain_mode",
            "path_cost_mode",
            "raycast_mode",
            "strict_no_prediction_write_note",
        }
        assert required_fields.issubset(set(data.files)), data.files
        feature_names = [str(v) for v in data["feature_names"]]
        assert feature_names == FEATURE_NAMES, feature_names
        assert data["candidate_features"].shape == (16, len(FEATURE_NAMES))
        assert int(data["expert_action"]) == expert_action

    jsonl_count = sum(1 for line in Path(paths["jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip())
    assert jsonl_count == len(result["all_candidates"]), (jsonl_count, len(result["all_candidates"]))

    diagnostics = result["diagnostics"]
    assert diagnostics["prediction_written_to_observed_state"] is False
    assert diagnostics["target_lr_used"] is False
    assert diagnostics["target_hr_used"] is False
    assert diagnostics["ground_truth_used"] is False
    assert diagnostics["rl_or_training_used"] is False
    assert diagnostics["optimizer_used"] is False
    assert diagnostics["policy_training_used"] is False

    after_hash = sha256_file(observed_state_path)
    after_state = np.load(observed_state_path)
    assert after_hash == before_hash, "observed_state_step2.npy hash changed"
    assert np.array_equal(after_state, before_state), "observed_state_step2.npy contents changed"

    best = result["best_candidate"]
    print("Stage 4A-2 simulator expert smoke test passed.")
    print(f"observed_state: {observed_state_path}")
    print(f"shape: {before_state.shape}")
    print(
        "counts: "
        f"unknown={int(np.count_nonzero(before_state == UNKNOWN))} "
        f"free={int(np.count_nonzero(before_state == FREE))} "
        f"occupied={int(np.count_nonzero(before_state == OCCUPIED))}"
    )
    print(f"frontier_count: {len(frontier_voxels)}")
    print(f"frontier_adjacent_free_count: {len(adjacent_free)}")
    print(f"candidate_count: {len(result['all_candidates'])}")
    print(f"expert_action: {expert_action}")
    print(
        "best: "
        f"id={best.id} score={best.final_score:.6f} "
        f"gain_exp={best.gain_exp:.1f} gain_sc={best.gain_sc:.1f} "
        f"gain_hybrid={best.gain_hybrid:.1f} path_cost={best.path_cost:.6f} "
        f"grid={best.grid_position} world={best.world_position} yaw={best.yaw:.6f}"
    )
    print(f"npz: {paths['npz']}")
    print(f"json: {paths['json']}")
    print(f"jsonl: {paths['jsonl']}")
    print(f"topdown: {paths['topdown']}")
    print(f"score_bar: {paths['score_bar']}")
    print("observed_state_modified: no")
    print("rl_or_optimizer_or_policy_training_run: no")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Smoke tests for strict paper-faithful Stage 2B expert scoring."""

import tempfile
from pathlib import Path

import numpy as np

from prediction_layer import PredictionLayer
from sc_explorer_paper_expert import (
    CandidateView,
    build_measured_mask_from_sensor_npz,
    build_prediction_set_mask,
    compute_candidate_cost_and_utilities,
    compute_paper_gains,
    evaluate_candidates,
    raycast_visible_voxels,
    run_paper_expert_scoring,
    set_candidate_final_score,
)


SAMPLE_NPZ = Path(
    "/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/"
    "NYUtest_npz/NYU0670_0000_voxels.npz"
)
PREDICTION_NPZ = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference/"
    "NYU0670_0000_voxels_prediction.npz"
)


def _make_prediction_layer(shape):
    pred_class = np.zeros(shape, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)
    occupied_prob = np.zeros(shape, dtype=np.float32)
    free_prob = np.ones(shape, dtype=np.float32)
    return PredictionLayer(
        pred_class=pred_class,
        confidence=confidence,
        occupied_prob=occupied_prob,
        free_prob=free_prob,
    )


def test_gain_formulas():
    shape = (4, 4, 4)
    measured_mask = np.zeros(shape, dtype=bool)
    measured_mask[0, 0, 0] = True

    layer = _make_prediction_layer(shape)
    layer.confidence[1, 0, 0] = 0.9
    layer.occupied_prob[1, 0, 0] = 0.8
    layer.free_prob[1, 0, 0] = 0.2
    layer.confidence[2, 0, 0] = 0.9
    layer.occupied_prob[2, 0, 0] = 0.2
    layer.free_prob[2, 0, 0] = 0.8
    layer.confidence[3, 0, 0] = 0.05
    layer.occupied_prob[3, 0, 0] = 0.9
    layer.free_prob[3, 0, 0] = 0.1

    visible = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
    gains = compute_paper_gains(visible, measured_mask, layer, tau=0.1)
    assert gains["gain_exp"] == 3.0
    assert gains["gain_sc"] == 2.0
    assert gains["gain_hybrid"] == 5.0
    assert gains["gain_occ"] == 1.0
    assert np.isclose(gains["gain_conf"], 0.6)
    assert gains["visible_count"] == 4
    assert gains["measured_visible_count"] == 1
    assert gains["predicted_unmeasured_visible_count"] == 2

    p_mask = build_prediction_set_mask(layer, measured_mask, tau=0.1)
    assert p_mask[1, 0, 0]
    assert p_mask[2, 0, 0]
    assert not p_mask[0, 0, 0]
    assert not p_mask[3, 0, 0]


def test_raycast_modes():
    shape = (10, 10, 10)
    measured_mask = np.zeros(shape, dtype=bool)
    measured_mask[2, 5, 5] = True
    layer = _make_prediction_layer(shape)
    layer.confidence[3, 5, 5] = 1.0
    layer.occupied_prob[3, 5, 5] = 1.0
    layer.free_prob[3, 5, 5] = 0.0

    candidate = CandidateView(id=0, position=(2, 5, 5), yaw=0.0)
    non_blocking = raycast_visible_voxels(
        candidate,
        measured_mask,
        layer,
        measured_occupied_mask=None,
        mode="non_blocking",
        max_range=5,
        num_yaw=1,
        num_pitch=1,
        tau=0.1,
    )
    sc_blocking = raycast_visible_voxels(
        candidate,
        measured_mask,
        layer,
        measured_occupied_mask=None,
        mode="sc_blocking",
        max_range=5,
        num_yaw=1,
        num_pitch=1,
        tau=0.1,
    )
    assert (3, 5, 5) in non_blocking
    assert (6, 5, 5) in non_blocking
    assert (3, 5, 5) in sc_blocking
    assert (6, 5, 5) not in sc_blocking


def test_cost_and_utility_are_finite():
    candidate = CandidateView(id=0, position=(1, 0, 0), yaw=np.pi / 2)
    candidate.gain_exp = 4.0
    candidate.gain_sc = 2.0
    candidate.gain_hybrid = 6.0
    candidate.gain_occ = 1.0
    candidate.gain_conf = 0.5
    compute_candidate_cost_and_utilities(
        candidate,
        start_position=(0, 0, 0),
        voxel_size=0.08,
        v_max=1.0,
        yaw_rate=np.pi / 2,
    )
    set_candidate_final_score(candidate, gain_mode="hybrid")
    assert candidate.path_cost > 0.0
    assert np.isfinite(candidate.utility_exp)
    assert np.isfinite(candidate.utility_sc)
    assert np.isfinite(candidate.utility_hybrid)
    assert np.isfinite(candidate.utility_occ)
    assert np.isfinite(candidate.utility_conf)
    assert np.isfinite(candidate.final_score)


def test_measured_mask_and_scoring_do_not_need_target_lr(prediction_layer):
    with np.load(SAMPLE_NPZ) as sample:
        tsdf_lr = np.array(sample["tsdf_lr"])
        position = np.array(sample["position"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        no_target_npz = Path(tmp_dir) / "sensor_only.npz"
        target_changed_npz = Path(tmp_dir) / "sensor_with_changed_target.npz"
        np.savez_compressed(no_target_npz, tsdf_lr=tsdf_lr, position=position)
        np.savez_compressed(
            target_changed_npz,
            tsdf_lr=tsdf_lr,
            position=position,
            target_lr=np.full(tsdf_lr.shape, 255, dtype=np.uint8),
            target_hr=np.zeros((1,), dtype=np.uint8),
        )

        mask_without_target = build_measured_mask_from_sensor_npz(
            no_target_npz, mode="tsdf_lr"
        )
        mask_with_changed_target = build_measured_mask_from_sensor_npz(
            target_changed_npz, mode="tsdf_lr"
        )
        assert np.array_equal(mask_without_target, mask_with_changed_target)
        assert mask_without_target.shape == prediction_layer.shape()

        result = run_paper_expert_scoring(
            sample_npz=no_target_npz,
            prediction_npz=PREDICTION_NPZ,
            output_dir=Path(tmp_dir) / "paper_output",
            num_candidates=4,
            top_n=2,
            tau=0.1,
            measured_mode="tsdf_lr",
            raycast_mode="non_blocking",
            gain_mode="hybrid",
            max_range=8,
            num_yaw=4,
            num_pitch=3,
            seed=7,
        )
        assert 0 <= result["expert_action"] < len(result["top_candidates"])
        assert np.isfinite(result["best_candidate"].final_score)

        output_npz = Path(result["npz_path"])
        output_jsonl = Path(result["jsonl_path"])
        assert output_npz.is_file()
        assert output_jsonl.is_file()
        with np.load(output_npz) as saved:
            required = (
                "candidate_features",
                "candidate_positions",
                "candidate_yaws",
                "valid_mask",
                "expert_action",
                "expert_scores",
                "gain_mode",
                "measured_mode",
                "raycast_mode",
                "sample_npz",
                "prediction_npz",
            )
            missing = [key for key in required if key not in saved.files]
            assert not missing, missing


def main():
    if not SAMPLE_NPZ.is_file():
        raise FileNotFoundError(f"Missing sample fixture: {SAMPLE_NPZ}")
    if not PREDICTION_NPZ.is_file():
        raise FileNotFoundError(f"Missing prediction fixture: {PREDICTION_NPZ}")

    prediction_layer = PredictionLayer.from_npz(PREDICTION_NPZ)
    print(f"loaded prediction: {PREDICTION_NPZ}")
    print(f"prediction shape: {prediction_layer.shape()}")

    with np.load(SAMPLE_NPZ) as sample:
        print(f"sample npz loaded: {SAMPLE_NPZ}")
        print(f"sample fields: {sample.files}")

    measured_mask = build_measured_mask_from_sensor_npz(SAMPLE_NPZ, mode="tsdf_lr")
    print(
        "measured mask built without target_lr: "
        f"shape={measured_mask.shape} measured={int(np.count_nonzero(measured_mask))} "
        f"unmeasured={int(measured_mask.size - np.count_nonzero(measured_mask))}"
    )
    assert measured_mask.shape == prediction_layer.shape()

    test_gain_formulas()
    print("gain formulas: passed")
    test_raycast_modes()
    print("raycast modes: passed")
    test_cost_and_utility_are_finite()
    print("cost/utilities: passed")
    test_measured_mask_and_scoring_do_not_need_target_lr(prediction_layer)
    print("target_lr not needed by measured mask/scoring path: passed")

    result = evaluate_candidates(
        measured_mask=measured_mask,
        prediction_layer=prediction_layer,
        measured_occupied_mask=None,
        num_candidates=8,
        top_n=4,
        tau=0.1,
        raycast_mode="non_blocking",
        gain_mode="hybrid",
        max_range=8,
        num_yaw=4,
        num_pitch=3,
        seed=3,
    )
    assert 0 <= result["expert_action"] < len(result["top_candidates"])
    assert np.isfinite(result["best_candidate"].final_score)
    print(
        "expert action: "
        f"{result['expert_action']} best_score={result['best_candidate'].final_score:.6f}"
    )

    print("Paper expert smoke test passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Smoke test for offline SSCNet prediction output and PredictionLayer."""

from pathlib import Path

import numpy as np

from offline_infer_npz import DEFAULT_CHECKPOINT, run_inference
from prediction_layer import PredictionLayer


TEST_ROOT = Path("/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/NYUtest_npz")
OUTPUT_DIR = Path("/home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference_smoke")


def first_test_npz():
    files = sorted(TEST_ROOT.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {TEST_ROOT}")
    return files[0]


def main():
    input_npz = first_test_npz()
    stats = run_inference(
        checkpoint=DEFAULT_CHECKPOINT,
        input_npz=input_npz,
        output_dir=OUTPUT_DIR,
        save_probs=False,
        save_logits=False,
        print_stats=True,
    )

    layer = PredictionLayer.from_npz(stats["output_path"])
    print(f"layer shape: {layer.shape()}")

    rng = np.random.default_rng(2019)
    shape = layer.shape()
    for idx in range(10):
        voxel = tuple(int(rng.integers(0, high)) for high in shape)
        print(
            f"voxel {idx}: index={voxel}, "
            f"pred_class={layer.get_pred_class(voxel)}, "
            f"confidence={layer.get_confidence(voxel):.6f}, "
            f"occupied_prob={layer.get_occupied_prob(voxel):.6f}, "
            f"free_prob={layer.get_free_prob(voxel):.6f}"
        )

    predicted_count = int(np.count_nonzero(layer.confidence >= 0.1))
    occupied_count = int(np.count_nonzero(layer.occupied_prob >= 0.5))
    free_count = int(np.count_nonzero(layer.free_prob >= 0.5))
    print(f"confidence >= 0.1 voxels: {predicted_count}")
    print(f"occupied_prob >= 0.5 voxels: {occupied_count}")
    print(f"free_prob >= 0.5 voxels: {free_count}")

    assert layer.pred_class.shape == layer.confidence.shape
    assert layer.pred_class.shape == layer.occupied_prob.shape
    assert layer.pred_class.shape == layer.free_prob.shape
    assert np.min(layer.confidence) >= 0.0 and np.max(layer.confidence) <= 1.0
    assert np.min(layer.occupied_prob) >= 0.0 and np.max(layer.occupied_prob) <= 1.0
    assert np.min(layer.free_prob) >= 0.0 and np.max(layer.free_prob) <= 1.0
    assert np.min(layer.pred_class) >= 0 and np.max(layer.pred_class) <= 11
    assert predicted_count > 0
    print("PredictionLayer smoke test passed.")


if __name__ == "__main__":
    main()

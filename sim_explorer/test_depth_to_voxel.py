#!/usr/bin/env python3
"""Pure Python smoke tests for depth_to_voxel.py."""

from __future__ import annotations

import inspect

import numpy as np

from depth_to_voxel import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    create_observed_grid,
    integrate_depth_frame,
    summarize_observed_grid,
)


def _synthetic_inputs(yaw_deg: float = 0.0):
    depth = np.full((12, 16), 1.6, dtype=np.float32)
    depth[0:2, :] = 2.6
    pose = {"position": [0.0, 0.0, 1.0], "yaw_deg": yaw_deg}
    intrinsics = {"fx": 10.0, "fy": 10.0, "cx": 7.5, "cy": 5.5, "max_depth": 3.0}
    map_bounds = {"x": [-1.0, 3.0], "y": [-1.5, 3.0], "z": [0.0, 2.0]}
    return depth, pose, intrinsics, map_bounds


def test_single_frame_updates_observed_map() -> None:
    depth, pose, intrinsics, map_bounds = _synthetic_inputs()
    observed = create_observed_grid(map_bounds, voxel_size=0.2)
    observed = integrate_depth_frame(
        observed,
        depth,
        pose,
        intrinsics,
        map_bounds=map_bounds,
        voxel_size=0.2,
        pixel_stride=2,
    )
    summary = summarize_observed_grid(observed)

    assert observed.shape == (20, 23, 10), observed.shape
    assert summary["free_count"] > 0, summary
    assert summary["occupied_count"] > 0, summary
    assert summary["unknown_count"] > 0, summary
    assert set(np.unique(observed).tolist()).issubset({int(UNKNOWN), int(FREE), int(OCCUPIED)})


def test_multi_frame_fusion_does_not_reduce_observed_count() -> None:
    depth0, pose0, intrinsics, map_bounds = _synthetic_inputs(yaw_deg=0.0)
    depth1, pose1, _, _ = _synthetic_inputs(yaw_deg=90.0)
    observed = create_observed_grid(map_bounds, voxel_size=0.2)
    observed = integrate_depth_frame(
        observed,
        depth0,
        pose0,
        intrinsics,
        map_bounds=map_bounds,
        voxel_size=0.2,
        pixel_stride=2,
    )
    observed_after_first = summarize_observed_grid(observed)["observed_count"]
    observed = integrate_depth_frame(
        observed,
        depth1,
        pose1,
        intrinsics,
        map_bounds=map_bounds,
        voxel_size=0.2,
        pixel_stride=2,
    )
    observed_after_second = summarize_observed_grid(observed)["observed_count"]
    assert observed_after_second >= observed_after_first


def test_no_prediction_interface_or_values() -> None:
    signature = inspect.signature(integrate_depth_frame)
    forbidden = {"prediction", "predicted", "target_lr", "target_hr", "ground_truth", "gt"}
    assert forbidden.isdisjoint(signature.parameters)

    depth, pose, intrinsics, map_bounds = _synthetic_inputs()
    observed = create_observed_grid(map_bounds, voxel_size=0.2)
    observed = integrate_depth_frame(
        observed,
        depth,
        pose,
        intrinsics,
        map_bounds=map_bounds,
        voxel_size=0.2,
        pixel_stride=4,
    )
    assert set(np.unique(observed).tolist()).issubset({int(UNKNOWN), int(FREE), int(OCCUPIED)})


def main() -> None:
    test_single_frame_updates_observed_map()
    test_multi_frame_fusion_does_not_reduce_observed_count()
    test_no_prediction_interface_or_values()
    print("depth_to_voxel tests passed")


if __name__ == "__main__":
    main()

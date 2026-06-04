#!/usr/bin/env python3
"""Preprocess one Isaac depth frame into SSCNet depth and position tensors.

This is a Stage 4A-5 smoke-only bridge from Isaac `distance_to_image_plane`
depth to the PALNet/SSCNet input shape used by the existing checkpoint. It
does not use targets, simulator ground truth, expert decisions, rollouts, or
prediction writeback.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:  # OpenCV is available in env_isaaclab, but keep a tiny fallback.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - fallback only
    cv2 = None


DEFAULT_OUTPUT_HW = (480, 640)
DEFAULT_HIGHRES_DIMS = (240, 144, 240)
DEFAULT_HIGHRES_VOXEL_SIZE = 0.02
DEFAULT_LOCAL_VOLUME_M = (4.8, 2.88, 4.8)
DEFAULT_ALIGNMENT_CONVENTION = "current_default_v0"
POSITION_AXIS_ORDER = "flat high-res np.ravel_multi_index((x_right_idx,y_up_idx,z_forward_idx),(240,144,240))"
SSCNET_OUTPUT_AXIS_ORDER = "network output arrays are (z_forward,y_up,x_right) under current_default_v0"
DOMAIN_SHIFT_NOTE = (
    "Smoke-only Isaac-to-SSCNet preprocessing. Isaac scripted depth, camera "
    "intrinsics, and local volume convention are not guaranteed to match the "
    "original NYU preprocessing distribution."
)


ALIGNMENT_CONVENTIONS: dict[str, dict[str, Any]] = {
    "current_default_v0": {
        "name": "current_default_v0",
        "description": "Stage 4A-5/6 behavior: Isaac position is flattened as (x_right,y_up,z_forward); Project2Dto3D then yields output axes (z_forward,y_up,x_right).",
        "input_position_flatten_order": ("x_right", "y_up", "z_forward"),
        "output_axis_order": ("z_forward", "y_up", "x_right"),
        "x_right_sign": 1,
        "y_up_sign": 1,
        "z_forward_sign": 1,
        "local_origin": "x_right and y_up centered; z_forward starts at camera plane",
        "voxel_center_formula": {
            "x_right": "-volume_x/2 + (idx + 0.5) * voxel_size",
            "y_up": "-volume_y/2 + (idx + 0.5) * voxel_size",
            "z_forward": "(idx + 0.5) * voxel_size",
        },
        "diagnostic_only": False,
    },
    "xz_swap_diagnostic": {
        "name": "xz_swap_diagnostic",
        "description": "Stage 4A-6.2 diagnostic reprojection: reuse old local predictions but interpret output axes as (x_right,y_up,z_forward).",
        "input_position_flatten_order": ("x_right", "y_up", "z_forward"),
        "output_axis_order": ("x_right", "y_up", "z_forward"),
        "x_right_sign": 1,
        "y_up_sign": 1,
        "z_forward_sign": 1,
        "local_origin": "x_right and y_up centered; z_forward starts at camera plane",
        "voxel_center_formula": {
            "x_right": "-volume_x/2 + (idx + 0.5) * voxel_size",
            "y_up": "-volume_y/2 + (idx + 0.5) * voxel_size",
            "z_forward": "(idx + 0.5) * voxel_size",
        },
        "diagnostic_only": True,
    },
    "code_consistent_v1": {
        "name": "code_consistent_v1",
        "description": "Projection-layer/C++-consistent convention: position flatten follows voxel_util.cpp z*W*H + y*W + x; after Project2Dto3D view+permute, network output axes are (x_right,y_up,z_forward).",
        "input_position_flatten_order": ("z_forward", "y_up", "x_right"),
        "output_axis_order": ("x_right", "y_up", "z_forward"),
        "x_right_sign": 1,
        "y_up_sign": 1,
        "z_forward_sign": 1,
        "local_origin": "x_right and y_up centered; z_forward starts at camera plane",
        "voxel_center_formula": {
            "x_right": "-volume_x/2 + (idx + 0.5) * voxel_size",
            "y_up": "-volume_y/2 + (idx + 0.5) * voxel_size",
            "z_forward": "(idx + 0.5) * voxel_size",
        },
        "diagnostic_only": False,
    },
}

ALIGNMENT_CONVENTION_ALIASES = {
    "current_default": "current_default_v0",
    "default_v0": "current_default_v0",
    "xz_swap_variant": "xz_swap_diagnostic",
    "xz_swap_fixed_v1": "code_consistent_v1",
    "projection_layer_consistent_v1": "code_consistent_v1",
}


def canonical_alignment_convention(convention: str | None) -> str:
    name = str(convention or DEFAULT_ALIGNMENT_CONVENTION)
    name = ALIGNMENT_CONVENTION_ALIASES.get(name, name)
    if name not in ALIGNMENT_CONVENTIONS:
        valid = sorted(ALIGNMENT_CONVENTIONS)
        raise ValueError(f"Unknown alignment convention {convention!r}; valid: {valid}")
    return name


def alignment_convention_metadata(convention: str | None) -> dict[str, Any]:
    name = canonical_alignment_convention(convention)
    data = dict(ALIGNMENT_CONVENTIONS[name])
    data["input_position_flatten_order"] = list(data["input_position_flatten_order"])
    data["output_axis_order"] = list(data["output_axis_order"])
    data["canonical_name"] = name
    return data


def _axis_center(
    axis_name: str,
    index_values: np.ndarray,
    local_volume_m: tuple[float, float, float],
    voxel_size: float,
) -> np.ndarray:
    if axis_name == "x_right":
        return -0.5 * float(local_volume_m[0]) + (index_values + 0.5) * float(voxel_size)
    if axis_name == "y_up":
        return -0.5 * float(local_volume_m[1]) + (index_values + 0.5) * float(voxel_size)
    if axis_name == "z_forward":
        return (index_values + 0.5) * float(voxel_size)
    raise ValueError(f"Unsupported local axis: {axis_name}")


def local_index_to_camera_coords(
    index: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[int, int, int],
    convention: str | None = DEFAULT_ALIGNMENT_CONVENTION,
    voxel_size: float = 0.08,
    local_volume_m: tuple[float, float, float] = DEFAULT_LOCAL_VOLUME_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert SSCNet output indices to local camera coordinates.

    Returns `(x_right, y_up, z_forward)` in meters. The input index order is
    always the array index order of the prediction volume; the named
    convention determines what semantic axis each array index controls.
    """

    name = canonical_alignment_convention(convention)
    axis_order = ALIGNMENT_CONVENTIONS[name]["output_axis_order"]
    arrays = [np.asarray(v, dtype=np.float32) for v in index]
    if len(arrays) != 3:
        raise ValueError("index must contain three arrays/scalars")
    coords: dict[str, np.ndarray] = {}
    for axis_name, values in zip(axis_order, arrays):
        coords[axis_name] = _axis_center(axis_name, values, local_volume_m, float(voxel_size))
    for axis_name in ("x_right", "y_up", "z_forward"):
        if axis_name not in coords:
            raise ValueError(f"Convention {name} does not map an output axis to {axis_name}")
    return coords["x_right"], coords["y_up"], coords["z_forward"]


def local_index_grid_to_camera_coords(
    shape: tuple[int, int, int],
    convention: str | None = DEFAULT_ALIGNMENT_CONVENTION,
    voxel_size: float = 0.08,
    local_volume_m: tuple[float, float, float] = DEFAULT_LOCAL_VOLUME_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.indices(tuple(int(v) for v in shape), dtype=np.float32)
    return local_index_to_camera_coords(
        (idx[0], idx[1], idx[2]),
        convention=convention,
        voxel_size=float(voxel_size),
        local_volume_m=local_volume_m,
    )


def camera_coords_to_world(
    local_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    pose: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map local `(x_right,y_up,z_forward)` camera coordinates to world xyz."""

    if "position" not in pose:
        raise KeyError("pose JSON missing position")
    origin = np.asarray(pose["position"], dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError(f"pose position must have shape (3,), got {origin.shape}")
    if "yaw_rad" in pose:
        yaw = float(pose["yaw_rad"])
    elif "yaw_deg" in pose:
        yaw = math.radians(float(pose["yaw_deg"]))
    else:
        raise KeyError("pose JSON missing yaw_rad or yaw_deg")

    x_right, y_up, z_forward = local_xyz
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = origin[0] + cos_yaw * z_forward - sin_yaw * x_right
    world_y = origin[1] + sin_yaw * z_forward + cos_yaw * x_right
    world_z = origin[2] + y_up
    return world_x, world_y, world_z


def world_to_global_grid(
    world_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    bounds: dict[str, tuple[float, float]] | dict[str, list[float]],
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds_norm = {
        axis: (float(bounds[axis][0]), float(bounds[axis][1]))
        for axis in ("x", "y", "z")
    }
    world_x, world_y, world_z = world_xyz
    gx = np.floor((world_x - bounds_norm["x"][0]) / float(voxel_size)).astype(np.int64)
    gy = np.floor((world_y - bounds_norm["y"][0]) / float(voxel_size)).astype(np.int64)
    gz = np.floor((world_z - bounds_norm["z"][0]) / float(voxel_size)).astype(np.int64)
    return gx, gy, gz


def flatten_highres_position_indices(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    dims: tuple[int, int, int] = DEFAULT_HIGHRES_DIMS,
    convention: str | None = DEFAULT_ALIGNMENT_CONVENTION,
) -> np.ndarray:
    """Flatten high-res `(x_right,y_up,z_forward)` indices for Project2Dto3D."""

    name = canonical_alignment_convention(convention)
    order = tuple(ALIGNMENT_CONVENTIONS[name]["input_position_flatten_order"])
    dims = tuple(int(v) for v in dims)
    ix = np.asarray(ix, dtype=np.int64)
    iy = np.asarray(iy, dtype=np.int64)
    iz = np.asarray(iz, dtype=np.int64)
    if order == ("x_right", "y_up", "z_forward"):
        return np.ravel_multi_index((ix, iy, iz), dims).astype(np.int64)
    if order == ("z_forward", "y_up", "x_right"):
        if dims[0] != dims[2]:
            raise ValueError("z,y,x flatten convention assumes equal x/z dimensions used by SSCNet")
        return (iz * (dims[0] * dims[1]) + iy * dims[0] + ix).astype(np.int64)
    raise ValueError(f"Unsupported position flatten order for {name}: {order}")


def position_axis_order_note(convention: str | None = DEFAULT_ALIGNMENT_CONVENTION) -> str:
    name = canonical_alignment_convention(convention)
    order = tuple(ALIGNMENT_CONVENTIONS[name]["input_position_flatten_order"])
    if order == ("x_right", "y_up", "z_forward"):
        return "flat high-res np.ravel_multi_index((x_right_idx,y_up_idx,z_forward_idx),(240,144,240))"
    if order == ("z_forward", "y_up", "x_right"):
        return "flat high-res z_forward_idx*(240*144) + y_up_idx*240 + x_right_idx"
    return f"flat high-res order {order}"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _resize_array(array: np.ndarray, output_hw: tuple[int, int], interpolation: str) -> np.ndarray:
    out_h, out_w = output_hw
    if cv2 is not None:
        interp = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
        return cv2.resize(array, (out_w, out_h), interpolation=interp)

    in_h, in_w = array.shape
    scale_h = out_h / float(in_h)
    scale_w = out_w / float(in_w)
    if scale_h.is_integer() and scale_w.is_integer():
        resized = np.repeat(np.repeat(array, int(scale_h), axis=0), int(scale_w), axis=1)
        return resized[:out_h, :out_w]
    raise RuntimeError("cv2 is unavailable and non-integer resize was requested")


def _intrinsics_from_camera_info(
    camera_info: dict[str, Any],
    input_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> tuple[float, float, float, float]:
    in_h, in_w = input_hw
    out_h, out_w = output_hw

    if "intrinsic_matrix" in camera_info:
        matrix = np.asarray(camera_info["intrinsic_matrix"], dtype=np.float64)
        if matrix.shape == (3, 3):
            fx = float(matrix[0, 0])
            fy = float(matrix[1, 1])
            cx = float(matrix[0, 2])
            cy = float(matrix[1, 2])
        else:
            raise ValueError(f"intrinsic_matrix must be 3x3, got {matrix.shape}")
    elif all(key in camera_info for key in ("fx", "fy", "cx", "cy")):
        fx = float(camera_info["fx"])
        fy = float(camera_info["fy"])
        cx = float(camera_info["cx"])
        cy = float(camera_info["cy"])
    else:
        horizontal_fov_deg = float(camera_info.get("horizontal_fov_deg", 90.0))
        fx = (float(in_w) * 0.5) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
        fy = fx
        cx = float(in_w) * 0.5
        cy = float(in_h) * 0.5

    scale_x = float(out_w) / float(in_w)
    scale_y = float(out_h) / float(in_h)
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


def _validate_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Isaac depth must be HxW or HxWx1, got {depth.shape}")
    return depth


def _stats(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | None]:
    values = array[mask] if mask is not None else array.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def preprocess_isaac_depth_for_sscnet(
    depth: np.ndarray,
    camera_info: dict[str, Any],
    pose: dict[str, Any],
    output_hw: tuple[int, int] = DEFAULT_OUTPUT_HW,
    ssc_highres_dims: tuple[int, int, int] = DEFAULT_HIGHRES_DIMS,
    highres_voxel_size: float = DEFAULT_HIGHRES_VOXEL_SIZE,
    local_volume_m: tuple[float, float, float] = DEFAULT_LOCAL_VOLUME_M,
    alignment_convention: str | None = DEFAULT_ALIGNMENT_CONVENTION,
) -> dict[str, Any]:
    """Return SSCNet depth input and position map for one Isaac depth frame.

    Invalid or non-finite depth is set to 0 in the network depth input. The
    position tensor also maps invalid/out-of-volume pixels to 0, matching the
    existing NYU preprocessing behavior, while `valid_position_mask` records
    which pixels actually landed inside the local SSC volume.
    """

    convention = canonical_alignment_convention(alignment_convention)
    convention_meta = alignment_convention_metadata(convention)
    depth_raw = _validate_depth(depth)
    input_hw = tuple(int(v) for v in depth_raw.shape)
    output_hw = tuple(int(v) for v in output_hw)
    if output_hw != DEFAULT_OUTPUT_HW:
        raise ValueError(f"SSCNet checkpoint expects output_hw {DEFAULT_OUTPUT_HW}, got {output_hw}")

    valid_depth_raw = np.isfinite(depth_raw) & (depth_raw > 0.0)
    depth_clean = np.where(valid_depth_raw, depth_raw, 0.0).astype(np.float32)
    resized_depth = _resize_array(depth_clean, output_hw, interpolation="linear").astype(np.float32)
    resized_valid_depth = _resize_array(valid_depth_raw.astype(np.uint8), output_hw, interpolation="nearest") > 0
    resized_valid_depth &= np.isfinite(resized_depth) & (resized_depth > 0.0)

    fx, fy, cx, cy = _intrinsics_from_camera_info(camera_info, input_hw=input_hw, output_hw=output_hw)
    out_h, out_w = output_hw
    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))

    z_forward = resized_depth.astype(np.float32)
    x_right = ((uu - np.float32(cx)) * z_forward / np.float32(fx)).astype(np.float32)
    y_up = (-(vv - np.float32(cy)) * z_forward / np.float32(fy)).astype(np.float32)

    volume_x, volume_y, volume_z = (float(v) for v in local_volume_m)
    x_min = -0.5 * volume_x
    y_min = -0.5 * volume_y
    z_min = 0.0

    valid_position_mask = (
        resized_valid_depth
        & (x_right >= x_min)
        & (x_right < x_min + volume_x)
        & (y_up >= y_min)
        & (y_up < y_min + volume_y)
        & (z_forward >= z_min)
        & (z_forward < z_min + volume_z)
    )

    ix = np.floor((x_right - x_min) / float(highres_voxel_size)).astype(np.int64)
    iy = np.floor((y_up - y_min) / float(highres_voxel_size)).astype(np.int64)
    iz = np.floor((z_forward - z_min) / float(highres_voxel_size)).astype(np.int64)

    dims = tuple(int(v) for v in ssc_highres_dims)
    valid_position_mask &= (
        (ix >= 0)
        & (ix < dims[0])
        & (iy >= 0)
        & (iy < dims[1])
        & (iz >= 0)
        & (iz < dims[2])
    )

    position = np.zeros(output_hw, dtype=np.int64)
    valid = valid_position_mask
    if np.any(valid):
        position[valid] = flatten_highres_position_indices(
            ix[valid],
            iy[valid],
            iz[valid],
            dims=dims,
            convention=convention,
        ).astype(np.int64)

    notes = {
        "invalid_depth_policy": "non-finite or non-positive Isaac depth is set to 0 in sscnet_depth_input",
        "position_invalid_policy": "invalid or out-of-volume pixels map to 0; valid_position_mask disambiguates",
        "position_axis_order": position_axis_order_note(convention),
        "sscnet_output_axis_order": ", ".join(convention_meta["output_axis_order"]),
        "alignment_convention": convention,
        "alignment_convention_metadata": convention_meta,
        "position_flatten_order": convention_meta["input_position_flatten_order"],
        "pixel_coordinate_note": "uses integer pixel coordinates like the existing NYU dataloader _depth2voxel",
        "local_volume_convention": (
            "x_right [-2.4,2.4] m, y_up [-1.44,1.44] m, "
            "z_forward [0,4.8] m, high-res voxel 0.02 m"
        ),
        "domain_shift_note": DOMAIN_SHIFT_NOTE,
    }

    stats = {
        "input_depth_shape": list(input_hw),
        "output_depth_shape": list(resized_depth.shape),
        "position_shape": list(position.shape),
        "position_dtype": str(position.dtype),
        "position_min": int(position.min()),
        "position_max": int(position.max()),
        "position_zero_count": int(np.count_nonzero(position == 0)),
        "valid_depth_pixels_raw": int(np.count_nonzero(valid_depth_raw)),
        "valid_depth_pixels_resized": int(np.count_nonzero(resized_valid_depth)),
        "valid_position_pixels": int(np.count_nonzero(valid_position_mask)),
        "depth_raw_stats": _stats(depth_raw, valid_depth_raw),
        "depth_resized_stats": _stats(resized_depth, resized_valid_depth),
        "intrinsics_output": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "highres_dims": list(dims),
        "highres_voxel_size": float(highres_voxel_size),
        "local_volume_m": list(local_volume_m),
        "pose_index": pose.get("index"),
        "pose_yaw_rad": pose.get("yaw_rad"),
        "alignment_convention": convention,
        "alignment_convention_metadata": convention_meta,
    }

    return {
        "sscnet_depth_input": resized_depth.astype(np.float32),
        "sscnet_position": position.astype(np.int64),
        "valid_position_mask": valid_position_mask.astype(bool),
        "original_depth": depth_raw.astype(np.float32),
        "resized_valid_depth_mask": resized_valid_depth.astype(bool),
        "local_points": {
            "x_right": x_right.astype(np.float32),
            "y_up": y_up.astype(np.float32),
            "z_forward": z_forward.astype(np.float32),
        },
        "stats": stats,
        "notes": notes,
    }


def save_preprocessing_debug(
    output_dir: str | Path,
    preprocess_result: dict[str, Any],
    depth_source: str | Path | None = None,
    pose_source: str | Path | None = None,
    camera_info_source: str | Path | None = None,
) -> dict[str, str]:
    """Save Stage 4A-5 preprocessing arrays and debug metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_npy = output_path / "sscnet_depth_input.npy"
    position_npy = output_path / "sscnet_position.npy"
    valid_mask_npy = output_path / "valid_position_mask.npy"
    debug_npz = output_path / "sscnet_input_debug.npz"

    np.save(depth_npy, preprocess_result["sscnet_depth_input"].astype(np.float32))
    np.save(position_npy, preprocess_result["sscnet_position"].astype(np.int64))
    np.save(valid_mask_npy, preprocess_result["valid_position_mask"].astype(bool))

    local_points = preprocess_result["local_points"]
    np.savez_compressed(
        debug_npz,
        sscnet_depth_input=preprocess_result["sscnet_depth_input"].astype(np.float32),
        sscnet_position=preprocess_result["sscnet_position"].astype(np.int64),
        valid_position_mask=preprocess_result["valid_position_mask"].astype(bool),
        resized_valid_depth_mask=preprocess_result["resized_valid_depth_mask"].astype(bool),
        original_depth=preprocess_result["original_depth"].astype(np.float32),
        local_x_right=local_points["x_right"].astype(np.float32),
        local_y_up=local_points["y_up"].astype(np.float32),
        local_z_forward=local_points["z_forward"].astype(np.float32),
        preprocessing_stats_json=json.dumps(preprocess_result["stats"], sort_keys=True),
        preprocessing_notes_json=json.dumps(preprocess_result["notes"], sort_keys=True),
        depth_source=str(depth_source) if depth_source else "",
        pose_source=str(pose_source) if pose_source else "",
        camera_info_source=str(camera_info_source) if camera_info_source else "",
    )

    return {
        "sscnet_depth_input": str(depth_npy),
        "sscnet_position": str(position_npy),
        "valid_position_mask": str(valid_mask_npy),
        "sscnet_input_debug": str(debug_npz),
    }

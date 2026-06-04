#!/usr/bin/env python3
"""Document SSCNet projection/output axis conventions for Isaac map_predict.

This script is audit-only. It reads source code and sample metadata, writes a
small report, and does not run training, planning, expert scoring, or Isaac.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from isaac_sscnet_preprocess import (
    ALIGNMENT_CONVENTIONS,
    alignment_convention_metadata,
    flatten_highres_position_indices,
)


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
SSC_ROOT = WORKSPACE / "ssc_exploration" / "ssc_network"
PROJECTION_LAYER = SSC_ROOT / "utils" / "projection_layer.py"
DATALOADER = SSC_ROOT / "dataloaders" / "dataloader.py"
MODEL = SSC_ROOT / "models" / "SSCNet.py"
OFFLINE_INFER = SSC_ROOT / "offline_infer_npz.py"
ROS_NODE = SSC_ROOT / "src" / "ssc_network_node.py"
VOXEL_CPP = SSC_ROOT / "voxel_utils" / "voxel_util.cpp"


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def contains(path: Path, needle: str) -> bool:
    return needle in read_text(path)


def _decode_flat(flat: int) -> dict[str, Any]:
    dims = (240, 144, 240)
    c_order = tuple(int(v) for v in np.unravel_index(int(flat), dims))
    zyx = {
        "z_forward_idx": int(flat // (dims[0] * dims[1])),
        "rem": int(flat % (dims[0] * dims[1])),
    }
    zyx["y_up_idx"] = int(zyx["rem"] // dims[0])
    zyx["x_right_idx"] = int(zyx["rem"] % dims[0])
    del zyx["rem"]
    return {
        "flat": int(flat),
        "decode_if_np_ravel_xyz": {
            "x_right_idx": c_order[0],
            "y_up_idx": c_order[1],
            "z_forward_idx": c_order[2],
        },
        "decode_if_cpp_zyx": zyx,
    }


def sample_position_examples(nyu_sample: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not nyu_sample.is_file():
        return {"available": False, "path": str(nyu_sample), "examples_csv": ""}

    with np.load(nyu_sample) as data:
        position = np.asarray(data["position"], dtype=np.int64)
        depth = np.asarray(data["depth"])
        if depth.ndim == 3:
            depth = depth[0]
        target_lr_shape = list(data["target_lr"].shape) if "target_lr" in data.files else None
        tsdf_lr_shape = list(data["tsdf_lr"].shape) if "tsdf_lr" in data.files else None

    h, w = position.shape
    probe_pixels = [
        (0, 0),
        (h // 2, w // 2),
        (h // 2, w // 4),
        (h // 2, 3 * w // 4),
        (h // 4, w // 2),
        (3 * h // 4, w // 2),
        (h - 1, w - 1),
    ]
    ys, xs = np.nonzero((position > 0) & (position < 240 * 144 * 240))
    for py, px in list(zip(ys[:8], xs[:8])):
        probe_pixels.append((int(py), int(px)))

    for py, px in probe_pixels:
        flat = int(position[py, px])
        decoded = _decode_flat(flat) if 0 <= flat < 240 * 144 * 240 else {}
        row = {
            "pixel_v": int(py),
            "pixel_u": int(px),
            "depth_m": float(depth[py, px]) if np.isfinite(depth[py, px]) else None,
            "flat_position": flat,
        }
        row.update({f"xyz_{k}": v for k, v in decoded.get("decode_if_np_ravel_xyz", {}).items()})
        row.update({f"cpp_{k}": v for k, v in decoded.get("decode_if_cpp_zyx", {}).items()})
        rows.append(row)

    csv_path = output_dir / "position_flatten_examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = position[(position > 0) & (position < 240 * 144 * 240)]
    return {
        "available": True,
        "path": str(nyu_sample),
        "position_shape": list(position.shape),
        "position_dtype": str(position.dtype),
        "position_nonzero_ratio": float(np.count_nonzero(position) / position.size),
        "position_valid_ratio": float(valid.size / position.size),
        "position_min": int(position.min()),
        "position_max": int(position.max()),
        "target_lr_shape": target_lr_shape,
        "target_lr_T_shape": list(reversed(target_lr_shape)) if target_lr_shape else None,
        "tsdf_lr_shape": tsdf_lr_shape,
        "examples_csv": str(csv_path),
    }


def synthetic_flatten_examples() -> list[dict[str, Any]]:
    dims = (240, 144, 240)
    examples = [(0, 0, 0), (4, 2, 7), (37, 18, 41), (239, 143, 239)]
    rows = []
    for ix, iy, iz in examples:
        current = int(flatten_highres_position_indices(np.array([ix]), np.array([iy]), np.array([iz]), dims, "current_default_v0")[0])
        code = int(flatten_highres_position_indices(np.array([ix]), np.array([iy]), np.array([iz]), dims, "code_consistent_v1")[0])
        rows.append(
            {
                "x_right_idx": ix,
                "y_up_idx": iy,
                "z_forward_idx": iz,
                "current_default_v0_flat": current,
                "code_consistent_v1_flat": code,
                "current_project2d_output_axis": "(z_forward,y_up,x_right)",
                "code_project2d_output_axis": "(x_right,y_up,z_forward)",
            }
        )
    return rows


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    nyu_sample = Path(args.nyu_sample).resolve()

    source_checks = {
        "projection_scatter_view_w_h_d": contains(PROJECTION_LAYER, "x3d.view(bs, c, self.w, self.h, self.d)"),
        "projection_permute_d_h_w": contains(PROJECTION_LAYER, "permute(0, 1, 4, 3, 2)"),
        "dataloader_target_lr_T": contains(DATALOADER, "target_lr.T"),
        "dataloader_python_np_ravel_xyz": contains(DATALOADER, "np.ravel_multi_index(point_grid[h, w, :], voxel_size)"),
        "model_logits_shape_comment": contains(MODEL, "(BS, 12L, 60L, 36L, 60L)"),
        "offline_infer_preserves_output_order": contains(OFFLINE_INFER, "pred_class_t.squeeze(0)"),
        "ros_node_uses_compute_tsdf_depth_mapping": contains(ROS_NODE, "depth_mapping_idxs"),
        "voxel_cpp_zyx_flatten": contains(VOXEL_CPP, "vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x"),
    }

    sample_info = sample_position_examples(nyu_sample, output_dir)
    synthetic_rows = synthetic_flatten_examples()
    synthetic_csv = output_dir / "synthetic_flatten_examples.csv"
    with synthetic_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(synthetic_rows[0].keys()))
        writer.writeheader()
        writer.writerows(synthetic_rows)

    conclusions = {
        "position_flattening": (
            "There are two source-code conventions. The raw Python dataloader branch uses "
            "np.ravel_multi_index((x,y,z),(240,144,240)), but the deployed C++/ROS projection "
            "path uses z*(240*144)+y*240+x. The repackaged .npz branch loads precomputed "
            "position and is therefore most consistent with the C++/ROS path used for online inference."
        ),
        "project2d_to_3d": (
            "Project2Dto3D scatters into a flat buffer, views it as (W,H,D), then permutes to (D,H,W). "
            "If flat indices encode (x,y,z), output axes are (z,y,x). If flat indices encode (z,y,x), "
            "output axes are (x,y,z)."
        ),
        "model_output": (
            "SSCNet returns logits (B,12,60,36,60). Axis semantics depend on the position-flattening "
            "path used to generate the scatter indices."
        ),
        "dataloader_T": (
            "target_lr.T reverses target axes before loss flattening. Under the C++/repackaged "
            "interpretation, stored target axes (z,y,x) become training target axes (x,y,z), matching "
            "Project2Dto3D output after z,y,x scatter."
        ),
        "stage4a5_assumption": (
            "Stage 4A-5 used the raw-Python interpretation: Isaac position flattened as (x,y,z), "
            "local prediction interpreted as (z_forward,y_up,x_right). That reproduces current outputs "
            "but is not the best match to the C++/ROS/repackaged SSCNet convention."
        ),
        "code_consistent_convention": "code_consistent_v1",
    }

    audit = {
        "stage": "Stage 4A-6.3 SSCNet axis convention audit",
        "output_dir": str(output_dir),
        "source_files": {
            "projection_layer": str(PROJECTION_LAYER),
            "dataloader": str(DATALOADER),
            "model": str(MODEL),
            "offline_infer": str(OFFLINE_INFER),
            "ros_node": str(ROS_NODE),
            "voxel_cpp": str(VOXEL_CPP),
        },
        "source_checks": source_checks,
        "nyu_sample": sample_info,
        "synthetic_flatten_examples_csv": str(synthetic_csv),
        "conventions": {
            name: alignment_convention_metadata(name)
            for name in ("current_default_v0", "xz_swap_diagnostic", "code_consistent_v1")
        },
        "answers": conclusions,
        "recommended_code_consistent_convention": "code_consistent_v1",
        "evaluation_only": True,
        "planning_or_training_used": False,
        "prediction_writeback": False,
        "future_observations_used_for_planning": False,
    }
    save_json(output_dir / "axis_convention_audit.json", audit)

    diagram = [
        "SSCNet projection axis diagram",
        "",
        "Raw tensor path:",
        "  x2d -> scatter flat index -> view(B,C,W,H,D) -> permute(B,C,D,H,W) -> Conv3d/output",
        "",
        "current_default_v0:",
        "  position flat = x*(H*D) + y*D + z",
        "  output index axes = (z_forward, y_up, x_right)",
        "",
        "code_consistent_v1:",
        "  position flat = z*(W*H) + y*W + x",
        "  output index axes = (x_right, y_up, z_forward)",
        "",
        "global map:",
        "  local x_right/y_up/z_forward -> yaw-only camera pose -> observed_state axes (world_x,world_y,world_z)",
    ]
    (output_dir / "axis_diagram_text.txt").write_text("\n".join(diagram) + "\n", encoding="utf-8")

    md = [
        "# Stage 4A-6.3 SSCNet Axis Convention Audit",
        "",
        "## Position Flattening",
        conclusions["position_flattening"],
        "",
        "## Project2Dto3D",
        conclusions["project2d_to_3d"],
        "",
        "## Model Output",
        conclusions["model_output"],
        "",
        "## Dataloader `.T`",
        conclusions["dataloader_T"],
        "",
        "## Stage 4A-5 Assumption",
        conclusions["stage4a5_assumption"],
        "",
        "## Code-Consistent Convention",
        "`code_consistent_v1`: input position flatten `(z_forward,y_up,x_right)`, output axes `(x_right,y_up,z_forward)`.",
        "",
        "## Safety",
        "Audit only. No planning, rollout scoring, training, checkpoint writes, observed_state writes, or prediction fusion.",
    ]
    (output_dir / "axis_convention_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Axis convention audit complete.")
    print(f"output_dir: {output_dir}")
    print("recommended_code_consistent_convention: code_consistent_v1")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyu_sample", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    run_audit(parse_args())


if __name__ == "__main__":
    main()

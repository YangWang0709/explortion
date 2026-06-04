#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""CLI for strict paper-faithful Stage 2B expert scoring."""

import argparse
from pathlib import Path

from sc_explorer_paper_expert import (
    MEASURED_NOTE,
    STRICT_NO_TARGET_NOTE,
    TREE_LIMITATION_NOTE,
    format_top_candidates,
    run_paper_expert_scoring,
)


DEFAULT_SAMPLE_NPZ = (
    "/home/ubuntu22/sc_explorer_ws/data/real_nyu_npz/"
    "NYUtest_npz/NYU0670_0000_voxels.npz"
)
DEFAULT_PREDICTION_NPZ = (
    "/home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference/"
    "NYU0670_0000_voxels_prediction.npz"
)
DEFAULT_OUTPUT_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/paper_expert"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict paper-faithful SC-Explorer expert candidate scorer"
    )
    parser.add_argument("--sample_npz", default=DEFAULT_SAMPLE_NPZ)
    parser.add_argument("--prediction_npz", default=DEFAULT_PREDICTION_NPZ)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--print_topn", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_paper_expert_scoring(
        sample_npz=args.sample_npz,
        prediction_npz=args.prediction_npz,
        output_dir=args.output_dir,
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

    best = result["best_candidate"]
    print("Stage 2B paper-faithful expert scoring complete.")
    print(f"sample_npz: {Path(args.sample_npz)}")
    print(f"prediction_npz: {Path(args.prediction_npz)}")
    print(f"measured_mode: {args.measured_mode}")
    print(f"raycast_mode: {args.raycast_mode}")
    print(f"gain_mode: {args.gain_mode}")
    print(f"measured_note: {MEASURED_NOTE}")
    print(f"strict_no_target_note: {STRICT_NO_TARGET_NOTE}")
    print(f"tree_limitation_note: {TREE_LIMITATION_NOTE}")
    print(f"measured_summary: {result['measured_summary']}")
    print(f"prediction_summary: {result['prediction_summary']}")
    print(f"candidate_count: {len(result['candidates'])}")
    print(f"expert_action: {result['expert_action']}")
    print(
        "best_candidate: "
        f"id={best.id} position={best.position} yaw={best.yaw:.6f} "
        f"score={best.final_score:.6f} "
        f"gain_exp={best.gain_exp:.1f} gain_sc={best.gain_sc:.1f} "
        f"gain_hybrid={best.gain_hybrid:.1f} gain_occ={best.gain_occ:.1f} "
        f"gain_conf={best.gain_conf:.6f} path_cost={best.path_cost:.6f}"
    )
    print(f"output_npz: {result['npz_path']}")
    print(f"output_jsonl: {result['jsonl_path']}")

    if args.print_topn:
        print("top_candidates:")
        print(format_top_candidates(result["top_candidates"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Stage 4A-6.6b GUI / visual inspection outputs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a66_manifest.json",
    "loaded_stage4a66_manifest.md",
    "loaded_stage4a66a_audit_manifest.json",
    "loaded_stage4a66a_audit_manifest.md",
    "gui_capability_report.json",
    "gui_capability_report.md",
    "gui_attempt_report.json",
    "gui_attempt_report.md",
    "user_viewing_instructions.md",
    "gui_start_command.sh",
    "gui_stop_instructions.md",
    "fallback_visual_package_report.json",
    "fallback_visual_package_report.md",
    "visual_inspection_summary.json",
    "visual_inspection_summary.md",
    "inspection_pose_manifest.json",
    "inspection_pose_manifest.md",
    "view_coverage_report.json",
    "view_coverage_report.md",
    "human_visual_review_checklist.md",
    "human_visual_review_checklist.json",
    "manual_review_gate.json",
    "manual_review_gate.md",
    "no_expert_sampling_report.json",
    "no_expert_sampling_report.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "visual_inspection_index.html",
    "recommended_next_faithful_step.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_IMAGES = [
    "scene_layout_topdown_human.png",
    "room_corridor_opening_labels_topdown.png",
    "obstacle_labels_topdown.png",
    "start_variants_labeled_topdown.png",
    "validation_poses_labeled_topdown.png",
    "topology_graph_labeled.png",
    "audit_warning_regions_topdown.png",
    "inspection_rgb_grid.png",
    "inspection_depth_grid.png",
    "closeup_corridor_east_spur.png",
    "closeup_room_j.png",
    "closeup_loop_junction.png",
    "closeup_dead_end_branch.png",
    "closeup_narrow_passage_examples.png",
    "closeup_obstacle_dense_spur_rooms.png",
    "closeup_start_locations.png",
]

FORBIDDEN_EXACT = [
    "transitions.jsonl",
    "expert_dataset_manifest.jsonl",
    "expert_dataset_manifest.json",
    "rollout_index.html",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "global_prediction_layer.npz",
    "selected_action_report.json",
    "action_execution_report.json",
]

FORBIDDEN_SUBSTRINGS = [
    "replay_buffer",
    "policy_checkpoint",
    "frame003",
    "action002",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"


def assert_png(path: Path, *, require_nonblank: bool = False) -> None:
    assert_file(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not PNG: {path}"
    image = Image.open(path).convert("RGB")
    assert image.size[0] > 0 and image.size[1] > 0, f"invalid image size: {path}"
    if require_nonblank:
        arr = np.asarray(image)
        assert int(arr.max()) > 2 and float(arr.std()) >= 1.0, f"blank or nearly uniform RGB: {path}"


def test_required_files(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_FILES:
        assert_file(output_dir / name)
    for name in REQUIRED_IMAGES:
        assert_png(output_dir / name)
    return {"passed": True, "required_files": len(REQUIRED_FILES), "required_images": len(REQUIRED_IMAGES)}


def test_capture_files(output_dir: Path, min_views: int) -> dict[str, Any]:
    rgb_paths = sorted(output_dir.glob("inspection_rgb_*.png"))
    depth_paths = sorted(output_dir.glob("inspection_depth_*.npy"))
    depth_color_paths = sorted(output_dir.glob("inspection_depth_color_*.png"))
    pose_paths = sorted(output_dir.glob("inspection_pose_*.json"))
    assert len(rgb_paths) >= min_views, f"RGB view count {len(rgb_paths)} < {min_views}"
    assert len(depth_paths) >= min_views, f"depth NPY count {len(depth_paths)} < {min_views}"
    assert len(depth_color_paths) >= min_views, f"depth color count {len(depth_color_paths)} < {min_views}"
    assert len(pose_paths) >= min_views, f"pose JSON count {len(pose_paths)} < {min_views}"

    nonblank = 0
    depth_positive = 0
    for idx in range(min_views):
        assert_png(output_dir / f"inspection_rgb_{idx:03d}.png", require_nonblank=True)
        assert_png(output_dir / f"inspection_depth_color_{idx:03d}.png")
        assert_file(output_dir / f"inspection_pose_{idx:03d}.json")
        depth = np.load(output_dir / f"inspection_depth_{idx:03d}.npy")
        assert depth.ndim == 2, f"depth shape invalid for {idx}: {depth.shape}"
        finite_positive = np.isfinite(depth) & (depth > 0.0)
        assert int(np.count_nonzero(finite_positive)) > 0, f"no finite positive depth for {idx}"
        depth_positive += 1
        rgb = np.asarray(Image.open(output_dir / f"inspection_rgb_{idx:03d}.png").convert("RGB"))
        if int(rgb.max()) > 2 and float(rgb.std()) >= 1.0:
            nonblank += 1
    return {"passed": True, "rgb_count": len(rgb_paths), "depth_count": len(depth_paths), "nonblank_checked": nonblank, "depth_positive_checked": depth_positive}


def test_reports_and_gates(output_dir: Path, min_views: int) -> dict[str, Any]:
    gate = load_json(output_dir / "manual_review_gate.json")
    checklist = load_json(output_dir / "human_visual_review_checklist.json")
    gui = load_json(output_dir / "gui_attempt_report.json")
    fallback = load_json(output_dir / "fallback_visual_package_report.json")
    summary = load_json(output_dir / "visual_inspection_summary.json")
    coverage = load_json(output_dir / "view_coverage_report.json")
    no_expert = load_json(output_dir / "no_expert_sampling_report.json")
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_map = load_json(output_dir / "no_map_predict_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")

    assert gate["human_visual_inspection_done"] is False
    assert gate["user_needs_to_review_visuals"] is True
    assert gate["visual_approval_required_before_stage4a67"] is True
    assert gate["formal_expert_sampling_ready_full_dataset"] is False
    assert gate["stage4a67_executed"] is False
    assert checklist["human_visual_inspection_done"] is False
    assert checklist["user_needs_to_review_visuals"] is True
    assert len(checklist["items"]) >= 12

    assert gui["gui_attempt_status"] in {"success", "failed", "skipped_no_display", "skipped_display_unverified", "visibility_unconfirmed"}
    assert fallback["fallback_visual_package_created"] is True
    assert int(fallback["inspection_view_count"]) >= min_views
    assert int(fallback["rgb_nonblank_count"]) >= min_views
    assert int(fallback["depth_positive_count"]) >= min_views
    assert summary["answers"]["formal_expert_sampling_ready_full_dataset"] is False
    assert summary["answers"]["no_expert_sampling"] is True
    assert summary["answers"]["no_rollout"] is True
    assert summary["answers"]["no_map_predict_or_sscnet_inference"] is True
    assert summary["answers"]["no_rl_gdpo_ppo_bc_il"] is True
    assert summary["answers"]["user_visual_review_still_required"] is True
    assert coverage["inspection_view_count"] >= min_views
    assert coverage["all_requested_categories_have_views"] is True

    assert no_expert["formal_expert_sampling_run"] is False
    assert no_expert["expert_dataset_generated"] is False
    assert no_rollout["rollout_run"] is False
    assert no_rollout["transitions_jsonl_created"] is False
    assert no_map["map_predict_called"] is False
    assert no_map["sscnet_inference_called"] is False
    assert no_map["prediction_npz_created"] is False
    assert no_rl["rl_run"] is False
    assert no_rl["gdpo_run"] is False
    assert no_rl["policy_checkpoint_created"] is False
    assert no_rl["checkpoint_modified"] is False
    assert "future directions only" in (output_dir / "long_term_rl_gdpo_note.md").read_text(encoding="utf-8")
    return {"passed": True}


def test_video_or_frames(output_dir: Path) -> dict[str, Any]:
    mp4 = output_dir / "larger_complex_scene_v1_flythrough.mp4"
    gif = output_dir / "larger_complex_scene_v1_flythrough.gif"
    frames = sorted((output_dir / "flythrough_frames").glob("frame_*.png"))
    if mp4.is_file():
        assert mp4.stat().st_size > 0, "empty MP4"
        return {"passed": True, "mode": "mp4", "frames": len(frames)}
    if gif.is_file():
        assert gif.stat().st_size > 0, "empty GIF"
        return {"passed": True, "mode": "gif", "frames": len(frames)}
    assert len(frames) > 0, "no flythrough video, GIF, or frames"
    assert_file(output_dir / "video_generation_skipped.md")
    for frame in frames[:3]:
        assert_png(frame, require_nonblank=True)
    return {"passed": True, "mode": "frames", "frames": len(frames)}


def test_hashes_unchanged(output_dir: Path, stage4a66_dir: Path, stage4a66a_dir: Path) -> dict[str, Any]:
    for manifest_name, root in [
        ("loaded_stage4a66_manifest.json", stage4a66_dir),
        ("loaded_stage4a66a_audit_manifest.json", stage4a66a_dir),
    ]:
        manifest = load_json(output_dir / manifest_name)
        assert Path(manifest["root"]).resolve() == root.resolve(), f"manifest root mismatch: {manifest_name}"
        for item in manifest["files"]:
            path = root / item["relative_path"]
            assert_file(path)
            assert sha256_file(path) == item["sha256"], f"input hash changed: {path}"
    return {"passed": True}


def test_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    hits = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in FORBIDDEN_EXACT:
            hits.append(str(path))
        if name.startswith("prediction") and name.endswith(".npz"):
            hits.append(str(path))
        for token in FORBIDDEN_SUBSTRINGS:
            if token in name:
                hits.append(str(path))
    assert not hits, f"forbidden outputs present: {hits}"
    return {"passed": True, "forbidden_hits": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6.6b visual inspection package.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage4a66_dir", required=True)
    parser.add_argument("--stage4a66a_dir", required=True)
    parser.add_argument("--expect_no_expert_sampling", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--min_inspection_views", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    stage4a66_dir = Path(args.stage4a66_dir).resolve()
    stage4a66a_dir = Path(args.stage4a66a_dir).resolve()
    if not (args.expect_no_expert_sampling and args.expect_no_rollout and args.expect_no_map_predict and args.expect_no_rl_gdpo):
        raise AssertionError("negative-scope expectation flags are required")

    results = {
        "required_files": test_required_files(output_dir),
        "capture_files": test_capture_files(output_dir, int(args.min_inspection_views)),
        "reports_and_gates": test_reports_and_gates(output_dir, int(args.min_inspection_views)),
        "video_or_frames": test_video_or_frames(output_dir),
        "hashes_unchanged": test_hashes_unchanged(output_dir, stage4a66_dir, stage4a66a_dir),
        "forbidden_outputs": test_forbidden_outputs(output_dir),
    }
    summary = {"all_passed": True, "results": results}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

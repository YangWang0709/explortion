#!/usr/bin/env python3
"""Reusable Isaac depth -> SSCNet -> simulator prediction wrapper.

Stage 4A-6 keeps map prediction read-only. This helper owns the SSCNet model
for a rollout, loads the checkpoint once, and returns a `SimPredictionLayer`
whose arrays match the measured-only `observed_state` shape.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaac_sscnet_preprocess import (
    DEFAULT_LOCAL_VOLUME_M,
    DOMAIN_SHIFT_NOTE,
    SSCNET_OUTPUT_AXIS_ORDER,
    alignment_convention_metadata,
    canonical_alignment_convention,
    preprocess_isaac_depth_for_sscnet,
    save_json,
    save_preprocessing_debug,
)
from run_isaac_map_predict_single import (
    FREE_CLASS_ID,
    NUM_CLASSES,
    align_local_prediction_to_global,
    array_stats,
    format_unique_counts,
    load_sscnet_model,
    save_global_prediction_layer,
    save_local_prediction,
)
from sim_paper_expert import UNKNOWN
from sim_prediction_layer import SimPredictionLayer


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


class IsaacMapPredictor:
    """Load SSCNet once and run per-step read-only map prediction."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        tau: float = 0.1,
        torch_num_threads: int | None = None,
        alignment_convention: str | None = "current_default_v0",
    ):
        self.checkpoint = Path(checkpoint).resolve()
        self.tau = float(tau)
        self.alignment_convention = canonical_alignment_convention(alignment_convention)
        self.alignment_convention_metadata = alignment_convention_metadata(self.alignment_convention)
        if torch_num_threads is not None:
            torch.set_num_threads(int(torch_num_threads))

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested for SSCNet inference, but torch.cuda.is_available() is false")
        if self.device.type != "cuda":
            raise RuntimeError(
                f"Stage 4A-6 requires GPU SSCNet inference on RTX hardware; requested device={self.device}"
            )

        self.checkpoint_stat_before = self._checkpoint_stat()
        load_start = time.perf_counter()
        self.model = load_sscnet_model(self.checkpoint, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.model_load_time = float(time.perf_counter() - load_start)
        self.model_loaded_once = True
        self.steps_predicted = 0
        self.gpu_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None
        self.gpu_memory_after_load = (
            int(torch.cuda.memory_allocated(self.device)) if self.device.type == "cuda" else None
        )

    def _checkpoint_stat(self) -> dict[str, Any]:
        stat = self.checkpoint.stat()
        return {
            "path": str(self.checkpoint),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def checkpoint_unchanged(self) -> bool:
        return self._checkpoint_stat() == self.checkpoint_stat_before

    def _run_inference(
        self,
        sscnet_depth_input: np.ndarray,
        sscnet_position: np.ndarray,
        save_probs: bool = False,
    ) -> dict[str, Any]:
        depth_t = torch.from_numpy(sscnet_depth_input[None, None, :, :].astype(np.float32)).to(self.device)
        position_t = torch.from_numpy(sscnet_position[None, :, :].astype(np.int64)).long().to(self.device)

        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        with torch.no_grad():
            logits_t = self.model(x_depth=depth_t, x_tsdf=None, p=position_t, x_rgb=None)
            class_prob_t = torch.softmax(logits_t, dim=1)
            confidence_t, pred_class_t = torch.max(class_prob_t, dim=1)
            free_prob_t = class_prob_t[:, FREE_CLASS_ID]
            occupied_prob_t = 1.0 - free_prob_t
        torch.cuda.synchronize(self.device)
        inference_time = time.perf_counter() - start_time

        inference: dict[str, Any] = {
            "logits_shape": tuple(int(v) for v in logits_t.shape),
            "pred_class": pred_class_t.squeeze(0).detach().cpu().numpy().astype(np.uint8),
            "confidence": confidence_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "free_prob": free_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "occupied_prob": occupied_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32),
            "device": str(self.device),
            "inference_time": float(inference_time),
            "loaded_strict": True,
            "gpu_memory_peak": int(torch.cuda.max_memory_allocated(self.device)),
        }
        if save_probs:
            inference["class_prob"] = class_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return inference

    def predict_step(
        self,
        depth: np.ndarray,
        pose: dict[str, Any],
        camera_info: dict[str, Any],
        observed_state: np.ndarray,
        map_bounds: dict[str, Any],
        voxel_size: float,
        output_dir: str | Path,
        step: int,
        save_probs: bool = False,
        save_viz: bool = False,
        observed_state_path: str | Path | None = None,
        depth_source: str | Path | None = None,
        pose_source: str | Path | None = None,
        camera_info_source: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run one read-only prediction step and return a SimPredictionLayer."""

        step_start = time.perf_counter()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        observed_state = np.asarray(observed_state)
        observed_hash_before = sha256_array(observed_state)

        preprocess_start = time.perf_counter()
        preprocess = preprocess_isaac_depth_for_sscnet(
            depth=depth,
            camera_info=camera_info,
            pose=pose,
            alignment_convention=self.alignment_convention,
        )
        preprocess_paths = save_preprocessing_debug(
            output_dir=output_path,
            preprocess_result=preprocess,
            depth_source=depth_source,
            pose_source=pose_source,
            camera_info_source=camera_info_source,
        )
        preprocess_time = time.perf_counter() - preprocess_start

        inference = self._run_inference(
            sscnet_depth_input=preprocess["sscnet_depth_input"],
            sscnet_position=preprocess["sscnet_position"],
            save_probs=bool(save_probs),
        )
        if inference["logits_shape"] != (1, NUM_CLASSES, 60, 36, 60):
            raise RuntimeError(f"Unexpected SSCNet logits shape: {inference['logits_shape']}")

        local_prediction_path = save_local_prediction(
            output_dir=output_path,
            inference=inference,
            checkpoint=self.checkpoint,
            depth_source=depth_source or f"step_{int(step):03d}_depth_in_memory",
            pose_source=pose_source or f"step_{int(step):03d}_pose_in_memory",
            preprocessing_notes=preprocess["notes"],
            save_probs=bool(save_probs),
        )

        alignment_start = time.perf_counter()
        aligned = align_local_prediction_to_global(
            pred_class=inference["pred_class"],
            confidence=inference["confidence"],
            free_prob=inference["free_prob"],
            occupied_prob=inference["occupied_prob"],
            observed_shape=tuple(int(v) for v in observed_state.shape),
            pose=pose,
            map_bounds=map_bounds,
            global_voxel_size=float(voxel_size),
            alignment_convention=self.alignment_convention,
        )
        global_prediction_path = save_global_prediction_layer(
            output_dir=output_path,
            aligned=aligned,
            observed_state_path=observed_state_path or f"step_{int(step):03d}_observed_state_in_memory.npy",
            local_prediction_path=local_prediction_path,
            checkpoint=self.checkpoint,
        )
        alignment_time = time.perf_counter() - alignment_start

        tau = float(self.tau)
        valid_tau = aligned["global_prediction_valid"] & (aligned["global_confidence"] >= tau)
        predicted_occupied = valid_tau & (aligned["global_occupied_prob"] >= 0.5)
        predicted_unmeasured = valid_tau & (observed_state == UNKNOWN)
        observed_hash_after = sha256_array(observed_state)

        summary = {
            "stage": "Stage 4A-6 dynamic read-only map_predict step",
            "step": int(step),
            "checkpoint": str(self.checkpoint),
            "checkpoint_stat_before_rollout": self.checkpoint_stat_before,
            "checkpoint_stat_after_step": self._checkpoint_stat(),
            "checkpoint_unchanged": self.checkpoint_unchanged(),
            "depth_source": str(depth_source) if depth_source is not None else "",
            "pose_source": str(pose_source) if pose_source is not None else "",
            "camera_info_source": str(camera_info_source) if camera_info_source is not None else "",
            "observed_state_source": str(observed_state_path)
            if observed_state_path is not None
            else f"step_{int(step):03d}_observed_state_in_memory.npy",
            "model_loaded_once": bool(self.model_loaded_once),
            "model_load_time": float(self.model_load_time),
            "device": str(self.device),
            "gpu_name": self.gpu_name,
            "observed_state_shape": [int(v) for v in observed_state.shape],
            "local_prediction_shape": [int(v) for v in inference["pred_class"].shape],
            "global_prediction_shape": [int(v) for v in aligned["global_pred_class"].shape],
            "depth_input_shape": [int(v) for v in preprocess["sscnet_depth_input"].shape],
            "position_shape": [int(v) for v in preprocess["sscnet_position"].shape],
            "valid_position_pixels": int(np.count_nonzero(preprocess["valid_position_mask"])),
            "logits_shape": [int(v) for v in inference["logits_shape"]],
            "local_confidence": array_stats(inference["confidence"]),
            "local_occupied_prob": array_stats(inference["occupied_prob"]),
            "local_free_prob": array_stats(inference["free_prob"]),
            "local_pred_class_unique_counts": format_unique_counts(inference["pred_class"]),
            "global_valid_prediction_count": int(np.count_nonzero(aligned["global_prediction_valid"])),
            "global_predicted_occupied_count": int(np.count_nonzero(predicted_occupied)),
            "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
            "tau": tau,
            "alignment_convention": self.alignment_convention,
            "alignment_convention_metadata": self.alignment_convention_metadata,
            "gpu_memory_peak": inference["gpu_memory_peak"],
            "gpu_memory_after_model_load": self.gpu_memory_after_load,
            "preprocessing_axis_convention": preprocess["notes"]["position_axis_order"],
            "preprocessing_alignment_convention": preprocess["notes"]["alignment_convention"],
            "local_prediction_axis_convention": ",".join(
                aligned["align_stats"]["alignment_convention_metadata"]["output_axis_order"]
            ),
            "global_prediction_axis_convention": "global arrays match observed_state axis order (world_x,world_y,world_z)",
            "local_volume_convention": preprocess["notes"]["local_volume_convention"],
            "local_volume_m": list(DEFAULT_LOCAL_VOLUME_M),
            "domain_shift_note": DOMAIN_SHIFT_NOTE,
            "alignment_stats": aligned["align_stats"],
            "strict_no_observed_write": observed_hash_before == observed_hash_after,
            "observed_state_sha256_before": observed_hash_before,
            "observed_state_sha256_after": observed_hash_after,
            "expert_used_prediction": False,
            "rollout_will_use_prediction_for_information_gain_only": True,
            "rl_or_ppo_training": False,
            "optimizer_step": False,
            "behavior_cloning_training": False,
            "imitation_learning_training": False,
            "sscnet_training": False,
            "prediction_written_to_observed_state": False,
            "prediction_used_for_collision_or_traversability": False,
            "paths": {
                "sscnet_input_debug": preprocess_paths["sscnet_input_debug"],
                "sscnet_depth_input": preprocess_paths["sscnet_depth_input"],
                "sscnet_position": preprocess_paths["sscnet_position"],
                "valid_position_mask": preprocess_paths["valid_position_mask"],
                "local_prediction": local_prediction_path,
                "global_prediction_layer": global_prediction_path,
            },
        }
        timing = {
            "preprocess_time": float(preprocess_time),
            "inference_time": float(inference["inference_time"]),
            "alignment_time": float(alignment_time),
            "total_time": float(time.perf_counter() - step_start),
        }
        summary["timing"] = timing
        summary_json = output_path / "prediction_alignment_summary.json"
        save_json(summary_json, summary)

        viz_summary: dict[str, Any] = {}
        if save_viz:
            viz_start = time.perf_counter()
            from visualize_isaac_prediction_alignment import generate_visualizations

            viz_summary = generate_visualizations(output_dir=output_path, tau=tau)
            timing["viz_time"] = float(time.perf_counter() - viz_start)

        prediction_layer = SimPredictionLayer.from_npz(global_prediction_path)
        self.steps_predicted += 1
        return {
            "prediction_layer": prediction_layer,
            "prediction_npz": str(global_prediction_path),
            "global_prediction_npz": str(global_prediction_path),
            "local_prediction_npz": str(local_prediction_path),
            "summary": summary,
            "summary_json": str(summary_json),
            "timing": timing,
            "preprocess_paths": preprocess_paths,
            "visualizations": viz_summary,
        }

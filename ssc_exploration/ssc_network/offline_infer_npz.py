#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Offline SSCNet checkpoint inference for one repackaged NYU .npz sample."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from models import make_model


DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)
DEFAULT_OUTPUT_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/sscnet_inference"
NUM_CLASSES = 12
FREE_CLASS_ID = 0


def load_npz_sample(input_npz):
    """Load fields exactly as the current NYUDataset npz branch does."""
    input_path = Path(input_npz)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input npz not found: {input_path}")

    required = ("rgb", "depth", "tsdf_hr", "tsdf_lr", "target_lr", "position")
    with np.load(input_path) as npz_file:
        missing = [key for key in required if key not in npz_file.files]
        if missing:
            raise KeyError(f"{input_path} is missing required fields: {missing}")
        sample = {key: np.array(npz_file[key]) for key in required}

    return sample


def _batch_depth(depth):
    if depth.ndim == 2:
        depth = depth[None, :, :]
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError(f"Expected depth shape (1,H,W) or (H,W), got {depth.shape}")
    return depth[None, ...]


def _batch_rgb(rgb):
    if rgb.ndim != 3 or rgb.shape[0] not in (1, 3):
        raise ValueError(f"Expected rgb shape (C,H,W), got {rgb.shape}")
    return rgb[None, ...]


def _batch_position(position):
    if position.ndim != 2:
        raise ValueError(f"Expected position shape (H,W), got {position.shape}")
    return position[None, ...]


def sample_to_tensors(sample, device):
    """Build tensors matching DataLoader output shapes used by test.py."""
    depth = torch.from_numpy(_batch_depth(sample["depth"])).float().to(device)
    rgb = torch.from_numpy(_batch_rgb(sample["rgb"])).float().to(device)
    position = torch.from_numpy(_batch_position(sample["position"])).long().to(device)
    return {"depth": depth, "rgb": rgb, "position": position}


def build_model(device):
    model = make_model("sscnet", num_classes=NUM_CLASSES)
    return model.to(device)


def load_checkpoint(model, checkpoint, device):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cp_states = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in cp_states:
        raise KeyError(f"Checkpoint missing 'state_dict': {checkpoint_path}")
    model.load_state_dict(cp_states["state_dict"], strict=True)
    model.eval()


def _format_unique_counts(pred_class):
    values, counts = np.unique(pred_class, return_counts=True)
    return ", ".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))


def _array_stats(array):
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def save_prediction(
    output_path,
    result,
    input_npz,
    checkpoint,
    save_probs=False,
    save_logits=False,
):
    payload = {
        "pred_class": result["pred_class"].astype(np.uint8),
        "confidence": result["confidence"].astype(np.float32),
        "occupied_prob": result["occupied_prob"].astype(np.float32),
        "free_prob": result["free_prob"].astype(np.float32),
        "input_npz": str(input_npz),
        "checkpoint": str(checkpoint),
        "free_class_id": np.array(FREE_CLASS_ID, dtype=np.int64),
        "free_class_assumption": "class 0 is empty/free; occupied_prob = 1 - prob[class 0]",
    }
    if save_probs:
        payload["class_prob"] = result["class_prob"].astype(np.float16)
    if save_logits:
        payload["logits"] = result["logits"].astype(np.float16)
    np.savez_compressed(output_path, **payload)


def run_inference(
    checkpoint=DEFAULT_CHECKPOINT,
    input_npz=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    device=None,
    save_probs=False,
    save_logits=False,
    print_stats=True,
):
    if input_npz is None:
        raise ValueError("--input_npz is required")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sample = load_npz_sample(input_npz)
    tensors = sample_to_tensors(sample, device)

    model = build_model(device)
    load_checkpoint(model, checkpoint, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits_t = model(
            x_depth=tensors["depth"],
            x_rgb=tensors["rgb"],
            p=tensors["position"],
        )
        class_prob_t = torch.softmax(logits_t, dim=1)
        confidence_t, pred_class_t = torch.max(class_prob_t, dim=1)
        free_prob_t = class_prob_t[:, FREE_CLASS_ID]
        occupied_prob_t = 1.0 - free_prob_t
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_time = time.perf_counter() - start_time

    logits = logits_t.squeeze(0).detach().cpu().numpy()
    class_prob = class_prob_t.squeeze(0).detach().cpu().numpy()
    pred_class = pred_class_t.squeeze(0).detach().cpu().numpy().astype(np.uint8)
    confidence = confidence_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    free_prob = free_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    occupied_prob = occupied_prob_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_npz)
    output_path = output_dir / f"{input_path.stem}_prediction.npz"

    result = {
        "logits": logits,
        "class_prob": class_prob,
        "pred_class": pred_class,
        "confidence": confidence,
        "occupied_prob": occupied_prob,
        "free_prob": free_prob,
    }
    save_prediction(
        output_path,
        result,
        input_npz=input_path,
        checkpoint=checkpoint,
        save_probs=save_probs,
        save_logits=save_logits,
    )

    gpu_memory_peak = None
    if device.type == "cuda":
        gpu_memory_peak = torch.cuda.max_memory_allocated(device)

    stats = {
        "input_npz": str(input_path),
        "checkpoint": str(checkpoint),
        "output_path": str(output_path),
        "device": str(device),
        "depth_shape": tuple(sample["depth"].shape),
        "position_shape": tuple(sample["position"].shape),
        "logits_shape": tuple(logits_t.shape),
        "pred_class_shape": tuple(pred_class.shape),
        "pred_class_unique_counts": _format_unique_counts(pred_class),
        "confidence": _array_stats(confidence),
        "occupied_prob": _array_stats(occupied_prob),
        "free_prob": _array_stats(free_prob),
        "inference_time": inference_time,
        "gpu_memory_peak": gpu_memory_peak,
        "loaded_strict": True,
        "free_class_id": FREE_CLASS_ID,
    }

    if print_stats:
        print(f"input path: {stats['input_npz']}")
        print(f"checkpoint: {stats['checkpoint']}")
        print(f"output path: {stats['output_path']}")
        print(f"device: {stats['device']}")
        print(f"depth shape: {stats['depth_shape']}")
        print(f"position shape: {stats['position_shape']}")
        print(f"logits shape: {stats['logits_shape']}")
        print(f"pred_class shape: {stats['pred_class_shape']}")
        print(f"pred_class unique counts: {stats['pred_class_unique_counts']}")
        print(
            "confidence min/max/mean: "
            f"{stats['confidence']['min']:.6f}/"
            f"{stats['confidence']['max']:.6f}/"
            f"{stats['confidence']['mean']:.6f}"
        )
        print(
            "occupied_prob min/max/mean: "
            f"{stats['occupied_prob']['min']:.6f}/"
            f"{stats['occupied_prob']['max']:.6f}/"
            f"{stats['occupied_prob']['mean']:.6f}"
        )
        print(
            "free_prob min/max/mean: "
            f"{stats['free_prob']['min']:.6f}/"
            f"{stats['free_prob']['max']:.6f}/"
            f"{stats['free_prob']['mean']:.6f}"
        )
        print(f"inference time: {stats['inference_time']:.6f}s")
        if stats["gpu_memory_peak"] is not None:
            print(f"GPU memory peak: {stats['gpu_memory_peak']} bytes")
        print(
            "free_class_id assumption: "
            f"{FREE_CLASS_ID} means empty/free; occupied_prob = 1 - prob[:, 0]"
        )

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Offline SSCNet inference for one .npz sample")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_probs", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--print_stats", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(
        checkpoint=args.checkpoint,
        input_npz=args.input_npz,
        output_dir=args.output_dir,
        device=args.device,
        save_probs=args.save_probs,
        save_logits=args.save_logits,
        print_stats=args.print_stats,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""PyTorch Dataset/DataLoader helpers for Stage 2C paper expert samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


FORBIDDEN_FIELDS = {"target_lr", "target_hr", "gt", "ground_truth"}
REQUIRED_NPZ_FIELDS = {
    "candidate_features",
    "feature_names",
    "candidate_positions",
    "candidate_yaws",
    "valid_mask",
    "expert_action",
    "expert_scores",
    "sample_id",
}


def _read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def _resolve_dataset_path(dataset_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return dataset_dir / path


def _to_feature_list(feature_names) -> list[str]:
    return [str(name) for name in np.asarray(feature_names).tolist()]


class PaperExpertDataset(Dataset):
    """Read Stage 2C paper expert `.npz` files without opening raw scene data."""

    def __init__(
        self,
        dataset_dir,
        split="all",
        normalize=False,
        feature_mean=None,
        feature_std=None,
        require_strict_no_target=True,
    ):
        if split != "all":
            raise ValueError("Stage 3A only supports split='all'")

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.normalize = bool(normalize)
        self.require_strict_no_target = bool(require_strict_no_target)

        metadata_path = self.dataset_dir / "metadata.json"
        manifest_path = self.dataset_dir / "manifest.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata.json missing: {metadata_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest.jsonl missing: {manifest_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.require_strict_no_target and self.metadata.get("strict_no_target_lr") is not True:
            raise ValueError("metadata strict_no_target_lr must be true")

        self.records = [
            record for record in _read_jsonl(manifest_path) if record.get("status") == "ok"
        ]
        if not self.records:
            raise ValueError(f"No ok samples found in manifest: {manifest_path}")

        self.feature_mean = None
        self.feature_std = None
        if self.normalize:
            if feature_mean is None or feature_std is None:
                raise ValueError(
                    "feature_mean and feature_std are required when normalize=True"
                )
            self.feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32)
            self.feature_std = torch.clamp(
                torch.as_tensor(feature_std, dtype=torch.float32), min=1e-6
            )

    def __len__(self):
        return len(self.records)

    def _expert_path(self, record) -> Path:
        if "expert_npz" not in record:
            raise KeyError(f"ok manifest record missing expert_npz: {record}")
        return _resolve_dataset_path(self.dataset_dir, record["expert_npz"])

    def _load_npz(self, expert_npz: Path):
        if not expert_npz.is_file():
            raise FileNotFoundError(f"expert_npz missing: {expert_npz}")
        return np.load(expert_npz, allow_pickle=False)

    def __getitem__(self, index):
        record = self.records[int(index)]
        expert_npz = self._expert_path(record)

        with self._load_npz(expert_npz) as data:
            fields = set(data.files)
            forbidden = sorted(fields & FORBIDDEN_FIELDS)
            if forbidden:
                raise ValueError(f"{expert_npz} contains forbidden fields: {forbidden}")

            missing = sorted(REQUIRED_NPZ_FIELDS - fields)
            if missing:
                raise KeyError(f"{expert_npz} missing required fields: {missing}")

            candidate_features_np = np.asarray(data["candidate_features"], dtype=np.float32)
            candidate_positions_np = np.asarray(data["candidate_positions"], dtype=np.int64)
            candidate_yaws_np = np.asarray(data["candidate_yaws"], dtype=np.float32)
            valid_mask_np = np.asarray(data["valid_mask"], dtype=bool)
            expert_scores_np = np.asarray(data["expert_scores"], dtype=np.float32)
            expert_action = int(np.asarray(data["expert_action"]).item())
            sample_id = str(np.asarray(data["sample_id"]).item())
            feature_names = _to_feature_list(data["feature_names"])

        if candidate_features_np.ndim != 2:
            raise ValueError(
                f"{expert_npz} candidate_features must be [N,D], "
                f"got {candidate_features_np.shape}"
            )
        num_candidates, num_features = candidate_features_np.shape
        if candidate_positions_np.shape != (num_candidates, 3):
            raise ValueError(
                f"{expert_npz} candidate_positions shape {candidate_positions_np.shape} "
                f"!= ({num_candidates}, 3)"
            )
        if candidate_yaws_np.shape != (num_candidates,):
            raise ValueError(
                f"{expert_npz} candidate_yaws shape {candidate_yaws_np.shape} "
                f"!= ({num_candidates},)"
            )
        if valid_mask_np.shape != (num_candidates,):
            raise ValueError(
                f"{expert_npz} valid_mask shape {valid_mask_np.shape} "
                f"!= ({num_candidates},)"
            )
        if expert_scores_np.shape != (num_candidates,):
            raise ValueError(
                f"{expert_npz} expert_scores shape {expert_scores_np.shape} "
                f"!= ({num_candidates},)"
            )
        if len(feature_names) != num_features:
            raise ValueError(
                f"{expert_npz} feature_names length {len(feature_names)} != D "
                f"{num_features}"
            )
        if expert_action < 0 or expert_action >= num_candidates:
            raise ValueError(
                f"{expert_npz} expert_action {expert_action} out of range "
                f"[0,{num_candidates})"
            )
        if not bool(valid_mask_np[expert_action]):
            raise ValueError(f"{expert_npz} expert_action is not valid")
        if not np.all(np.isfinite(candidate_features_np)):
            raise ValueError(f"{expert_npz} has non-finite candidate_features")
        if not np.all(np.isfinite(expert_scores_np)):
            raise ValueError(f"{expert_npz} has non-finite expert_scores")

        candidate_features = torch.from_numpy(
            np.ascontiguousarray(candidate_features_np)
        ).float()
        if self.normalize:
            if self.feature_mean.numel() != num_features:
                raise ValueError(
                    f"feature_mean length {self.feature_mean.numel()} != D {num_features}"
                )
            candidate_features = (
                candidate_features - self.feature_mean
            ) / self.feature_std

        return {
            "candidate_features": candidate_features,
            "candidate_positions": torch.from_numpy(
                np.ascontiguousarray(candidate_positions_np)
            ).long(),
            "candidate_yaws": torch.from_numpy(np.ascontiguousarray(candidate_yaws_np)).float(),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask_np)).bool(),
            "expert_action": torch.tensor(expert_action, dtype=torch.long),
            "expert_scores": torch.from_numpy(np.ascontiguousarray(expert_scores_np)).float(),
            "sample_id": sample_id,
            "feature_names": feature_names,
        }


def _assert_shared_features(feature_name_lists: Iterable[list[str]]) -> list[str]:
    iterator = iter(feature_name_lists)
    try:
        first = list(next(iterator))
    except StopIteration:
        raise ValueError("Cannot collate an empty batch")
    for feature_names in iterator:
        if list(feature_names) != first:
            raise ValueError("feature_names differ across batch samples")
    return first


def collate_paper_expert_batch(batch):
    """Stack fixed-size candidate batches, padding variable candidate counts."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")

    feature_names = _assert_shared_features(item["feature_names"] for item in batch)
    batch_size = len(batch)
    max_candidates = max(int(item["candidate_features"].shape[0]) for item in batch)
    num_features = int(batch[0]["candidate_features"].shape[1])

    candidate_features = torch.zeros(
        (batch_size, max_candidates, num_features), dtype=torch.float32
    )
    candidate_positions = torch.zeros(
        (batch_size, max_candidates, 3), dtype=torch.long
    )
    candidate_yaws = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    expert_scores = torch.full(
        (batch_size, max_candidates), float("-inf"), dtype=torch.float32
    )
    expert_action = torch.empty((batch_size,), dtype=torch.long)
    sample_id = []

    for row, item in enumerate(batch):
        features = item["candidate_features"]
        positions = item["candidate_positions"]
        yaws = item["candidate_yaws"]
        mask = item["valid_mask"]
        scores = item["expert_scores"]
        action = int(item["expert_action"].item())
        num_candidates = int(features.shape[0])

        if features.shape[1] != num_features:
            raise ValueError("candidate feature dimensions differ across batch samples")
        if action < 0 or action >= num_candidates:
            raise ValueError(f"expert_action {action} out of range for batch row {row}")
        if not bool(mask[action]):
            raise ValueError(f"expert_action {action} is invalid for batch row {row}")

        candidate_features[row, :num_candidates] = features
        candidate_positions[row, :num_candidates] = positions
        candidate_yaws[row, :num_candidates] = yaws
        valid_mask[row, :num_candidates] = mask
        expert_scores[row, :num_candidates] = scores
        expert_action[row] = action
        sample_id.append(item["sample_id"])

    return {
        "candidate_features": candidate_features,
        "candidate_positions": candidate_positions,
        "candidate_yaws": candidate_yaws,
        "valid_mask": valid_mask,
        "expert_action": expert_action,
        "expert_scores": expert_scores,
        "sample_id": sample_id,
        "feature_names": feature_names,
    }


def compute_feature_stats(dataset):
    """Compute feature mean/std over valid candidates only."""

    feature_chunks = []
    feature_names = None
    for index in range(len(dataset)):
        item = dataset[index]
        if feature_names is None:
            feature_names = list(item["feature_names"])
        elif list(item["feature_names"]) != feature_names:
            raise ValueError("feature_names differ across dataset samples")

        valid_features = item["candidate_features"][item["valid_mask"]]
        if valid_features.numel() > 0:
            feature_chunks.append(valid_features)

    if not feature_chunks:
        raise ValueError("No valid candidates available for feature stats")

    all_features = torch.cat(feature_chunks, dim=0)
    mean = all_features.mean(dim=0)
    std = torch.clamp(all_features.std(dim=0, unbiased=False), min=1e-6)
    return {"mean": mean, "std": std, "feature_names": feature_names}


def save_feature_stats(path, stats):
    """Save feature statistics in a compact npz file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=stats["mean"].detach().cpu().numpy().astype(np.float32),
        std=stats["std"].detach().cpu().numpy().astype(np.float32),
        feature_names=np.array(stats["feature_names"]),
    )
    return output_path

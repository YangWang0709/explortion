#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Read Stage 4A-7.0 BC-ready candidate-set datasets without training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FORBIDDEN_FIELDS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "reward",
    "policy_logits",
    "replay_buffer",
    "optimizer",
    "training_state",
}

REQUIRED_FIELDS = {
    "sample_id",
    "candidate_features_model",
    "candidate_valid_mask",
    "expert_action_index_primary",
    "expert_action_index_measured_shadow",
    "expert_action_index_lambda48_shadow",
    "expert_action_index_confidence_gated_shadow",
    "quality_keep_mask",
    "missing_feature_mask",
}


class SimExpertBCDataset(Dataset):
    """Dataset wrapper for Stage 4A-7.0 candidate-set BC tensors."""

    def __init__(self, npz_path, strict_keep_only=False):
        self.npz_path = Path(npz_path)
        if not self.npz_path.is_file():
            raise FileNotFoundError(f"BC dataset NPZ missing: {self.npz_path}")

        with np.load(self.npz_path, allow_pickle=False) as data:
            fields = set(data.files)
            forbidden = sorted(fields & FORBIDDEN_FIELDS)
            if forbidden:
                raise ValueError(f"{self.npz_path} contains forbidden fields: {forbidden}")
            missing = sorted(REQUIRED_FIELDS - fields)
            if missing:
                raise KeyError(f"{self.npz_path} missing required fields: {missing}")

            self.sample_id = np.asarray(data["sample_id"]).astype(str)
            self.candidate_features_model = np.asarray(data["candidate_features_model"], dtype=np.float32)
            self.candidate_valid_mask = np.asarray(data["candidate_valid_mask"], dtype=bool)
            self.expert_action_index_primary = np.asarray(data["expert_action_index_primary"], dtype=np.int64)
            self.expert_action_index_measured_shadow = np.asarray(data["expert_action_index_measured_shadow"], dtype=np.int64)
            self.expert_action_index_lambda48_shadow = np.asarray(data["expert_action_index_lambda48_shadow"], dtype=np.int64)
            self.expert_action_index_confidence_gated_shadow = np.asarray(data["expert_action_index_confidence_gated_shadow"], dtype=np.int64)
            self.quality_keep_mask = np.asarray(data["quality_keep_mask"], dtype=bool)
            self.missing_feature_mask = np.asarray(data["missing_feature_mask"], dtype=bool)

        if self.candidate_features_model.ndim != 3:
            raise ValueError("candidate_features_model must be [N,C,D]")
        if self.candidate_valid_mask.shape != self.candidate_features_model.shape[:2]:
            raise ValueError("candidate_valid_mask shape must match candidate_features_model [N,C]")
        if self.missing_feature_mask.shape != self.candidate_features_model.shape:
            raise ValueError("missing_feature_mask shape must match candidate_features_model")
        if not np.all(np.isfinite(self.candidate_features_model)):
            raise ValueError("candidate_features_model must be finite after imputation")

        sample_count, candidate_count, _feature_dim = self.candidate_features_model.shape
        for name in (
            "expert_action_index_primary",
            "expert_action_index_measured_shadow",
            "expert_action_index_lambda48_shadow",
            "expert_action_index_confidence_gated_shadow",
        ):
            labels = getattr(self, name)
            if labels.shape != (sample_count,):
                raise ValueError(f"{name} shape {labels.shape} != ({sample_count},)")
            invalid = [
                int(i)
                for i, label in enumerate(labels)
                if label < 0 or label >= candidate_count or not self.candidate_valid_mask[i, label]
            ]
            if invalid and name == "expert_action_index_primary":
                raise ValueError(f"primary labels invalid at sample indices: {invalid[:20]}")

        if strict_keep_only:
            self.indices = np.flatnonzero(self.quality_keep_mask).astype(np.int64)
        else:
            self.indices = np.arange(sample_count, dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, index):
        sample_index = int(self.indices[int(index)])
        return {
            "sample_id": str(self.sample_id[sample_index]),
            "candidate_features_model": torch.from_numpy(
                np.ascontiguousarray(self.candidate_features_model[sample_index])
            ).float(),
            "candidate_valid_mask": torch.from_numpy(
                np.ascontiguousarray(self.candidate_valid_mask[sample_index])
            ).bool(),
            "expert_action_index_primary": torch.tensor(
                int(self.expert_action_index_primary[sample_index]), dtype=torch.long
            ),
            "expert_action_index_measured_shadow": torch.tensor(
                int(self.expert_action_index_measured_shadow[sample_index]), dtype=torch.long
            ),
            "expert_action_index_lambda48_shadow": torch.tensor(
                int(self.expert_action_index_lambda48_shadow[sample_index]), dtype=torch.long
            ),
            "expert_action_index_confidence_gated_shadow": torch.tensor(
                int(self.expert_action_index_confidence_gated_shadow[sample_index]), dtype=torch.long
            ),
            "quality_keep": torch.tensor(bool(self.quality_keep_mask[sample_index])),
            "missing_feature_mask": torch.from_numpy(
                np.ascontiguousarray(self.missing_feature_mask[sample_index])
            ).bool(),
        }

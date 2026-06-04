#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Smoke test for Stage 3A paper expert Dataset/DataLoader."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .paper_expert_dataset import (
    FORBIDDEN_FIELDS,
    PaperExpertDataset,
    collate_paper_expert_batch,
    compute_feature_stats,
)
from .policy import CandidateMLPPolicy


DEFAULT_DATASET_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/paper_expert_dataset_smoke"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3A dataset smoke test")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def _check_no_forbidden_fields(dataset):
    forbidden_hits = []
    for record in dataset.records:
        expert_npz = Path(record["expert_npz"])
        with np.load(expert_npz, allow_pickle=False) as data:
            hits = sorted(set(data.files) & FORBIDDEN_FIELDS)
            if hits:
                forbidden_hits.append((str(expert_npz), hits))
    if forbidden_hits:
        raise AssertionError(f"forbidden target fields found: {forbidden_hits}")


def main():
    args = parse_args()
    dataset = PaperExpertDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        normalize=False,
        require_strict_no_target=True,
    )
    assert len(dataset) == 5, f"expected 5 smoke samples, got {len(dataset)}"

    first = dataset[0]
    assert tuple(first["candidate_features"].shape) == (16, 15)
    assert tuple(first["valid_mask"].shape) == (16,)
    assert first["expert_action"].ndim == 0
    assert len(first["feature_names"]) == 15
    action = int(first["expert_action"].item())
    assert 0 <= action < first["candidate_features"].shape[0]
    assert bool(first["valid_mask"][action])
    assert torch.isfinite(first["candidate_features"]).all()
    assert torch.isfinite(first["expert_scores"]).all()

    _check_no_forbidden_fields(dataset)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_paper_expert_batch,
    )
    batch = next(iter(loader))
    assert tuple(batch["candidate_features"].shape) == (2, 16, 15)
    assert tuple(batch["valid_mask"].shape) == (2, 16)
    assert tuple(batch["expert_action"].shape) == (2,)
    assert len(batch["feature_names"]) == 15

    stats = compute_feature_stats(dataset)
    assert tuple(stats["mean"].shape) == (15,)
    assert tuple(stats["std"].shape) == (15,)
    assert len(stats["feature_names"]) == 15
    assert torch.isfinite(stats["mean"]).all()
    assert torch.isfinite(stats["std"]).all()
    assert torch.all(stats["std"] >= 1e-6)

    policy = CandidateMLPPolicy(input_dim=15, hidden_dim=64)
    logits = policy(batch["candidate_features"], batch["valid_mask"])
    assert tuple(logits.shape) == (2, 16)
    loss = F.cross_entropy(logits, batch["expert_action"])
    assert torch.isfinite(loss)

    print("Stage 3A IL dataset smoke test passed.")
    print(f"dataset size: {len(dataset)}")
    print(f"first candidate_features shape: {tuple(first['candidate_features'].shape)}")
    print(f"batch candidate_features shape: {tuple(batch['candidate_features'].shape)}")
    print(f"feature_names length: {len(first['feature_names'])}")
    print(f"expert_action valid: {bool(first['valid_mask'][action])}")
    print(f"feature_stats mean shape: {tuple(stats['mean'].shape)}")
    print(f"feature_stats std min: {float(stats['std'].min().item()):.6f}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"cross_entropy loss: {float(loss.detach().item()):.6f}")
    print("optimizer step performed: no")
    print("forbidden target fields: none")


if __name__ == "__main__":
    main()

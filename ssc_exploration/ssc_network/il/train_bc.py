#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Behavior cloning script skeleton for Stage 3A.

Training is intentionally disabled in this stage. With --dry_run, this script
loads one batch, runs a policy forward pass, computes CE loss, and exits
without optimizer.step() or model saving.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .paper_expert_dataset import (
    PaperExpertDataset,
    collate_paper_expert_batch,
    compute_feature_stats,
    save_feature_stats,
)
from .policy import CandidateMLPPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3A BC skeleton")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--feature_stats_out")
    parser.add_argument("--hidden_dim", type=int, default=128)
    return parser.parse_args()


def _make_dataset(args):
    raw_dataset = PaperExpertDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        normalize=False,
        require_strict_no_target=True,
    )

    stats = None
    if args.normalize or args.feature_stats_out:
        stats = compute_feature_stats(raw_dataset)
        if args.feature_stats_out:
            stats_path = save_feature_stats(args.feature_stats_out, stats)
            print(f"feature_stats_out: {stats_path}")
            print(f"feature_stats_mean_shape: {tuple(stats['mean'].shape)}")
            print(f"feature_stats_std_min: {float(stats['std'].min().item()):.6f}")

    if not args.normalize:
        return raw_dataset, stats

    dataset = PaperExpertDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        normalize=True,
        feature_mean=stats["mean"],
        feature_std=stats["std"],
        require_strict_no_target=True,
    )
    return dataset, stats


def run_dry_run(args):
    dataset, _stats = _make_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_paper_expert_batch,
    )
    batch = next(iter(loader))

    candidate_features = batch["candidate_features"]
    valid_mask = batch["valid_mask"]
    expert_action = batch["expert_action"]
    batch_size, num_candidates, num_features = candidate_features.shape

    policy = CandidateMLPPolicy(input_dim=num_features, hidden_dim=args.hidden_dim)
    logits = policy(candidate_features, valid_mask)
    loss = F.cross_entropy(logits, expert_action)

    print("Stage 3A BC dry-run complete.")
    print(f"dataset size: {len(dataset)}")
    print(f"B,N,D: {batch_size},{num_candidates},{num_features}")
    print(f"expert_action: {expert_action.tolist()}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {float(loss.detach().item()):.6f}")
    print("optimizer step performed: no")
    print("model saved: no")


def main():
    args = parse_args()
    print("Stage 3A behavior cloning skeleton")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"normalize: {args.normalize}")
    print(f"hidden_dim: {args.hidden_dim}")

    if not args.dry_run:
        print("Training is intentionally disabled in Stage 3A. Re-run in Stage 3B.")
        return

    run_dry_run(args)


if __name__ == "__main__":
    main()

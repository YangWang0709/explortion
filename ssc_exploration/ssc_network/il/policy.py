#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Candidate-scoring policy skeleton for Stage 3A forward-only checks."""

from __future__ import annotations

import torch
import torch.nn as nn


class CandidateMLPPolicy(nn.Module):
    """Shared MLP over candidate features, producing one logit per candidate."""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.candidate_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, candidate_features, valid_mask):
        if candidate_features.ndim != 3:
            raise ValueError(
                "candidate_features must be [B,N,D], "
                f"got {tuple(candidate_features.shape)}"
            )
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must be [B,N], got {tuple(valid_mask.shape)}")
        if candidate_features.shape[:2] != valid_mask.shape:
            raise ValueError(
                "candidate_features [B,N] and valid_mask shape differ: "
                f"{tuple(candidate_features.shape[:2])} vs {tuple(valid_mask.shape)}"
            )
        if candidate_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"candidate_features D={candidate_features.shape[-1]} != "
                f"input_dim={self.input_dim}"
            )

        logits = self.candidate_mlp(candidate_features).squeeze(-1)
        return logits.masked_fill(~valid_mask.bool(), -1.0e9)

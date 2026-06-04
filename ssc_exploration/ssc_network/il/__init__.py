"""Stage 3A imitation-learning data utilities.

This package is intentionally limited to dataset loading, batching, feature
statistics, and forward-only policy smoke checks. Training is disabled until
Stage 3B.
"""

from .paper_expert_dataset import (
    PaperExpertDataset,
    collate_paper_expert_batch,
    compute_feature_stats,
    save_feature_stats,
)
from .policy import CandidateMLPPolicy

__all__ = [
    "CandidateMLPPolicy",
    "PaperExpertDataset",
    "collate_paper_expert_batch",
    "compute_feature_stats",
    "save_feature_stats",
]

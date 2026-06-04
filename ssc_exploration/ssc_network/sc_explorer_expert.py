#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Deprecated Stage 2B prototype entry.

The earlier mock-observed-map expert has been disabled because strict Stage 2B
must not use ground-truth labels for scoring. Use sc_explorer_paper_expert.py.
"""

from sc_explorer_paper_expert import CandidateView  # re-export harmless dataclass


def _raise_disabled(*_args, **_kwargs):
    raise RuntimeError(
        "Deprecated Stage 2B prototype disabled. Use "
        "sc_explorer_paper_expert.py / run_paper_expert_offline.py instead; "
        "strict expert scoring must not use ground-truth labels."
    )


build_mock_observed_map_from_nyu_npz = _raise_disabled
run_expert_scoring = _raise_disabled
evaluate_candidates = _raise_disabled
save_expert_outputs = _raise_disabled

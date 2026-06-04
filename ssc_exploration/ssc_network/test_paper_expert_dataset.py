#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Validate Stage 2C paper expert dataset artifacts.

This is a format and leakage smoke test only. It does not create or train an
imitation-learning policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FORBIDDEN_FIELDS = {"target_lr", "target_hr", "gt", "ground_truth"}
VALID_STATUSES = {"ok", "failed"}
REQUIRED_FEATURES = {
    "gain_exp",
    "gain_sc",
    "gain_hybrid",
    "gain_occ",
    "gain_conf",
    "path_cost",
    "utility_exp",
    "utility_sc",
    "utility_hybrid",
    "utility_occ",
    "utility_conf",
    "final_score",
    "visible_count",
    "measured_visible_count",
    "predicted_unmeasured_visible_count",
}
REQUIRED_NPZ_FIELDS = {
    "candidate_features",
    "feature_names",
    "candidate_positions",
    "candidate_yaws",
    "valid_mask",
    "expert_action",
    "expert_scores",
    "top_candidate_ids",
    "gain_mode",
    "measured_mode",
    "raycast_mode",
    "sample_npz",
    "prediction_npz",
    "sample_id",
    "strict_no_target_note",
    "tree_limitation_note",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Stage 2C paper expert dataset files"
    )
    parser.add_argument("--dataset_dir", required=True)
    return parser.parse_args()


def _load_manifest(manifest_path):
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            status = record.get("status")
            if status not in VALID_STATUSES:
                raise AssertionError(
                    f"manifest line {line_number} has invalid status: {status}"
                )
            records.append(record)
    return records


def _scalar_int(array):
    value = np.asarray(array)
    if value.shape != ():
        raise AssertionError(f"expected scalar int, got shape {value.shape}")
    return int(value.item())


def validate_expert_npz(expert_npz):
    with np.load(expert_npz, allow_pickle=False) as data:
        fields = set(data.files)
        missing = sorted(REQUIRED_NPZ_FIELDS - fields)
        if missing:
            raise AssertionError(f"{expert_npz} missing fields: {missing}")

        forbidden = sorted(fields & FORBIDDEN_FIELDS)
        if forbidden:
            raise AssertionError(f"{expert_npz} contains forbidden fields: {forbidden}")

        candidate_features = np.asarray(data["candidate_features"])
        if candidate_features.ndim != 2:
            raise AssertionError(
                f"{expert_npz} candidate_features must be [N,D], "
                f"got {candidate_features.shape}"
            )
        if candidate_features.dtype != np.float32:
            raise AssertionError(
                f"{expert_npz} candidate_features must be float32, "
                f"got {candidate_features.dtype}"
            )
        num_candidates, num_features = candidate_features.shape
        if num_candidates <= 0 or num_features <= 0:
            raise AssertionError(f"{expert_npz} has empty candidate_features")

        feature_names = [str(name) for name in data["feature_names"].tolist()]
        if len(feature_names) != num_features:
            raise AssertionError(
                f"{expert_npz} feature_names length {len(feature_names)} != D "
                f"{num_features}"
            )
        missing_required_features = sorted(REQUIRED_FEATURES - set(feature_names))
        if missing_required_features:
            raise AssertionError(
                f"{expert_npz} missing required features: {missing_required_features}"
            )

        valid_mask = np.asarray(data["valid_mask"])
        if valid_mask.shape != (num_candidates,):
            raise AssertionError(
                f"{expert_npz} valid_mask shape {valid_mask.shape} != "
                f"({num_candidates},)"
            )
        if valid_mask.dtype != np.bool_:
            raise AssertionError(f"{expert_npz} valid_mask must be bool")

        expert_scores = np.asarray(data["expert_scores"])
        if expert_scores.shape != (num_candidates,):
            raise AssertionError(
                f"{expert_npz} expert_scores shape {expert_scores.shape} != "
                f"({num_candidates},)"
            )
        if expert_scores.dtype != np.float32:
            raise AssertionError(
                f"{expert_npz} expert_scores must be float32, got {expert_scores.dtype}"
            )

        expert_action = _scalar_int(data["expert_action"])
        if expert_action < 0 or expert_action >= num_candidates:
            raise AssertionError(
                f"{expert_npz} expert_action {expert_action} out of range "
                f"[0,{num_candidates})"
            )
        if not bool(valid_mask[expert_action]):
            raise AssertionError(f"{expert_npz} expert_action is not valid")

        final_score_index = feature_names.index("final_score")
        final_scores = candidate_features[:, final_score_index]
        if not np.all(np.isfinite(final_scores)):
            raise AssertionError(f"{expert_npz} has non-finite final_score values")
        if not np.all(np.isfinite(expert_scores)):
            raise AssertionError(f"{expert_npz} has non-finite expert_scores")

        candidate_positions = np.asarray(data["candidate_positions"])
        if candidate_positions.shape != (num_candidates, 3):
            raise AssertionError(
                f"{expert_npz} candidate_positions shape {candidate_positions.shape} "
                f"!= ({num_candidates},3)"
            )
        if candidate_positions.dtype != np.int32:
            raise AssertionError(
                f"{expert_npz} candidate_positions must be int32, "
                f"got {candidate_positions.dtype}"
            )

        candidate_yaws = np.asarray(data["candidate_yaws"])
        if candidate_yaws.shape != (num_candidates,):
            raise AssertionError(
                f"{expert_npz} candidate_yaws shape {candidate_yaws.shape} "
                f"!= ({num_candidates},)"
            )
        if candidate_yaws.dtype != np.float32:
            raise AssertionError(
                f"{expert_npz} candidate_yaws must be float32, "
                f"got {candidate_yaws.dtype}"
            )

        top_candidate_ids = np.asarray(data["top_candidate_ids"])
        if top_candidate_ids.shape != (num_candidates,):
            raise AssertionError(
                f"{expert_npz} top_candidate_ids shape {top_candidate_ids.shape} "
                f"!= ({num_candidates},)"
            )
        if top_candidate_ids.dtype != np.int64:
            raise AssertionError(
                f"{expert_npz} top_candidate_ids must be int64, "
                f"got {top_candidate_ids.dtype}"
            )

    return {"num_candidates": int(num_candidates), "num_features": int(num_features)}


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    manifest_path = dataset_dir / "manifest.jsonl"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json missing: {metadata_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.jsonl missing: {manifest_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("strict_no_target_lr") is not True:
        raise AssertionError("metadata strict_no_target_lr must be true")
    if metadata.get("tree_utility_eq12_implemented") is not False:
        raise AssertionError("metadata tree_utility_eq12_implemented must be false")

    metadata_forbidden = sorted(set(metadata.keys()) & FORBIDDEN_FIELDS)
    if metadata_forbidden:
        raise AssertionError(f"metadata contains forbidden keys: {metadata_forbidden}")

    records = _load_manifest(manifest_path)
    if not records:
        raise AssertionError("manifest is empty")

    ok_records = [record for record in records if record["status"] == "ok"]
    failed_records = [record for record in records if record["status"] == "failed"]
    if not ok_records:
        raise AssertionError("no ok samples in manifest")

    checked = []
    for record in ok_records:
        expert_npz = Path(record["expert_npz"])
        if not expert_npz.is_file():
            raise FileNotFoundError(f"ok expert_npz missing: {expert_npz}")
        checked.append(validate_expert_npz(expert_npz))

    for record in failed_records:
        if "error" not in record:
            raise AssertionError(f"failed manifest record missing error: {record}")

    print("Stage 2C paper expert dataset validation passed.")
    print(f"dataset_dir: {dataset_dir}")
    print(f"manifest records: {len(records)}")
    print(f"ok samples: {len(ok_records)}")
    print(f"failed samples: {len(failed_records)}")
    print(f"first shape: N={checked[0]['num_candidates']} D={checked[0]['num_features']}")
    print("forbidden target fields check: passed")


if __name__ == "__main__":
    main()

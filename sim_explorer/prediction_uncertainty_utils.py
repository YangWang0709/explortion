#!/usr/bin/env python3
"""Utilities for compact prediction-derived uncertainty artifacts."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


UNKNOWN = np.int8(-1)
NUM_CLASSES = 12
FREE_CLASS_ID = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def file_record(path: str | Path, include_hash: bool = True) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat() if path.exists() else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": int(stat.st_size) if stat else None,
        "mtime_ns": int(stat.st_mtime_ns) if stat else None,
        "sha256": sha256_file(path) if include_hash and path.is_file() else None,
    }


def markdown_kv(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| Field | Value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(jsonable(value))
        lines.append(f"| `{key}` | {html.escape(rendered)} |")
    lines.append("")
    return "\n".join(lines)


def markdown_rows(title: str, rows: list[dict[str, Any]], max_rows: int = 40) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        lines.extend(["No rows.", ""])
        return "\n".join(lines)
    fields: list[str] = []
    for row in rows[:max_rows]:
        for key in row:
            if key not in fields:
                fields.append(key)
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows[:max_rows]:
        values = []
        for key in fields:
            value = row.get(key, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
            values.append(html.escape(str(jsonable(value))))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(rows)} rows.")
    lines.append("")
    return "\n".join(lines)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def as_int(value: Any) -> int | None:
    v = as_float(value)
    return int(v) if v is not None else None


def summarize(values: list[Any]) -> dict[str, Any]:
    clean = np.asarray([float(v) for v in values if as_float(v) is not None], dtype=np.float64)
    if clean.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(clean.size),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p50": float(np.percentile(clean, 50)),
        "p90": float(np.percentile(clean, 90)),
    }


def pearson(x_values: list[Any], y_values: list[Any]) -> dict[str, Any]:
    pairs = [(as_float(x), as_float(y)) for x, y in zip(x_values, y_values)]
    clean = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(clean) < 2:
        return {"status": "insufficient_pairs", "n": len(clean), "pearson": None}
    x = np.asarray([p[0] for p in clean], dtype=np.float64)
    y = np.asarray([p[1] for p in clean], dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"status": "degenerate_zero_variance", "n": int(x.size), "pearson": None}
    return {"status": "computed", "n": int(x.size), "pearson": float(np.corrcoef(x, y)[0, 1])}


def compute_entropy_margin_from_prob(class_prob: np.ndarray, eps: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray]:
    prob = np.asarray(class_prob, dtype=np.float32)
    entropy = -np.sum(prob * np.log(prob + float(eps)), axis=0)
    entropy_norm = np.clip(entropy / math.log(float(prob.shape[0])), 0.0, 1.0).astype(np.float32)
    top2 = np.partition(prob, kth=-2, axis=0)[-2:]
    top2.sort(axis=0)
    margin = np.clip(top2[-1] - top2[-2], 0.0, 1.0).astype(np.float32)
    return entropy_norm, margin


def dense_fields_from_aligned(
    *,
    pred_class: np.ndarray,
    confidence: np.ndarray,
    entropy_norm: np.ndarray,
    margin: np.ndarray,
    occupied_prob: np.ndarray,
    free_prob: np.ndarray,
    raw_valid_mask: np.ndarray,
    observed_state: np.ndarray,
    tau: float,
) -> dict[str, np.ndarray]:
    confidence = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
    entropy_norm = np.clip(np.asarray(entropy_norm, dtype=np.float32), 0.0, 1.0)
    margin = np.clip(np.asarray(margin, dtype=np.float32), 0.0, 1.0)
    occupied_prob = np.clip(np.asarray(occupied_prob, dtype=np.float32), 0.0, 1.0)
    free_prob = np.clip(np.asarray(free_prob, dtype=np.float32), 0.0, 1.0)
    valid_mask = np.asarray(raw_valid_mask, dtype=bool) & (confidence >= float(tau))
    predicted_unmeasured = valid_mask & (np.asarray(observed_state) == UNKNOWN)
    return {
        "pred_class_uint8": np.asarray(pred_class, dtype=np.uint8),
        "confidence_float16": confidence.astype(np.float16),
        "entropy_norm_float16": entropy_norm.astype(np.float16),
        "margin_float16": margin.astype(np.float16),
        "occupied_prob_float16": occupied_prob.astype(np.float16),
        "free_prob_float16": free_prob.astype(np.float16),
        "valid_mask_bool": valid_mask.astype(bool),
        "predicted_unmeasured_mask_bool": predicted_unmeasured.astype(bool),
    }


def dense_stats(fields: dict[str, np.ndarray], observed_state: np.ndarray | None = None) -> dict[str, Any]:
    valid = np.asarray(fields["valid_mask_bool"], dtype=bool)
    predicted_unmeasured = np.asarray(fields["predicted_unmeasured_mask_bool"], dtype=bool)
    stats: dict[str, Any] = {
        "shape": [int(v) for v in fields["pred_class_uint8"].shape],
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "predicted_occupied_count": int(np.count_nonzero(valid & (np.asarray(fields["occupied_prob_float16"], dtype=np.float32) >= 0.5))),
        "source_occ_free_count": int(np.count_nonzero(predicted_unmeasured)),
    }
    if observed_state is not None:
        stats["observed_count"] = int(np.count_nonzero(np.asarray(observed_state) != UNKNOWN))
        stats["observed_reference_hash"] = sha256_array(np.asarray(observed_state))
    for name, key in (
        ("confidence", "confidence_float16"),
        ("entropy_norm", "entropy_norm_float16"),
        ("margin", "margin_float16"),
        ("occupied_prob", "occupied_prob_float16"),
        ("free_prob", "free_prob_float16"),
    ):
        values = np.asarray(fields[key], dtype=np.float32)[valid]
        stats.update({f"{name}_{k}": v for k, v in summarize(values.tolist()).items()})
    return stats


def save_dense_uncertainty_npz(
    path: str | Path,
    fields: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {key: value for key, value in fields.items()}
    for key, value in metadata.items():
        if isinstance(value, (dict, list, tuple)):
            payload[key] = np.array(json.dumps(jsonable(value), sort_keys=True))
        elif isinstance(value, str):
            payload[key] = np.array(value)
        elif isinstance(value, bool):
            payload[key] = np.array(value, dtype=bool)
        elif isinstance(value, int):
            payload[key] = np.array(value, dtype=np.int64)
        elif isinstance(value, float):
            payload[key] = np.array(value, dtype=np.float64)
        elif value is None:
            payload[key] = np.array("")
        else:
            payload[key] = np.array(str(value))
    np.savez_compressed(path, **payload)


def assert_no_nan_inf_in_arrays(arrays: dict[str, np.ndarray], names: list[str]) -> None:
    for name in names:
        values = np.asarray(arrays[name])
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN/Inf")

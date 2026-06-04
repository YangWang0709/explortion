#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Paper-faithful offline SC-Explorer expert candidate scorer.

Stage 2B strict version:
- S is approximated only from sensor-derived NYU fields: tsdf_lr and/or position.
- P is predicted-by-scene-completion and not measured.
- target_lr/target_hr are never read by this module and never affect scoring.
- The default ray caster is non-blocking with respect to scene completion.

This is a per-candidate gain/cost scorer. It does not implement the full
SC-Explorer RRT tree utility/replanning equation; that remains future work.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from prediction_layer import PredictionLayer
except ImportError:  # pragma: no cover - allows package-style imports later.
    from .prediction_layer import PredictionLayer


LR_SHAPE = (60, 36, 60)
HR_SHAPE = (240, 144, 240)
POSITION_DOWNSAMPLE = 4

TSDF_EMPTY = np.float32(0.001)
TSDF_SURFACE = np.float32(1.0)

MEASURED_NOTE = (
    "offline NYU measured set S approximation only; real exploration must use "
    "the online sensor-integrated measured map"
)
STRICT_NO_TARGET_NOTE = (
    "target_lr/target_hr are ground truth and are not read or used for expert scoring"
)
TREE_LIMITATION_NOTE = (
    "per-candidate paper-faithful gain/cost scorer; full RRT tree utility Eq. 12 "
    "is future work"
)

FEATURE_NAMES = [
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
]


@dataclass
class CandidateView:
    id: int
    position: tuple[int, int, int]
    yaw: float = 0.0
    pitch: float = 0.0
    valid: bool = True
    invalid_reason: str = ""

    visible_voxels: list | None = None

    gain_exp: float = 0.0
    gain_sc: float = 0.0
    gain_hybrid: float = 0.0
    gain_occ: float = 0.0
    gain_conf: float = 0.0

    path_cost: float = 0.0
    utility_exp: float = 0.0
    utility_sc: float = 0.0
    utility_hybrid: float = 0.0
    utility_occ: float = 0.0
    utility_conf: float = 0.0

    final_score: float = 0.0

    visible_count: int = 0
    measured_visible_count: int = 0
    predicted_unmeasured_visible_count: int = 0


def _load_sensor_fields(sample_npz, required):
    """Load only sensor-derived fields.

    This helper deliberately refuses target_lr/target_hr so an accidental
    scoring path cannot leak ground truth into measured S, P, gain, cost, or
    expert_action.
    """

    forbidden = {"target_lr", "target_hr"}
    requested_forbidden = sorted(set(required) & forbidden)
    if requested_forbidden:
        raise ValueError(f"Forbidden ground-truth fields requested: {requested_forbidden}")

    sample_path = Path(sample_npz)
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample npz not found: {sample_path}")

    with np.load(sample_path) as data:
        missing = [field for field in required if field not in data.files]
        if missing:
            raise KeyError(f"{sample_path} is missing required sensor fields: {missing}")
        return {field: np.array(data[field]) for field in required}


def _position_mask(position, grid_shape=LR_SHAPE):
    """Convert depth-pixel high-res voxel hits into low-res measured voxels."""

    mask = np.zeros(tuple(grid_shape), dtype=bool)
    flat_position = np.asarray(position).reshape(-1)
    valid = flat_position[(flat_position > 0) & (flat_position < np.prod(HR_SHAPE))]
    if valid.size == 0:
        return mask

    hr_coords = np.column_stack(np.unravel_index(valid.astype(np.int64), HR_SHAPE))
    lr_coords = hr_coords // POSITION_DOWNSAMPLE
    shape = np.asarray(grid_shape, dtype=np.int64)
    in_bounds = np.all((lr_coords >= 0) & (lr_coords < shape), axis=1)
    lr_coords = lr_coords[in_bounds]
    if lr_coords.size:
        mask[lr_coords[:, 0], lr_coords[:, 1], lr_coords[:, 2]] = True
    return mask


def _tsdf_measured_mask(tsdf_lr):
    """Approximate measured S from low-res TSDF without labels.

    The local dataloader comments/construction indicate:
    - 0.001: empty/free-like sensor-observed cell
    - 1.0: surface cell
    - -0.001: occluded/unmeasured cell

    For strict no-label scoring, S is approximated as non-negative TSDF cells:
    free-like empty cells plus surface cells. Negative occluded cells remain
    unmeasured. This is conservative with respect to ground-truth leakage.
    """

    tsdf_lr = np.asarray(tsdf_lr, dtype=np.float32)
    return np.isfinite(tsdf_lr) & (tsdf_lr >= TSDF_EMPTY)


def _tsdf_measured_occupied_mask(tsdf_lr):
    """Approximate measured occupied surfaces from TSDF only."""

    tsdf_lr = np.asarray(tsdf_lr, dtype=np.float32)
    return np.isfinite(tsdf_lr) & np.isclose(tsdf_lr, TSDF_SURFACE)


def build_measured_mask_from_sensor_npz(sample_npz, mode="tsdf_lr"):
    """Build measured set S using only sensor-derived fields.

    Modes:
    - tsdf_lr: measured iff tsdf_lr >= 0.001.
    - position: measured iff a depth pixel projects to the voxel.
    - union: union of both masks.
    """

    if mode == "tsdf_lr":
        fields = _load_sensor_fields(sample_npz, required=("tsdf_lr",))
        mask = _tsdf_measured_mask(fields["tsdf_lr"])
    elif mode == "position":
        fields = _load_sensor_fields(sample_npz, required=("position",))
        mask = _position_mask(fields["position"], grid_shape=LR_SHAPE)
    elif mode == "union":
        fields = _load_sensor_fields(sample_npz, required=("tsdf_lr", "position"))
        mask = _tsdf_measured_mask(fields["tsdf_lr"])
        mask |= _position_mask(fields["position"], grid_shape=fields["tsdf_lr"].shape)
    else:
        raise ValueError("mode must be one of: tsdf_lr, position, union")

    if mask.shape != LR_SHAPE:
        raise ValueError(f"Measured mask must have shape {LR_SHAPE}, got {mask.shape}")
    return mask.astype(bool, copy=False)


def build_measured_occupied_mask_from_sensor_npz(sample_npz, mode="tsdf_lr"):
    """Build a sensor-derived blocking mask, never using target labels."""

    if mode == "tsdf_lr":
        fields = _load_sensor_fields(sample_npz, required=("tsdf_lr",))
        mask = _tsdf_measured_occupied_mask(fields["tsdf_lr"])
    elif mode == "position":
        fields = _load_sensor_fields(sample_npz, required=("position",))
        mask = _position_mask(fields["position"], grid_shape=LR_SHAPE)
    elif mode == "union":
        fields = _load_sensor_fields(sample_npz, required=("tsdf_lr", "position"))
        mask = _tsdf_measured_occupied_mask(fields["tsdf_lr"])
        mask |= _position_mask(fields["position"], grid_shape=fields["tsdf_lr"].shape)
    else:
        raise ValueError("mode must be one of: tsdf_lr, position, union")
    return mask.astype(bool, copy=False)


def summarize_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    return {
        "shape": tuple(int(v) for v in mask.shape),
        "measured": int(np.count_nonzero(mask)),
        "unmeasured": int(mask.size - np.count_nonzero(mask)),
        "measured_fraction": float(np.count_nonzero(mask) / mask.size),
        "measured_note": MEASURED_NOTE,
        "strict_no_target_note": STRICT_NO_TARGET_NOTE,
    }


def build_prediction_set_mask(prediction_layer, measured_mask, tau=0.1):
    measured_mask = np.asarray(measured_mask, dtype=bool)
    if tuple(measured_mask.shape) != tuple(prediction_layer.shape()):
        raise ValueError(
            f"measured_mask shape {measured_mask.shape} differs from "
            f"PredictionLayer shape {prediction_layer.shape()}"
        )
    predicted_mask = prediction_layer.confidence >= float(tau)
    return predicted_mask & ~measured_mask


def _choose_start_voxel(measured_voxels):
    center = np.median(measured_voxels.astype(np.float32), axis=0)
    distances = np.linalg.norm(measured_voxels.astype(np.float32) - center, axis=1)
    return tuple(int(v) for v in measured_voxels[int(np.argmin(distances))])


def sample_candidate_views(
    measured_mask,
    grid_shape,
    num_candidates=64,
    seed=0,
):
    """Sample candidate poses from sensor-measured voxels.

    This offline NYU scorer has no reliable online collision-free robot map.
    Therefore candidates are sampled from S itself, and collision/path planning
    is explicitly not implemented here. No target labels are used.
    """

    measured_mask = np.asarray(measured_mask, dtype=bool)
    grid_shape = tuple(int(v) for v in grid_shape)
    if measured_mask.shape != grid_shape:
        raise ValueError(f"measured_mask shape {measured_mask.shape} != {grid_shape}")

    measured_voxels = np.argwhere(measured_mask)
    if measured_voxels.size == 0:
        raise ValueError("No measured voxels available for candidate sampling")

    rng = np.random.default_rng(seed)
    sample_count = min(int(num_candidates), len(measured_voxels))
    selected_ids = rng.choice(len(measured_voxels), size=sample_count, replace=False)
    selected = measured_voxels[selected_ids]

    candidates = []
    for candidate_id, position_arr in enumerate(selected):
        position = tuple(int(v) for v in position_arr)
        yaw = float(rng.uniform(0.0, 2.0 * math.pi))
        candidates.append(CandidateView(id=candidate_id, position=position, yaw=yaw))
    return candidates


def _sample_directions(num_yaw, num_pitch):
    if num_yaw <= 0 or num_pitch <= 0:
        raise ValueError("num_yaw and num_pitch must be positive")

    yaws = np.linspace(0.0, 2.0 * math.pi, int(num_yaw), endpoint=False)
    if int(num_pitch) == 1:
        pitches = np.array([0.0], dtype=np.float32)
    else:
        pitches = np.deg2rad(np.linspace(-30.0, 30.0, int(num_pitch)))

    directions = []
    for pitch in pitches:
        cos_pitch = math.cos(float(pitch))
        sin_pitch = math.sin(float(pitch))
        for yaw in yaws:
            direction = np.array(
                [
                    cos_pitch * math.cos(float(yaw)),
                    sin_pitch,
                    cos_pitch * math.sin(float(yaw)),
                ],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                directions.append(direction / norm)
    return directions


def _in_bounds(voxel, shape):
    return (
        0 <= voxel[0] < shape[0]
        and 0 <= voxel[1] < shape[1]
        and 0 <= voxel[2] < shape[2]
    )


def raycast_visible_voxels(
    candidate,
    measured_mask,
    prediction_layer,
    measured_occupied_mask=None,
    mode="non_blocking",
    max_range=20,
    num_yaw=16,
    num_pitch=5,
    tau=0.1,
):
    """Compute visible voxels with SC-Explorer-style ray casting.

    non_blocking:
        Scene-completion predictions do not block rays. Only a supplied
        sensor-derived measured_occupied_mask may stop a ray.
    sc_blocking:
        Paper ablation mode. Predicted occupied voxels may also stop rays.

    target labels are never used for blocking.
    """

    measured_mask = np.asarray(measured_mask, dtype=bool)
    if tuple(measured_mask.shape) != tuple(prediction_layer.shape()):
        raise ValueError(
            f"measured_mask shape {measured_mask.shape} differs from "
            f"PredictionLayer shape {prediction_layer.shape()}"
        )
    if measured_occupied_mask is not None:
        measured_occupied_mask = np.asarray(measured_occupied_mask, dtype=bool)
        if measured_occupied_mask.shape != measured_mask.shape:
            raise ValueError("measured_occupied_mask shape differs from measured_mask")
    if mode not in ("non_blocking", "sc_blocking"):
        raise ValueError("mode must be non_blocking or sc_blocking")
    if not candidate.valid:
        return []
    if not _in_bounds(candidate.position, measured_mask.shape):
        raise ValueError(f"Candidate position out of bounds: {candidate.position}")

    origin = np.asarray(candidate.position, dtype=np.float32) + 0.5
    visible = set()
    step_size = 0.5
    directions = _sample_directions(num_yaw=num_yaw, num_pitch=num_pitch)

    for direction in directions:
        distance = step_size
        last_voxel = None
        while distance <= float(max_range):
            point = origin + direction * distance
            voxel = tuple(int(math.floor(v)) for v in point)
            if not _in_bounds(voxel, measured_mask.shape):
                break
            if voxel == last_voxel:
                distance += step_size
                continue
            last_voxel = voxel
            visible.add(voxel)

            if measured_occupied_mask is not None and measured_occupied_mask[voxel]:
                break
            if mode == "sc_blocking" and prediction_layer.is_predicted_occupied(
                voxel, tau=tau
            ):
                break

            distance += step_size

    return sorted(visible)


def compute_paper_gains(
    visible_voxels,
    measured_mask,
    prediction_layer,
    tau=0.1,
):
    measured_mask = np.asarray(measured_mask, dtype=bool)
    if tuple(measured_mask.shape) != tuple(prediction_layer.shape()):
        raise ValueError(
            f"measured_mask shape {measured_mask.shape} differs from "
            f"PredictionLayer shape {prediction_layer.shape()}"
        )

    gains = {
        "gain_exp": 0.0,
        "gain_sc": 0.0,
        "gain_hybrid": 0.0,
        "gain_occ": 0.0,
        "gain_conf": 0.0,
        "visible_count": 0,
        "measured_visible_count": 0,
        "predicted_unmeasured_visible_count": 0,
    }

    for voxel in visible_voxels:
        voxel = tuple(int(v) for v in voxel)
        if not _in_bounds(voxel, measured_mask.shape):
            continue

        gains["visible_count"] += 1
        in_s = bool(measured_mask[voxel])
        in_p = bool(prediction_layer.is_predicted(voxel, tau=tau) and not in_s)

        if in_s:
            gains["measured_visible_count"] += 1
        else:
            gains["gain_exp"] += 1.0

        if in_p:
            gains["gain_sc"] += 1.0
            gains["predicted_unmeasured_visible_count"] += 1
            if prediction_layer.is_predicted_occupied(voxel, tau=tau):
                gains["gain_occ"] += 1.0
            gains["gain_conf"] += abs(0.5 - prediction_layer.get_occupied_prob(voxel))

        gains["gain_hybrid"] += (0.0 if in_s else 1.0) + (1.0 if in_p else 0.0)

    return gains


def _wrap_angle(angle):
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def compute_candidate_cost_and_utilities(
    candidate,
    start_position,
    start_yaw=0.0,
    voxel_size=0.08,
    v_max=1.0,
    yaw_rate=math.pi / 2.0,
    eps=1e-6,
):
    distance_vox = float(
        np.linalg.norm(
            np.asarray(candidate.position, dtype=np.float32)
            - np.asarray(start_position, dtype=np.float32)
        )
    )
    distance_m = distance_vox * float(voxel_size)
    time_pos = distance_m / max(float(v_max), eps)
    time_yaw = abs(_wrap_angle(candidate.yaw - float(start_yaw))) / max(float(yaw_rate), eps)
    candidate.path_cost = float(time_pos + time_yaw)

    denom = max(candidate.path_cost, eps)
    candidate.utility_exp = float(candidate.gain_exp / denom)
    candidate.utility_sc = float(candidate.gain_sc / denom)
    candidate.utility_hybrid = float(candidate.gain_hybrid / denom)
    candidate.utility_occ = float(candidate.gain_occ / denom)
    candidate.utility_conf = float(candidate.gain_conf / denom)
    return candidate


def set_candidate_final_score(candidate, gain_mode="hybrid"):
    score_attr = {
        "exp": "utility_exp",
        "sc": "utility_sc",
        "hybrid": "utility_hybrid",
        "occ": "utility_occ",
        "conf": "utility_conf",
    }.get(gain_mode)
    if score_attr is None:
        raise ValueError("gain_mode must be one of: exp, sc, hybrid, occ, conf")
    candidate.final_score = float(getattr(candidate, score_attr))
    return candidate


def apply_gains_to_candidate(candidate, gains):
    candidate.gain_exp = float(gains["gain_exp"])
    candidate.gain_sc = float(gains["gain_sc"])
    candidate.gain_hybrid = float(gains["gain_hybrid"])
    candidate.gain_occ = float(gains["gain_occ"])
    candidate.gain_conf = float(gains["gain_conf"])
    candidate.visible_count = int(gains["visible_count"])
    candidate.measured_visible_count = int(gains["measured_visible_count"])
    candidate.predicted_unmeasured_visible_count = int(
        gains["predicted_unmeasured_visible_count"]
    )
    return candidate


def select_expert_action(candidates, top_n=16):
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate.valid and np.isfinite(candidate.final_score)
    ]
    if not valid_candidates:
        raise ValueError("No valid candidates available for expert action selection")

    top_candidates = sorted(
        valid_candidates, key=lambda candidate: (-candidate.final_score, candidate.id)
    )[: int(top_n)]
    return top_candidates[0], top_candidates, 0


def candidate_feature_vector(candidate):
    return np.array(
        [
            candidate.gain_exp,
            candidate.gain_sc,
            candidate.gain_hybrid,
            candidate.gain_occ,
            candidate.gain_conf,
            candidate.path_cost,
            candidate.utility_exp,
            candidate.utility_sc,
            candidate.utility_hybrid,
            candidate.utility_occ,
            candidate.utility_conf,
            candidate.final_score,
            candidate.visible_count,
            candidate.measured_visible_count,
            candidate.predicted_unmeasured_visible_count,
        ],
        dtype=np.float32,
    )


def _candidate_json(candidate):
    payload = asdict(candidate)
    payload["position"] = list(candidate.position)
    payload["feature_vector"] = candidate_feature_vector(candidate).astype(float).tolist()
    payload.pop("visible_voxels", None)
    return payload


def evaluate_candidates(
    measured_mask,
    prediction_layer,
    measured_occupied_mask=None,
    num_candidates=64,
    top_n=16,
    tau=0.1,
    raycast_mode="non_blocking",
    gain_mode="hybrid",
    voxel_size=0.08,
    v_max=1.0,
    yaw_rate=math.pi / 2.0,
    max_range=20,
    num_yaw=16,
    num_pitch=5,
    seed=0,
):
    measured_mask = np.asarray(measured_mask, dtype=bool)
    if tuple(measured_mask.shape) != tuple(prediction_layer.shape()):
        raise ValueError(
            f"measured_mask shape {measured_mask.shape} differs from "
            f"PredictionLayer shape {prediction_layer.shape()}"
        )

    candidates = sample_candidate_views(
        measured_mask=measured_mask,
        grid_shape=prediction_layer.shape(),
        num_candidates=num_candidates,
        seed=seed,
    )
    measured_voxels = np.argwhere(measured_mask)
    start_position = _choose_start_voxel(measured_voxels)

    for candidate in candidates:
        candidate.visible_voxels = raycast_visible_voxels(
            candidate=candidate,
            measured_mask=measured_mask,
            prediction_layer=prediction_layer,
            measured_occupied_mask=measured_occupied_mask,
            mode=raycast_mode,
            max_range=max_range,
            num_yaw=num_yaw,
            num_pitch=num_pitch,
            tau=tau,
        )
        gains = compute_paper_gains(
            visible_voxels=candidate.visible_voxels,
            measured_mask=measured_mask,
            prediction_layer=prediction_layer,
            tau=tau,
        )
        apply_gains_to_candidate(candidate, gains)
        compute_candidate_cost_and_utilities(
            candidate=candidate,
            start_position=start_position,
            start_yaw=0.0,
            voxel_size=voxel_size,
            v_max=v_max,
            yaw_rate=yaw_rate,
        )
        set_candidate_final_score(candidate, gain_mode=gain_mode)

    best_candidate, top_candidates, expert_action = select_expert_action(
        candidates, top_n=top_n
    )
    return {
        "candidates": candidates,
        "best_candidate": best_candidate,
        "top_candidates": top_candidates,
        "expert_action": expert_action,
        "start_position": start_position,
    }


def save_expert_outputs(
    output_dir,
    sample_npz,
    prediction_npz,
    top_candidates,
    expert_action,
    measured_summary,
    prediction_summary,
    gain_mode,
    measured_mode,
    raycast_mode,
    all_candidate_count,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = Path(sample_npz).stem

    candidate_features = np.stack(
        [candidate_feature_vector(candidate) for candidate in top_candidates], axis=0
    ).astype(np.float32)
    candidate_positions = np.array(
        [candidate.position for candidate in top_candidates], dtype=np.int32
    )
    candidate_yaws = np.array([candidate.yaw for candidate in top_candidates], dtype=np.float32)
    valid_mask = np.array([candidate.valid for candidate in top_candidates], dtype=bool)
    expert_scores = np.array(
        [candidate.final_score for candidate in top_candidates], dtype=np.float32
    )
    top_candidate_ids = np.array([candidate.id for candidate in top_candidates], dtype=np.int64)

    npz_path = output_dir / f"paper_expert_decision_{sample_id}.npz"
    np.savez_compressed(
        npz_path,
        candidate_features=candidate_features,
        feature_names=np.array(FEATURE_NAMES),
        candidate_positions=candidate_positions,
        candidate_yaws=candidate_yaws,
        valid_mask=valid_mask,
        expert_action=np.array(int(expert_action), dtype=np.int64),
        expert_scores=expert_scores,
        top_candidate_ids=top_candidate_ids,
        gain_mode=np.array(str(gain_mode)),
        measured_mode=np.array(str(measured_mode)),
        raycast_mode=np.array(str(raycast_mode)),
        sample_npz=np.array(str(sample_npz)),
        prediction_npz=np.array(str(prediction_npz)),
        measured_note=np.array(MEASURED_NOTE),
        strict_no_target_note=np.array(STRICT_NO_TARGET_NOTE),
        tree_limitation_note=np.array(TREE_LIMITATION_NOTE),
        all_candidate_count=np.array(int(all_candidate_count), dtype=np.int64),
    )

    jsonl_path = output_dir / "paper_expert_decisions.jsonl"
    record = {
        "sample_id": sample_id,
        "sample_npz": str(sample_npz),
        "prediction_npz": str(prediction_npz),
        "output_npz": str(npz_path),
        "feature_names": FEATURE_NAMES,
        "expert_action": int(expert_action),
        "best_candidate_id": int(top_candidates[int(expert_action)].id),
        "gain_mode": gain_mode,
        "measured_mode": measured_mode,
        "raycast_mode": raycast_mode,
        "all_candidate_count": int(all_candidate_count),
        "measured_summary": measured_summary,
        "prediction_summary": prediction_summary,
        "measured_note": MEASURED_NOTE,
        "strict_no_target_note": STRICT_NO_TARGET_NOTE,
        "tree_limitation_note": TREE_LIMITATION_NOTE,
        "top_candidates": [_candidate_json(candidate) for candidate in top_candidates],
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {"npz_path": str(npz_path), "jsonl_path": str(jsonl_path)}


def run_paper_expert_scoring(
    sample_npz,
    prediction_npz,
    output_dir=None,
    num_candidates=64,
    top_n=16,
    tau=0.1,
    measured_mode="tsdf_lr",
    raycast_mode="non_blocking",
    gain_mode="hybrid",
    voxel_size=0.08,
    v_max=1.0,
    yaw_rate_deg=90.0,
    max_range=20,
    num_yaw=16,
    num_pitch=5,
    seed=0,
):
    prediction_layer = PredictionLayer.from_npz(prediction_npz)
    measured_mask = build_measured_mask_from_sensor_npz(sample_npz, mode=measured_mode)
    measured_occupied_mask = build_measured_occupied_mask_from_sensor_npz(
        sample_npz, mode=measured_mode
    )
    p_mask = build_prediction_set_mask(prediction_layer, measured_mask, tau=tau)
    measured_summary = summarize_mask(measured_mask)
    measured_summary["measured_occupied_for_blocking"] = int(
        np.count_nonzero(measured_occupied_mask)
    )
    prediction_summary = {
        "predicted_confident": int(np.count_nonzero(prediction_layer.confidence >= tau)),
        "predicted_unmeasured_P": int(np.count_nonzero(p_mask)),
        "tau": float(tau),
        "P_definition": "prediction_layer.confidence >= tau and not in measured S",
    }

    result = evaluate_candidates(
        measured_mask=measured_mask,
        prediction_layer=prediction_layer,
        measured_occupied_mask=measured_occupied_mask,
        num_candidates=num_candidates,
        top_n=top_n,
        tau=tau,
        raycast_mode=raycast_mode,
        gain_mode=gain_mode,
        voxel_size=voxel_size,
        v_max=v_max,
        yaw_rate=math.radians(float(yaw_rate_deg)),
        max_range=max_range,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
        seed=seed,
    )
    result["measured_mask"] = measured_mask
    result["measured_occupied_mask"] = measured_occupied_mask
    result["prediction_set_mask"] = p_mask
    result["measured_summary"] = measured_summary
    result["prediction_summary"] = prediction_summary
    result["gain_mode"] = gain_mode
    result["measured_mode"] = measured_mode
    result["raycast_mode"] = raycast_mode

    if output_dir is not None:
        output_paths = save_expert_outputs(
            output_dir=output_dir,
            sample_npz=sample_npz,
            prediction_npz=prediction_npz,
            top_candidates=result["top_candidates"],
            expert_action=result["expert_action"],
            measured_summary=measured_summary,
            prediction_summary=prediction_summary,
            gain_mode=gain_mode,
            measured_mode=measured_mode,
            raycast_mode=raycast_mode,
            all_candidate_count=len(result["candidates"]),
        )
        result.update(output_paths)

    return result


def format_top_candidates(top_candidates: Iterable[CandidateView]):
    lines = [
        "rank id position yaw score gain_exp gain_sc gain_hybrid "
        "gain_occ gain_conf path_cost visible measured_visible P_visible"
    ]
    for rank, candidate in enumerate(top_candidates):
        lines.append(
            f"{rank:02d} {candidate.id:03d} {candidate.position} "
            f"{candidate.yaw:.3f} {candidate.final_score:.6f} "
            f"{candidate.gain_exp:.1f} {candidate.gain_sc:.1f} "
            f"{candidate.gain_hybrid:.1f} {candidate.gain_occ:.1f} "
            f"{candidate.gain_conf:.6f} {candidate.path_cost:.6f} "
            f"{candidate.visible_count} {candidate.measured_visible_count} "
            f"{candidate.predicted_unmeasured_visible_count}"
        )
    return "\n".join(lines)

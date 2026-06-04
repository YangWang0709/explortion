#!/usr/bin/env python3
"""Read-only simulator-native prediction layer for Stage 4A-5."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SimPredictionLayer:
    """Small read-only wrapper around global simulator prediction arrays.

    The array shape is expected to match `observed_state`, with axis order
    `(world_x, world_y, world_z)`. This class never writes to observed maps or
    performs traversability/collision queries.
    """

    def __init__(
        self,
        pred_class: np.ndarray,
        confidence: np.ndarray,
        occupied_prob: np.ndarray,
        free_prob: np.ndarray,
        valid: np.ndarray,
        source_npz: str | None = None,
    ):
        self.pred_class = np.asarray(pred_class, dtype=np.uint8)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.occupied_prob = np.asarray(occupied_prob, dtype=np.float32)
        self.free_prob = np.asarray(free_prob, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)
        self.source_npz = source_npz
        self._validate()

        for array in (self.pred_class, self.confidence, self.occupied_prob, self.free_prob, self.valid):
            array.setflags(write=False)

    @classmethod
    def from_npz(cls, path: str | Path) -> "SimPredictionLayer":
        prediction_path = Path(path)
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Global prediction layer npz not found: {prediction_path}")
        with np.load(prediction_path) as data:
            required = (
                "global_pred_class",
                "global_confidence",
                "global_occupied_prob",
                "global_free_prob",
                "global_prediction_valid",
            )
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{prediction_path} is missing required fields: {missing}")
            return cls(
                pred_class=np.array(data["global_pred_class"]),
                confidence=np.array(data["global_confidence"]),
                occupied_prob=np.array(data["global_occupied_prob"]),
                free_prob=np.array(data["global_free_prob"]),
                valid=np.array(data["global_prediction_valid"]),
                source_npz=str(prediction_path),
            )

    def _validate(self) -> None:
        arrays = (self.pred_class, self.confidence, self.occupied_prob, self.free_prob, self.valid)
        if any(array.shape == () for array in arrays):
            raise ValueError("Prediction arrays are required")
        if self.pred_class.ndim != 3:
            raise ValueError(f"Prediction arrays must be 3D, got {self.pred_class.shape}")
        shapes = {array.shape for array in arrays}
        if len(shapes) != 1:
            raise ValueError(f"Prediction array shapes differ: {sorted(shapes)}")
        for name, array in (
            ("confidence", self.confidence),
            ("occupied_prob", self.occupied_prob),
            ("free_prob", self.free_prob),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")

    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.pred_class.shape)

    def _index(self, index: tuple[int, int, int] | list[int] | np.ndarray) -> tuple[int, int, int]:
        if len(index) != 3:
            raise IndexError(f"Expected a 3D voxel index, got {index}")
        idx = tuple(int(v) for v in index)
        shape = self.shape()
        if not (0 <= idx[0] < shape[0] and 0 <= idx[1] < shape[1] and 0 <= idx[2] < shape[2]):
            raise IndexError(f"Index {idx} is outside prediction shape {shape}")
        return idx

    def get_pred_class(self, index: tuple[int, int, int] | list[int] | np.ndarray) -> int:
        return int(self.pred_class[self._index(index)])

    def get_confidence(self, index: tuple[int, int, int] | list[int] | np.ndarray) -> float:
        return float(self.confidence[self._index(index)])

    def get_occupied_prob(self, index: tuple[int, int, int] | list[int] | np.ndarray) -> float:
        return float(self.occupied_prob[self._index(index)])

    def get_free_prob(self, index: tuple[int, int, int] | list[int] | np.ndarray) -> float:
        return float(self.free_prob[self._index(index)])

    def is_predicted(self, index: tuple[int, int, int] | list[int] | np.ndarray, tau: float = 0.1) -> bool:
        idx = self._index(index)
        return bool(self.valid[idx] and self.confidence[idx] >= float(tau))

    def is_predicted_occupied(
        self,
        index: tuple[int, int, int] | list[int] | np.ndarray,
        tau: float = 0.1,
    ) -> bool:
        idx = self._index(index)
        return bool(self.is_predicted(idx, tau=tau) and self.occupied_prob[idx] >= 0.5)

    def is_predicted_free(
        self,
        index: tuple[int, int, int] | list[int] | np.ndarray,
        tau: float = 0.1,
    ) -> bool:
        idx = self._index(index)
        return bool(self.is_predicted(idx, tau=tau) and self.free_prob[idx] >= 0.5)

    def get_prediction_gain(
        self,
        index: tuple[int, int, int] | list[int] | np.ndarray,
        tau: float = 0.1,
    ) -> float:
        confidence = self.get_confidence(index)
        return confidence if self.is_predicted(index, tau=tau) else 0.0


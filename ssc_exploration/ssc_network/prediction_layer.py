#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Lightweight read-only wrapper for offline SSCNet prediction arrays."""

from pathlib import Path

import numpy as np


class PredictionLayer:
    def __init__(
        self,
        prediction_npz_path=None,
        pred_class=None,
        confidence=None,
        occupied_prob=None,
        free_prob=None,
        free_class_id=0,
    ):
        if prediction_npz_path and all(
            value is None for value in (pred_class, confidence, occupied_prob, free_prob)
        ):
            loaded = self._load_npz_arrays(prediction_npz_path)
            pred_class = loaded["pred_class"]
            confidence = loaded["confidence"]
            occupied_prob = loaded["occupied_prob"]
            free_prob = loaded["free_prob"]
            free_class_id = loaded["free_class_id"]

        self.prediction_npz_path = str(prediction_npz_path) if prediction_npz_path else None
        self.pred_class = np.asarray(pred_class, dtype=np.uint8)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.occupied_prob = np.asarray(occupied_prob, dtype=np.float32)
        self.free_prob = np.asarray(free_prob, dtype=np.float32)
        self.free_class_id = int(free_class_id)
        self._validate()

    @classmethod
    def from_npz(cls, path):
        return cls(prediction_npz_path=path)

    @staticmethod
    def _load_npz_arrays(path):
        prediction_path = Path(path)
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Prediction npz not found: {prediction_path}")

        with np.load(prediction_path) as data:
            required = ("pred_class", "confidence", "occupied_prob", "free_prob")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{prediction_path} is missing required fields: {missing}")
            free_class_id = int(data["free_class_id"].item()) if "free_class_id" in data.files else 0
            return {
                "pred_class": np.array(data["pred_class"]),
                "confidence": np.array(data["confidence"]),
                "occupied_prob": np.array(data["occupied_prob"]),
                "free_prob": np.array(data["free_prob"]),
                "free_class_id": free_class_id,
            }

    def _validate(self):
        if any(
            array.shape == ()
            for array in (self.pred_class, self.confidence, self.occupied_prob, self.free_prob)
        ):
            raise ValueError("Prediction arrays are required")
        if self.pred_class.shape != self.confidence.shape:
            raise ValueError("pred_class and confidence shapes differ")
        if self.pred_class.shape != self.occupied_prob.shape:
            raise ValueError("pred_class and occupied_prob shapes differ")
        if self.pred_class.shape != self.free_prob.shape:
            raise ValueError("pred_class and free_prob shapes differ")
        if self.pred_class.ndim != 3:
            raise ValueError(f"Prediction arrays must be 3D, got {self.pred_class.shape}")

    def shape(self):
        return self.pred_class.shape

    def _index(self, index):
        if len(index) != 3:
            raise IndexError(f"Expected a 3D voxel index, got {index}")
        return tuple(int(i) for i in index)

    def get_pred_class(self, index):
        return int(self.pred_class[self._index(index)])

    def get_confidence(self, index):
        return float(self.confidence[self._index(index)])

    def get_occupied_prob(self, index):
        return float(self.occupied_prob[self._index(index)])

    def get_free_prob(self, index):
        return float(self.free_prob[self._index(index)])

    def is_predicted(self, index, tau=0.1):
        return self.get_confidence(index) >= tau

    def is_predicted_occupied(self, index, tau=0.1):
        return self.get_occupied_prob(index) >= 0.5 and self.get_confidence(index) >= tau

    def is_predicted_free(self, index, tau=0.1):
        return self.get_free_prob(index) >= 0.5 and self.get_confidence(index) >= tau

    def get_prediction_gain(self, index, tau=0.1):
        confidence = self.get_confidence(index)
        return confidence if confidence >= tau else 0.0

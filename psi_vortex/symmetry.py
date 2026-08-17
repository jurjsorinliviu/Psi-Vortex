"""Data-driven odd/even symmetry detection for complete trajectory data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import Trajectory
from .initialization import physics_aware_initialize, symmetry_orthogonal_initialize


@dataclass(frozen=True)
class SymmetryResult:
    symmetry: str
    confidence: float
    odd_score: float
    even_score: float
    matched_pairs: int


class AutoSymmetryDetector:
    def __init__(self, tolerance: float = 0.15, min_confidence: float = 0.7):
        if tolerance <= 0 or min_confidence < 0 or min_confidence > 1:
            raise ValueError("invalid symmetry detector thresholds")
        self.tolerance = tolerance
        self.min_confidence = min_confidence

    def detect(self, x: torch.Tensor, y: torch.Tensor) -> SymmetryResult:
        inputs = x.detach().cpu().reshape(-1).to(torch.float64)
        outputs = y.detach().cpu().reshape(-1).to(torch.float64)
        if inputs.numel() != outputs.numel() or inputs.numel() < 10:
            raise ValueError("symmetry detection requires at least ten matching values")
        positive = torch.nonzero(inputs > self.tolerance, as_tuple=False).flatten()
        negative = torch.nonzero(inputs < -self.tolerance, as_tuple=False).flatten()
        odd_errors: list[float] = []
        even_errors: list[float] = []
        scale = max(float(outputs.std(unbiased=False)), np.finfo(float).eps)
        for index in positive:
            distances = (inputs[negative].abs() - inputs[index].abs()).abs()
            closest = int(torch.argmin(distances))
            if float(distances[closest]) <= self.tolerance * max(abs(float(inputs[index])), 1.0):
                reflected = outputs[negative[closest]]
                odd_errors.append(abs(float(outputs[index] + reflected)) / scale)
                even_errors.append(abs(float(outputs[index] - reflected)) / scale)
        if len(odd_errors) < 5:
            return SymmetryResult("none", 0.0, 0.0, 0.0, len(odd_errors))
        odd_score = max(0.0, 1.0 - float(np.mean(odd_errors)))
        even_score = max(0.0, 1.0 - float(np.mean(even_errors)))
        best = max(odd_score, even_score)
        symmetry = "none"
        if best >= self.min_confidence:
            symmetry = "odd" if odd_score > even_score else "even"
        return SymmetryResult(symmetry, best, odd_score, even_score, len(odd_errors))

    def detect_trajectories(
        self, trajectories: list[Trajectory], feature_index: int = 0, output_index: int = 0
    ) -> SymmetryResult:
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        x = torch.cat([item.features[:, feature_index] for item in trajectories])
        y = torch.cat([item.targets[:, output_index] for item in trajectories])
        return self.detect(x, y)


def apply_auto_physics_initialization(
    model: nn.Module,
    trajectories: list[Trajectory],
    *,
    time_constant: float,
    delta_t: float,
    feature_index: int = 0,
    output_index: int = 0,
) -> SymmetryResult:
    """Detect symmetry, encode its Equation-5 mask, then set physical retention."""
    result = AutoSymmetryDetector().detect_trajectories(
        trajectories, feature_index, output_index
    )
    symmetry_orthogonal_initialize(
        model, result.symmetry, input_feature=feature_index, preserve_recurrence=True
    )
    physics_aware_initialize(model, time_constant, delta_t)
    return result

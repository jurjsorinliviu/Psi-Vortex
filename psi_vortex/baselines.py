"""Matched-split baselines with the same chronological tensor interface."""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import torch
from torch import nn

from .contracts import require_sequence
from .data import Trajectory, iter_batches


class StaticMLP(nn.Module):
    """Explicit pointwise control; timesteps are never reordered."""

    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        require_sequence(sequence)
        return self.network(sequence)


class VanillaLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, sequence: torch.Tensor, state=None):
        require_sequence(sequence)
        hidden, state = self.recurrent(sequence, state)
        return self.readout(hidden), state, {"recurrent": hidden}


class ChebyshevKANLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, degree: int = 3):
        super().__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(torch.empty(output_size, input_size, degree + 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(value)
        terms = [torch.ones_like(bounded), bounded]
        for _ in range(2, self.degree + 1):
            terms.append(2 * bounded * terms[-1] - terms[-2])
        basis = torch.stack(terms[: self.degree + 1], dim=-1)
        return torch.einsum("...id,oid->...o", basis, self.coefficients)


class PIKAN(nn.Module):
    """Physics-compatible KAN baseline without a third-party binary dependency."""

    def __init__(self, input_size: int, hidden_size: int = 16, output_size: int = 1, degree: int = 3):
        super().__init__()
        self.first = ChebyshevKANLayer(input_size, hidden_size, degree)
        self.second = ChebyshevKANLayer(hidden_size, output_size, degree)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        require_sequence(sequence)
        return self.second(torch.tanh(self.first(sequence)))


def train_static_model(
    model: nn.Module,
    trajectories: list[Trajectory],
    *,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 4,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> nn.Module:
    torch.manual_seed(seed)
    # Construction happens before this function; reset here so the declared seed
    # controls model initialization as well as chronological batch order.
    model.apply(
        lambda module: module.reset_parameters()
        if hasattr(module, "reset_parameters")
        else None
    )
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        generator = __import__("random").Random(seed + epoch)
        for batch in iter_batches(trajectories, batch_size, shuffle_trajectories=True, generator=generator):
            batch = batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch.features)
            loss = nn.functional.mse_loss(prediction, batch.targets)
            loss.backward()
            optimizer.step()
    return model


def _powers(feature_count: int, degree: int) -> list[tuple[int, ...]]:
    terms: list[tuple[int, ...]] = [()]
    for current_degree in range(1, degree + 1):
        terms.extend(combinations_with_replacement(range(feature_count), current_degree))
    return terms


def polynomial_library(values: np.ndarray, degree: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("polynomial library input must be [observations, features]")
    powers = _powers(array.shape[1], degree)
    columns = [np.ones(array.shape[0])]
    for power in powers[1:]:
        column = np.ones(array.shape[0])
        for index in power:
            column *= array[:, index]
        columns.append(column)
    return np.column_stack(columns), powers


class SINDyRegressor:
    """Thresholded polynomial static baseline evaluated on matched chronological splits."""

    def __init__(self, degree: int = 3, threshold: float = 1e-4, iterations: int = 8):
        self.degree = degree
        self.threshold = threshold
        self.iterations = iterations
        self.coefficients: np.ndarray | None = None
        self.powers: list[tuple[int, ...]] | None = None

    def fit(self, trajectories: list[Trajectory]) -> "SINDyRegressor":
        if not trajectories:
            raise ValueError("at least one training trajectory is required")
        features = np.concatenate([item.features.detach().cpu().numpy() for item in trajectories])
        targets = np.concatenate([item.targets.detach().cpu().numpy() for item in trajectories])
        library, self.powers = polynomial_library(features, self.degree)
        coefficients, _, _, _ = np.linalg.lstsq(library, targets, rcond=None)
        for _ in range(self.iterations):
            small = np.abs(coefficients) < self.threshold
            coefficients[small] = 0
            for output in range(targets.shape[1]):
                active = ~small[:, output]
                if active.any():
                    coefficients[active, output], _, _, _ = np.linalg.lstsq(
                        library[:, active], targets[:, output], rcond=None
                    )
        self.coefficients = coefficients
        return self

    def predict(self, trajectory: Trajectory) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("SINDyRegressor must be fitted before prediction")
        library, _ = polynomial_library(
            trajectory.features.detach().cpu().numpy(), self.degree
        )
        return library @ self.coefficients

    @property
    def active_terms(self) -> int:
        if self.coefficients is None:
            raise RuntimeError("SINDyRegressor is not fitted")
        return int(np.count_nonzero(self.coefficients))

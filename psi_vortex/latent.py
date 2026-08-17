"""Latent-coupling recovery and prespecified acceptance safeguards."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import r2_score


@dataclass(frozen=True)
class CouplingEstimate:
    value: float
    intercept: float
    r2: float
    accepted: bool
    threshold: float
    observations: int


def fit_coupling(
    response: np.ndarray,
    driver: np.ndarray,
    *,
    nuisance: np.ndarray | None = None,
    r2_threshold: float = 0.8,
    driver_band: float | None = None,
) -> CouplingEstimate:
    """Free-intercept OLS; never emits an accepted coefficient below the R² gate."""
    y = np.asarray(response, dtype=float).reshape(-1)
    x = np.asarray(driver, dtype=float).reshape(-1)
    if y.shape != x.shape or y.size < 3:
        raise ValueError("response and driver require matching lengths of at least three")
    if driver_band is not None and driver_band < 0:
        raise ValueError("driver_band must be non-negative")
    finite = np.isfinite(y) & np.isfinite(x)
    nuisance_array = None
    if nuisance is not None:
        nuisance_array = np.asarray(nuisance, dtype=float)
        if nuisance_array.shape[0] != y.size:
            raise ValueError("nuisance rows must match response length")
        if nuisance_array.ndim == 1:
            nuisance_array = nuisance_array[:, None]
        if nuisance_array.ndim != 2:
            raise ValueError("nuisance must be one- or two-dimensional")
        finite &= np.isfinite(nuisance_array).all(axis=1)
    selected = finite
    if driver_band is not None:
        outside_band = finite & (np.abs(x) > driver_band)
        if int(outside_band.sum()) >= 5:
            selected = outside_band
    if int(selected.sum()) < 3:
        raise ValueError("at least three finite observations are required")
    y = y[selected]
    x = x[selected]
    if nuisance_array is not None:
        nuisance_array = nuisance_array[selected]
    columns = [np.ones_like(x), x]
    if nuisance_array is not None:
        columns.extend(nuisance_array[:, index] for index in range(nuisance_array.shape[1]))
    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ coefficients
    score = r2_score(prediction, y)
    return CouplingEstimate(
        value=float(coefficients[1]),
        intercept=float(coefficients[0]),
        r2=score,
        accepted=bool(score >= r2_threshold),
        threshold=float(r2_threshold),
        observations=int(y.size),
    )


def relative_error(estimate: float, truth: float) -> float:
    if truth == 0:
        return abs(float(estimate))
    return abs(float(estimate) - float(truth)) / abs(float(truth))


def aggregate_estimates(estimates: list[CouplingEstimate]) -> dict[str, float | int]:
    if not estimates:
        raise ValueError("at least one estimate is required")
    values = np.asarray([item.value for item in estimates])
    scores = np.asarray([item.r2 for item in estimates])
    return {
        "value_mean": float(values.mean()),
        "value_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "r2_mean": float(scores.mean()),
        "r2_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "accepted_seeds": int(sum(item.accepted for item in estimates)),
        "total_seeds": len(estimates),
    }

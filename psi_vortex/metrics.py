"""Element-weighted and trajectory-safe evaluation metrics."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    return float((prediction - target).square().sum() / target.numel())


def r2_score(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    pred = np.asarray(prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else prediction, dtype=float)
    true = np.asarray(target.detach().cpu() if isinstance(target, torch.Tensor) else target, dtype=float)
    if pred.shape != true.shape:
        raise ValueError("prediction and target shapes must match")
    residual = float(np.square(pred - true).sum())
    total = float(np.square(true - true.mean()).sum())
    return 0.0 if total <= np.finfo(float).eps else 1.0 - residual / total


def correlation(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    pred = np.asarray(prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else prediction, dtype=float).ravel()
    true = np.asarray(target.detach().cpu() if isinstance(target, torch.Tensor) else target, dtype=float).ravel()
    if pred.size != true.size or pred.size < 2:
        raise ValueError("correlation requires matching arrays with at least two elements")
    if np.std(pred) == 0 or np.std(true) == 0:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


def nrmse(prediction: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    pred = np.asarray(prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else prediction, dtype=float)
    true = np.asarray(target.detach().cpu() if isinstance(target, torch.Tensor) else target, dtype=float)
    if pred.shape != true.shape:
        raise ValueError("prediction and target shapes must match")
    span = float(true.max() - true.min())
    root = math.sqrt(float(np.square(pred - true).mean()))
    return root / max(span, np.finfo(float).eps)


def aggregate_element_mse(pairs: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
    squared = 0.0
    count = 0
    for prediction, target in pairs:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        squared += float((prediction - target).square().sum())
        count += target.numel()
    if count == 0:
        raise ValueError("at least one prediction/target pair is required")
    return squared / count

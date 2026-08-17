"""Sequence invariants shared by every public Ψ-Vortex path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

State = Any


def require_sequence(x: torch.Tensor, *, name: str = "input", min_length: int = 2) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, sequence_length, features], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[2] < 1:
        raise ValueError(f"{name} must have non-empty batch and feature dimensions")
    if x.shape[1] < min_length:
        raise ValueError(f"{name} sequence_length must be >= {min_length}, got {x.shape[1]}")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} must contain only finite values")
    return x


def require_chronological(time: torch.Tensor) -> torch.Tensor:
    require_sequence(time, name="time")
    if time.shape[-1] != 1:
        raise ValueError("time must have exactly one feature")
    if torch.any(time[:, 1:] <= time[:, :-1]):
        raise ValueError("timesteps must be strictly increasing within each trajectory/window")
    return time


def detach_state(state: State) -> State:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_state(v) for v in state)
    if isinstance(state, list):
        return [detach_state(v) for v in state]
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    raise TypeError(f"unsupported recurrent state type: {type(state)!r}")


@dataclass(frozen=True)
class SequenceBatch:
    features: torch.Tensor
    targets: torch.Tensor
    time: torch.Tensor
    trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sequence(self.features, name="features")
        require_sequence(self.targets, name="targets")
        require_chronological(self.time)
        if self.features.shape[:2] != self.targets.shape[:2] or self.features.shape[:2] != self.time.shape[:2]:
            raise ValueError("features, targets, and time must share batch and sequence dimensions")
        if len(self.trajectory_ids) != self.features.shape[0]:
            raise ValueError("one trajectory_id is required per batch element")

    def to(self, device: torch.device | str) -> "SequenceBatch":
        return SequenceBatch(self.features.to(device), self.targets.to(device), self.time.to(device), self.trajectory_ids)

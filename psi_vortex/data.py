"""Chronological trajectory and window construction."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator, Sequence

import torch

from .contracts import SequenceBatch, require_chronological, require_sequence


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    features: torch.Tensor  # [time, features]
    targets: torch.Tensor   # [time, outputs]
    time: torch.Tensor      # [time, 1]
    source_trajectory_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_trajectory_id is None: object.__setattr__(self,"source_trajectory_id",self.trajectory_id)
        if self.features.ndim != 2 or self.targets.ndim != 2 or self.time.ndim != 2:
            raise ValueError("trajectory tensors must have shape [sequence_length, features]")
        require_sequence(self.features.unsqueeze(0), name="features")
        require_chronological(self.time.unsqueeze(0))
        if self.features.shape[0] != self.targets.shape[0] or self.features.shape[0] != self.time.shape[0]:
            raise ValueError("trajectory tensors must have equal sequence lengths")


def make_windows(trajectory: Trajectory, window_length: int, stride: int | None = None) -> list[Trajectory]:
    if window_length < 2:
        raise ValueError("window_length must be > 1 for recurrent experiments")
    stride = window_length if stride is None else stride
    if stride < 1:
        raise ValueError("stride must be positive")
    windows = []
    for start in range(0, len(trajectory.time) - window_length + 1, stride):
        stop = start + window_length
        windows.append(Trajectory(f"{trajectory.trajectory_id}:{start}:{stop}", trajectory.features[start:stop],
                                  trajectory.targets[start:stop], trajectory.time[start:stop],trajectory.source_trajectory_id))
    return windows


def iter_batches(trajectories: Sequence[Trajectory], batch_size: int, *, shuffle_trajectories: bool = False,
                 generator: random.Random | None = None) -> Iterator[SequenceBatch]:
    """Shuffle whole trajectories/windows only; never reorder their timesteps."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    order = list(range(len(trajectories)))
    if shuffle_trajectories:
        (generator or random).shuffle(order)
    # Bucket only complete items by length; this supports heterogeneous trajectories
    # without padding, truncation, or movement of a timestep across its source item.
    buckets: dict[int, list[Trajectory]] = {}
    groups: list[list[Trajectory]] = []
    for index in order:
        item = trajectories[index]
        bucket = buckets.setdefault(len(item.time), [])
        bucket.append(item)
        if len(bucket) == batch_size:
            groups.append(bucket.copy())
            bucket.clear()
    groups.extend(bucket.copy() for bucket in buckets.values() if bucket)
    for group in groups:
        yield SequenceBatch(torch.stack([x.features for x in group]), torch.stack([x.targets for x in group]),
                            torch.stack([x.time for x in group]), tuple(x.trajectory_id for x in group))


def contiguous_chunks(batch: SequenceBatch, chunk_length: int) -> Iterator[tuple[int, SequenceBatch]]:
    """Yield ordered TBPTT chunks without ever creating a length-one sequence.

    When the requested chunk size would leave one final timestep, that timestep is
    appended to the current chunk. The resulting chunk may therefore be one sample
    longer than ``chunk_length``. This keeps every sample and preserves order.
    """
    if chunk_length < 2:
        raise ValueError("chunk_length must be > 1")
    length = batch.features.shape[1]
    start = 0
    while start < length:
        stop = min(start + chunk_length, length)
        if length - stop == 1:
            stop = length
        yield start, SequenceBatch(batch.features[:, start:stop], batch.targets[:, start:stop],
                                   batch.time[:, start:stop], batch.trajectory_ids)
        start = stop

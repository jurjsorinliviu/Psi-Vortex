"""Validation-only recurrent teacher architecture selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import random

import torch

from .data import Trajectory, iter_batches
from .models import PsiXLSTMTeacher
from .pipeline import assert_source_disjoint
from .trainer import SequenceTrainer, evaluate


@dataclass(frozen=True)
class ArchitectureCandidate:
    teacher_hidden: int
    teacher_blocks: int
    teacher_memory_size: int


@dataclass(frozen=True)
class ArchitectureScore:
    candidate: ArchitectureCandidate
    validation_mse: float
    test_mse: float | None = None


def select_architecture(
    candidates: Iterable[ArchitectureCandidate],
    train: list[Trajectory],
    validation: list[Trajectory],
    test: list[Trajectory],
    *,
    seed: int,
    teacher_epochs: int,
    batch_size: int,
    chunk_length: int | None,
    device: str,
    input_size: int,
    output_size: int,
    learning_rate: float = 1e-3,
    max_grad_norm: float | None = None,
) -> tuple[PsiXLSTMTeacher, ArchitectureScore, list[ArchitectureScore]]:
    """Select ``(hidden, blocks, memory)`` on validation; reveal test afterward."""
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("at least one architecture candidate is required")
    assert_source_disjoint(train, validation, test)
    trials: list[tuple[PsiXLSTMTeacher, ArchitectureScore]] = []
    for candidate in candidate_list:
        torch.manual_seed(seed)
        model = PsiXLSTMTeacher(
            input_size,
            candidate.teacher_hidden,
            output_size,
            num_blocks=candidate.teacher_blocks,
            memory_size=candidate.teacher_memory_size,
        ).to(device)
        trainer = SequenceTrainer(
            model,
            torch.optim.Adam(model.parameters(), lr=learning_rate),
            chunk_length,
            max_grad_norm,
        )
        for epoch in range(teacher_epochs):
            trainer.train_epoch(
                iter_batches(
                    train,
                    batch_size,
                    shuffle_trajectories=True,
                    generator=random.Random(seed + epoch),
                )
            )
        validation_mse = evaluate(model, iter_batches(validation, 1))["mse"]
        trials.append(
            (model, ArchitectureScore(candidate, validation_mse, None))
        )
    selected_model, selected_score = min(trials, key=lambda item: item[1].validation_mse)
    selected_score = ArchitectureScore(
        selected_score.candidate,
        selected_score.validation_mse,
        evaluate(selected_model, iter_batches(test, 1))["mse"],
    )
    scores = [
        selected_score if score.candidate == selected_score.candidate else score
        for _, score in trials
    ]
    return selected_model, selected_score, scores


def candidate_record(score: ArchitectureScore) -> dict[str, object]:
    return {
        **asdict(score.candidate),
        "validation_mse": score.validation_mse,
        "test_mse": score.test_mse,
    }

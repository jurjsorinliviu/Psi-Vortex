"""End-to-end sequence pipeline with source-disjoint experimental splits."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Literal

import torch

from .bic import select_cluster_count
from .data import Trajectory, iter_batches
from .export import export_torchscript
from .models import LowRankRecurrentStudent, PsiXLSTMTeacher, RecurrentStudent
from .trainer import SequenceTrainer, VortexTrainer, evaluate


def assert_source_disjoint(*splits: list[Trajectory]) -> None:
    identities = [{item.source_trajectory_id for item in split} for split in splits]
    for left in range(len(identities)):
        for right in range(left + 1, len(identities)):
            overlap = identities[left] & identities[right]
            if overlap:
                raise ValueError(
                    "train, validation, and test source_trajectory_ids must be disjoint; "
                    f"overlap={sorted(overlap)}"
                )


def _independent_evaluation_batches(trajectories: list[Trajectory]):
    """Use one trajectory per batch so arbitrary lengths remain independent."""
    return iter_batches(trajectories, 1, shuffle_trajectories=False)


class EndToEndPipeline:
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        teacher_hidden: int = 64,
        student_hidden: int = 16,
        teacher_blocks: int = 4,
        *,
        teacher_memory_size: int | None = None,
        student_type: Literal["gru", "low_rank"] = "gru",
        student_layers: int = 1,
        student_rank: int = 4,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.device = torch.device(device)
        self.teacher = PsiXLSTMTeacher(
            input_size, teacher_hidden, output_size, num_blocks=teacher_blocks,
            memory_size=teacher_memory_size,
        ).to(self.device)
        if student_type == "gru":
            self.student = RecurrentStudent(
                input_size, student_hidden, output_size, num_layers=student_layers
            ).to(self.device)
        elif student_type == "low_rank":
            self.student = LowRankRecurrentStudent(
                input_size, student_hidden, student_rank, output_size
            ).to(self.device)
        else:
            raise ValueError("student_type must be 'gru' or 'low_rank'")
        self.student_type = student_type
        self.selected_cluster_count: int | None = None

    def fit(
        self,
        train_trajectories: list[Trajectory],
        validation_trajectories: list[Trajectory],
        test_trajectories: list[Trajectory],
        *,
        teacher_epochs: int = 10,
        student_epochs: int = 10,
        batch_size: int = 4,
        chunk_length: int | None = None,
        lr: float | None = None,
        teacher_lr: float | None = None,
        student_lr: float | None = None,
        max_grad_norm: float | None = None,
        cluster_candidates: tuple[int, ...] = (4, 8, 16),
        lambda_bic: float = 0.01,
        bic_start_epoch: int = 0,
        rrad_weights: tuple[float, float, float, float] = (1.0, 0.5, 1.0, 0.5),
        parameter_regularizer: tuple[str, float] | None = None,
        evaluate_test: bool = True,
    ) -> dict[str, dict[str, float]]:
        if not train_trajectories or not validation_trajectories or not test_trajectories:
            raise ValueError("explicit non-empty train, validation, and test splits are required")
        if teacher_epochs < 0 or student_epochs < 0:
            raise ValueError("epoch counts cannot be negative")
        if bic_start_epoch < 0 or bic_start_epoch > student_epochs:
            raise ValueError("bic_start_epoch must be between zero and student_epochs")
        assert_source_disjoint(train_trajectories, validation_trajectories, test_trajectories)
        common_lr = 1e-3 if lr is None else lr
        teacher_rate = common_lr if teacher_lr is None else teacher_lr
        student_rate = common_lr if student_lr is None else student_lr
        teacher_trainer = SequenceTrainer(
            self.teacher,
            torch.optim.Adam(self.teacher.parameters(), lr=teacher_rate),
            chunk_length,
            max_grad_norm,
        )
        for epoch in range(teacher_epochs):
            teacher_trainer.train_epoch(
                iter_batches(
                    train_trajectories,
                    batch_size,
                    shuffle_trajectories=True,
                    generator=random.Random(self.seed + epoch),
                )
            )
        vortex = VortexTrainer(
            self.teacher,
            self.student,
            torch.optim.Adam(self.student.parameters(), lr=student_rate),
            chunk_length,
            lambda_bic,
            rrad_weights,
            parameter_regularizer,
            max_grad_norm,
        )
        for epoch in range(student_epochs):
            vortex.lambda_bic = lambda_bic if epoch >= bic_start_epoch else 0.0
            vortex.train_epoch(
                iter_batches(
                    train_trajectories,
                    batch_size,
                    shuffle_trajectories=True,
                    generator=random.Random(self.seed + teacher_epochs + epoch),
                )
            )
        if cluster_candidates:
            observations = sum(item.targets.numel() for item in validation_trajectories)

            def validation_loss(candidate):
                squared_error: torch.Tensor | None = None
                count = 0
                for batch in _independent_evaluation_batches(validation_trajectories):
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        prediction, _, _ = candidate(batch.features, None)
                    batch_sse = (prediction - batch.targets).square().sum()
                    squared_error = batch_sse if squared_error is None else squared_error + batch_sse
                    count += batch.targets.numel()
                assert squared_error is not None
                return squared_error / count

            _, requested_count, clustered = select_cluster_count(
                self.student, list(cluster_candidates), validation_loss, observations
            )
            self.student = clustered
            self.selected_cluster_count = clustered.materialized_cluster_count
            if self.selected_cluster_count > requested_count:
                raise AssertionError("materialized cluster count cannot exceed requested count")
        metrics = {
            "validation": evaluate(
                self.student, _independent_evaluation_batches(validation_trajectories)
            )
        }
        if evaluate_test:
            metrics["test"] = evaluate(
                self.student, _independent_evaluation_batches(test_trajectories)
            )
        return metrics

    def evaluate(self, trajectories: list[Trajectory]) -> dict[str, float]:
        if not trajectories:
            raise ValueError("at least one evaluation trajectory is required")
        return evaluate(self.student, _independent_evaluation_batches(trajectories))

    def export(self, example_sequence: torch.Tensor, destination: str | Path):
        return export_torchscript(self.student, example_sequence, destination)

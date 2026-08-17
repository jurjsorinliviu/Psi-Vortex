"""Teacher, distillation, and BIC stages with trajectory-safe state handling."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn
from .bic import DifferentiableBIC
from .contracts import SequenceBatch, detach_state
from .data import contiguous_chunks
from .rrad import RRADLoss


@dataclass
class TrainMetrics:
    loss: float = 0.0
    chunks: int = 0


class SequenceTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 chunk_length: int | None = None, max_grad_norm: float | None = None):
        self.model, self.optimizer, self.chunk_length = model, optimizer, chunk_length
        self.max_grad_norm = max_grad_norm

    def _chunks(self, batch: SequenceBatch):
        return contiguous_chunks(batch, self.chunk_length) if self.chunk_length else [(0, batch)]

    def train_epoch(self, batches: Iterable[SequenceBatch]) -> TrainMetrics:
        self.model.train(); result = TrainMetrics()
        for batch in batches:
            batch=batch.to(next(self.model.parameters()).device)
            state = None  # mandatory reset at every independent batch of trajectories
            for _, chunk in self._chunks(batch):
                self.optimizer.zero_grad()
                prediction, state, _ = self.model(chunk.features, state)
                loss = nn.functional.mse_loss(prediction, chunk.targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite recurrent training loss")
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                finite_gradients = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in self.model.parameters()
                )
                if finite_gradients:
                    self.optimizer.step()
                else:
                    raise FloatingPointError("non-finite recurrent training gradient")
                state = detach_state(state)  # propagate only to the next contiguous chunk
                result.loss += float(loss.detach()); result.chunks += 1
        result.loss /= max(result.chunks, 1)
        return result


class VortexTrainer:
    """RRAD distillation plus BIC, with identical sequence/state semantics."""
    def __init__(self, teacher: nn.Module, student: nn.Module, optimizer: torch.optim.Optimizer,
                 chunk_length: int | None = None, lambda_bic: float = .01,
                 rrad_weights: tuple[float, float, float, float] = (1.0, .5, 1.0, .5),
                 parameter_regularizer: tuple[str, float] | None = None,
                 max_grad_norm: float | None = None):
        self.teacher, self.student, self.optimizer = teacher, student, optimizer
        self.chunk_length, self.lambda_bic = chunk_length, lambda_bic
        self.parameter_regularizer = parameter_regularizer
        self.max_grad_norm = max_grad_norm
        teacher_hidden=teacher.readout.in_features; student_hidden=student.readout.in_features
        device=next(student.parameters()).device
        self.rrad, self.bic = RRADLoss(teacher_hidden,student_hidden,*rrad_weights).to(device), DifferentiableBIC().to(device)
        optimizer.add_param_group({"params": self.rrad.parameters()})
        for parameter in teacher.parameters(): parameter.requires_grad_(False)

    def train_epoch(self, batches: Iterable[SequenceBatch]) -> TrainMetrics:
        self.teacher.eval(); self.student.train(); result = TrainMetrics()
        for batch in batches:
            batch=batch.to(next(self.student.parameters()).device)
            teacher_state = student_state = None
            chunks = list(contiguous_chunks(batch, self.chunk_length)) if self.chunk_length else [(0, batch)]
            # BIC N must count the same scalar observations used by the MSE denominator.
            total_observations=batch.targets.numel()
            self.optimizer.zero_grad(); supervised_sse=[]; supervised_counts=[]; rrad_statistics=[]; previous=None
            for _, chunk in chunks:
                with torch.no_grad(): teacher_y, teacher_state, teacher_aux = self.teacher(chunk.features, teacher_state)
                student_y, student_state, student_aux = self.student(chunk.features, student_state)
                _,rrad_components=self.rrad(student_y,teacher_y,chunk.time,student_aux["recurrent"],teacher_aux["recurrent"],previous)
                supervised = nn.functional.mse_loss(student_y, chunk.targets)
                supervised_sse.append((student_y-chunk.targets).square().sum()); supervised_counts.append(chunk.targets.numel())
                rrad_statistics.append(rrad_components["statistics"])
                previous=detach_state({"student":student_y[:,-1:],"teacher":teacher_y[:,-1:],"student_hidden":student_aux["recurrent"][:,-1:],"teacher_hidden":teacher_aux["recurrent"][:,-1:],"time":chunk.time[:,-1:]})
                teacher_state, student_state = detach_state(teacher_state), detach_state(student_state)
            mean_mse=sum(supervised_sse)/sum(supervised_counts); rrad=self.rrad.aggregate(rrad_statistics)
            # One complete-trajectory update holds parameters fixed across the TBPTT
            # partition and uses the full N log(MSE) + log(N) DoF score.
            structural=self.bic(self.student,mean_mse,total_observations) if self.lambda_bic else mean_mse.new_zeros(())
            penalty=mean_mse.new_zeros(())
            if self.parameter_regularizer is not None:
                kind,weight=self.parameter_regularizer
                if kind not in ("l1","l2") or weight<0: raise ValueError("regularizer must be ('l1'|'l2', nonnegative weight)")
                values=torch.cat([parameter.reshape(-1) for parameter in self.student.parameters() if parameter.requires_grad])
                penalty=weight*(values.abs().sum() if kind=="l1" else values.square().sum())
            loss=mean_mse+rrad+self.lambda_bic*structural+penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite distillation/BIC loss")
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    [parameter for group in self.optimizer.param_groups for parameter in group["params"]],
                    self.max_grad_norm,
                )
            optimized = [
                parameter for group in self.optimizer.param_groups for parameter in group["params"]
            ]
            if all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in optimized
            ):
                self.optimizer.step()
            else:
                raise FloatingPointError("non-finite distillation/BIC gradient")
            result.loss += float(loss.detach())*len(chunks); result.chunks += len(chunks)
        result.loss /= max(result.chunks, 1)
        return result


@torch.no_grad()
def evaluate(model: nn.Module, batches: Iterable[SequenceBatch]) -> dict[str, float]:
    model.eval(); squared = count = 0
    for batch in batches:
        batch=batch.to(next(model.parameters()).device)
        prediction, _, _ = model(batch.features, None)  # reset per independent trajectory batch
        squared += float(torch.sum((prediction-batch.targets)**2)); count += batch.targets.numel()
    return {"mse": squared/count}

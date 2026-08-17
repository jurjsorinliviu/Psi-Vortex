"""Runtime-guarded recurrent export and unambiguous compression accounting."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from .contracts import require_sequence
from .models import LowRankRecurrentStudent, RecurrentStudent


class ScriptedGRUDeployment(nn.Module):
    __annotations__ = {}
    def __init__(self, recurrent: nn.GRU, readout: nn.Linear):
        super().__init__()
        self.recurrent = recurrent
        self.readout = readout

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        torch._assert(sequence.dim() == 3, "input must be [batch, sequence_length, features]")
        torch._assert(sequence.size(1) > 1, "sequence_length must be greater than one")
        hidden, _ = self.recurrent(sequence)
        return self.readout(hidden)

    @torch.jit.export
    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            self.recurrent.num_layers,
            batch_size,
            self.recurrent.hidden_size,
            dtype=self.readout.weight.dtype,
            device=self.readout.weight.device,
        )

    @torch.jit.export
    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        torch._assert(x_t.dim() == 2, "step input must be [batch, features]")
        torch._assert(state.dim() == 3, "state must be [layers, batch, hidden]")
        hidden, next_state = self.recurrent(x_t.unsqueeze(1), state)
        return self.readout(hidden[:, 0]), next_state


class ScriptedLowRankDeployment(nn.Module):
    __annotations__ = {}
    def __init__(self, model: LowRankRecurrentStudent):
        super().__init__()
        self.encode = model.encode
        self.decode = model.decode
        self.readout = model.readout
        self.hidden_size = model.hidden_size

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        torch._assert(sequence.dim() == 3, "input must be [batch, sequence_length, features]")
        torch._assert(sequence.size(1) > 1, "sequence_length must be greater than one")
        state = torch.zeros(
            sequence.size(0), self.hidden_size, dtype=sequence.dtype, device=sequence.device
        )
        outputs = torch.jit.annotate(list[torch.Tensor], [])
        for index in range(sequence.size(1)):
            state = torch.tanh(
                self.decode(self.encode(torch.cat((sequence[:, index], state), dim=-1)))
            )
            outputs.append(self.readout(state))
        return torch.stack(outputs, dim=1)

    @torch.jit.export
    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.hidden_size,
            dtype=self.readout.weight.dtype,
            device=self.readout.weight.device,
        )

    @torch.jit.export
    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        torch._assert(x_t.dim() == 2, "step input must be [batch, features]")
        torch._assert(state.dim() == 2, "state must be [batch, hidden]")
        next_state = torch.tanh(self.decode(self.encode(torch.cat((x_t, state), dim=-1))))
        return self.readout(next_state), next_state


def export_torchscript(model: nn.Module, example_sequence: torch.Tensor, destination: str | Path) -> Path:
    require_sequence(example_sequence, name="export example")
    if hasattr(model, "bake"):
        export_model = model.bake()
    else:
        export_model = copy.deepcopy(model)
    # Deployment artifacts are intentionally device-portable.  Exporting from a
    # CUDA training run must neither mutate the caller's model nor serialize a
    # CUDA-bound module that fails when loaded with ordinary CPU tensors.
    export_model = export_model.cpu().eval()
    if isinstance(export_model, RecurrentStudent):
        deployment: nn.Module = ScriptedGRUDeployment(
            export_model.recurrent, export_model.readout
        )
    elif isinstance(export_model, LowRankRecurrentStudent):
        deployment = ScriptedLowRankDeployment(export_model)
    else:
        raise TypeError("deployment export supports GRU and low-rank recurrent students")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(deployment).save(str(destination))
    return destination


def compression_report(model: nn.Module, artifact: str | Path | None = None) -> dict[str, int | None]:
    effective = int(
        getattr(
            model,
            "materialized_cluster_count",
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        )
    )
    expanded_model = model.bake() if hasattr(model, "bake") else model
    expanded = sum(parameter.numel() for parameter in expanded_model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in expanded_model.parameters()
    )
    artifact_bytes = Path(artifact).stat().st_size if artifact is not None else None
    return {
        "effective_trainable_values": effective,
        "expanded_parameters": expanded,
        "expanded_parameter_bytes": parameter_bytes,
        "serialized_artifact_bytes": artifact_bytes,
    }

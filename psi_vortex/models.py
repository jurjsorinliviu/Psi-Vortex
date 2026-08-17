"""Faithful chronological Ψ-xLSTM teacher and recurrent students."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .contracts import require_sequence


class MatrixLSTMBlock(nn.Module):
    """Canonical matrix-memory equations evaluated in true time order."""

    def __init__(self, input_size: int, hidden_size: int, memory_size: int | None = None):
        super().__init__()
        memory_size = memory_size or hidden_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, memory_size)
        self.W_q = nn.Linear(input_size, memory_size)
        self.W_v = nn.Linear(input_size, memory_size)
        self.R_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_k = nn.Linear(hidden_size, memory_size, bias=False)
        self.R_q = nn.Linear(hidden_size, memory_size, bias=False)
        self.R_v = nn.Linear(hidden_size, memory_size, bias=False)
        self.output_proj = nn.Linear(memory_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        require_sequence(x, name="matrix-LSTM input")
        batch = x.shape[0]
        if state is None:
            hidden = x.new_zeros(batch, self.hidden_size)
            memory = x.new_zeros(batch, self.memory_size, self.memory_size)
        else:
            hidden, memory = state
            if hidden.shape != (batch, self.hidden_size):
                raise ValueError("invalid matrix-LSTM hidden-state shape")
            if memory.shape != (batch, self.memory_size, self.memory_size):
                raise ValueError("invalid matrix-LSTM matrix-state shape")

        outputs: list[torch.Tensor] = []
        memories: list[torch.Tensor] = []
        for index in range(x.shape[1]):
            value_t = x[:, index]
            input_gate = torch.exp(
                (self.W_i(value_t) + self.R_i(hidden)).clamp(max=10)
            )
            forget_gate = torch.sigmoid(self.W_f(value_t) + self.R_f(hidden))
            output_gate = torch.sigmoid(self.W_o(value_t) + self.R_o(hidden))
            key = self.W_k(value_t) + self.R_k(hidden)
            query = self.W_q(value_t) + self.R_q(hidden)
            value = torch.tanh(self.W_v(value_t) + self.R_v(hidden))
            key = key / (torch.linalg.vector_norm(key, dim=-1, keepdim=True) + 1e-8)
            scalar_forget = forget_gate.mean(dim=-1, keepdim=True).unsqueeze(-1)
            scalar_input = input_gate.mean(dim=-1, keepdim=True).unsqueeze(-1)
            outer = torch.bmm(value.unsqueeze(2), key.unsqueeze(1))
            memory = scalar_forget * memory + scalar_input * outer
            recalled = torch.bmm(memory, query.unsqueeze(2)).squeeze(2)
            hidden = output_gate * torch.tanh(self.norm(self.output_proj(recalled)))
            outputs.append(hidden)
            memories.append(memory)
        return torch.stack(outputs, dim=1), (hidden, memory), torch.stack(memories, dim=1)


class ScalarLSTMBlock(nn.Module):
    """Canonical enhanced scalar-memory block threaded across sequences."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.memory_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        require_sequence(x, name="scalar-LSTM input")
        batch = x.shape[0]
        if state is None:
            hidden = cell = x.new_zeros(batch, self.hidden_size)
        else:
            hidden, cell = state
            if hidden.shape != (batch, self.hidden_size) or cell.shape != hidden.shape:
                raise ValueError("invalid scalar-LSTM state shape")
        outputs: list[torch.Tensor] = []
        for index in range(x.shape[1]):
            hidden_t, cell_t = self.cell(x[:, index], (hidden, cell))
            gate, candidate = self.memory_proj(hidden_t).chunk(2, dim=-1)
            cell = cell_t + torch.sigmoid(gate) * torch.tanh(candidate)
            hidden = self.norm(hidden_t)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), (hidden, cell)


def _canonical_block_widths(hidden_size: int, num_blocks: int) -> list[int]:
    if hidden_size < 2:
        raise ValueError("teacher hidden_size must be at least two")
    if num_blocks < 1:
        raise ValueError("teacher num_blocks must be positive")
    split = max(1, num_blocks // 2)
    return [hidden_size // 2 if index < split else hidden_size for index in range(num_blocks)]


class PsiXLSTMTeacher(nn.Module):
    """Canonical block/fusion topology with chronological state threading."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 1,
        num_blocks: int = 4,
        memory_size: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.block_widths = _canonical_block_widths(hidden_size, num_blocks)
        blocks: list[nn.Module] = []
        previous = input_size
        for index, width in enumerate(self.block_widths):
            if index % 2 == 0:
                blocks.append(
                    MatrixLSTMBlock(
                        previous,
                        width,
                        memory_size=memory_size if memory_size is not None else max(16, width // 2),
                    )
                )
            else:
                blocks.append(ScalarLSTMBlock(previous, width))
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.fusion = nn.Linear(sum(self.block_widths), hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        state: list[Any] | None = None,
    ) -> tuple[torch.Tensor, list[Any], dict[str, Any]]:
        require_sequence(x)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"input feature size must be {self.input_size}")
        if state is not None and len(state) != len(self.blocks):
            raise ValueError(f"state must contain {len(self.blocks)} block states")
        incoming = [None] * len(self.blocks) if state is None else state
        current = x
        block_outputs: list[torch.Tensor] = []
        states: list[Any] = []
        matrix_memories: list[torch.Tensor] = []
        for block, old_state in zip(self.blocks, incoming):
            if isinstance(block, MatrixLSTMBlock):
                current, new_state, memory_sequence = block(current, old_state)
                matrix_memories.append(memory_sequence)
            else:
                current, new_state = block(current, old_state)
            block_outputs.append(current)
            states.append(new_state)
        fused = torch.tanh(self.norm(self.fusion(torch.cat(block_outputs, dim=-1))))
        auxiliary = {
            "recurrent": fused,
            "block_outputs": tuple(block_outputs),
            "matrix_memories": tuple(matrix_memories),
            "matrix_memory": matrix_memories[-1] if matrix_memories else fused.unsqueeze(-1),
        }
        return self.readout(fused), states, auxiliary


class RecurrentStudent(nn.Module):
    """Compact GRU student with batch-sequence and physical streaming APIs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        output_size: int = 1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.recurrent = nn.GRU(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        require_sequence(x)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"input feature size must be {self.input_size}")
        hidden, next_state = self.recurrent(x, state)
        return self.readout(hidden), next_state, {"recurrent": hidden}

    def step(self, x: torch.Tensor, state: torch.Tensor):
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ValueError(f"step input must have shape [batch, {self.input_size}]")
        if state.shape != (self.num_layers, x.shape[0], self.hidden_size):
            raise ValueError("invalid GRU streaming state shape")
        hidden, next_state = self.recurrent(x.unsqueeze(1), state)
        return self.readout(hidden[:, 0]), next_state

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None):
        parameter = next(self.parameters())
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype,
        )


class LowRankRecurrentStudent(nn.Module):
    """Low-rank recurrent bottleneck student with sequence and streaming APIs."""

    def __init__(self, input_size: int, hidden_size: int = 16, rank: int = 4, output_size: int = 1):
        super().__init__()
        if rank < 1 or rank > input_size + hidden_size:
            raise ValueError("rank must be between one and input_size + hidden_size")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rank = rank
        self.encode = nn.Linear(input_size + hidden_size, rank)
        self.decode = nn.Linear(rank, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        require_sequence(x)
        if x.shape[-1] != self.input_size:
            raise ValueError(f"input feature size must be {self.input_size}")
        hidden = x.new_zeros(x.shape[0], self.hidden_size) if state is None else state
        if hidden.shape != (x.shape[0], self.hidden_size):
            raise ValueError("invalid low-rank recurrent state shape")
        values: list[torch.Tensor] = []
        for index in range(x.shape[1]):
            hidden = torch.tanh(self.decode(self.encode(torch.cat((x[:, index], hidden), -1))))
            values.append(hidden)
        sequence = torch.stack(values, dim=1)
        return self.readout(sequence), hidden, {"recurrent": sequence}

    def step(self, x: torch.Tensor, state: torch.Tensor):
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ValueError(f"step input must have shape [batch, {self.input_size}]")
        if state.shape != (x.shape[0], self.hidden_size):
            raise ValueError("invalid low-rank streaming state shape")
        hidden = torch.tanh(self.decode(self.encode(torch.cat((x, state), -1))))
        return self.readout(hidden), hidden

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None):
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype,
        )


RecurrentTeacher = PsiXLSTMTeacher

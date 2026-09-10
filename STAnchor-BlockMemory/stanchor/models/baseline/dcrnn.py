"""Official DCRNN encoder-decoder with a project-native [B, H, N, 1] interface."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from stanchor.data.graph import GraphData


def dense_adjacency_from_graph(graph: GraphData) -> torch.Tensor:
    """Recover the dense adjacency used by official DCRNN, including self-loops."""
    adjacency = torch.zeros((graph.num_nodes, graph.num_nodes), dtype=torch.float32)
    target, source = graph.edge_index.detach().cpu()
    adjacency[target, source] = graph.edge_weight.detach().cpu().float()
    return adjacency


def _random_walk_matrix(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=1)
    inv_degree = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
    return inv_degree.unsqueeze(1) * adjacency


def _scaled_laplacian(adjacency: torch.Tensor, lambda_max: float = 2.0) -> torch.Tensor:
    symmetric = torch.maximum(adjacency, adjacency.transpose(0, 1))
    degree = symmetric.sum(dim=1)
    inv_sqrt = torch.pow(degree.clamp_min(0.0), -0.5)
    inv_sqrt = torch.where(torch.isfinite(inv_sqrt), inv_sqrt, torch.zeros_like(inv_sqrt))
    normalized = symmetric * inv_sqrt.unsqueeze(1) * inv_sqrt.unsqueeze(0)
    identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
    laplacian = identity - normalized
    return (2.0 / lambda_max) * laplacian - identity


def build_diffusion_supports(adjacency: torch.Tensor, filter_type: str) -> torch.Tensor:
    """Build DCRNN diffusion supports, including the official transpose convention."""
    if filter_type == "laplacian":
        supports = [_scaled_laplacian(adjacency)]
    elif filter_type == "random_walk":
        supports = [_random_walk_matrix(adjacency).transpose(0, 1)]
    elif filter_type == "dual_random_walk":
        supports = [
            _random_walk_matrix(adjacency).transpose(0, 1),
            _random_walk_matrix(adjacency.transpose(0, 1)).transpose(0, 1),
        ]
    else:
        raise ValueError(f"unsupported DCRNN filter_type: {filter_type}")
    return torch.stack(supports, dim=0)


class DiffusionGraphConv(nn.Module):
    """Chebyshev diffusion convolution used inside DCGRU gates and candidates."""

    def __init__(
        self,
        input_dim: int,
        output_size: int,
        supports: torch.Tensor,
        max_diffusion_step: int,
        bias_start: float = 0.0,
    ) -> None:
        super().__init__()
        if supports.ndim != 3 or supports.shape[-1] != supports.shape[-2]:
            raise ValueError("supports must be [S, N, N]")
        if max_diffusion_step < 0:
            raise ValueError("max_diffusion_step must be non-negative")
        self.input_dim = input_dim
        self.output_size = output_size
        self.max_diffusion_step = max_diffusion_step
        self.num_matrices = supports.shape[0] * max_diffusion_step + 1
        self.register_buffer("supports", supports.float())
        self.weight = nn.Parameter(torch.empty(input_dim * self.num_matrices, output_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, bias_start)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or state.ndim != 3:
            raise ValueError("DiffusionGraphConv expects [B, N, D] inputs and state")
        batch_size, num_nodes, _ = inputs.shape
        fused = torch.cat((inputs, state), dim=-1)
        if fused.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected concat dim {self.input_dim}, got {fused.shape[-1]}"
            )
        # Official TF mutates x0 across supports; keep that recurrence.
        x0 = fused.permute(1, 2, 0).reshape(num_nodes, self.input_dim * batch_size)
        pieces = [x0]
        if self.max_diffusion_step > 0:
            for support in self.supports.to(device=fused.device, dtype=fused.dtype):
                x1 = support @ x0
                pieces.append(x1)
                for _ in range(2, self.max_diffusion_step + 1):
                    x2 = 2.0 * (support @ x1) - x0
                    pieces.append(x2)
                    x1, x0 = x2, x1
        if len(pieces) != self.num_matrices:
            raise RuntimeError(
                f"expected {self.num_matrices} diffusion terms, got {len(pieces)}"
            )
        stacked = torch.stack(pieces, dim=0)
        stacked = stacked.reshape(self.num_matrices, num_nodes, self.input_dim, batch_size)
        stacked = stacked.permute(3, 1, 2, 0).reshape(
            batch_size * num_nodes, self.input_dim * self.num_matrices
        )
        output = stacked @ self.weight.to(dtype=stacked.dtype) + self.bias.to(dtype=stacked.dtype)
        return output.reshape(batch_size, num_nodes, self.output_size)


class DCGRUCell(nn.Module):
    """Graph-convolution GRU cell from the official DCRNN source."""

    def __init__(
        self,
        input_dim: int,
        num_units: int,
        supports: torch.Tensor,
        max_diffusion_step: int,
        num_proj: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_units = num_units
        self.num_proj = num_proj
        fused_dim = input_dim + num_units
        self.gate = DiffusionGraphConv(
            fused_dim, 2 * num_units, supports, max_diffusion_step, bias_start=1.0
        )
        self.candidate = DiffusionGraphConv(
            fused_dim, num_units, supports, max_diffusion_step, bias_start=0.0
        )
        if num_proj is not None:
            self.project = nn.Linear(num_units, num_proj, bias=False)
            nn.init.xavier_uniform_(self.project.weight)
        else:
            self.project = None

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reset_update = torch.sigmoid(self.gate(inputs, state))
        reset_gate, update_gate = torch.split(reset_update, self.num_units, dim=-1)
        candidate = torch.tanh(self.candidate(inputs, reset_gate * state))
        new_state = update_gate * state + (1.0 - update_gate) * candidate
        output = new_state
        if self.project is not None:
            output = self.project(new_state)
        return output, new_state


class DCRNNForecastBackbone(nn.Module):
    """Official DCRNN seq2seq model with curriculum teacher forcing."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        num_nodes: int,
        input_channels: int,
        output_channels: int,
        adjacency: torch.Tensor,
        rnn_units: int = 64,
        rnn_layers: int = 2,
        max_diffusion_step: int = 2,
        filter_type: str = "dual_random_walk",
        use_curriculum_learning: bool = True,
        cl_decay_steps: int = 2000,
        use_time_in_day: bool = True,
        slots_per_day: int = 288,
    ) -> None:
        super().__init__()
        if input_channels != 1 or output_channels != 1:
            raise ValueError("DCRNN adapter currently supports one input/output channel")
        if adjacency.ndim != 2 or adjacency.shape[0] != num_nodes or adjacency.shape[1] != num_nodes:
            raise ValueError("adjacency must be [N, N]")
        if rnn_layers < 1:
            raise ValueError("dcrnn_rnn_layers must be >= 1")
        if slots_per_day <= 0:
            raise ValueError("slots_per_day must be positive")
        self.context_length = context_length
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.output_dim = output_channels
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps
        self.use_time_in_day = use_time_in_day
        self.slots_per_day = slots_per_day
        self.encoder_input_dim = input_channels + int(use_time_in_day)
        supports = build_diffusion_supports(adjacency.float(), filter_type)
        self.register_buffer("adjacency", adjacency.float())
        self.register_buffer("supports", supports)
        self.batches_seen = 0

        encoder_dims = [self.encoder_input_dim] + [rnn_units] * (rnn_layers - 1)
        self.encoder_cells = nn.ModuleList(
            [
                DCGRUCell(input_dim, rnn_units, supports, max_diffusion_step)
                for input_dim in encoder_dims
            ]
        )
        decoder_cells: list[DCGRUCell] = []
        for layer_index in range(rnn_layers):
            layer_input = output_channels if layer_index == 0 else rnn_units
            num_proj = output_channels if layer_index == rnn_layers - 1 else None
            decoder_cells.append(
                DCGRUCell(layer_input, rnn_units, supports, max_diffusion_step, num_proj=num_proj)
            )
        self.decoder_cells = nn.ModuleList(decoder_cells)

    def _time_in_day(self, x: torch.Tensor, tod: torch.Tensor) -> torch.Tensor:
        if tod.ndim == 2:
            tod = tod.unsqueeze(-1).expand(-1, -1, x.shape[2])
        if tod.shape[:3] != x.shape[:3]:
            raise ValueError("tod must broadcast to [B, T, N]")
        return tod.to(dtype=x.dtype) / float(self.slots_per_day)

    def _encoder_inputs(self, x: torch.Tensor, tod: torch.Tensor | None) -> torch.Tensor:
        if not self.use_time_in_day:
            return x
        if tod is None:
            raise ValueError("DCRNN requires tod when use_time_in_day is enabled")
        return torch.cat((x, self._time_in_day(x, tod).unsqueeze(-1)), dim=-1)

    def _curriculum_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (
            self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps)
        )

    def _encode(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        current = inputs
        states: list[torch.Tensor] = []
        for cell in self.encoder_cells:
            state = current.new_zeros(current.shape[0], self.num_nodes, self.rnn_units)
            outputs = []
            for time_index in range(current.shape[1]):
                output, state = cell(current[:, time_index], state)
                outputs.append(output)
            current = torch.stack(outputs, dim=1)
            states.append(state)
        return states

    def forward(
        self,
        x: torch.Tensor,
        tod: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.context_length or x.shape[2] != self.num_nodes or x.shape[-1] != 1:
            raise ValueError("DCRNN expects x as [B, T, N, 1]")
        if labels is not None and (
            labels.shape[0] != x.shape[0]
            or labels.shape[1] != self.horizon
            or labels.shape[2] != self.num_nodes
            or labels.shape[-1] != self.output_dim
        ):
            raise ValueError("DCRNN labels must be [B, H, N, 1]")
        encoder_inputs = self._encoder_inputs(x, tod)
        states = self._encode(encoder_inputs)
        decoder_input = x.new_zeros(x.shape[0], self.num_nodes, self.output_dim)
        outputs = []
        teacher_force = (
            self.training and self.use_curriculum_learning and labels is not None
        )
        threshold = self._curriculum_threshold(self.batches_seen)
        for step in range(self.horizon):
            current = decoder_input
            next_states = []
            for cell, state in zip(self.decoder_cells, states):
                current, state = cell(current, state)
                next_states.append(state)
            outputs.append(current)
            states = next_states
            if teacher_force and float(np.random.uniform(0.0, 1.0)) < threshold:
                decoder_input = labels[:, step]
            else:
                decoder_input = current
        if self.training and labels is not None:
            self.batches_seen += 1
        return torch.stack(outputs, dim=1)

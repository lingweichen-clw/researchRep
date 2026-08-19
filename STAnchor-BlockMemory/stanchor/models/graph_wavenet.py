"""Project-native Graph WaveNet forecasting backbone.

The implementation follows the supplied IJCAI 2019 reference while using
Conv2d for the reference's four-dimensional ``[B,C,N,T]`` tensors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from stanchor.data.graph import GraphData


def build_graph_wavenet_supports(graph: GraphData) -> tuple[torch.Tensor, ...]:
    graph.validate()
    adjacency = torch.zeros(
        (graph.num_nodes, graph.num_nodes),
        dtype=torch.float32,
        device=graph.edge_index.device,
    )
    target, source = graph.edge_index
    non_self = target != source
    adjacency[target[non_self], source[non_self]] = graph.edge_weight[non_self].float()

    def transition(matrix: torch.Tensor) -> torch.Tensor:
        degree = matrix.sum(dim=1)
        inverse = torch.zeros_like(degree)
        inverse[degree > 0] = degree[degree > 0].reciprocal()
        return inverse[:, None] * matrix

    return transition(adjacency), transition(adjacency.transpose(0, 1))


class _NConv(nn.Module):
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ncvl,vw->ncwl", x, adjacency).contiguous()


class _GraphConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, dropout: float, support_len: int, order: int = 2) -> None:
        super().__init__()
        self.nconv = _NConv()
        self.mlp = nn.Conv2d((order * support_len + 1) * channels_in, channels_out, kernel_size=1)
        self.dropout = float(dropout)
        self.order = int(order)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        outputs = [x]
        for adjacency in supports:
            first = self.nconv(x, adjacency)
            outputs.append(first)
            current = first
            for _ in range(2, self.order + 1):
                current = self.nconv(current, adjacency)
                outputs.append(current)
        result = self.mlp(torch.cat(outputs, dim=1))
        return functional.dropout(result, self.dropout, training=self.training)


class _GraphWaveNet(nn.Module):
    def __init__(
        self,
        device: torch.device,
        num_nodes: int,
        supports: tuple[torch.Tensor, ...],
        input_channels: int,
        horizon: int,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        end_channels: int,
        kernel_size: int,
        blocks: int,
        layers: int,
        dropout: float,
        adaptive_adj: bool,
        adaptive_dim: int,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.dropout = float(dropout)
        self.adaptive_adj = bool(adaptive_adj)
        self.start_conv = nn.Conv2d(input_channels, residual_channels, kernel_size=1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.support_names: list[str] = []
        for index, support in enumerate(supports):
            name = f"support_{index}"
            self.register_buffer(name, support.to(device))
            self.support_names.append(name)
        support_len = len(supports) + (1 if adaptive_adj else 0)
        if adaptive_adj:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, adaptive_dim, device=device))
            self.nodevec2 = nn.Parameter(torch.randn(adaptive_dim, num_nodes, device=device))

        for _ in range(int(blocks)):
            dilation = 1
            for _ in range(int(layers)):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=(1, dilation)))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=(1, dilation)))
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, 1))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, 1))
                self.batch_norms.append(nn.BatchNorm2d(residual_channels))
                if support_len:
                    self.graph_convs.append(_GraphConv(dilation_channels, residual_channels, dropout, support_len=support_len))
                dilation *= 2
        self.layers_total = len(self.filter_convs)
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, 1)
        self.end_conv_2 = nn.Conv2d(end_channels, horizon, 1)

    def _supports(self) -> list[torch.Tensor]:
        supports = [getattr(self, name) for name in self.support_names]
        if self.adaptive_adj:
            adaptive = functional.softmax(functional.relu(self.nodevec1 @ self.nodevec2), dim=1)
            supports.append(adaptive)
        return supports

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start_conv(x)
        supports = self._supports()
        skip: torch.Tensor | None = None
        graph_index = 0
        for index in range(self.layers_total):
            residual = x
            filter_value = torch.tanh(self.filter_convs[index](residual))
            gate_value = torch.sigmoid(self.gate_convs[index](residual))
            x = filter_value * gate_value
            current_skip = self.skip_convs[index](x)
            if skip is None:
                skip = current_skip
            else:
                skip = current_skip + skip[..., -current_skip.shape[-1] :]
            if self.graph_convs:
                x = self.graph_convs[graph_index](x, supports)
                graph_index += 1
            else:
                x = self.residual_convs[index](x)
            x = x + residual[..., -x.shape[-1] :]
            x = self.batch_norms[index](x)
        if skip is None:
            raise RuntimeError("Graph WaveNet has no temporal layers")
        return self.end_conv_2(functional.relu(self.end_conv_1(functional.relu(skip))))


class GraphWaveNetForecastBackbone(nn.Module):
    """Graph WaveNet with the STAnchor ``[B,T,N,C]`` forecast contract."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        input_channels: int,
        output_channels: int,
        graph: GraphData,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        dropout: float = 0.3,
        adaptive_adj: bool = True,
        adaptive_dim: int = 10,
    ) -> None:
        super().__init__()
        if output_channels != 1:
            raise ValueError("Graph WaveNet adapter currently supports one output channel")
        if context_length <= 0 or horizon <= 0 or kernel_size <= 1 or blocks <= 0 or layers <= 0:
            raise ValueError("invalid Graph WaveNet dimensions")
        self.network = _GraphWaveNet(
            torch.device(graph.edge_index.device),
            graph.num_nodes,
            build_graph_wavenet_supports(graph),
            input_channels,
            horizon,
            residual_channels,
            dilation_channels,
            skip_channels,
            end_channels,
            kernel_size,
            blocks,
            layers,
            dropout,
            adaptive_adj,
            adaptive_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must be [B,T,N,C]")
        network_input = x.permute(0, 3, 2, 1).contiguous()
        network_input = functional.pad(network_input, (1, 0, 0, 0))
        output = self.network(network_input)
        if output.shape[-1] != 1:
            output = output[..., -1:]
        return output

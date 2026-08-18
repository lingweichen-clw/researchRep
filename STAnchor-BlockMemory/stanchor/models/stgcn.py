"""Static-graph STGCN forecasting backbone.

The block structure follows the STGCN implementation already present in the
local ST-TTC repository.  This copy is self-contained so checkpoints do not
depend on a sibling repository import path.
"""

from __future__ import annotations

import torch
from torch import nn

from stanchor.data.graph import GraphData


def build_stgcn_gso(graph: GraphData) -> torch.Tensor:
    """Build the symmetric scaled-normalized-Laplacian GSO used by STGCN.

    ``GraphData`` stores target/source edges and includes self-loops for the
    encoder.  STGCN uses the physical neighbor graph, so self-loops are
    removed here before symmetrization and normalized Laplacian construction.
    The result has shape ``[N, N]`` and is finite for isolated nodes.
    """

    graph.validate()
    adjacency = torch.zeros(
        (graph.num_nodes, graph.num_nodes),
        dtype=torch.float32,
        device=graph.edge_index.device,
    )
    target, source = graph.edge_index
    adjacency[target, source] = graph.edge_weight.float()
    adjacency.fill_diagonal_(0.0)
    adjacency = torch.maximum(adjacency, adjacency.transpose(0, 1))

    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = degree[nonzero].rsqrt()
    normalized_adjacency = (
        inv_sqrt_degree[:, None]
        * adjacency
        * inv_sqrt_degree[None, :]
    )
    identity = torch.eye(graph.num_nodes, dtype=adjacency.dtype, device=adjacency.device)
    laplacian = identity - normalized_adjacency
    lambda_max = torch.linalg.eigvalsh(laplacian).amax().clamp_min(1.0e-6)
    return ((2.0 / lambda_max) * laplacian - identity).contiguous()


class _Align(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.projection = (
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
            if input_channels > output_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_channels == self.output_channels:
            return x
        if self.input_channels > self.output_channels:
            assert self.projection is not None
            return self.projection(x)
        padding = torch.zeros(
            x.shape[0],
            self.output_channels - self.input_channels,
            x.shape[2],
            x.shape[3],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat((x, padding), dim=1)


class _TemporalConvLayer(nn.Module):
    """Causal gated temporal convolution with no future padding."""

    def __init__(self, kernel_size: int, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.align = _Align(input_channels, output_channels)
        self.conv = nn.Conv2d(
            input_channels,
            2 * output_channels,
            kernel_size=(kernel_size, 1),
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] < self.kernel_size:
            raise ValueError("temporal input is shorter than the STGCN kernel")
        # x: [B, C, T, N] -> [B, C_out, T-Kt+1, N]
        aligned = self.align(x)[:, :, self.kernel_size - 1 :, :]
        gated = self.conv(x)
        value, gate = gated.chunk(2, dim=1)
        return (value + aligned) * torch.sigmoid(gate)


class _ChebyshevGraphConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        graph_kernel: int,
        gso: torch.Tensor,
    ) -> None:
        super().__init__()
        if graph_kernel <= 0:
            raise ValueError("graph_kernel must be positive")
        self.graph_kernel = graph_kernel
        self.register_buffer("gso", gso.float().contiguous())
        self.weight = nn.Parameter(
            torch.empty(graph_kernel, input_channels, output_channels)
        )
        self.bias = nn.Parameter(torch.empty(output_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N] -> [B, T, N, C]
        values = x.permute(0, 2, 3, 1).contiguous()
        chebyshev = [values]
        if self.graph_kernel >= 2:
            first = torch.einsum("ij,btjc->btic", self.gso, values)
            chebyshev.append(first)
            for order in range(2, self.graph_kernel):
                chebyshev.append(
                    torch.einsum(
                        "ij,btjc->btic",
                        2.0 * self.gso,
                        chebyshev[-1],
                    )
                    - chebyshev[-2]
                )
        stacked = torch.stack(chebyshev, dim=2)
        result = torch.einsum("btkic,kco->btio", stacked, self.weight)
        return result + self.bias


class _GraphConvLayer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        graph_kernel: int,
        gso: torch.Tensor,
    ) -> None:
        super().__init__()
        self.align = _Align(input_channels, output_channels)
        self.graph = _ChebyshevGraphConv(
            output_channels,
            output_channels,
            graph_kernel,
            gso,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aligned = self.align(x)
        graph_output = self.graph(aligned).permute(0, 3, 1, 2).contiguous()
        return graph_output + aligned


class _STConvBlock(nn.Module):
    def __init__(
        self,
        temporal_kernel: int,
        graph_kernel: int,
        node_count: int,
        input_channels: int,
        hidden_channels: int,
        bottleneck_channels: int,
        output_channels: int,
        gso: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal1 = _TemporalConvLayer(
            temporal_kernel, input_channels, hidden_channels
        )
        self.graph = _GraphConvLayer(
            hidden_channels, bottleneck_channels, graph_kernel, gso
        )
        self.temporal2 = _TemporalConvLayer(
            temporal_kernel, bottleneck_channels, output_channels
        )
        self.normalization = nn.LayerNorm([node_count, output_channels])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal1(x)
        x = self.graph(x)
        x = self.activation(x)
        x = self.temporal2(x)
        x = self.normalization(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class _OutputBlock(nn.Module):
    def __init__(
        self,
        temporal_kernel: int,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        node_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal = _TemporalConvLayer(
            temporal_kernel, input_channels, hidden_channels
        )
        self.normalization = nn.LayerNorm([node_count, hidden_channels])
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, output_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.normalization(x.permute(0, 2, 3, 1))
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # The output temporal length is one: [B, 1, N, H*C].
        return x[:, 0]


class STGCNForecastBackbone(nn.Module):
    """STGCN adapted to the project's ``[B,T,N,C]`` forecast contract."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        input_channels: int,
        output_channels: int,
        graph: GraphData,
        temporal_kernel: int = 3,
        graph_kernel: int = 3,
        block_num: int = 2,
        hidden_channels: int = 64,
        bottleneck_channels: int = 16,
        output_hidden_channels: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if context_length <= 0 or horizon <= 0:
            raise ValueError("context_length and horizon must be positive")
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("input and output channels must be positive")
        if temporal_kernel < 2 or graph_kernel < 1 or block_num < 1:
            raise ValueError("STGCN kernel sizes and block_num must be positive")
        for name, value in (
            ("hidden_channels", hidden_channels),
            ("bottleneck_channels", bottleneck_channels),
            ("output_hidden_channels", output_hidden_channels),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        graph.validate()
        output_temporal_length = context_length - 2 * (temporal_kernel - 1) * block_num
        if output_temporal_length < 1:
            raise ValueError(
                "context_length is too short for the requested STGCN blocks"
            )

        self.context_length = context_length
        self.horizon = horizon
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.node_count = graph.num_nodes
        self.temporal_kernel = temporal_kernel
        self.graph_kernel = graph_kernel
        self.block_num = block_num
        self.output_temporal_length = output_temporal_length
        gso = build_stgcn_gso(graph)
        self.register_buffer("gso", gso)

        blocks = []
        last_channels = input_channels
        for _ in range(block_num):
            block = _STConvBlock(
                temporal_kernel,
                graph_kernel,
                graph.num_nodes,
                last_channels,
                hidden_channels,
                bottleneck_channels,
                hidden_channels,
                gso,
                dropout,
            )
            blocks.append(block)
            last_channels = hidden_channels
        self.st_blocks = nn.ModuleList(blocks)
        self.output = _OutputBlock(
            output_temporal_length,
            last_channels,
            output_hidden_channels,
            horizon * output_channels,
            graph.num_nodes,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must be [B,T,N,C]")
        batch, time, nodes, channels = x.shape
        if (time, nodes, channels) != (
            self.context_length,
            self.node_count,
            self.input_channels,
        ):
            raise ValueError("x does not match the STGCN configuration")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf")
        # [B,T,N,C] -> [B,C,T,N]
        hidden = x.permute(0, 3, 1, 2).contiguous()
        for block in self.st_blocks:
            hidden = block(hidden)
        # [B, N, H*C] -> [B,H,N,C]
        output = self.output(hidden)
        return output.view(batch, nodes, self.horizon, self.output_channels).permute(
            0, 2, 1, 3
        ).contiguous()

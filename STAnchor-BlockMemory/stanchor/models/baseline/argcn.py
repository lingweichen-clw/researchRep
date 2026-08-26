"""AGCN/AGCRN-compatible forecasting baseline.

Only the reusable AGCN and AGCRNCell components from ST-SSDL are ported here.
ST-SSDL-specific prototype, hypernetwork, dynamic-graph and curriculum-learning
branches are deliberately excluded.
"""

from __future__ import annotations

import torch
from torch import nn


class AGCN(nn.Module):
    """Chebyshev graph convolution used by the reference AGCRN cell."""

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, supports: torch.Tensor) -> None:
        super().__init__()
        if supports.ndim != 3 or supports.shape[-1] != supports.shape[-2]:
            raise ValueError("supports must have shape [num_supports, N, N]")
        if cheb_k < 1:
            raise ValueError("cheb_k must be >= 1")
        self.cheb_k = cheb_k
        self.num_supports = int(supports.shape[0])
        self.node_num = int(supports.shape[-1])
        self.register_buffer("supports", supports.float())
        self.weights = nn.Parameter(torch.empty(self.num_supports * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.node_num:
            raise ValueError("x must have shape [B, N, Din]")
        supports = self.supports.to(device=x.device, dtype=x.dtype)
        graph_features = []
        for support in supports:
            eye = torch.eye(self.node_num, device=x.device, dtype=x.dtype)
            basis = [eye]
            if self.cheb_k > 1:
                basis.append(support)
            for _ in range(2, self.cheb_k):
                basis.append(2.0 * support @ basis[-1] - basis[-2])
            graph_features.extend(torch.einsum("nm,bmc->bnc", item, x) for item in basis)
        features = torch.cat(graph_features, dim=-1)
        return torch.einsum("bni,io->bno", features, self.weights.to(x.dtype)) + self.bias.to(x.dtype)


class AGCRNCell(nn.Module):
    """AGCN-gated recurrent cell copied from the reference implementation."""

    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, supports: torch.Tensor) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + dim_out, 2 * dim_out, cheb_k, supports)
        self.update = AGCN(dim_in + dim_out, dim_out, cheb_k, supports)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        input_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_state))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate))
        return r * state + (1.0 - r) * hc


class ADCRNNEncoder(nn.Module):
    """Time-unrolled stacked AGCRN encoder from the reference code."""

    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, rnn_layers: int, supports: torch.Tensor) -> None:
        super().__init__()
        if rnn_layers < 1:
            raise ValueError("rnn_layers must be >= 1")
        self.node_num = node_num
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in if i == 0 else dim_out, dim_out, cheb_k, supports) for i in range(rnn_layers)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 4 or x.shape[2] != self.node_num:
            raise ValueError("x must have shape [B, T, N, Din]")
        current = x
        final_states = []
        for cell in self.dcrnn_cells:
            state = x.new_zeros(x.shape[0], self.node_num, cell.hidden_dim)
            outputs = []
            for t in range(x.shape[1]):
                state = cell(current[:, t], state)
                outputs.append(state)
            current = torch.stack(outputs, dim=1)
            final_states.append(state)
        return current, final_states


class ARGCNForecastBackbone(nn.Module):
    """Independent AGCRN-compatible multi-horizon forecasting baseline.

    The project keeps the requested ``argcn`` name for experiment compatibility.
    This is not the complete ST-SSDL model.
    """

    def __init__(self, context_length: int, horizon: int, num_nodes: int, input_channels: int,
                 output_channels: int, graph_support: torch.Tensor, hidden_dim: int = 128,
                 num_layers: int = 1, cheb_k: int = 3) -> None:
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon
        self.nodes = num_nodes
        self.input_channels = input_channels
        self.output_channels = output_channels
        # The reference AGCRN cell consumes the raw input channels directly
        # when no ST-SSDL-specific STE branch is enabled.
        self.encoder = ADCRNNEncoder(num_nodes, input_channels, hidden_dim, cheb_k, num_layers, graph_support)
        self.proj = nn.Linear(hidden_dim, horizon * output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.context_length, self.nodes, self.input_channels):
            raise ValueError("x does not match the ARGCN configuration")
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf")
        encoded, _ = self.encoder(x)
        output = self.proj(encoded[:, -1])
        return output.view(x.shape[0], self.nodes, self.horizon, self.output_channels).permute(0, 2, 1, 3).contiguous()

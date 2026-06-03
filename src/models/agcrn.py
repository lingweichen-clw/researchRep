"""AGCRN building blocks adapted from ST-SSDL with minimal changes."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


class AGCN(nn.Module):
    """Chebyshev graph convolution supporting static and batch graphs."""

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, num_support: int):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.empty(num_support * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.empty(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor, supports: Sequence[torch.Tensor]) -> torch.Tensor:
        x_g: List[torch.Tensor] = []
        for support in supports:
            if support.dim() == 2:
                support_ks = [
                    torch.eye(support.shape[0], device=support.device, dtype=support.dtype),
                    support,
                ]
                for _ in range(2, self.cheb_k):
                    support_ks.append(2 * support @ support_ks[-1] - support_ks[-2])
                for graph in support_ks:
                    x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
            else:
                eye = torch.eye(support.shape[1], device=support.device, dtype=support.dtype)
                support_ks = [eye.unsqueeze(0).expand(support.shape[0], -1, -1), support]
                for _ in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
                for graph in support_ks:
                    x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g_cat = torch.cat(x_g, dim=-1)
        return torch.einsum("bni,io->bno", x_g_cat, self.weights) + self.bias


class AGCRNCell(nn.Module):
    """Graph-convolutional GRU cell."""

    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, num_support: int):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + dim_out, 2 * dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in + dim_out, dim_out, cheb_k, num_support)

    def forward(self, x: torch.Tensor, state: torch.Tensor, supports: Sequence[torch.Tensor]) -> torch.Tensor:
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        return r * state + (1.0 - r) * hc

    def init_hidden_state(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device)


class ADCRNNEncoder(nn.Module):
    """Multi-layer AGCRN encoder."""

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        rnn_layers: int,
        num_support: int,
    ):
        super().__init__()
        if rnn_layers < 1:
            raise ValueError("rnn_layers must be at least 1.")
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, x: torch.Tensor, init_state: Sequence[torch.Tensor], supports: Sequence[torch.Tensor]):
        if x.shape[2] != self.node_num or x.shape[3] != self.input_dim:
            raise ValueError(f"Unexpected encoder input shape: {tuple(x.shape)}")
        current_inputs = x
        output_hidden = []
        for layer_idx, cell in enumerate(self.dcrnn_cells):
            state = init_state[layer_idx]
            inner_states = []
            for t in range(current_inputs.shape[1]):
                state = cell(current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> List[torch.Tensor]:
        return [cell.init_hidden_state(batch_size, device=device) for cell in self.dcrnn_cells]


class ADCRNNDecoder(nn.Module):
    """Multi-layer AGCRN decoder used one future step at a time."""

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        rnn_layers: int,
        num_support: int,
    ):
        super().__init__()
        if rnn_layers < 1:
            raise ValueError("rnn_layers must be at least 1.")
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, xt: torch.Tensor, init_state: Sequence[torch.Tensor], supports: Sequence[torch.Tensor]):
        if xt.shape[1] != self.node_num or xt.shape[2] != self.input_dim:
            raise ValueError(f"Unexpected decoder input shape: {tuple(xt.shape)}")
        current_inputs = xt
        output_hidden = []
        for layer_idx, cell in enumerate(self.dcrnn_cells):
            state = cell(current_inputs, init_state[layer_idx], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden

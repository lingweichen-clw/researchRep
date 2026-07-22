"""Factorized temporal attention and edge-sparse graph attention encoder."""

from __future__ import annotations

import math

import torch
from torch import nn

from stanchor.data.graph import GraphData


class SparseGraphAttention(nn.Module):
    """Multi-head attention evaluated only on graph edges."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        graph_bias: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.graph_bias = float(graph_bias)
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: GraphData) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must be [B, P, N, D]")
        batch, patches, nodes, hidden = x.shape
        if hidden != self.hidden_dim or nodes != graph.num_nodes:
            raise ValueError("encoder tensor and graph dimensions do not match")
        target, source = graph.edge_index
        edges = target.numel()
        q = self.q_projection(x).view(batch, patches, nodes, self.num_heads, self.head_dim)
        k = self.k_projection(x).view(batch, patches, nodes, self.num_heads, self.head_dim)
        v = self.v_projection(x).view(batch, patches, nodes, self.num_heads, self.head_dim)
        q_edge = q.index_select(2, target)
        k_edge = k.index_select(2, source)
        scores = (q_edge * k_edge).sum(dim=-1) / math.sqrt(self.head_dim)  # [B, P, E, heads]
        if self.graph_bias != 0.0:
            scores = scores + self.graph_bias * torch.log(graph.edge_weight.clamp_min(1.0e-8))[None, None, :, None]

        scatter_index = target.view(1, 1, edges, 1).expand(batch, patches, edges, self.num_heads)
        maxima = torch.full(
            (batch, patches, nodes, self.num_heads),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        maxima.scatter_reduce_(2, scatter_index, scores, reduce="amax", include_self=True)
        stable = scores - maxima.gather(2, scatter_index)
        exponent = torch.exp(stable)
        denominator = torch.zeros_like(maxima)
        denominator.scatter_add_(2, scatter_index, exponent)
        attention = exponent / denominator.gather(2, scatter_index).clamp_min(1.0e-8)
        attention = self.attention_dropout(attention)

        weighted = attention.unsqueeze(-1) * v.index_select(2, source)
        output = torch.zeros(
            (batch, patches, nodes, self.num_heads, self.head_dim),
            dtype=x.dtype,
            device=x.device,
        )
        output_index = target.view(1, 1, edges, 1, 1).expand_as(weighted)
        output.scatter_add_(2, output_index, weighted)
        return self.output_projection(output.reshape(batch, patches, nodes, hidden))


class FactorizedSTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout: float,
        graph_bias: float,
    ) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.spatial_attention = SparseGraphAttention(
            hidden_dim,
            num_heads,
            dropout,
            graph_bias,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: GraphData) -> torch.Tensor:
        batch, patches, nodes, hidden = x.shape
        temporal_input = self.temporal_norm(x).permute(0, 2, 1, 3).reshape(batch * nodes, patches, hidden)
        temporal_output, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            need_weights=False,
        )
        temporal_output = temporal_output.reshape(batch, nodes, patches, hidden).permute(0, 2, 1, 3)
        x = x + self.dropout(temporal_output)
        x = x + self.dropout(self.spatial_attention(self.spatial_norm(x), graph))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class FactorizedSTEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_multiplier: int = 2,
        dropout: float = 0.1,
        graph_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList(
            [
                FactorizedSTBlock(
                    hidden_dim,
                    num_heads,
                    ffn_multiplier,
                    dropout,
                    graph_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, graph: GraphData) -> torch.Tensor:
        if tokens.ndim != 4 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError("tokens must be [B, P, N, D]")
        if tokens.shape[2] != graph.num_nodes:
            raise ValueError("token node dimension does not match graph")
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden, graph)
        return self.output_norm(hidden)


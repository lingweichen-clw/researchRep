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


class MixedRangeRouteAttention(nn.Module):
    """History-conditioned route over first-order and remote nodes.

    The route is query-conditioned but node-ID independent.  It selects a
    fixed-size mixture of direct graph neighbors and non-neighbors, then
    reuses the selected sources for every temporal patch.
    """

    def __init__(
        self,
        hidden_dim: int,
        route_dim: int = 16,
        route_top_k: int = 10,
        route_local_quota: int = 4,
        prior_weight: float = 0.25,
        temperature: float = 0.1,
        gate_bias: float = -2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or route_dim <= 0:
            raise ValueError("hidden_dim and route_dim must be positive")
        if route_top_k <= 0:
            raise ValueError("route_top_k must be positive")
        if not 0 <= route_local_quota <= route_top_k:
            raise ValueError("route_local_quota must be in [0, route_top_k]")
        if prior_weight < 0.0:
            raise ValueError("prior_weight must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if gate_bias >= 0.0:
            raise ValueError("gate_bias must be negative")

        self.hidden_dim = int(hidden_dim)
        self.route_dim = int(route_dim)
        self.route_top_k = int(route_top_k)
        self.route_local_quota = int(route_local_quota)
        self.prior_weight = float(prior_weight)
        self.temperature = float(temperature)
        self.summary_norm = nn.LayerNorm(2 * hidden_dim)
        self.query_projection = nn.Linear(2 * hidden_dim, route_dim)
        self.key_projection = nn.Linear(2 * hidden_dim, route_dim)
        # Values travel through a route_dim basis before returning to D.  This
        # keeps each route a low-rank residual instead of a second full D x D
        # spatial projection.
        self.value_down = nn.Linear(hidden_dim, route_dim)
        self.value_up = nn.Linear(route_dim, hidden_dim)
        self.target_norm = nn.LayerNorm(hidden_dim)
        self.route_norm = nn.LayerNorm(hidden_dim)
        self.gate_projection = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.gate_projection.bias, gate_bias)

    @staticmethod
    def _candidate_masks(graph: GraphData, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = graph.num_nodes
        target, source = graph.edge_index.to(device=device)
        direct = torch.zeros((nodes, nodes), dtype=torch.bool, device=device)
        direct[target, source] = True
        direct.fill_diagonal_(False)
        remote = ~direct
        remote.fill_diagonal_(False)
        return direct, remote

    def _select_indices_reference(
        self,
        scores: torch.Tensor,
        graph: GraphData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select route indices with a mixed-range quota.

        Args:
            scores: [B, N, N] route scores before masking.
            graph: static graph used to identify first-order neighbors.

        Returns:
            indices: [B, N, K] source indices.
            valid: [B, N, K] slots that contain a source.
            local: [B, N, K] slots selected from first-order neighbors.
        """
        if scores.ndim != 3 or scores.shape[1] != scores.shape[2]:
            raise ValueError("scores must be [B, N, N]")
        batch, nodes, _ = scores.shape
        if nodes != graph.num_nodes:
            raise ValueError("route score node dimension does not match graph")
        effective_k = min(self.route_top_k, max(nodes - 1, 0))
        if effective_k == 0:
            empty = torch.empty((batch, nodes, 0), dtype=torch.long, device=scores.device)
            valid = torch.empty((batch, nodes, 0), dtype=torch.bool, device=scores.device)
            return empty, valid, valid

        local_quota = min(self.route_local_quota, effective_k)
        remote_quota = effective_k - local_quota
        direct, remote = self._candidate_masks(graph, scores.device)
        indices = torch.zeros((batch, nodes, effective_k), dtype=torch.long, device=scores.device)
        valid = torch.zeros((batch, nodes, effective_k), dtype=torch.bool, device=scores.device)
        local_slots = torch.zeros_like(valid)

        # N is at most a few hundred for the supported traffic graphs.  The
        # per-target loop keeps quota fallback explicit and avoids dense masks
        # whose valid count differs across nodes.
        for target_node in range(nodes):
            local_ids = torch.nonzero(direct[target_node], as_tuple=False).flatten()
            remote_ids = torch.nonzero(remote[target_node], as_tuple=False).flatten()
            local_take = min(local_quota, int(local_ids.numel()))
            remote_take = min(remote_quota, int(remote_ids.numel()))
            local_deficit = local_quota - local_take
            remote_deficit = remote_quota - remote_take
            remote_take = min(int(remote_ids.numel()), remote_take + local_deficit)
            local_take = min(int(local_ids.numel()), local_take + remote_deficit)

            position = 0
            if local_take > 0:
                local_values = scores[:, target_node, local_ids]
                _, local_order = torch.topk(local_values, k=local_take, dim=-1)
                chosen = local_ids[local_order]
                indices[:, target_node, position : position + local_take] = chosen
                valid[:, target_node, position : position + local_take] = True
                local_slots[:, target_node, position : position + local_take] = True
                position += local_take
            if remote_take > 0:
                remote_values = scores[:, target_node, remote_ids]
                _, remote_order = torch.topk(remote_values, k=remote_take, dim=-1)
                chosen = remote_ids[remote_order]
                indices[:, target_node, position : position + remote_take] = chosen
                valid[:, target_node, position : position + remote_take] = True

        return indices, valid, local_slots

    def select_indices_vectorized(
        self,
        scores: torch.Tensor,
        graph: GraphData,
        candidate_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select mixed-range route slots without a Python target-node loop."""
        if scores.ndim != 3 or scores.shape[1] != scores.shape[2]:
            raise ValueError("scores must be [B, N, N]")
        batch, nodes, _ = scores.shape
        if nodes != graph.num_nodes:
            raise ValueError("route score node dimension does not match graph")
        effective_k = min(self.route_top_k, max(nodes - 1, 0))
        if effective_k == 0:
            empty = torch.empty((batch, nodes, 0), dtype=torch.long, device=scores.device)
            valid = torch.empty((batch, nodes, 0), dtype=torch.bool, device=scores.device)
            return empty, valid, valid

        if candidate_indices is None:
            candidate_indices = graph.mixed_range_candidate_indices()
        local_ids, local_valid, remote_ids, remote_valid = (
            tensor.to(device=scores.device) for tensor in candidate_indices
        )
        candidate_width = local_ids.shape[1]
        if candidate_width < effective_k or remote_ids.shape[1] < effective_k:
            raise ValueError("graph candidate index width is smaller than route top-k")
        local_quota = min(self.route_local_quota, effective_k)
        remote_quota = effective_k - local_quota

        gather_index = local_ids.unsqueeze(0).expand(batch, -1, -1)
        local_scores = scores.gather(2, gather_index).masked_fill(
            ~local_valid.unsqueeze(0), -torch.inf
        )
        gather_index = remote_ids.unsqueeze(0).expand(batch, -1, -1)
        remote_scores = scores.gather(2, gather_index).masked_fill(
            ~remote_valid.unsqueeze(0), -torch.inf
        )
        local_count = local_valid.sum(dim=1).to(dtype=torch.long)
        remote_count = remote_valid.sum(dim=1).to(dtype=torch.long)
        local_take = local_count.clamp_max(local_quota)
        remote_take = remote_count.clamp_max(remote_quota)
        local_deficit = local_quota - local_take
        remote_deficit = remote_quota - remote_take
        remote_take = (remote_take + local_deficit).clamp_max(remote_count)
        local_take = (local_take + remote_deficit).clamp_max(local_count)

        local_order = torch.topk(local_scores, k=effective_k, dim=-1).indices
        remote_order = torch.topk(remote_scores, k=effective_k, dim=-1).indices
        local_rank_ids = local_ids.unsqueeze(0).expand(batch, -1, -1).gather(2, local_order)
        remote_rank_ids = remote_ids.unsqueeze(0).expand(batch, -1, -1).gather(2, remote_order)

        slots = torch.arange(effective_k, device=scores.device).view(1, 1, -1)
        local_final = local_take.view(1, nodes, 1)
        remote_final = remote_take.view(1, nodes, 1)
        local_slot = (slots < local_final).expand(batch, -1, -1)
        remote_slot = ((slots >= local_final) & (slots - local_final < remote_final)).expand(
            batch, -1, -1
        )
        local_position = slots.expand(batch, nodes, -1).clamp_max(effective_k - 1)
        remote_position = (slots - local_final).clamp_min(0).expand(batch, -1, -1).clamp_max(effective_k - 1)
        selected_local = local_rank_ids.gather(2, local_position)
        selected_remote = remote_rank_ids.gather(2, remote_position)
        indices = torch.where(
            local_slot,
            selected_local,
            torch.where(remote_slot, selected_remote, torch.zeros_like(selected_local)),
        )
        valid = local_slot | remote_slot
        local_slots = local_slot
        return indices, valid, local_slots

    def select_indices(
        self,
        scores: torch.Tensor,
        graph: GraphData,
        candidate_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.select_indices_vectorized(scores, graph, candidate_indices)

    def forward(
        self,
        x: torch.Tensor,
        graph: GraphData,
        diffusion_prior: torch.Tensor | None = None,
        candidate_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must be [B, P, N, D]")
        batch, patches, nodes, hidden = x.shape
        if hidden != self.hidden_dim or nodes != graph.num_nodes:
            raise ValueError("route tensor and graph dimensions do not match")
        if not torch.isfinite(x).all():
            raise ValueError("route input contains NaN or Inf")

        summary = torch.cat((x.mean(dim=1), x[:, -1] - x[:, 0]), dim=-1)  # [B, N, 2D]
        summary = self.summary_norm(summary)
        query = self.query_projection(summary)
        key = self.key_projection(summary)
        scores = torch.einsum("bnr,bmr->bnm", query, key) / math.sqrt(self.route_dim)
        if diffusion_prior is None:
            diffusion_prior = graph.random_walk_diffusion_prior()
        diffusion_prior = diffusion_prior.to(device=x.device, dtype=x.dtype)
        if diffusion_prior.shape != (nodes, nodes):
            raise ValueError("diffusion prior shape does not match graph")
        scores = scores + self.prior_weight * torch.log1p(diffusion_prior.clamp_min(0.0))[None]
        scores = scores.masked_fill(torch.eye(nodes, dtype=torch.bool, device=x.device)[None], -torch.inf)

        indices, valid, _ = self.select_indices(scores, graph, candidate_indices)
        if indices.shape[-1] == 0:
            return torch.zeros_like(x)
        route_scores = scores.gather(2, indices)
        route_scores = route_scores.masked_fill(~valid, -torch.inf)
        weights = torch.softmax(route_scores / self.temperature, dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        values = self.value_down(x).unsqueeze(3)
        route_k = indices.shape[-1]
        gather_index = indices[:, None, :, :, None].expand(
            batch,
            patches,
            nodes,
            route_k,
            self.route_dim,
        )
        source_values = values.expand(
            batch,
            patches,
            nodes,
            route_k,
            self.route_dim,
        ).gather(2, gather_index)
        # Gathered route messages are [B, P, N, K, route_dim].
        route_output = (weights[:, None, :, :, None] * source_values).sum(dim=3)
        route_output = self.value_up(route_output)
        gate_input = torch.cat((self.target_norm(x), self.route_norm(route_output)), dim=-1)
        gate = torch.sigmoid(self.gate_projection(gate_input))
        return self.dropout(gate * route_output)


class FactorizedSTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout: float,
        graph_bias: float,
        route_enabled: bool = False,
        route_dim: int = 16,
        route_top_k: int = 10,
        route_local_quota: int = 4,
        route_prior_weight: float = 0.25,
        route_temperature: float = 0.1,
        route_gate_bias: float = -2.0,
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
        self.route_attention = (
            MixedRangeRouteAttention(
                hidden_dim=hidden_dim,
                route_dim=route_dim,
                route_top_k=route_top_k,
                route_local_quota=route_local_quota,
                prior_weight=route_prior_weight,
                temperature=route_temperature,
                gate_bias=route_gate_bias,
                dropout=dropout,
            )
            if route_enabled
            else None
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        graph: GraphData,
        diffusion_prior: torch.Tensor | None = None,
        candidate_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
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
        spatial_input = self.spatial_norm(x)
        spatial_output = self.spatial_attention(spatial_input, graph)
        if self.route_attention is not None:
            # The route branch supplements local graph attention with a fixed
            # 4-direct + 6-remote candidate set selected from history only.
            spatial_output = spatial_output + self.route_attention(
                spatial_input,
                graph,
                diffusion_prior,
                candidate_indices,
            )
        x = x + self.dropout(spatial_output)
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
        route_enabled: bool = False,
        route_dim: int = 16,
        route_top_k: int = 10,
        route_local_quota: int = 4,
        route_prior_weight: float = 0.25,
        route_temperature: float = 0.1,
        route_gate_bias: float = -2.0,
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
                    route_enabled,
                    route_dim,
                    route_top_k,
                    route_local_quota,
                    route_prior_weight,
                    route_temperature,
                    route_gate_bias,
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
        diffusion_prior = (
            graph.random_walk_diffusion_prior()
            if self.blocks[0].route_attention is not None
            else None
        )
        candidate_indices = (
            (
                graph.higher_order_candidate_indices()
                if self.blocks[0].route_attention.route_local_quota == 0
                else graph.mixed_range_candidate_indices()
            )
            if self.blocks[0].route_attention is not None
            else None
        )
        for block in self.blocks:
            hidden = block(hidden, graph, diffusion_prior, candidate_indices)
        return self.output_norm(hidden)

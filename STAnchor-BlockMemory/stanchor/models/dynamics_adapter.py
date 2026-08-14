"""History-only local and graph dynamics adapter for retrieval pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from stanchor.data.graph import GraphData


@dataclass(frozen=True)
class DynamicsAdapterOutput:
    hidden: torch.Tensor  # [B, P, N, D]
    local_dynamics: torch.Tensor  # [B, P, N, D]
    local_valid: torch.Tensor  # [B, P, N]
    graph_dynamics: torch.Tensor | None  # [B, P, N, D]
    graph_valid: torch.Tensor | None  # [B, P, N]
    spatial_gate: torch.Tensor | None  # [B, P, N, 1]
    fusion_gate: torch.Tensor  # [B, P, N, 1] or [B, P, N, G]
    residual: torch.Tensor  # [B, P, N, D], zero at invalid adapter positions
    adapter_valid: torch.Tensor  # [B, P, N]
    modulation: torch.Tensor | None = None  # [B, P, N, Db]
    low_rank_residual: torch.Tensor | None = None  # [B, P, N, D]
    direct_residual: torch.Tensor | None = None  # [B, P, N, D]


def summarize_adapter_output(output: DynamicsAdapterOutput) -> dict[str, float]:
    """Return deployment-safe scalar diagnostics for one adapter forward pass."""
    with torch.no_grad():
        valid = output.adapter_valid.bool()
        valid_count = int(valid.sum().item())
        total_count = valid.numel()
        hidden_dim = output.hidden.shape[-1]
        gate_groups = output.fusion_gate.shape[-1]
        if hidden_dim % gate_groups != 0:
            raise ValueError("fusion gate groups must divide the hidden dimension")
        expanded_gate = output.fusion_gate.repeat_interleave(
            hidden_dim // gate_groups,
            dim=-1,
        )
        if valid_count == 0:
            fusion_gate_mean = contribution_ratio = 0.0
            group_gate_std = modulation_abs_mean = modulation_token_std = 0.0
            low_rank_ratio = direct_ratio = 0.0
        else:
            selected_gate = output.fusion_gate[valid]
            fusion_gate_mean = float(selected_gate.mean().detach())
            group_gate_std = float(
                selected_gate.std(dim=-1, unbiased=False).mean().detach()
            )
            contribution = expanded_gate * output.residual
            base_hidden = output.hidden - contribution
            ratio = contribution.norm(dim=-1) / base_hidden.norm(dim=-1).clamp_min(1.0e-8)
            contribution_ratio = float(ratio.masked_select(valid).mean().detach())
            base_norm = base_hidden.norm(dim=-1).clamp_min(1.0e-8)

            if output.modulation is None:
                modulation_abs_mean = modulation_token_std = 0.0
            else:
                selected_modulation = output.modulation[valid]
                modulation_abs_mean = float(selected_modulation.abs().mean().detach())
                modulation_token_std = float(
                    selected_modulation.std(dim=0, unbiased=False).mean().detach()
                )

            def component_ratio(component: torch.Tensor | None) -> float:
                if component is None:
                    return 0.0
                component_norm = component.norm(dim=-1) / base_norm
                return float(component_norm.masked_select(valid).mean().detach())

            low_rank_ratio = component_ratio(output.low_rank_residual)
            direct_ratio = component_ratio(output.direct_residual)

        spatial_gate_mean = 0.0
        if output.spatial_gate is not None and output.graph_valid is not None:
            graph_valid = output.graph_valid.bool()
            if bool(graph_valid.any()):
                spatial_gate_mean = float(
                    output.spatial_gate.squeeze(-1)
                    .masked_select(graph_valid)
                    .mean()
                    .detach()
                )
        return {
            "valid_fraction": valid_count / max(total_count, 1),
            "fusion_gate_mean": fusion_gate_mean,
            "spatial_gate_mean": spatial_gate_mean,
            "contribution_ratio": contribution_ratio,
            "modulation_abs_mean": modulation_abs_mean,
            "modulation_token_std": modulation_token_std,
            "group_gate_mean": fusion_gate_mean,
            "group_gate_std": group_gate_std,
            "low_rank_contribution_ratio": low_rank_ratio,
            "direct_contribution_ratio": direct_ratio,
            "total_contribution_ratio": contribution_ratio,
        }


class HistoryDynamicsAdapter(nn.Module):
    """Inject visible historical increments into encoded patch tokens.

    ``local`` uses only each node's increments. ``local_graph`` additionally
    aggregates valid non-self neighbors over the configured static graph.
    Query future values are not accepted by this interface.
    """

    VALID_MODES = frozenset({"local", "local_graph", "context_conditioned"})

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        hidden_dim: int,
        bottleneck_dim: int = 16,
        mode: str = "local_graph",
        gate_bias: float = -2.0,
        gate_groups: int = 1,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or patch_size <= 0 or hidden_dim <= 0:
            raise ValueError("input_channels, patch_size, and hidden_dim must be positive")
        if bottleneck_dim <= 0 or bottleneck_dim > hidden_dim:
            raise ValueError("bottleneck_dim must be in [1, hidden_dim]")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        if gate_bias >= 0.0:
            raise ValueError("gate_bias must be negative so the adapter starts conservative")
        if mode == "context_conditioned" and (
            gate_groups <= 0 or hidden_dim % gate_groups != 0
        ):
            raise ValueError("gate_groups must be positive and divide hidden_dim")

        self.input_channels = int(input_channels)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.mode = mode
        self.gate_groups = int(gate_groups if mode == "context_conditioned" else 1)

        self.delta_projection = nn.Linear(
            self.patch_size * self.input_channels,
            self.hidden_dim,
            bias=False,
        )
        self.spatial_gate_projection = (
            nn.Linear(2 * self.hidden_dim, 1)
            if mode in {"local_graph", "context_conditioned"}
            else None
        )
        self.residual_down = nn.Linear(self.hidden_dim, self.bottleneck_dim)
        self.residual_up = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        self.fusion_gate_projection = nn.Linear(2 * self.hidden_dim, self.gate_groups)
        self.activation = nn.GELU()
        if mode == "context_conditioned":
            self.context_hidden_norm = nn.LayerNorm(self.hidden_dim)
            self.context_dynamics_norm = nn.LayerNorm(self.hidden_dim)
            self.residual_norm = nn.LayerNorm(self.hidden_dim)
            self.modulation_projection = nn.Linear(
                2 * self.hidden_dim,
                self.bottleneck_dim,
            )
            self.direct_scale = nn.Parameter(torch.zeros(self.hidden_dim))
        else:
            self.context_hidden_norm = None
            self.context_dynamics_norm = None
            self.residual_norm = None
            self.modulation_projection = None
            self.direct_scale = None

        # Exact identity at initialization: the adapter may learn only when
        # pretraining losses provide useful gradients.
        nn.init.zeros_(self.residual_up.weight)
        nn.init.zeros_(self.residual_up.bias)
        if self.modulation_projection is not None:
            nn.init.zeros_(self.modulation_projection.weight)
            nn.init.zeros_(self.modulation_projection.bias)
        nn.init.zeros_(self.fusion_gate_projection.weight)
        nn.init.constant_(self.fusion_gate_projection.bias, gate_bias)
        if self.spatial_gate_projection is not None:
            nn.init.zeros_(self.spatial_gate_projection.weight)
            nn.init.constant_(self.spatial_gate_projection.bias, gate_bias)

    def _project_local_dynamics(
        self,
        normalized_history: torch.Tensor,
        observed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, nodes, channels = normalized_history.shape
        pair_valid = observed[:, 1:].bool() & observed[:, :-1].bool()
        increments = normalized_history[:, 1:] - normalized_history[:, :-1]
        increments = torch.where(pair_valid, increments, torch.zeros_like(increments))

        # The first position has no preceding observation and is therefore an
        # explicitly invalid zero difference.
        increments = torch.cat((torch.zeros_like(normalized_history[:, :1]), increments), dim=1)
        difference_valid = torch.cat((torch.zeros_like(observed[:, :1], dtype=torch.bool), pair_valid), dim=1)

        patches = time // self.patch_size
        patch_values = increments.reshape(
            batch, patches, self.patch_size, nodes, channels
        ).permute(0, 1, 3, 2, 4)
        patch_values = patch_values.reshape(
            batch, patches, nodes, self.patch_size * channels
        )
        patch_valid = difference_valid.reshape(
            batch, patches, self.patch_size, nodes, channels
        ).any(dim=(2, 4))
        projected = self.delta_projection(patch_values)
        projected = torch.where(
            patch_valid.unsqueeze(-1), projected, torch.zeros_like(projected)
        )
        return projected, patch_valid

    @staticmethod
    def _aggregate_graph_dynamics(
        local_dynamics: torch.Tensor,
        local_valid: torch.Tensor,
        graph: GraphData,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, patches, nodes, hidden = local_dynamics.shape
        target, source = graph.edge_index
        non_self = target != source
        target = target[non_self]
        source = source[non_self]
        edge_weight = graph.edge_weight[non_self].to(
            device=local_dynamics.device,
            dtype=local_dynamics.dtype,
        )
        if target.numel() == 0:
            return (
                torch.zeros_like(local_dynamics),
                torch.zeros_like(local_valid, dtype=torch.bool),
            )

        source_valid = local_valid.index_select(2, source)
        effective_weight = source_valid.to(local_dynamics.dtype) * edge_weight.view(1, 1, -1)
        source_features = local_dynamics.index_select(2, source)
        weighted_features = source_features * effective_weight.unsqueeze(-1)

        numerator = torch.zeros(
            (batch, patches, nodes, hidden),
            dtype=local_dynamics.dtype,
            device=local_dynamics.device,
        )
        feature_index = target.view(1, 1, -1, 1).expand_as(weighted_features)
        numerator.scatter_add_(2, feature_index, weighted_features)
        denominator = torch.zeros(
            (batch, patches, nodes),
            dtype=local_dynamics.dtype,
            device=local_dynamics.device,
        )
        weight_index = target.view(1, 1, -1).expand_as(effective_weight)
        denominator.scatter_add_(2, weight_index, effective_weight)
        graph_valid = denominator > 0
        aggregated = numerator / denominator.clamp_min(1.0e-8).unsqueeze(-1)
        aggregated = torch.where(
            graph_valid.unsqueeze(-1), aggregated, torch.zeros_like(aggregated)
        )
        return aggregated, graph_valid

    def forward(
        self,
        hidden: torch.Tensor,
        normalized_history: torch.Tensor,
        observed: torch.Tensor,
        graph: GraphData,
    ) -> DynamicsAdapterOutput:
        if hidden.ndim != 4:
            raise ValueError("hidden must be [B, P, N, D]")
        if normalized_history.ndim != 4 or observed.shape != normalized_history.shape:
            raise ValueError("normalized_history and observed must be [B, T, N, C]")
        batch, time, nodes, channels = normalized_history.shape
        if channels != self.input_channels:
            raise ValueError("history channel dimension does not match input_channels")
        if time % self.patch_size != 0:
            raise ValueError("history length must be divisible by patch_size")
        expected_hidden = (batch, time // self.patch_size, nodes, self.hidden_dim)
        if hidden.shape != expected_hidden:
            raise ValueError(f"hidden must have shape {expected_hidden}")
        if graph.num_nodes != nodes:
            raise ValueError("history node dimension does not match graph")

        local_dynamics, local_valid = self._project_local_dynamics(
            normalized_history, observed.bool()
        )
        graph_dynamics: torch.Tensor | None = None
        graph_valid: torch.Tensor | None = None
        spatial_gate: torch.Tensor | None = None
        combined_dynamics = local_dynamics
        adapter_valid = local_valid

        if self.spatial_gate_projection is not None:
            graph_dynamics, graph_valid = self._aggregate_graph_dynamics(
                local_dynamics, local_valid, graph
            )
            spatial_gate = torch.sigmoid(
                self.spatial_gate_projection(
                    torch.cat((local_dynamics, graph_dynamics), dim=-1)
                )
            )
            combined_dynamics = local_dynamics + spatial_gate * graph_dynamics
            adapter_valid = local_valid | graph_valid

        modulation: torch.Tensor | None = None
        low_rank_residual: torch.Tensor | None = None
        direct_residual: torch.Tensor | None = None
        if self.mode == "context_conditioned":
            if (
                self.context_hidden_norm is None
                or self.context_dynamics_norm is None
                or self.residual_norm is None
                or self.modulation_projection is None
                or self.direct_scale is None
            ):
                raise RuntimeError("context-conditioned adapter is not initialized")
            hidden_context = self.context_hidden_norm(hidden)
            dynamics_context = self.context_dynamics_norm(combined_dynamics)
            modulation = torch.tanh(
                self.modulation_projection(
                    torch.cat((hidden_context, dynamics_context), dim=-1)
                )
            )
            dynamic_factors = self.activation(
                self.residual_down(combined_dynamics)
            )
            modulated_factors = dynamic_factors * (1.0 + modulation)
            raw_low_rank = self.residual_up(modulated_factors)
            raw_direct = combined_dynamics * self.direct_scale.view(1, 1, 1, -1)
            low_rank_residual = torch.where(
                adapter_valid.unsqueeze(-1),
                raw_low_rank,
                torch.zeros_like(raw_low_rank),
            )
            direct_residual = torch.where(
                adapter_valid.unsqueeze(-1),
                raw_direct,
                torch.zeros_like(raw_direct),
            )
            modulation = torch.where(
                adapter_valid.unsqueeze(-1),
                modulation,
                torch.zeros_like(modulation),
            )
            residual = low_rank_residual + direct_residual
            fusion_gate = torch.sigmoid(
                self.fusion_gate_projection(
                    torch.cat(
                        (hidden_context, self.residual_norm(residual)),
                        dim=-1,
                    )
                )
            )
            expanded_gate = fusion_gate.repeat_interleave(
                self.hidden_dim // self.gate_groups,
                dim=-1,
            )
        else:
            raw_residual = self.residual_up(
                self.activation(self.residual_down(combined_dynamics))
            )
            residual = torch.where(
                adapter_valid.unsqueeze(-1),
                raw_residual,
                torch.zeros_like(raw_residual),
            )
            fusion_gate = torch.sigmoid(
                self.fusion_gate_projection(torch.cat((hidden, residual), dim=-1))
            )
            expanded_gate = fusion_gate
            low_rank_residual = residual
        adapted_hidden = hidden + expanded_gate * residual
        return DynamicsAdapterOutput(
            hidden=adapted_hidden,
            local_dynamics=local_dynamics,
            local_valid=local_valid,
            graph_dynamics=graph_dynamics,
            graph_valid=graph_valid,
            spatial_gate=spatial_gate,
            fusion_gate=fusion_gate,
            residual=residual,
            adapter_valid=adapter_valid,
            modulation=modulation,
            low_rank_residual=low_rank_residual,
            direct_residual=direct_residual,
        )

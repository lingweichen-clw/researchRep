"""Lightweight forecasting, mirage confidence, and exact fallback fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from stanchor.modes import (
    BASE_ONLY,
    HORIZON_ONLY_MODES,
    LEARNED_TOPK_CONFIDENCE,
    validate_downstream_mode,
)
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


class LightweightForecastBackbone(nn.Module):
    """Shared per-node residual MLP used as the v1 simple downstream model."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        input_channels: int,
        output_channels: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels != output_channels:
            raise ValueError("v1 residual backbone requires input_channels == output_channels")
        self.context_length = context_length
        self.horizon = horizon
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.network = nn.Sequential(
            nn.Linear(context_length * input_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * output_channels),
        )
        output_layer = self.network[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must be [B, T, N, C]")
        batch, time, nodes, channels = x.shape
        if time != self.context_length or channels != self.input_channels:
            raise ValueError("x does not match backbone configuration")
        node_history = x.permute(0, 2, 1, 3).reshape(batch, nodes, time * channels)
        residual = self.network(node_history).view(batch, nodes, self.horizon, self.output_channels)
        residual = residual.permute(0, 2, 1, 3).contiguous()
        return x[:, -1:, :, :].expand(-1, self.horizon, -1, -1) + residual


def build_confidence_features(
    candidates: NodeCandidates,
    aggregation: AggregationOutput,
    base_prediction: torch.Tensor,
    level_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return six diagnostics [B, H, N, 6] and memory validity [B, H, N, 1]."""
    if level_temperature <= 0:
        raise ValueError("level_temperature must be positive")
    if aggregation.prediction.shape != base_prediction.shape:
        raise ValueError("memory and base predictions must have identical shapes")
    batch, horizon, nodes, channels = base_prediction.shape
    if candidates.weights.shape[:2] != (batch, nodes):
        raise ValueError("candidate and prediction dimensions do not align")
    valid = candidates.valid
    shape_values = candidates.shape_scores.masked_fill(~valid, -torch.inf)
    best_shape = shape_values.amax(dim=-1)
    best_shape = torch.where(torch.isfinite(best_shape), best_shape, torch.zeros_like(best_shape))

    top1 = candidates.total_scores[..., 0]
    top2 = candidates.total_scores[..., 1] if candidates.total_scores.shape[-1] > 1 else torch.zeros_like(top1)
    has_two = valid.sum(dim=-1) >= 2
    margin = torch.where(has_two, top1 - top2, torch.zeros_like(top1))

    weights = candidates.weights
    top_k = weights.shape[-1]
    if top_k > 1:
        entropy = -(weights * torch.log(weights.clamp_min(1.0e-8))).sum(dim=-1) / math.log(top_k)
        concentration = 1.0 - entropy
    else:
        concentration = torch.ones_like(best_shape)
    concentration = torch.where(valid.any(dim=-1), concentration, torch.zeros_like(concentration))

    future_disagreement = torch.log1p(aggregation.variance.mean(dim=-1))
    weighted_level = (weights * candidates.level_distances.masked_fill(~valid, 0.0)).sum(dim=-1)
    level_match = torch.exp(-weighted_level / level_temperature)
    level_match = torch.where(valid.any(dim=-1), level_match, torch.zeros_like(level_match))
    source_disagreement = (aggregation.prediction - base_prediction).abs().mean(dim=-1)

    def expand_node_feature(value: torch.Tensor) -> torch.Tensor:
        return value[:, None, :, None].expand(-1, horizon, -1, -1)

    features = torch.cat(
        (
            expand_node_feature(best_shape),
            expand_node_feature(margin),
            expand_node_feature(concentration),
            future_disagreement.unsqueeze(-1),
            expand_node_feature(level_match),
            source_disagreement.unsqueeze(-1),
        ),
        dim=-1,
    )
    # One confidence/fusion value is shared by all channels, so every channel
    # must have a valid memory value before the history path can be enabled.
    memory_valid = aggregation.valid.all(dim=-1, keepdim=True)
    if features.shape != (batch, horizon, nodes, 6):
        raise RuntimeError("confidence feature construction produced an invalid shape")
    return features, memory_valid


class ConfidenceHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, memory_valid: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-1] != 6:
            raise ValueError("features must be [B, H, N, 6]")
        if memory_valid.shape != features.shape[:-1] + (1,):
            raise ValueError("memory_valid must be [B, H, N, 1]")
        probability = torch.sigmoid(self.network(features))
        return torch.where(memory_valid, probability, torch.zeros_like(probability))


class SafeResidualFusion(nn.Module):
    def __init__(self, horizon: int, initial_max_weight: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < initial_max_weight < 1.0:
            raise ValueError("initial_max_weight must be in (0, 1)")
        initial_logit = math.log(initial_max_weight / (1.0 - initial_max_weight))
        self.horizon_logits = nn.Parameter(torch.full((horizon,), initial_logit))

    def forward(
        self,
        base: torch.Tensor,
        memory: torch.Tensor,
        confidence: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base.shape != memory.shape:
            raise ValueError("base and memory predictions must have the same shape")
        if confidence.shape != base.shape[:-1] + (1,) or memory_valid.shape != confidence.shape:
            raise ValueError("confidence and memory_valid must be [B, H, N, 1]")
        if base.shape[1] != self.horizon_logits.numel():
            raise ValueError("prediction horizon mismatch")
        horizon_limit = torch.sigmoid(self.horizon_logits).view(1, -1, 1, 1)
        weight = horizon_limit * confidence
        weight = torch.where(memory_valid, weight, torch.zeros_like(weight))
        return base + weight * (memory - base), weight


def confidence_soft_target(
    base: torch.Tensor,
    memory: torch.Tensor,
    target: torch.Tensor,
    memory_valid: torch.Tensor,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    if base.shape != memory.shape or target.shape != base.shape:
        raise ValueError("base, memory, and target must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gain = (base - target).abs().mean(dim=-1) - (memory - target).abs().mean(dim=-1)
    soft = torch.sigmoid((gain - margin) / temperature).unsqueeze(-1)
    return torch.where(memory_valid, soft, torch.zeros_like(soft))


@dataclass(frozen=True)
class DownstreamOutput:
    base_prediction: torch.Tensor
    memory_prediction: torch.Tensor
    confidence_features: torch.Tensor
    confidence: torch.Tensor
    fusion_weight: torch.Tensor
    final_prediction: torch.Tensor
    memory_valid: torch.Tensor


class STAnchorDownstreamModel(nn.Module):
    def __init__(
        self,
        backbone: LightweightForecastBackbone,
        confidence_head: ConfidenceHead,
        fusion: SafeResidualFusion,
        confidence_level_temperature: float,
        mode: str = LEARNED_TOPK_CONFIDENCE,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.confidence_head = confidence_head
        self.fusion = fusion
        self.confidence_level_temperature = confidence_level_temperature
        self.mode = validate_downstream_mode(mode)

    def forward(
        self,
        x: torch.Tensor,
        candidates: NodeCandidates | None,
        aggregation: AggregationOutput | None,
    ) -> DownstreamOutput:
        base = self.backbone(x)
        batch, horizon, nodes, _ = base.shape
        node_shape = (batch, horizon, nodes, 1)
        if self.mode == BASE_ONLY:
            zeros = torch.zeros(node_shape, dtype=base.dtype, device=base.device)
            return DownstreamOutput(
                base_prediction=base,
                memory_prediction=torch.zeros_like(base),
                confidence_features=torch.zeros(
                    (*node_shape[:-1], 6), dtype=base.dtype, device=base.device
                ),
                confidence=zeros,
                fusion_weight=zeros,
                final_prediction=base,
                memory_valid=torch.zeros(node_shape, dtype=torch.bool, device=base.device),
            )
        if aggregation is None:
            raise ValueError(f"{self.mode} requires a historical aggregation")
        memory_valid = aggregation.valid.all(dim=-1, keepdim=True)
        if self.mode in HORIZON_ONLY_MODES:
            confidence = torch.where(
                memory_valid,
                torch.ones(node_shape, dtype=base.dtype, device=base.device),
                torch.zeros(node_shape, dtype=base.dtype, device=base.device),
            )
            final, fusion_weight = self.fusion(
                base,
                aggregation.prediction.detach(),
                confidence,
                memory_valid,
            )
            return DownstreamOutput(
                base_prediction=base,
                memory_prediction=aggregation.prediction,
                confidence_features=torch.zeros(
                    (*node_shape[:-1], 6), dtype=base.dtype, device=base.device
                ),
                confidence=confidence,
                fusion_weight=fusion_weight,
                final_prediction=final,
                memory_valid=memory_valid,
            )
        if candidates is None:
            raise ValueError(f"{self.mode} requires node candidates for confidence features")
        features, memory_valid = build_confidence_features(
            candidates,
            aggregation,
            base.detach(),
            self.confidence_level_temperature,
        )
        confidence = self.confidence_head(features.detach(), memory_valid)
        final, fusion_weight = self.fusion(
            base,
            aggregation.prediction.detach(),
            confidence,
            memory_valid,
        )
        return DownstreamOutput(
            base_prediction=base,
            memory_prediction=aggregation.prediction,
            confidence_features=features,
            confidence=confidence,
            fusion_weight=fusion_weight,
            final_prediction=final,
            memory_valid=memory_valid,
        )

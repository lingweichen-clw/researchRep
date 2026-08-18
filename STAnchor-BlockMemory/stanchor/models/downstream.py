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
    LEARNED_TOPK_ERROR_AWARE,
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


class PredictedBaseRisk(nn.Module):
    """Estimate per-horizon base-model error from visible history and base output."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channels: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if context_length <= 0 or horizon <= 0 or channels <= 0 or hidden_dim <= 0:
            raise ValueError("risk-head dimensions must be positive")
        self.context_length = context_length
        self.horizon = horizon
        self.channels = channels
        input_dim = (context_length + horizon) * channels
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, history: torch.Tensor, base_prediction: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or base_prediction.ndim != 4:
            raise ValueError("history and base_prediction must be [B,T/H,N,C]")
        batch, time, nodes, channels = history.shape
        if time != self.context_length or channels != self.channels:
            raise ValueError("history does not match risk-head configuration")
        if base_prediction.shape != (batch, self.horizon, nodes, channels):
            raise ValueError("base prediction does not match risk-head configuration")
        node_history = history.permute(0, 2, 1, 3)
        mean = node_history.mean(dim=2, keepdim=True)
        std = node_history.std(dim=2, keepdim=True, unbiased=False).clamp_min(1.0e-3)
        normalized_history = (node_history - mean) / std
        node_base = base_prediction.permute(0, 2, 1, 3)
        risk_input = torch.cat(
            (
                normalized_history.reshape(batch, nodes, -1),
                node_base.reshape(batch, nodes, -1),
            ),
            dim=-1,
        )
        risk = torch.nn.functional.softplus(self.network(risk_input))
        return risk.permute(0, 2, 1).unsqueeze(-1).contiguous()


def build_error_aware_features(
    candidates: NodeCandidates,
    aggregation: AggregationOutput,
    base_prediction: torch.Tensor,
    predicted_base_risk: torch.Tensor,
    level_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build nine non-redundant deployment-available diagnostics.

    Latent48 Banks expose one retrieval similarity.  Profile and latent
    sub-scores are absent in this mainline, so using both fallback values
    would duplicate the same ``shape_scores`` signal.
    """
    if level_temperature <= 0:
        raise ValueError("level_temperature must be positive")
    if aggregation.prediction.shape != base_prediction.shape:
        raise ValueError("memory and base predictions must have identical shapes")
    batch, horizon, nodes, channels = base_prediction.shape
    if predicted_base_risk.shape != (batch, horizon, nodes, 1):
        raise ValueError("predicted_base_risk must be [B,H,N,1]")
    if candidates.weights.shape[:2] != (batch, nodes):
        raise ValueError("candidate weights do not align with predictions")

    valid_candidates = candidates.valid
    weights = torch.where(
        valid_candidates, candidates.weights, torch.zeros_like(candidates.weights)
    )
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    normalized_weights = weights / weight_sum

    def weighted_node_score(values: torch.Tensor | None) -> torch.Tensor:
        if values is None:
            values = candidates.shape_scores
        return (normalized_weights * torch.where(valid_candidates, values, torch.zeros_like(values))).sum(dim=-1)

    retrieval_similarity = weighted_node_score(candidates.shape_scores)
    top1 = candidates.total_scores[..., 0]
    if candidates.total_scores.shape[-1] > 1:
        top2 = candidates.total_scores[..., 1]
        margin = torch.where(valid_candidates.sum(dim=-1) >= 2, top1 - top2, torch.zeros_like(top1))
    else:
        margin = torch.zeros_like(top1)
    top_k = weights.shape[-1]
    effective_support = 1.0 / normalized_weights.square().sum(dim=-1).clamp_min(1.0e-8)
    effective_support = effective_support / float(top_k)
    effective_support = torch.where(valid_candidates.any(dim=-1), effective_support, torch.zeros_like(effective_support))
    weighted_level = (normalized_weights * candidates.level_distances.masked_fill(~valid_candidates, 0.0)).sum(dim=-1)
    level_match = torch.exp(-weighted_level / level_temperature)
    level_match = torch.where(valid_candidates.any(dim=-1), level_match, torch.zeros_like(level_match))

    payload_dispersion = torch.log1p(aggregation.variance.clamp_min(0.0).sqrt().mean(dim=-1))
    memory_disagreement = torch.log1p((aggregation.prediction - base_prediction).abs().mean(dim=-1))
    candidate_mask = aggregation.candidate_masks.bool()
    candidate_weights = normalized_weights[:, None, :, :, None]
    effective_candidate_weights = candidate_weights * candidate_mask.to(base_prediction.dtype)
    candidate_denominator = effective_candidate_weights.sum(dim=3).clamp_min(1.0e-8)
    correction_sign = torch.sign(aggregation.candidate_futures - base_prediction.unsqueeze(3))
    signed_agreement = (effective_candidate_weights * correction_sign).sum(dim=3) / candidate_denominator
    direction_agreement = signed_agreement.abs().mean(dim=-1)

    def expand_node(value: torch.Tensor) -> torch.Tensor:
        return value[:, None, :, None].expand(-1, horizon, -1, -1)

    horizon_position = torch.linspace(
        0.0, 1.0, horizon, dtype=base_prediction.dtype, device=base_prediction.device
    ).view(1, horizon, 1, 1).expand(batch, -1, nodes, -1)
    features = torch.cat(
        (
            predicted_base_risk,
            expand_node(retrieval_similarity),
            expand_node(margin),
            expand_node(effective_support),
            payload_dispersion.unsqueeze(-1),
            direction_agreement.unsqueeze(-1),
            expand_node(level_match),
            memory_disagreement.unsqueeze(-1),
            horizon_position,
        ),
        dim=-1,
    )
    memory_valid = aggregation.valid.all(dim=-1, keepdim=True)
    features = torch.where(memory_valid.expand_as(features), features, torch.zeros_like(features))
    if features.shape != (batch, horizon, nodes, 9):
        raise RuntimeError("error-aware feature construction produced an invalid shape")
    return features, memory_valid


class ErrorAwareAdditiveFusion(nn.Module):
    """Add interpretable per-feature logit contributions to one blend weight."""

    def __init__(
        self,
        num_features: int = 9,
        hidden_dim: int = 8,
        initial_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if num_features <= 0 or hidden_dim <= 0:
            raise ValueError("fusion dimensions must be positive")
        if not 0.0 < initial_weight < 1.0:
            raise ValueError("initial_weight must be in (0,1)")
        self.num_features = num_features
        self.shape_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_features)
            ]
        )
        for function in self.shape_functions:
            nn.init.zeros_(function[-1].weight)
            nn.init.zeros_(function[-1].bias)
        initial_logit = math.log(initial_weight / (1.0 - initial_weight))
        self.bias = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))

    def forward(
        self,
        base: torch.Tensor,
        memory: torch.Tensor,
        features: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base.shape != memory.shape:
            raise ValueError("base and memory must have identical shapes")
        if features.shape != base.shape[:-1] + (self.num_features,):
            raise ValueError("features have an invalid shape")
        if memory_valid.shape != base.shape[:-1] + (1,):
            raise ValueError("memory_valid must be [B,H,N,1]")
        contributions = torch.cat(
            [function(features[..., index : index + 1]) for index, function in enumerate(self.shape_functions)],
            dim=-1,
        )
        logit = self.bias + contributions.sum(dim=-1, keepdim=True)
        weight = torch.sigmoid(logit)
        weight = torch.where(memory_valid, weight, torch.zeros_like(weight))
        final = base + weight * (memory - base)
        return final, weight, contributions


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
    predicted_base_risk: torch.Tensor | None = None
    additive_contributions: torch.Tensor | None = None


class STAnchorDownstreamModel(nn.Module):
    def __init__(
        self,
        backbone: LightweightForecastBackbone,
        confidence_head: ConfidenceHead,
        fusion: SafeResidualFusion,
        confidence_level_temperature: float,
        mode: str = LEARNED_TOPK_CONFIDENCE,
        risk_head: PredictedBaseRisk | None = None,
        error_aware_fusion: ErrorAwareAdditiveFusion | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.confidence_head = confidence_head
        self.fusion = fusion
        self.confidence_level_temperature = confidence_level_temperature
        self.mode = validate_downstream_mode(mode)
        self.risk_head = risk_head
        self.error_aware_fusion = error_aware_fusion
        if self.mode == LEARNED_TOPK_ERROR_AWARE and (
            risk_head is None or error_aware_fusion is None
        ):
            raise ValueError("error-aware mode requires risk_head and error_aware_fusion")

    def train(self, mode: bool = True) -> "STAnchorDownstreamModel":
        super().train(mode)
        if mode:
            for module in (
                self.backbone,
                self.confidence_head,
                self.fusion,
                self.risk_head,
                self.error_aware_fusion,
            ):
                if module is not None and not any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    module.eval()
        return self

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
        if self.mode == LEARNED_TOPK_ERROR_AWARE:
            if self.risk_head is None or self.error_aware_fusion is None:
                raise RuntimeError("error-aware modules are not initialized")
            predicted_risk = self.risk_head(x, base.detach())
            features, memory_valid = build_error_aware_features(
                candidates,
                aggregation,
                base.detach(),
                predicted_risk,
                self.confidence_level_temperature,
            )
            final, fusion_weight, contributions = self.error_aware_fusion(
                base,
                aggregation.prediction.detach(),
                features,
                memory_valid,
            )
            return DownstreamOutput(
                base_prediction=base,
                memory_prediction=aggregation.prediction,
                confidence_features=features,
                confidence=fusion_weight,
                fusion_weight=fusion_weight,
                final_prediction=final,
                memory_valid=memory_valid,
                predicted_base_risk=predicted_risk,
                additive_contributions=contributions,
            )
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

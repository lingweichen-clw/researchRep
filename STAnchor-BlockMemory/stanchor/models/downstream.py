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


class HorizonAwareAggregationHead(nn.Module):
    """Learn horizon-specific candidate weights from deployment-visible evidence.

    The head consumes only retrieval-time signals and the current base forecast.
    It does not touch targets or the frozen encoder. The output keeps the same
    [B, H, N, C] interface as the legacy aggregation step.
    """

    def __init__(self, hidden_dim: int = 174) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidates: NodeCandidates,
        aggregation: AggregationOutput,
        base_prediction: torch.Tensor,
    ) -> AggregationOutput:
        if aggregation.prediction.shape != base_prediction.shape:
            raise ValueError("memory and base predictions must have identical shapes")
        if aggregation.candidate_futures.ndim != 5:
            raise ValueError("candidate futures must be [B,H,N,K,C]")
        batch, horizon, nodes, top_k, channels = aggregation.candidate_futures.shape
        if candidates.weights.shape != (batch, nodes, top_k):
            raise ValueError("candidate weights do not align with aggregation")
        if aggregation.candidate_masks.shape != aggregation.candidate_futures.shape:
            raise ValueError("candidate masks must match candidate futures")

        candidate_mask = aggregation.candidate_masks.bool()
        candidate_valid = candidate_mask.any(dim=-1)
        # Invalid Bank padding may contain NaN; never let 0 * NaN enter a reduction.
        safe_candidate_futures = torch.where(
            candidate_mask,
            torch.nan_to_num(aggregation.candidate_futures),
            torch.zeros_like(aggregation.candidate_futures),
        )
        base_weights = candidates.weights[:, None, :, :].expand(batch, horizon, nodes, top_k)
        prior = torch.where(candidate_valid, base_weights, torch.zeros_like(base_weights))
        prior_den = prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        provisional = (prior.unsqueeze(-1) * safe_candidate_futures).sum(dim=3) / prior_den

        candidate_offset = (safe_candidate_futures - base_prediction.unsqueeze(3)).abs().mean(dim=-1)
        provisional_offset = (safe_candidate_futures - provisional.unsqueeze(3)).abs().mean(dim=-1)
        shape_score = torch.nan_to_num(candidates.shape_scores[:, None, :, :].expand(batch, horizon, nodes, top_k))
        level_distance = torch.nan_to_num(candidates.level_distances[:, None, :, :].expand(batch, horizon, nodes, top_k))
        shape_score = torch.where(candidate_valid, shape_score, torch.zeros_like(shape_score))
        level_distance = torch.where(candidate_valid, level_distance, torch.zeros_like(level_distance))
        horizon_position = torch.linspace(
            0.0,
            1.0,
            horizon,
            dtype=base_prediction.dtype,
            device=base_prediction.device,
        ).view(1, horizon, 1, 1).expand(batch, -1, nodes, top_k)
        features = torch.stack(
            (
                prior,
                shape_score,
                -level_distance,
                -candidate_offset,
                -provisional_offset,
                horizon_position,
            ),
            dim=-1,
        )
        logits = self.network(features).squeeze(-1)
        logits = logits.masked_fill(~candidate_valid, -1.0e9)
        max_logits = logits.amax(dim=-1, keepdim=True)
        candidate_weights = torch.exp(logits - max_logits).masked_fill(~candidate_valid, 0.0)
        candidate_weights = candidate_weights / candidate_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        effective_weights = candidate_weights.unsqueeze(-1) * candidate_mask.to(candidate_weights.dtype)
        denominator = effective_weights.sum(dim=3)
        prediction = (effective_weights * safe_candidate_futures).sum(dim=3) / denominator.clamp_min(1.0e-8)
        variance = (
            effective_weights
            * (safe_candidate_futures - prediction.unsqueeze(3)).square()
        ).sum(dim=3) / denominator.clamp_min(1.0e-8)
        valid = denominator > 0
        prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
        variance = torch.where(valid, variance, torch.zeros_like(variance))
        return AggregationOutput(
            prediction=prediction,
            variance=variance,
            valid=valid,
            candidate_futures=aggregation.candidate_futures,
            candidate_masks=aggregation.candidate_masks,
        )


def build_error_aware_features(
    candidates: NodeCandidates,
    aggregation: AggregationOutput,
    base_prediction: torch.Tensor,
    predicted_base_risk: torch.Tensor,
    level_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the legacy nine deployment-available diagnostics.

    The candidate future tensor remains [B,H,N,K,C] while diagnostics are
    computed. This preserves horizon-specific evidence instead of copying
    node-level retrieval statistics to every forecast step.
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

    payload_dispersion = torch.log1p(aggregation.variance.clamp_min(1.0e-8).sqrt().mean(dim=-1))
    memory_disagreement = torch.log1p((aggregation.prediction - base_prediction).abs().mean(dim=-1))
    candidate_mask = aggregation.candidate_masks.bool()
    safe_candidate_futures = torch.where(
        candidate_mask,
        torch.nan_to_num(aggregation.candidate_futures),
        torch.zeros_like(aggregation.candidate_futures),
    )
    candidate_weights = normalized_weights[:, None, :, :, None]
    effective_candidate_weights = candidate_weights * candidate_mask.to(base_prediction.dtype)
    candidate_denominator = effective_candidate_weights.sum(dim=3).clamp_min(1.0e-8)
    correction_sign = torch.sign(safe_candidate_futures - base_prediction.unsqueeze(3))
    signed_agreement = (effective_candidate_weights * correction_sign).sum(dim=3) / candidate_denominator
    direction_agreement = signed_agreement.mean(dim=-1).abs()

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


class StructuredErrorCorrector(nn.Module):
    """Legacy risk/evidence corrector used by the post-hoc protocol.

    The module keeps the base forecaster frozen and learns a bounded residual
    weight from deployment-available history, base output, and nine retrieval
    diagnostics. With the documented default widths it remains close to the 300k-parameter TGGE encoder budget.
    """

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channels: int,
        risk_hidden_dim: int = 256,
        evidence_hidden_dim: int = 128,
        num_features: int = 9,
        initial_weight: float = 0.1,
        correction_variant: str = "scalar_gate",
    ) -> None:
        super().__init__()
        if min(context_length, horizon, channels, risk_hidden_dim, evidence_hidden_dim, num_features) <= 0:
            raise ValueError("structured corrector dimensions must be positive")
        if not 0.0 < initial_weight < 1.0:
            raise ValueError("initial_weight must be in (0,1)")
        self.context_length = context_length
        self.horizon = horizon
        self.channels = channels
        self.num_features = num_features
        self.correction_variant = correction_variant
        self.risk_repr_dim = risk_hidden_dim // 2
        if self.risk_repr_dim <= 0:
            raise ValueError("risk_hidden_dim must be at least 2")
        self.risk_encoder = nn.Sequential(
            nn.Linear((context_length + horizon) * channels, risk_hidden_dim),
            nn.GELU(),
            nn.Linear(risk_hidden_dim, self.risk_repr_dim),
            nn.GELU(),
        )
        self.risk_output = nn.Linear(self.risk_repr_dim, horizon)
        self.evidence_encoder = nn.Sequential(
            nn.Linear(num_features, evidence_hidden_dim),
            nn.GELU(),
            nn.Linear(evidence_hidden_dim, evidence_hidden_dim),
            nn.GELU(),
        )
        # The documented joint state concatenates 128-D risk and 128-D
        # evidence representations, then preserves the full 256-D state.
        self.joint_dim = self.risk_repr_dim + evidence_hidden_dim
        self.joint_encoder = nn.Sequential(
            nn.Linear(self.risk_repr_dim + evidence_hidden_dim, self.joint_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Linear(self.joint_dim, self.joint_dim), nn.Sigmoid())
        self.output = nn.Sequential(
            nn.Linear(self.joint_dim, evidence_hidden_dim),
            nn.GELU(),
            nn.Linear(evidence_hidden_dim, 1),
        )
        self.shape_functions = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(num_features)]
        )
        for function in self.shape_functions:
            nn.init.zeros_(function[-1].weight)
            nn.init.zeros_(function[-1].bias)
        initial_logit = math.log(initial_weight / (1.0 - initial_weight))
        self.horizon_logits = nn.Parameter(torch.full((horizon,), initial_logit))
        self.horizon_bias = nn.Parameter(torch.zeros(horizon))

    def _risk_state(self, history: torch.Tensor, base_prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 4 or base_prediction.ndim != 4:
            raise ValueError("history and base_prediction must be [B,T/H,N,C]")
        batch, time, nodes, channels = history.shape
        if time != self.context_length or channels != self.channels:
            raise ValueError("history does not match corrector configuration")
        if base_prediction.shape != (batch, self.horizon, nodes, channels):
            raise ValueError("base prediction does not match corrector configuration")
        node_history = history.permute(0, 2, 1, 3)
        mean = node_history.mean(dim=2, keepdim=True)
        std = node_history.std(dim=2, keepdim=True, unbiased=False).clamp_min(1.0e-3)
        normalized_history = (node_history - mean) / std
        node_base = base_prediction.permute(0, 2, 1, 3)
        risk_input = torch.cat((normalized_history.reshape(batch, nodes, -1), node_base.reshape(batch, nodes, -1)), dim=-1)
        return self.risk_encoder(risk_input), node_base

    def predict_risk(self, history: torch.Tensor, base_prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state, _ = self._risk_state(history, base_prediction)
        risk = torch.nn.functional.softplus(self.risk_output(state))
        return risk.permute(0, 2, 1).unsqueeze(-1).contiguous(), state

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        memory: torch.Tensor,
        features: torch.Tensor,
        memory_valid: torch.Tensor,
        risk_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if base.shape != memory.shape or features.shape != base.shape[:-1] + (self.num_features,):
            raise ValueError("base, memory, and features have incompatible shapes")
        if memory_valid.shape != base.shape[:-1] + (1,):
            raise ValueError("memory_valid must be [B,H,N,1]")
        if risk_state is None:
            risk_state, _ = self._risk_state(history, base)
        elif risk_state.shape != (base.shape[0], base.shape[2], self.risk_repr_dim):
            raise ValueError("risk_state has an invalid shape")
        evidence = self.evidence_encoder(features)
        risk_state = risk_state[:, None, :, :].expand(-1, self.horizon, -1, -1)
        joint = self.joint_encoder(torch.cat((risk_state, evidence), dim=-1))
        gated = joint * self.gate(joint)
        contributions = torch.cat(
            [function(features[..., index : index + 1]) for index, function in enumerate(self.shape_functions)], dim=-1
        )
        logit = (
            self.horizon_bias.view(1, -1, 1, 1)
            + self.output(gated)
            + contributions.sum(dim=-1, keepdim=True)
        )
        horizon_limit = torch.sigmoid(self.horizon_logits).view(1, -1, 1, 1)
        if self.correction_variant == "vector_residual":
            residual = 0.5 * torch.tanh(self.output(gated))
            residual = torch.where(memory_valid, residual, torch.zeros_like(residual))
            return base + residual, residual, contributions
        weight = horizon_limit * torch.sigmoid(logit)
        if self.correction_variant == "residual_additive":
            additive = 0.25 * torch.tanh(self.output(gated))
            final = base + weight * (memory - base) + additive
            final = torch.where(memory_valid, final, base)
            return final, weight, contributions
        weight = torch.where(memory_valid, weight, torch.zeros_like(weight))
        return base + weight * (memory - base), weight, contributions



class LegacyCandidateSetHorizonCorrector(nn.Module):
    """Two-module correction head: candidate set attention plus horizon mixer."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channels: int,
        hidden_dim: int = 64,
        state_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or state_dim <= 0:
            raise ValueError("corrector dimensions must be positive")
        self.context_length = context_length
        self.horizon = horizon
        self.channels = channels
        self.num_features = 9
        self.state_encoder = nn.Sequential(
            nn.Linear((context_length + horizon) * channels, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(channels * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_proj = nn.Linear(state_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.horizon_dw = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.horizon_pw = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)
        self.risk_probe = nn.Linear(state_dim, horizon)
        self.current_attention = None
        self.last_attention = None

    def _state(self, history, base):
        if history.ndim != 4 or base.ndim != 4:
            raise ValueError("history and base must be [B,T/H,N,C]")
        b, t, n, c = history.shape
        if t != self.context_length or c != self.channels:
            raise ValueError("history does not match corrector configuration")
        if base.shape != (b, self.horizon, n, c):
            raise ValueError("base does not match corrector configuration")
        node_h = history.permute(0, 2, 1, 3)
        finite = torch.isfinite(node_h)
        count = finite.sum(2, keepdim=True).clamp_min(1)
        safe = torch.where(finite, node_h, torch.zeros_like(node_h))
        mean = safe.sum(2, keepdim=True) / count.to(node_h.dtype)
        centered = torch.where(finite, node_h - mean, torch.zeros_like(node_h))
        variance = centered.square().sum(2, keepdim=True) / count.to(node_h.dtype)
        normalized = centered / (variance + 1.0e-6).sqrt()
        base_state = torch.nan_to_num(base.permute(0, 2, 1, 3))
        state_input = torch.cat((normalized, base_state), dim=2).reshape(b, n, -1)
        return self.state_encoder(state_input)

    def predict_risk(self, history, base):
        state = self._state(history, base)
        risk = torch.nn.functional.softplus(self.risk_probe(state))
        return risk.permute(0, 2, 1).unsqueeze(-1).contiguous(), state

    def forward(
        self,
        history,
        base,
        memory,
        features,
        memory_valid,
        risk_state=None,
        candidates=None,
        aggregation=None,
    ):
        if candidates is None or aggregation is None:
            raise ValueError("CandidateSetHorizonCorrector requires candidates and aggregation")
        if aggregation.candidate_futures.ndim != 5:
            raise ValueError("candidate futures must be [B,H,N,K,C]")
        cand = torch.nan_to_num(aggregation.candidate_futures)
        mask = aggregation.candidate_masks.bool()
        valid = mask.any(dim=-1)
        b, h, n, k, c = cand.shape
        if base.shape != (b, h, n, c):
            raise ValueError("base and candidate futures do not align")
        if risk_state is None:
            risk_state = self._state(history, base)
        delta = cand - base.unsqueeze(3)
        abs_delta = delta.abs()
        sim = torch.nan_to_num(
            candidates.shape_scores[:, None, :, :, None].expand(b, h, n, k, 1)
        )
        level = torch.nan_to_num(
            (-candidates.level_distances[:, None, :, :, None]).expand(b, h, n, k, 1)
        )
        pos = torch.linspace(0, 1, h, device=base.device, dtype=base.dtype)
        pos = pos.view(1, h, 1, 1, 1).expand(b, h, n, k, 1)
        token = self.candidate_encoder(
            torch.cat((delta, abs_delta, sim, level, pos), dim=-1)
        )
        token = torch.where(valid.unsqueeze(-1), token, torch.zeros_like(token))
        query = self.query_proj(risk_state)[:, None, :, :].expand(b, h, n, -1)
        logits = (
            self.key_proj(token) * query.unsqueeze(3)
        ).sum(-1) / (token.shape[-1] ** 0.5)
        logits = logits.masked_fill(~valid, -1.0e9)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * valid.to(attention.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1.0e-8)
        self.current_attention = attention
        self.last_attention = attention.detach()

        pooled = (attention.unsqueeze(-1) * token).sum(3)
        flat = pooled.permute(0, 2, 3, 1).reshape(b * n, token.shape[-1], h)
        mixed = self.horizon_dw(flat).transpose(1, 2)
        mixed = self.horizon_pw(mixed)
        mixed = mixed.reshape(b, n, h, token.shape[-1]).permute(0, 2, 1, 3)
        pooled = pooled + mixed

        alpha = torch.sigmoid(self.alpha_head(pooled))
        residual = (attention.unsqueeze(-1) * delta).sum(3)
        residual_mean = residual.unsqueeze(3)
        residual_variance = (
            attention.unsqueeze(-1) * (delta - residual_mean).square()
        ).sum(3)
        dispersion = (residual_variance + 1.0e-8).sqrt().mean(-1, keepdim=True)
        beta = 0.25 * dispersion * torch.tanh(self.beta_head(pooled))
        final = base + alpha * residual + beta
        valid_output = aggregation.valid.all(-1, keepdim=True)
        final = torch.where(valid_output, final, base)
        learned_memory = base + residual
        contributions = torch.cat(
            (residual.abs().mean(-1, keepdim=True), dispersion), dim=-1
        )
        return final, alpha, contributions, learned_memory


class CandidateSetHorizonCorrector(nn.Module):
    """Base-as-candidate residual mixture (K historical + one Base token).

    The shared token refiner adds one bounded nonlinear interaction block for
    both historical and Base tokens before the unified attention decision.
    """

    def __init__(
        self,
        context_length,
        horizon,
        channels,
        hidden_dim=224,
        state_dim=160,
        attention_heads=4,
        base_logit_init_bias=1.0,
        trajectory_hidden_dim=0,
        use_horizon_embedding=False,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.context_length, self.horizon, self.channels = context_length, horizon, channels
        self.hidden_dim = hidden_dim
        self.trajectory_hidden_dim = trajectory_hidden_dim
        self.use_horizon_embedding = use_horizon_embedding
        self.state_encoder = nn.Sequential(
            nn.Linear((context_length + horizon) * channels, state_dim), nn.GELU(),
            nn.Linear(state_dim, state_dim), nn.GELU())
        self.candidate_encoder = nn.Sequential(
            nn.Linear(channels * 2 + 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.base_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.query_proj = nn.Linear(state_dim, hidden_dim)
        self.base_type = nn.Parameter(torch.zeros(hidden_dim))
        self.base_bias = nn.Parameter(torch.tensor(float(base_logit_init_bias)))
        self.token_refiner = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Start as an exact identity refinement so the wider model does not
        # perturb the established Base fallback before receiving gradients.
        nn.init.zeros_(self.token_refiner[1].weight)
        nn.init.zeros_(self.token_refiner[1].bias)
        self.risk_probe = nn.Linear(state_dim, horizon)
        self.current_attention = None
        self.last_attention = None

    def _state(self, history, base):
        b, t, n, c = history.shape
        finite = torch.isfinite(history)
        count = finite.sum(1, keepdim=True).clamp_min(1)
        safe = torch.where(finite, history, torch.zeros_like(history))
        mean = safe.sum(1, keepdim=True) / count.to(history.dtype)
        centered = torch.where(finite, history - mean, torch.zeros_like(history))
        var = centered.square().sum(1, keepdim=True) / count.to(history.dtype)
        norm = centered / (var + 1e-6).sqrt()
        safe_base = torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        inp = torch.cat((norm, safe_base), dim=1).permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.state_encoder(inp)

    def predict_risk(self, history, base):
        state = self._state(history, base)
        return torch.nn.functional.softplus(self.risk_probe(state)).permute(0, 2, 1).unsqueeze(-1), state

    def forward(self, history, base, memory, features, memory_valid, risk_state=None,
                candidates=None, aggregation=None):
        if candidates is None or aggregation is None:
            raise ValueError("CandidateSetHorizonCorrector requires candidates and aggregation")
        cand = torch.nan_to_num(
            aggregation.candidate_futures, nan=0.0, posinf=0.0, neginf=0.0
        )
        masks = aggregation.candidate_masks.bool()
        valid = masks.any(-1)
        b, h, n, k, c = cand.shape
        if risk_state is None:
            risk_state = self._state(history, base)
        delta = cand - base.unsqueeze(3)
        sim = torch.nan_to_num(
            candidates.shape_scores[:, None, :, :, None].expand(b, h, n, k, 1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        level = torch.nan_to_num(
            (-candidates.level_distances)[:, None, :, :, None].expand(b, h, n, k, 1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        pos = torch.linspace(0, 1, h, device=base.device, dtype=base.dtype).view(1, h, 1, 1, 1).expand(b, h, n, k, 1)
        cand_tok = self.candidate_encoder(torch.cat((delta, delta.abs(), sim, level, pos), -1))
        cand_tok = torch.where(valid.unsqueeze(-1), cand_tok, torch.zeros_like(cand_tok))
        # Compute context volatility from observed values only.  Replacing an
        # Inf with torch.nan_to_num's default maximum would overflow this
        # statistic before the candidate mask can remove the affected token.
        history_finite = torch.isfinite(history)
        history_safe = torch.where(history_finite, history, torch.zeros_like(history))
        history_count = history_finite.sum(1).clamp_min(1).to(history.dtype)
        history_mean = history_safe.sum(1) / history_count
        history_centered = (
            history_safe - history_mean.unsqueeze(1)
        ) * history_finite.to(history.dtype)
        ctx_std = (
            history_centered.square().sum(1) / history_count
        ).clamp_min(0.0).sqrt().mean(-1, keepdim=True)
        base_risk = torch.nn.functional.softplus(self.risk_probe(risk_state)).permute(0, 2, 1).unsqueeze(-1)
        # All Base-token scalar features are [B,H,N,1]; encode one token per
        # horizon/node, yielding [B,H,N,D] before appending the K historical
        # candidate tokens.
        ctx_std = ctx_std[:, None, :, :].expand(-1, h, -1, -1)
        pos_scalar = torch.linspace(0, 1, h, device=base.device, dtype=base.dtype).view(1, h, 1, 1).expand(b, h, n, 1)
        base_type_scalar = torch.ones_like(pos_scalar)
        base_feat = torch.cat((base_risk, ctx_std, pos_scalar, base_type_scalar), dim=-1)
        base_tok = self.base_encoder(base_feat) + self.base_type
        query = self.query_proj(risk_state)[:, None, :, :].expand(b, h, n, -1)
        all_tok = torch.cat((cand_tok, base_tok.unsqueeze(3)), 3)
        all_tok = all_tok + self.token_refiner(all_tok)
        logits = (all_tok * query.unsqueeze(3)).sum(-1) / (self.hidden_dim ** 0.5)
        logits[..., -1] = logits[..., -1] + self.base_bias
        all_valid = torch.cat((valid, torch.ones(b, h, n, 1, dtype=torch.bool, device=base.device)), -1)
        attn = torch.softmax(logits.masked_fill(~all_valid, -1e9), -1)
        self.current_attention, self.last_attention = attn, attn.detach()
        hist_attn = attn[..., :k]
        residual = (hist_attn.unsqueeze(-1) * delta).sum(3)
        final = base + residual
        # Base is the explicit fallback token; only fall back when the whole
        # historical candidate set is invalid for a node/horizon.
        final = torch.where(valid.any(-1, keepdim=True), final, base)
        historical_mass = attn[..., :k].sum(-1, keepdim=True)
        contributions = torch.cat((residual.abs().mean(-1, keepdim=True), (hist_attn.unsqueeze(-1) * (delta - residual.unsqueeze(3)).square()).sum(3).sqrt().mean(-1, keepdim=True)), -1)
        learned_memory = base + residual
        return final, historical_mass, contributions, learned_memory


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
    candidate_attention: torch.Tensor | None = None
    candidate_futures: torch.Tensor | None = None
    candidate_masks: torch.Tensor | None = None
    routing_weights: torch.Tensor | None = None
    mha_attention_weights: torch.Tensor | None = None
    base_usage: torch.Tensor | None = None
    routing_entropy: torch.Tensor | None = None


class STAnchorDownstreamModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        confidence_head: ConfidenceHead,
        fusion: SafeResidualFusion,
        confidence_level_temperature: float,
        mode: str = LEARNED_TOPK_CONFIDENCE,
        error_corrector: StructuredErrorCorrector | None = None,
        horizon_aggregator: HorizonAwareAggregationHead | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.confidence_head = confidence_head
        self.fusion = fusion
        self.confidence_level_temperature = confidence_level_temperature
        self.mode = validate_downstream_mode(mode)
        self.error_corrector = error_corrector
        self.horizon_aggregator = horizon_aggregator
        if self.mode == LEARNED_TOPK_ERROR_AWARE and error_corrector is None:
            raise ValueError("error-aware mode requires StructuredErrorCorrector")

    def train(self, mode: bool = True) -> "STAnchorDownstreamModel":
        super().train(mode)
        if mode:
            for module in (
                self.backbone,
                self.confidence_head,
                self.fusion,
                self.error_corrector,
                self.horizon_aggregator,
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
        base_override: torch.Tensor | None = None,
        retrieval_node_keys: torch.Tensor | None = None,
    ) -> DownstreamOutput:
        base = self.backbone(x) if base_override is None else base_override
        if base.shape[0] != x.shape[0] or base.shape[2] != x.shape[2]:
            raise ValueError("base_override must match batch and node dimensions")
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
                aggregation.prediction,
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
            if self.error_corrector is None:
                raise RuntimeError("error-aware corrector is not initialized")
            predicted_risk, risk_state = self.error_corrector.predict_risk(x, base)
            is_candidate_router = (
                isinstance(self.error_corrector, CandidateSetHorizonCorrector)
                or bool(getattr(self.error_corrector, "uses_candidate_routing", False))
            )
            if is_candidate_router:
                memory_valid = aggregation.valid.all(dim=-1, keepdim=True)
                router_kwargs = dict(
                    risk_state=risk_state,
                    candidates=candidates,
                    aggregation=aggregation,
                )
                if getattr(self.error_corrector, "uses_retrieval_node_keys", False):
                    router_kwargs["retrieval_node_keys"] = retrieval_node_keys
                final, fusion_weight, contributions, learned_memory = self.error_corrector(
                    x, base, aggregation.prediction, None, memory_valid,
                    **router_kwargs,
                )
                features = torch.zeros((*base.shape[:-1], 9), dtype=base.dtype, device=base.device)
                memory_prediction = learned_memory
                candidate_attention = self.error_corrector.current_attention
                candidate_futures = aggregation.candidate_futures
                candidate_masks = aggregation.candidate_masks
            else:
                if self.horizon_aggregator is not None:
                    aggregation = self.horizon_aggregator(candidates, aggregation, base)
                features, memory_valid = build_error_aware_features(
                    candidates, aggregation, base, predicted_risk,
                    self.confidence_level_temperature,
                )
                final, fusion_weight, contributions = self.error_corrector(
                    x, base, aggregation.prediction, features, memory_valid,
                    risk_state=risk_state,
                )
                memory_prediction = aggregation.prediction
            return DownstreamOutput(
                base_prediction=base,
                memory_prediction=memory_prediction,
                confidence_features=features,
                confidence=fusion_weight,
                fusion_weight=fusion_weight,
                final_prediction=final,
                memory_valid=memory_valid,
                predicted_base_risk=predicted_risk,
                additive_contributions=contributions,
                candidate_attention=(candidate_attention if is_candidate_router else None),
                candidate_futures=(candidate_futures if is_candidate_router else None),
                candidate_masks=(candidate_masks if is_candidate_router else None),
                routing_weights=(getattr(self.error_corrector, "last_routing_weights", None) if is_candidate_router else None),
                mha_attention_weights=(getattr(self.error_corrector, "last_mha_attention", None) if is_candidate_router else None),
                base_usage=(getattr(self.error_corrector, "last_base_usage", None) if is_candidate_router else None),
                routing_entropy=(getattr(self.error_corrector, "last_routing_entropy", None) if is_candidate_router else None),
            )
        features, memory_valid = build_confidence_features(
            candidates,
            aggregation,
            base,
            self.confidence_level_temperature,
        )
        confidence = self.confidence_head(features, memory_valid)
        final, fusion_weight = self.fusion(
            base,
            aggregation.prediction,
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











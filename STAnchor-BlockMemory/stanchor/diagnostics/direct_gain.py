"""Deployment-available candidate features for direct gain policies."""

from __future__ import annotations

import torch

from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


def build_direct_gain_features(
    base_features: torch.Tensor,
    candidates: NodeCandidates,
    aggregation: AggregationOutput,
    base_prediction: torch.Tensor,
) -> torch.Tensor:
    """Append horizon-specific candidate correction statistics.

    Inputs use model space and contain no query future. Shapes are
    ``base_features [B,H,N,F]``, candidate futures ``[B,H,N,K,C]`` and
    candidate weights ``[B,N,K]``. The five appended features are weighted
    correction mean, correction standard deviation, signed direction
    agreement, positive-direction mass, and negative-direction mass.
    """
    if base_features.ndim != 4 or base_prediction.ndim != 4:
        raise ValueError("base features and prediction must be [B,H,N,F/C]")
    if aggregation.candidate_futures.shape[:3] != base_prediction.shape[:3]:
        raise ValueError("candidate futures do not align with base prediction")
    if candidates.weights.shape != candidates.valid.shape:
        raise ValueError("candidate weights and validity must align")
    valid = candidates.valid
    node_weights = torch.where(valid, candidates.weights, torch.zeros_like(candidates.weights))
    node_weights = node_weights / node_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    weights = node_weights[:, None, :, :, None]
    mask = aggregation.candidate_masks.bool()
    effective = weights * mask.to(base_prediction.dtype)
    denominator = effective.sum(dim=3).clamp_min(1.0e-8)
    delta = aggregation.candidate_futures - base_prediction.unsqueeze(3)
    mean = (effective * delta).sum(dim=3) / denominator
    variance = (effective * (delta - mean.unsqueeze(3)).square()).sum(dim=3) / denominator
    signed = (effective * torch.sign(delta)).sum(dim=3) / denominator
    positive = (effective * (delta > 0).to(delta.dtype)).sum(dim=3) / denominator
    negative = (effective * (delta < 0).to(delta.dtype)).sum(dim=3) / denominator
    appended = torch.stack(
        (
            mean.mean(dim=-1),
            variance.clamp_min(0.0).sqrt().mean(dim=-1),
            signed.mean(dim=-1),
            positive.mean(dim=-1),
            negative.mean(dim=-1),
        ),
        dim=-1,
    )
    memory_valid = aggregation.valid.all(dim=-1, keepdim=True)
    appended = torch.where(memory_valid.expand_as(appended), appended, torch.zeros_like(appended))
    return torch.cat((base_features, appended), dim=-1)

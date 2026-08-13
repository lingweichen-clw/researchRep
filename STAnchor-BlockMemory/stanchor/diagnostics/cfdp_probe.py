"""Frozen-encoder probes for attributing weak CFDP profile semantics.

The probes consume clean encoder representations only.  They are diagnostic
heads and never change the deployed retrieval checkpoint or Bank schema.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.stats import spearmanr
from torch import nn


class SharedPooledLinearProbe(nn.Module):
    """Original shared temporal pooling followed by a linear profile head."""

    def __init__(self, hidden_dim: int, profile_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_dim, profile_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 3:
            raise ValueError("pooled must be [B,N,D]")
        return self.head(pooled)


class SharedPooledMLPProbe(nn.Module):
    """Shared pooling with a small nonlinear profile decoder."""

    def __init__(self, hidden_dim: int, profile_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, profile_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 3:
            raise ValueError("pooled must be [B,N,D]")
        return self.head(pooled)


class HorizonSpecificPoolingProbe(nn.Module):
    """Use a separate temporal attention query for each profile position."""

    def __init__(self, hidden_dim: int, profile_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or profile_dim <= 0:
            raise ValueError("hidden_dim and profile_dim must be positive")
        self.token_projection = nn.Linear(hidden_dim, hidden_dim)
        self.horizon_queries = nn.Parameter(torch.empty(profile_dim, hidden_dim))
        self.output_weight = nn.Parameter(torch.empty(profile_dim, hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(profile_dim))
        nn.init.xavier_uniform_(self.horizon_queries)
        nn.init.xavier_uniform_(self.output_weight)

    def forward(
        self,
        hidden: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hidden.ndim != 4:
            raise ValueError("hidden must be [B,P,N,D]")
        batch, patches, nodes, dimension = hidden.shape
        if dimension != self.token_projection.in_features:
            raise ValueError("hidden feature dimension does not match probe")
        projected = torch.tanh(self.token_projection(hidden))
        # scores: [B,K,P,N], one independent temporal distribution per horizon.
        scores = torch.einsum("b p n d,k d->b k p n", projected, self.horizon_queries)
        weights = torch.softmax(scores, dim=2)
        horizon_hidden = torch.einsum("b k p n,b p n d->b k n d", weights, hidden)
        prediction = torch.einsum(
            "b k n d,k d->b n k", horizon_hidden, self.output_weight
        ) + self.output_bias.view(1, 1, -1)
        if return_weights:
            return prediction, weights
        return prediction


def masked_profile_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mask-aware SmoothL1 loss for tensors shaped ``[B,N,K]``."""
    if prediction.shape != teacher.shape or valid.shape != teacher.shape:
        raise ValueError("profile tensors must share [B,N,K]")
    mask = valid.bool() & torch.isfinite(prediction) & torch.isfinite(teacher)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    values = functional.smooth_l1_loss(prediction, teacher, reduction="none")
    return values.masked_select(mask).mean()


def _pairwise_profile_mae(
    profile: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_valid = valid[:, None] & valid[None, :]
    difference = (profile[:, None] - profile[None, :]).abs()
    count = pair_valid.sum(dim=-1)
    distance = torch.where(pair_valid, difference, torch.zeros_like(difference)).sum(dim=-1)
    return distance / count.clamp_min(1), count > 0


def _relation_metrics(
    key_distance: torch.Tensor,
    teacher_distance: torch.Tensor,
    candidate_mask: torch.Tensor,
    top_k: int,
) -> tuple[float, float, int, int]:
    valid = candidate_mask.bool() & torch.isfinite(key_distance) & torch.isfinite(teacher_distance)
    key_values = key_distance.masked_select(valid).detach().double().cpu().numpy()
    teacher_values = teacher_distance.masked_select(valid).detach().double().cpu().numpy()
    if key_values.size < 2 or np.ptp(key_values) == 0.0 or np.ptp(teacher_values) == 0.0:
        spearman = 0.0
    else:
        spearman = float(spearmanr(key_values, teacher_values).statistic)
    recalls: list[float] = []
    batch, _, nodes = candidate_mask.shape
    for anchor in range(batch):
        for node in range(nodes):
            candidates = torch.where(valid[anchor, :, node])[0]
            if candidates.numel() < top_k:
                continue
            key_ids = candidates[torch.topk(key_distance[anchor, candidates, node], top_k, largest=False).indices]
            teacher_ids = candidates[torch.topk(teacher_distance[anchor, candidates, node], top_k, largest=False).indices]
            recalls.append(float(torch.isin(key_ids, teacher_ids).sum()) / top_k)
    recall = float(np.mean(recalls)) if recalls else 0.0
    return spearman, recall, int(valid.sum().item()), len(recalls)


def profile_relation_metrics(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    od_distance: torch.Tensor,
    candidate_mask: torch.Tensor,
    top_k: int = 5,
) -> dict[str, float | int]:
    """Evaluate profile fit and profile-key geometry for one validation batch."""
    if prediction.ndim != 3 or teacher.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError("profile tensors must be [B,N,K]")
    if od_distance.ndim != 3 or candidate_mask.shape != od_distance.shape:
        raise ValueError("relation tensors must be [B,B,N]")
    point_valid = valid.bool() & torch.isfinite(prediction) & torch.isfinite(teacher)
    profile_mae = float((prediction - teacher).abs().masked_select(point_valid).mean()) if bool(point_valid.any()) else 0.0
    safe_prediction = torch.where(point_valid, prediction, torch.zeros_like(prediction))
    safe_teacher = torch.where(point_valid, teacher, torch.zeros_like(teacher))
    vector_valid = point_valid.any(dim=-1) & (safe_prediction.norm(dim=-1) > 1.0e-8) & (safe_teacher.norm(dim=-1) > 1.0e-8)
    cosine_values = functional.cosine_similarity(safe_prediction, safe_teacher, dim=-1, eps=1.0e-8)
    profile_cosine = float(cosine_values.masked_select(vector_valid).mean()) if bool(vector_valid.any()) else 0.0
    profile_teacher_distance, profile_pair_valid = _pairwise_profile_mae(teacher, valid)
    relation_mask = candidate_mask.bool() & profile_pair_valid
    normalized_prediction = functional.normalize(safe_prediction, dim=-1)
    profile_key_distance = 1.0 - torch.einsum("bnd,jnd->bjn", normalized_prediction, normalized_prediction)
    profile_spearman, profile_recall, profile_pairs, profile_anchors = _relation_metrics(
        profile_key_distance, profile_teacher_distance, relation_mask, top_k
    )
    od_spearman, od_recall, od_pairs, od_anchors = _relation_metrics(
        profile_key_distance, od_distance, candidate_mask, top_k
    )
    return {
        "profile_mae": profile_mae,
        "profile_cosine": profile_cosine,
        "profile_mae_relation_spearman": profile_spearman,
        "profile_mae_relation_recall_at_k": profile_recall,
        "od_relation_spearman": od_spearman,
        "od_relation_recall_at_k": od_recall,
        "profile_points": int(point_valid.sum().item()),
        "profile_vectors": int(vector_valid.sum().item()),
        "profile_relation_pairs": profile_pairs,
        "profile_relation_anchors": profile_anchors,
        "od_relation_pairs": od_pairs,
        "od_relation_anchors": od_anchors,
    }


def weighted_metric_average(records: list[dict[str, float | int]]) -> dict[str, float]:
    """Average batch metrics using the same valid-point/pair weighting as CFDP diagnostics."""
    weights = {
        "profile_mae": "profile_points",
        "profile_cosine": "profile_vectors",
        "profile_mae_relation_spearman": "profile_relation_pairs",
        "profile_mae_relation_recall_at_k": "profile_relation_anchors",
        "od_relation_spearman": "od_relation_pairs",
        "od_relation_recall_at_k": "od_relation_anchors",
    }
    output: dict[str, float] = {}
    for metric, weight_name in weights.items():
        denominator = sum(float(record[weight_name]) for record in records)
        output[metric] = (
            float(sum(float(record[metric]) * float(record[weight_name]) for record in records) / denominator)
            if denominator > 0
            else 0.0
        )
    return output


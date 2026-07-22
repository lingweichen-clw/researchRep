"""Masked reconstruction and future-guided retrieval objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from stanchor.data.normalization import normalize_future_with_context
from stanchor.models.pretraining import PretrainForwardOutput


@dataclass(frozen=True)
class RetrievalLossOutput:
    loss: torch.Tensor
    valid_anchors: int
    positive_pairs: int
    hard_negative_pairs: int


@dataclass(frozen=True)
class PretrainingLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    retrieval: torch.Tensor
    valid_retrieval_anchors: int
    positive_pairs: int
    hard_negative_pairs: int
    reconstruction_positions: int


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    artificial_mask: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or artificial_mask.shape != target.shape or observed.shape != target.shape:
        raise ValueError("all reconstruction tensors must have shape [B, T, N, C]")
    loss_mask = artificial_mask.bool() & observed.bool()
    count = loss_mask.sum()
    if int(count) == 0:
        return prediction.sum() * 0.0
    element_loss = functional.smooth_l1_loss(prediction, target, reduction="none")
    return element_loss.masked_select(loss_mask).mean()


def _masked_logsumexp(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(values.masked_fill(~mask, -torch.inf), dim=dim)


def _pairwise_masked_mae(
    values: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape != observed.shape or values.ndim != 4:
        raise ValueError("pairwise values and observed must be [B, T/H, N, C]")
    pair_observed = observed[:, None].bool() & observed[None, :].bool()
    absolute = (values[:, None] - values[None, :]).abs()
    count = pair_observed.sum(dim=(2, 4))  # [B, B, N]
    distance = torch.where(pair_observed, absolute, torch.zeros_like(absolute)).sum(dim=(2, 4))
    return distance / count.clamp_min(1), count > 0


def future_guided_retrieval_loss(
    node_keys: torch.Tensor,
    context_normalized: torch.Tensor,
    future_model: torch.Tensor,
    context_statistics,
    context_observed: torch.Tensor,
    future_observed: torch.Tensor,
    context_start: torch.Tensor,
    future_end: torch.Tensor,
    temperature: float = 0.1,
    positive_quantile: float = 0.1,
    context_quantile: float = 0.2,
    negative_quantile: float = 0.8,
    hard_negative_weight: float = 2.0,
) -> RetrievalLossOutput:
    """Build within-batch future positives and mirage hard negatives."""
    if node_keys.ndim != 3 or context_normalized.ndim != 4 or future_model.ndim != 4:
        raise ValueError("invalid retrieval loss tensor ranks")
    batch, nodes, _ = node_keys.shape
    if context_normalized.shape[0] != batch or context_normalized.shape[2] != nodes:
        raise ValueError("context and keys do not align")
    if future_model.shape[0] != batch or future_model.shape[2] != nodes:
        raise ValueError("future and keys do not align")
    if context_start.shape != (batch,) or future_end.shape != (batch,):
        raise ValueError("context_start and future_end must be [B]")
    future_signature = normalize_future_with_context(future_model, context_statistics)
    context_distance, context_pair_valid = _pairwise_masked_mae(
        context_normalized, context_observed
    )
    future_distance, future_pair_valid = _pairwise_masked_mae(
        future_signature, future_observed
    )

    non_overlap = (future_end[:, None] < context_start[None, :]) | (
        future_end[None, :] < context_start[:, None]
    )
    non_self = ~torch.eye(batch, dtype=torch.bool, device=node_keys.device)
    valid_pair = non_overlap & non_self
    valid_3d = (
        valid_pair.unsqueeze(-1).expand(-1, -1, nodes)
        & context_pair_valid
        & future_pair_valid
    )
    nan = torch.tensor(float("nan"), device=node_keys.device, dtype=context_distance.dtype)
    context_for_quantile = torch.where(valid_3d, context_distance, nan)
    future_for_quantile = torch.where(valid_3d, future_distance, nan)
    context_threshold = torch.nanquantile(context_for_quantile, context_quantile, dim=1)
    positive_threshold = torch.nanquantile(future_for_quantile, positive_quantile, dim=1)
    negative_threshold = torch.nanquantile(future_for_quantile, negative_quantile, dim=1)
    positive = valid_3d & (future_distance <= positive_threshold[:, None, :])
    hard_negative = (
        valid_3d
        & ~positive
        & (context_distance <= context_threshold[:, None, :])
        & (future_distance >= negative_threshold[:, None, :])
    )

    logits = torch.einsum("ind,jnd->ijn", node_keys, node_keys) / temperature
    denominator_logits = logits + hard_negative.to(logits.dtype) * torch.log(
        torch.tensor(hard_negative_weight, dtype=logits.dtype, device=logits.device)
    )
    numerator = _masked_logsumexp(logits, positive, dim=1)
    denominator = _masked_logsumexp(denominator_logits, valid_3d, dim=1)
    valid_anchor = positive.any(dim=1) & torch.isfinite(numerator) & torch.isfinite(denominator)
    if bool(valid_anchor.any()):
        loss = (denominator - numerator).masked_select(valid_anchor).mean()
    else:
        loss = node_keys.sum() * 0.0
    return RetrievalLossOutput(
        loss=loss,
        valid_anchors=int(valid_anchor.sum().item()),
        positive_pairs=int(positive.sum().item()),
        hard_negative_pairs=int(hard_negative.sum().item()),
    )


def compute_pretraining_loss(
    output: PretrainForwardOutput,
    future_model: torch.Tensor,
    observed_context: torch.Tensor,
    observed_future: torch.Tensor,
    context_start: torch.Tensor,
    future_end: torch.Tensor,
    retrieval_weight: float,
    retrieval_temperature: float,
    positive_quantile: float,
    context_quantile: float,
    negative_quantile: float,
    hard_negative_weight: float,
) -> PretrainingLoss:
    reconstruction = masked_reconstruction_loss(
        output.reconstruction,
        output.reconstruction_target,
        output.mask.value_mask,
        observed_context,
    )
    retrieval_output = future_guided_retrieval_loss(
        node_keys=output.clean.retrieval.node_keys,
        context_normalized=output.clean.statistics.normalized,
        future_model=future_model,
        context_statistics=output.clean.statistics,
        context_observed=observed_context,
        future_observed=observed_future,
        context_start=context_start,
        future_end=future_end,
        temperature=retrieval_temperature,
        positive_quantile=positive_quantile,
        context_quantile=context_quantile,
        negative_quantile=negative_quantile,
        hard_negative_weight=hard_negative_weight,
    )
    total = reconstruction + retrieval_weight * retrieval_output.loss
    reconstruction_positions = int(
        (output.mask.value_mask.bool() & observed_context.bool()).sum().item()
    )
    return PretrainingLoss(
        total=total,
        reconstruction=reconstruction,
        retrieval=retrieval_output.loss,
        valid_retrieval_anchors=retrieval_output.valid_anchors,
        positive_pairs=retrieval_output.positive_pairs,
        hard_negative_pairs=retrieval_output.hard_negative_pairs,
        reconstruction_positions=reconstruction_positions,
    )

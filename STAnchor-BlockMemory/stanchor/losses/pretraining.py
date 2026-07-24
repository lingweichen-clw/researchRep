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
    candidate_pairs: int = 0
    teacher_effective_support: float = 0.0
    student_effective_support: float = 0.0


@dataclass(frozen=True)
class FutureRelationTargets:
    """Future-derived teacher relations for one source pretraining batch."""

    future_distance: torch.Tensor  # [B, B, N]
    candidate_mask: torch.Tensor  # [B, B, N]
    teacher_distribution: torch.Tensor  # [B, B, N]
    valid_anchors: torch.Tensor  # [B, N]


@dataclass(frozen=True)
class PretrainingLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    retrieval: torch.Tensor
    valid_retrieval_anchors: int
    positive_pairs: int
    hard_negative_pairs: int
    reconstruction_positions: int
    relation_candidate_pairs: int = 0
    teacher_effective_support: float = 0.0
    student_effective_support: float = 0.0


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


def _masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid_anchors: torch.Tensor,
) -> torch.Tensor:
    """Softmax over candidates while keeping empty anchors finite and zeroed."""
    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have the same shape")
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    safe_logits = torch.where(
        valid_anchors[:, None, :],
        masked_logits,
        torch.zeros_like(masked_logits),
    )
    distribution = torch.softmax(safe_logits, dim=1)
    return distribution * mask.to(distribution.dtype)


def build_future_relation_targets(
    future_model: torch.Tensor,
    context_statistics,
    future_observed: torch.Tensor,
    context_start: torch.Tensor,
    future_end: torch.Tensor,
    teacher_temperature: float = 0.1,
) -> FutureRelationTargets:
    """Construct future-distance teacher distributions without model gradients."""
    if future_model.ndim != 4:
        raise ValueError("future_model must be [B, H, N, C]")
    if future_observed.shape != future_model.shape:
        raise ValueError("future_observed must align with future_model")
    batch, _, nodes, _ = future_model.shape
    if context_start.shape != (batch,) or future_end.shape != (batch,):
        raise ValueError("context_start and future_end must be [B]")
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")

    with torch.no_grad():
        future_signature = normalize_future_with_context(future_model, context_statistics)
        observed = future_observed.bool() & torch.isfinite(future_signature)
        future_distance, future_pair_valid = _pairwise_masked_mae(
            future_signature,
            observed,
        )
        non_overlap = (future_end[:, None] < context_start[None, :]) | (
            future_end[None, :] < context_start[:, None]
        )
        non_self = ~torch.eye(batch, dtype=torch.bool, device=future_model.device)
        candidate_mask = (
            non_overlap.unsqueeze(-1).expand(-1, -1, nodes)
            & non_self.unsqueeze(-1).expand(-1, -1, nodes)
            & future_pair_valid
        )
        candidate_count = candidate_mask.sum(dim=1)
        valid_anchors = candidate_count >= 2
        teacher_logits = -future_distance / teacher_temperature
        teacher_distribution = _masked_softmax(
            teacher_logits,
            candidate_mask,
            valid_anchors,
        )
    return FutureRelationTargets(
        future_distance=future_distance,
        candidate_mask=candidate_mask,
        teacher_distribution=teacher_distribution,
        valid_anchors=valid_anchors,
    )


def future_relation_retrieval_loss(
    node_keys: torch.Tensor,
    future_model: torch.Tensor,
    context_statistics,
    future_observed: torch.Tensor,
    context_start: torch.Tensor,
    future_end: torch.Tensor,
    teacher_temperature: float = 0.1,
    student_temperature: float = 0.1,
) -> RetrievalLossOutput:
    """Match node-key similarity distributions to future-distance relations."""
    if node_keys.ndim != 3:
        raise ValueError("node_keys must be [B, N, D]")
    batch, nodes, _ = node_keys.shape
    if future_model.shape[0] != batch or future_model.shape[2] != nodes:
        raise ValueError("future and node keys do not align")
    if student_temperature <= 0:
        raise ValueError("student_temperature must be positive")
    targets = build_future_relation_targets(
        future_model=future_model,
        context_statistics=context_statistics,
        future_observed=future_observed,
        context_start=context_start,
        future_end=future_end,
        teacher_temperature=teacher_temperature,
    )
    normalized_keys = functional.normalize(node_keys, dim=-1)
    student_logits = torch.einsum("ind,jnd->ijn", normalized_keys, normalized_keys)
    student_logits = student_logits / student_temperature
    student_distribution = _masked_softmax(
        student_logits,
        targets.candidate_mask,
        targets.valid_anchors,
    )
    log_student = torch.log(student_distribution.clamp_min(1.0e-12))
    cross_entropy = -(targets.teacher_distribution * log_student).sum(dim=1)
    valid_anchors = targets.valid_anchors
    if bool(valid_anchors.any()):
        loss = cross_entropy.masked_select(valid_anchors).mean()
        teacher_keff = 1.0 / targets.teacher_distribution.pow(2).sum(dim=1).clamp_min(1.0e-12)
        student_keff = 1.0 / student_distribution.pow(2).sum(dim=1).clamp_min(1.0e-12)
        teacher_support = float(teacher_keff.masked_select(valid_anchors).mean().detach())
        student_support = float(student_keff.masked_select(valid_anchors).mean().detach())
    else:
        loss = node_keys.sum() * 0.0
        teacher_support = 0.0
        student_support = 0.0
    valid_candidates = targets.candidate_mask & valid_anchors[:, None, :]
    return RetrievalLossOutput(
        loss=loss,
        valid_anchors=int(valid_anchors.sum().item()),
        positive_pairs=0,
        hard_negative_pairs=0,
        candidate_pairs=int(valid_candidates.sum().item()),
        teacher_effective_support=teacher_support,
        student_effective_support=student_support,
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
    retrieval_loss_mode: str = "hard_negative",
    relation_teacher_temperature: float = 0.1,
    relation_student_temperature: float = 0.1,
) -> PretrainingLoss:
    reconstruction = masked_reconstruction_loss(
        output.reconstruction,
        output.reconstruction_target,
        output.mask.value_mask,
        observed_context,
    )
    if retrieval_loss_mode == "hard_negative":
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
    elif retrieval_loss_mode == "relation":
        retrieval_output = future_relation_retrieval_loss(
            node_keys=output.clean.retrieval.node_keys,
            future_model=future_model,
            context_statistics=output.clean.statistics,
            future_observed=observed_future,
            context_start=context_start,
            future_end=future_end,
            teacher_temperature=relation_teacher_temperature,
            student_temperature=relation_student_temperature,
        )
    else:
        raise ValueError(f"unknown retrieval_loss_mode: {retrieval_loss_mode}")
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
        relation_candidate_pairs=retrieval_output.candidate_pairs,
        teacher_effective_support=retrieval_output.teacher_effective_support,
        student_effective_support=retrieval_output.student_effective_support,
    )

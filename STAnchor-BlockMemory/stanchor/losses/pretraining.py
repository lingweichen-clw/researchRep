"""Masked reconstruction and future-guided retrieval objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from stanchor.data.normalization import normalize_future_with_context
from stanchor.retrieval.semantic_profile import symmetric_geometric_mean_normalize
from stanchor.retrieval.semantic_profile import build_cfdp_teacher
from stanchor.models.pretraining import CleanEncoding, PretrainForwardOutput


@dataclass(frozen=True)
class RetrievalLossOutput:
    loss: torch.Tensor
    valid_anchors: int
    positive_pairs: int
    hard_negative_pairs: int
    candidate_pairs: int = 0
    teacher_effective_support: float = 0.0
    student_effective_support: float = 0.0
    rank_loss: torch.Tensor | None = None
    rank_pairs: int = 0
    relation_loss: torch.Tensor | None = None


@dataclass(frozen=True)
class RankingLossOutput:
    loss: torch.Tensor
    valid_pairs: int


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
    profile: torch.Tensor | None = None
    rank: torch.Tensor | None = None
    rank_pairs: int = 0
    relation: torch.Tensor | None = None


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


def _endpoint_level_from_context(
    context: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return endpoint levels and validity as ``[B,N,C]``."""
    if context.ndim != 4 or observed.shape != context.shape:
        raise ValueError("context and observed must be [B, T, N, C]")
    if context.shape[1] <= 0:
        raise ValueError("context time dimension must be positive")
    visible = observed.bool() & torch.isfinite(context)
    count = visible.sum(dim=1)
    safe_count = count.clamp_min(1)
    mean = torch.where(visible, context, torch.zeros_like(context)).sum(dim=1) / safe_count
    endpoint_visible = visible[:, -1]
    endpoint = torch.where(endpoint_visible, context[:, -1], mean)
    valid = count > 0
    return torch.where(valid, endpoint, torch.zeros_like(endpoint)), valid


def build_offset_decay_signature(
    future_model: torch.Tensor,
    future_observed: torch.Tensor,
    forecast_context: torch.Tensor,
    context_observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``Y - lambda * endpoint`` and its mask as ``[B,H,N,C]``."""
    if future_model.ndim != 4 or future_observed.shape != future_model.shape:
        raise ValueError("future_model and future_observed must be [B, H, N, C]")
    if forecast_context.ndim != 4 or context_observed.shape != forecast_context.shape:
        raise ValueError("forecast_context and context_observed must be [B, T, N, C]")
    if future_model.shape[0] != forecast_context.shape[0] or future_model.shape[2:] != forecast_context.shape[2:]:
        raise ValueError("future and forecast context batch/node/channel dimensions must align")
    endpoint, endpoint_valid = _endpoint_level_from_context(
        forecast_context,
        context_observed,
    )
    horizon = future_model.shape[1]
    decay = torch.linspace(
        1.0,
        0.0,
        horizon,
        dtype=future_model.dtype,
        device=future_model.device,
    ).view(1, horizon, 1, 1)
    valid = (
        future_observed.bool()
        & torch.isfinite(future_model)
        & endpoint_valid.unsqueeze(1)
    )
    signature = future_model - decay * endpoint.unsqueeze(1)
    return torch.where(valid, signature, torch.zeros_like(signature)), valid


def build_future_increment(
    future_model: torch.Tensor,
    future_observed: torch.Tensor,
    endpoint_level: torch.Tensor,
    endpoint_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build endpoint-to-H1 and adjacent future increments as ``[B,H,N,C]``."""
    if future_model.ndim != 4 or future_observed.shape != future_model.shape:
        raise ValueError("future_model and future_observed must be [B, H, N, C]")
    expected = future_model.shape[:1] + future_model.shape[2:]
    if endpoint_level.shape != expected or endpoint_valid.shape != expected:
        raise ValueError("endpoint tensors must be [B, N, C]")
    future_valid = future_observed.bool() & torch.isfinite(future_model)
    first_valid = future_valid[:, :1] & endpoint_valid.unsqueeze(1)
    first = future_model[:, :1] - endpoint_level.unsqueeze(1)
    if future_model.shape[1] == 1:
        valid = first_valid
        increment = first
    else:
        adjacent_valid = future_valid[:, 1:] & future_valid[:, :-1]
        adjacent = future_model[:, 1:] - future_model[:, :-1]
        valid = torch.cat((first_valid, adjacent_valid), dim=1)
        increment = torch.cat((first, adjacent), dim=1)
    return torch.where(valid, increment, torch.zeros_like(increment)), valid


def anchor_mean_normalize_distances(
    distances: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize candidate distances per sample-node anchor without changing order."""
    if distances.ndim != 3 or valid.shape != distances.shape:
        raise ValueError("distances and valid must be [B, B, N]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    finite_valid = valid.bool() & torch.isfinite(distances)
    count = finite_valid.sum(dim=1)
    total = torch.where(finite_valid, distances, torch.zeros_like(distances)).sum(dim=1)
    scale = total / count.clamp_min(1)
    normalized = distances / scale.unsqueeze(1).clamp_min(eps)
    return torch.where(finite_valid, normalized, torch.zeros_like(normalized))


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
    relation_teacher_mode: str = "context_normalized",
    forecast_context: torch.Tensor | None = None,
    forecast_context_observed: torch.Tensor | None = None,
    relation_distance_normalization: str = "none",
    future_increment_weight: float = 0.0,
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
    if relation_teacher_mode not in {
        "context_normalized",
        "offset_decay",
        "offset_decay_increment",
    }:
        raise ValueError(f"unknown relation_teacher_mode: {relation_teacher_mode}")
    if relation_distance_normalization not in {
        "none",
        "anchor_mean",
        "symmetric_geometric_mean",
    }:
        raise ValueError(
            "relation_distance_normalization must be none, anchor_mean, "
            "or symmetric_geometric_mean"
        )
    if not 0.0 <= future_increment_weight <= 1.0:
        raise ValueError("future_increment_weight must be in [0, 1]")

    with torch.no_grad():
        non_overlap = (future_end[:, None] < context_start[None, :]) | (
            future_end[None, :] < context_start[:, None]
        )
        non_self = ~torch.eye(batch, dtype=torch.bool, device=future_model.device)
        temporal_candidates = (
            non_overlap.unsqueeze(-1).expand(-1, -1, nodes)
            & non_self.unsqueeze(-1).expand(-1, -1, nodes)
        )
        if relation_teacher_mode == "context_normalized":
            future_signature = normalize_future_with_context(
                future_model,
                context_statistics,
            )
            observed = future_observed.bool() & torch.isfinite(future_signature)
            raw_distance, future_pair_valid = _pairwise_masked_mae(
                future_signature,
                observed,
            )
            candidate_mask = temporal_candidates & future_pair_valid
            future_distance = raw_distance
        else:
            if forecast_context is None or forecast_context_observed is None:
                raise ValueError(
                    "OffsetDecay relation teachers require forecast_context and its mask"
                )
            offset_signature, offset_observed = build_offset_decay_signature(
                future_model,
                future_observed,
                forecast_context,
                forecast_context_observed,
            )
            offset_distance, offset_pair_valid = _pairwise_masked_mae(
                offset_signature,
                offset_observed,
            )
            if relation_teacher_mode == "offset_decay":
                candidate_mask = temporal_candidates & offset_pair_valid
                future_distance = offset_distance
                if relation_distance_normalization == "anchor_mean":
                    future_distance = anchor_mean_normalize_distances(
                        future_distance,
                        candidate_mask,
                    )
                elif relation_distance_normalization == "symmetric_geometric_mean":
                    future_distance = symmetric_geometric_mean_normalize(
                        future_distance,
                        candidate_mask,
                    )
            else:
                endpoint, endpoint_valid = _endpoint_level_from_context(
                    forecast_context,
                    forecast_context_observed,
                )
                increment, increment_observed = build_future_increment(
                    future_model,
                    future_observed,
                    endpoint,
                    endpoint_valid,
                )
                increment_distance, increment_pair_valid = _pairwise_masked_mae(
                    increment,
                    increment_observed,
                )
                candidate_mask = (
                    temporal_candidates & offset_pair_valid & increment_pair_valid
                )
                if relation_distance_normalization == "anchor_mean":
                    offset_distance = anchor_mean_normalize_distances(
                        offset_distance,
                        candidate_mask,
                    )
                    increment_distance = anchor_mean_normalize_distances(
                        increment_distance,
                        candidate_mask,
                    )
                elif relation_distance_normalization == "symmetric_geometric_mean":
                    offset_distance = symmetric_geometric_mean_normalize(
                        offset_distance,
                        candidate_mask,
                    )
                    increment_distance = symmetric_geometric_mean_normalize(
                        increment_distance,
                        candidate_mask,
                    )
                future_distance = (
                    (1.0 - future_increment_weight) * offset_distance
                    + future_increment_weight * increment_distance
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


def hard_mirage_ranking_loss(
    student_similarity: torch.Tensor,
    future_distance: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_count: int = 2,
    negative_count: int = 2,
    future_gap: float = 0.05,
    margin: float = 0.05,
    temperature: float = 0.1,
) -> RankingLossOutput:
    """Correct local key-order inversions using the existing future teacher."""
    if student_similarity.shape != future_distance.shape:
        raise ValueError("student_similarity and future_distance must align")
    if candidate_mask.shape != future_distance.shape or future_distance.ndim != 3:
        raise ValueError("ranking tensors must be [B, B, N]")
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("positive_count and negative_count must be positive")
    if future_gap < 0.0 or margin < 0.0:
        raise ValueError("future_gap and margin must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    capacity = future_distance.shape[1]
    positive_k = min(positive_count, capacity)
    negative_k = min(negative_count, capacity)
    valid = candidate_mask.bool() & torch.isfinite(future_distance)
    positive_distance, positive_index = torch.topk(
        future_distance.masked_fill(~valid, torch.inf),
        k=positive_k,
        dim=1,
        largest=False,
    )
    positive_valid = torch.isfinite(positive_distance)
    positive_score = torch.gather(student_similarity, 1, positive_index)
    positive_cutoff = positive_distance.masked_fill(
        ~positive_valid, -torch.inf
    ).amax(dim=1, keepdim=True)
    negative_mask = valid & (future_distance >= positive_cutoff + future_gap)
    selected_negative_score, negative_index = torch.topk(
        student_similarity.masked_fill(~negative_mask, -torch.inf),
        k=negative_k,
        dim=1,
        largest=True,
    )
    negative_valid = torch.isfinite(selected_negative_score)
    negative_score = torch.gather(student_similarity, 1, negative_index)

    pair_valid = positive_valid.unsqueeze(2) & negative_valid.unsqueeze(1)
    score_gap = positive_score.unsqueeze(2) - negative_score.unsqueeze(1)
    pair_loss = functional.softplus((margin - score_gap) / temperature)
    if bool(pair_valid.any()):
        loss = pair_loss.masked_select(pair_valid).mean()
        valid_pairs = int(pair_valid.sum().item())
    else:
        loss = student_similarity.sum() * 0.0
        valid_pairs = 0
    return RankingLossOutput(loss=loss, valid_pairs=valid_pairs)


def future_relation_retrieval_loss(
    node_keys: torch.Tensor,
    future_model: torch.Tensor,
    context_statistics,
    future_observed: torch.Tensor,
    context_start: torch.Tensor,
    future_end: torch.Tensor,
    teacher_temperature: float = 0.1,
    student_temperature: float = 0.1,
    relation_teacher_mode: str = "context_normalized",
    forecast_context: torch.Tensor | None = None,
    forecast_context_observed: torch.Tensor | None = None,
    relation_distance_normalization: str = "none",
    future_increment_weight: float = 0.0,
    rank_loss_weight: float = 0.0,
    rank_positive_count: int = 2,
    rank_negative_count: int = 2,
    rank_future_gap: float = 0.05,
    rank_margin: float = 0.05,
    rank_temperature: float = 0.1,
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
        relation_teacher_mode=relation_teacher_mode,
        forecast_context=forecast_context,
        forecast_context_observed=forecast_context_observed,
        relation_distance_normalization=relation_distance_normalization,
        future_increment_weight=future_increment_weight,
    )
    normalized_keys = functional.normalize(node_keys, dim=-1)
    student_logits = torch.einsum("ind,jnd->ijn", normalized_keys, normalized_keys)
    student_similarity = student_logits
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
    relation_loss = loss
    rank_output = hard_mirage_ranking_loss(
        student_similarity=student_similarity,
        future_distance=targets.future_distance,
        candidate_mask=targets.candidate_mask,
        positive_count=rank_positive_count,
        negative_count=rank_negative_count,
        future_gap=rank_future_gap,
        margin=rank_margin,
        temperature=rank_temperature,
    )
    loss = loss + rank_loss_weight * rank_output.loss
    valid_candidates = targets.candidate_mask & valid_anchors[:, None, :]
    return RetrievalLossOutput(
        loss=loss,
        valid_anchors=int(valid_anchors.sum().item()),
        positive_pairs=0,
        hard_negative_pairs=0,
        candidate_pairs=int(valid_candidates.sum().item()),
        teacher_effective_support=teacher_support,
        student_effective_support=student_support,
        rank_loss=rank_output.loss,
        rank_pairs=rank_output.valid_pairs,
        relation_loss=relation_loss,
    )


def compute_relation_only_loss(
    clean: CleanEncoding,
    future_model: torch.Tensor,
    observed_future: torch.Tensor,
    context_start: torch.Tensor,
    future_end: torch.Tensor,
    retrieval_weight: float,
    relation_teacher_temperature: float,
    relation_student_temperature: float,
    forecast_context: torch.Tensor,
    forecast_context_observed: torch.Tensor,
    relation_teacher_mode: str,
    relation_distance_normalization: str,
    future_increment_weight: float = 0.0,
    rank_loss_weight: float = 0.0,
    rank_positive_count: int = 2,
    rank_negative_count: int = 2,
    rank_future_gap: float = 0.05,
    rank_margin: float = 0.05,
    rank_temperature: float = 0.1,
) -> PretrainingLoss:
    """Compute a clean-key-only future relation objective.

    The zero reconstruction term is connected to the clean key so callers can
    reuse the common loss record without constructing a masked view.
    """
    if retrieval_weight < 0.0:
        raise ValueError("retrieval_weight must be non-negative")
    retrieval_output = future_relation_retrieval_loss(
        node_keys=clean.retrieval.node_keys,
        future_model=future_model,
        context_statistics=clean.statistics,
        future_observed=observed_future,
        context_start=context_start,
        future_end=future_end,
        teacher_temperature=relation_teacher_temperature,
        student_temperature=relation_student_temperature,
        relation_teacher_mode=relation_teacher_mode,
        forecast_context=forecast_context,
        forecast_context_observed=forecast_context_observed,
        relation_distance_normalization=relation_distance_normalization,
        future_increment_weight=future_increment_weight,
        rank_loss_weight=rank_loss_weight,
        rank_positive_count=rank_positive_count,
        rank_negative_count=rank_negative_count,
        rank_future_gap=rank_future_gap,
        rank_margin=rank_margin,
        rank_temperature=rank_temperature,
    )
    reconstruction = clean.retrieval.node_keys.sum() * 0.0
    rank = retrieval_output.rank_loss if retrieval_output.rank_loss is not None else reconstruction
    total = reconstruction + retrieval_weight * retrieval_output.loss
    return PretrainingLoss(
        total=total,
        reconstruction=reconstruction,
        retrieval=retrieval_output.loss,
        valid_retrieval_anchors=retrieval_output.valid_anchors,
        positive_pairs=retrieval_output.positive_pairs,
        hard_negative_pairs=retrieval_output.hard_negative_pairs,
        reconstruction_positions=0,
        relation_candidate_pairs=retrieval_output.candidate_pairs,
        teacher_effective_support=retrieval_output.teacher_effective_support,
        student_effective_support=retrieval_output.student_effective_support,
        profile=None,
        rank=rank,
        rank_pairs=retrieval_output.rank_pairs,
        relation=(retrieval_output.relation_loss if retrieval_output.relation_loss is not None else retrieval_output.loss),
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
    forecast_context: torch.Tensor | None = None,
    forecast_context_observed: torch.Tensor | None = None,
    relation_teacher_mode: str = "context_normalized",
    relation_distance_normalization: str = "none",
    future_increment_weight: float = 0.0,
    rank_loss_weight: float = 0.0,
    rank_positive_count: int = 2,
    rank_negative_count: int = 2,
    rank_future_gap: float = 0.05,
    rank_margin: float = 0.05,
    rank_temperature: float = 0.1,
    profile_loss_weight: float = 0.0,
    profile_scale_floor: float = 0.1,
    reconstruction_weight: float = 1.0,
) -> PretrainingLoss:
    if reconstruction_weight < 0.0:
        raise ValueError("reconstruction_weight must be non-negative")
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
            relation_teacher_mode=relation_teacher_mode,
            forecast_context=forecast_context,
            forecast_context_observed=forecast_context_observed,
            relation_distance_normalization=relation_distance_normalization,
            future_increment_weight=future_increment_weight,
            rank_loss_weight=rank_loss_weight,
            rank_positive_count=rank_positive_count,
            rank_negative_count=rank_negative_count,
            rank_future_gap=rank_future_gap,
            rank_margin=rank_margin,
            rank_temperature=rank_temperature,
        )
    else:
        raise ValueError(f"unknown retrieval_loss_mode: {retrieval_loss_mode}")
    profile_loss = output.clean.retrieval.node_keys.sum() * 0.0
    profile_prediction = output.clean.retrieval.profile_prediction
    if profile_loss_weight > 0.0:
        if profile_prediction is None:
            raise ValueError("profile loss requires a profile-enabled retrieval head")
        if future_model.shape[-1] != 1:
            raise ValueError("CFDP profile supervision currently requires C=1")
        if forecast_context is None or forecast_context_observed is None:
            raise ValueError(
                "CFDP profile supervision requires forecast_context and "
                "forecast_context_observed"
            )
        target_profile, target_valid = build_cfdp_teacher(
            future=future_model,
            future_observed=observed_future,
            context=forecast_context,
            context_observed=forecast_context_observed,
            profile_size=profile_prediction.shape[-1],
            scale_floor=profile_scale_floor,
        )
        target_profile = target_profile.squeeze(-1).permute(0, 2, 1).contiguous()
        target_valid = target_valid.squeeze(-1).permute(0, 2, 1).contiguous()
        if profile_prediction.shape != target_profile.shape:
            raise ValueError("profile prediction and CFDP target shapes do not align")
        element_loss = functional.smooth_l1_loss(
            profile_prediction, target_profile, reduction="none"
        )
        profile_loss = (
            element_loss.masked_select(target_valid).mean()
            if bool(target_valid.any())
            else profile_prediction.sum() * 0.0
        )
    total = (
        reconstruction_weight * reconstruction
        + retrieval_weight * retrieval_output.loss
        + profile_loss_weight * profile_loss
    )
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
        profile=profile_loss,
        rank=(retrieval_output.rank_loss if retrieval_output.rank_loss is not None else reconstruction),
        rank_pairs=retrieval_output.rank_pairs,
        relation=(retrieval_output.relation_loss if retrieval_output.relation_loss is not None else retrieval_output.loss),
    )

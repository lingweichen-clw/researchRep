"""Shared historical candidate aggregation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stanchor.retrieval.retriever import AggregationOutput, EventCandidates


def calendar_event_candidates(
    bank: Any,
    weekday: torch.Tensor,
    slot: torch.Tensor,
    context_start: torch.Tensor,
    max_candidates: int,
    device: torch.device,
) -> EventCandidates:
    """Return every causal event in the exact weekday-slot bucket."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if weekday.ndim != 1 or slot.shape != weekday.shape or context_start.shape != weekday.shape:
        raise ValueError("weekday, slot, and context_start must be [B]")
    batch = int(weekday.shape[0])
    ids = torch.full((batch, max_candidates), -1, dtype=torch.long, device=device)
    scores = torch.full((batch, max_candidates), -torch.inf, dtype=torch.float32, device=device)
    valid = torch.zeros((batch, max_candidates), dtype=torch.bool, device=device)
    future_end = np.asarray(bank.future_end)
    for batch_index in range(batch):
        calendar_ids = np.asarray(
            bank.calendar.lookup(
                int(weekday[batch_index].item()),
                int(slot[batch_index].item()),
            ),
            dtype=np.int64,
        )
        legal = calendar_ids[
            future_end[calendar_ids] < int(context_start[batch_index].item())
        ]
        if legal.size > max_candidates:
            raise ValueError(
                "event_top_r truncates the legal calendar pool; increase it for a fair ablation"
            )
        if legal.size == 0:
            continue
        count = int(legal.size)
        ids[batch_index, :count] = torch.from_numpy(legal).to(device)
        scores[batch_index, :count] = 0.0
        valid[batch_index, :count] = True
    return EventCandidates(ids, scores, valid)


def uniform_candidate_aggregation(
    candidates: torch.Tensor,
    valid: torch.Tensor,
) -> AggregationOutput:
    """Uniformly aggregate candidate futures with mask-aware variance."""
    if candidates.ndim != 5 or valid.shape != candidates.shape:
        raise ValueError("candidates and valid must be [B, H, N, K, C]")
    mask = valid.bool()
    weights = mask.to(candidates.dtype)
    denominator = weights.sum(dim=3)
    prediction = (candidates * weights).sum(dim=3) / denominator.clamp_min(1.0)
    prediction_valid = denominator > 0
    prediction = torch.where(prediction_valid, prediction, torch.zeros_like(prediction))
    difference = candidates - prediction.unsqueeze(3)
    variance = (weights * difference.square()).sum(dim=3) / denominator.clamp_min(1.0)
    variance = torch.where(prediction_valid, variance, torch.zeros_like(variance))
    return AggregationOutput(
        prediction=prediction,
        variance=variance,
        valid=prediction_valid,
        candidate_futures=candidates,
        candidate_masks=mask,
    )


def masked_candidate_mean(
    candidates: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mask-aware uniform mean without discarding candidates."""
    result = uniform_candidate_aggregation(candidates, valid)
    return result.prediction, result.valid


def raw_l1_candidate_scores(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute node-wise context L1 distance for every candidate event."""
    if query.ndim != 4 or query_observed.shape != query.shape:
        raise ValueError("query and query_observed must be [B, T, N, C]")
    if candidates.ndim != 5 or candidate_observed.shape != candidates.shape:
        raise ValueError("candidates and candidate_observed must be [B, R, T, N, C]")
    batch, candidate_count, time, nodes, channels = candidates.shape
    if query.shape != (batch, time, nodes, channels):
        raise ValueError("query and candidate context dimensions do not align")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B, R]")

    common = query_observed[:, None].bool() & candidate_observed.bool()
    count = common.sum(dim=(2, 4))
    absolute = (query[:, None] - candidates).abs()
    distance = torch.where(common, absolute, torch.zeros_like(absolute)).sum(dim=(2, 4))
    score_valid = (count > 0) & event_valid[:, :, None].bool()
    scores = distance / count.clamp_min(1)
    scores = scores.permute(0, 2, 1).contiguous()
    score_valid = score_valid.permute(0, 2, 1).contiguous()
    return scores.masked_fill(~score_valid, torch.inf), score_valid


def select_candidate_set_by_node(
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    selected: torch.Tensor,
    selected_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K candidate futures independently for every query node."""
    if candidates.ndim != 5 or candidate_valid.shape != candidates.shape:
        raise ValueError("candidates and candidate_valid must be [B, H, N, R, C]")
    batch, horizon, nodes, _, channels = candidates.shape
    if selected.ndim != 3 or selected.shape[:2] != (batch, nodes):
        raise ValueError("selected must be [B, N, K]")
    if selected_valid.shape != selected.shape:
        raise ValueError("selected_valid must match selected")
    top_k = selected.shape[-1]
    gather_index = selected[:, None, :, :, None].expand(batch, horizon, nodes, top_k, channels)
    values = candidates.gather(3, gather_index)
    valid = candidate_valid.gather(3, gather_index)
    valid = valid & selected_valid[:, None, :, :, None]
    return torch.where(valid, values, torch.zeros_like(values)), valid


def candidate_contexts(
    bank: Any,
    event_ids: torch.Tensor,
    series: Any,
    scaler: Any,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and normalize candidate contexts as [B, R, T, N, C]."""
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    starts = np.asarray(bank.context_start)[safe_ids]
    indices = starts[..., None] + np.arange(context_length, dtype=np.int64)
    raw = np.asarray(series.values[indices], dtype=np.float32)
    observed = np.asarray(series.observed[indices], dtype=bool)
    mean = scaler.mean[None, None, None, ...]
    std = scaler.std[None, None, None, ...]
    model_values = (raw - mean) / (std + scaler.eps)
    model_values = np.where(observed, model_values, 0.0).astype(np.float32)
    return torch.from_numpy(model_values).to(device), torch.from_numpy(observed).to(device)


def event_candidate_futures(
    bank: Any,
    event_ids: torch.Tensor,
    event_valid: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load candidate futures as [B, H, N, R, C]."""
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    values = np.asarray(bank.future_values[safe_ids], dtype=np.float32)
    observed = np.asarray(bank.future_masks[safe_ids], dtype=np.uint8).astype(bool)
    values = values.transpose(0, 2, 3, 1, 4)
    observed = observed.transpose(0, 2, 3, 1, 4)
    future = torch.from_numpy(values).to(device)
    valid = torch.from_numpy(observed).to(device)
    valid = valid & event_valid[:, None, None, :, None]
    return future, valid


def weekly_mean_aggregation(
    bank: Any,
    events: EventCandidates,
    device: torch.device,
) -> AggregationOutput:
    futures, valid = event_candidate_futures(
        bank,
        events.event_ids,
        events.valid,
        device,
    )
    return uniform_candidate_aggregation(futures, valid)


def raw_l1_topk_aggregation(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    bank: Any,
    events: EventCandidates,
    series: Any,
    scaler: Any,
    context_length: int,
    top_k: int,
    device: torch.device,
) -> AggregationOutput:
    """Select node-wise raw-L1 contexts and uniformly aggregate their futures."""
    contexts, context_observed = candidate_contexts(
        bank,
        events.event_ids,
        series,
        scaler,
        context_length,
        device,
    )
    raw_scores, raw_valid = raw_l1_candidate_scores(
        query,
        query_observed,
        contexts,
        context_observed,
        events.valid,
    )
    if top_k <= 0 or top_k > raw_scores.shape[-1]:
        raise ValueError("top_k must fit the padded calendar candidate axis")
    top_scores, selected = torch.topk(raw_scores, top_k, dim=-1, largest=False)
    selected_valid = torch.isfinite(top_scores)
    futures, future_valid = event_candidate_futures(
        bank,
        events.event_ids,
        events.valid,
        device,
    )
    selected_futures, selected_future_valid = select_candidate_set_by_node(
        futures,
        future_valid,
        selected,
        selected_valid,
    )
    return uniform_candidate_aggregation(selected_futures, selected_future_valid)

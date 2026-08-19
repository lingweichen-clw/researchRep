"""Shared historical candidate aggregation strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from stanchor.retrieval.retriever import AggregationOutput, EventCandidates, NodeCandidates
from stanchor.retrieval.trend_residual import estimate_local_trend

CANDIDATE_PROTOCOLS = ("exact_calendar", "relaxed_calendar")


def validate_candidate_protocol(protocol: str) -> str:
    protocol = str(protocol).lower()
    if protocol not in CANDIDATE_PROTOCOLS:
        choices = ", ".join(CANDIDATE_PROTOCOLS)
        raise ValueError(f"candidate protocol must be one of: {choices}")
    return protocol


def calendar_event_candidates(
    bank: Any,
    weekday: torch.Tensor,
    slot: torch.Tensor,
    context_start: torch.Tensor,
    max_candidates: int,
    device: torch.device,
    candidate_protocol: str = "exact_calendar",
) -> EventCandidates:
    """Return causal events from the configured calendar candidate protocol.

    ``exact_calendar`` selects the same weekday and slot. ``relaxed_calendar``
    selects the same weekday and slots in ``slot - 1, slot, slot + 1``. Both
    protocols use only historical events whose future has ended before the
    query context starts.
    """
    candidate_protocol = validate_candidate_protocol(candidate_protocol)
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if weekday.ndim != 1 or slot.shape != weekday.shape or context_start.shape != weekday.shape:
        raise ValueError("weekday, slot, and context_start must be [B]")
    batch = int(weekday.shape[0])
    ids = torch.full((batch, max_candidates), -1, dtype=torch.long, device=device)
    scores = torch.full((batch, max_candidates), -torch.inf, dtype=torch.float32, device=device)
    valid = torch.zeros((batch, max_candidates), dtype=torch.bool, device=device)
    slots_per_day = int(
        getattr(getattr(bank, "manifest", None), "slots_per_day", 24 * 60 // 5)
    )
    if candidate_protocol == "exact_calendar" and hasattr(
        bank, "calendar_event_ids_padded"
    ):
        padded_ids = torch.from_numpy(bank.calendar_event_ids_padded).to(device)
        padded_future_end = torch.from_numpy(bank.calendar_future_end_padded).to(device)
        bucket = weekday.to(device=device, dtype=torch.long) * slots_per_day + slot.to(
            device=device, dtype=torch.long
        )
        row_ids = padded_ids.index_select(0, bucket)
        row_future_end = padded_future_end.index_select(0, bucket)
        legal_mask = (row_ids >= 0) & (
            row_future_end < context_start.to(device=device, dtype=torch.long).unsqueeze(1)
        )
        if bool((legal_mask.sum(dim=1) > max_candidates).any()):
            raise ValueError(
                "event_top_r truncates the legal calendar pool; increase it for a fair ablation"
            )
        width = row_ids.shape[1]
        if width:
            positions = torch.arange(width, device=device).view(1, width)
            compact_key = torch.where(legal_mask, positions, positions + width)
            order = torch.argsort(compact_key, dim=1)
            compact_ids = row_ids.gather(1, order)
            compact_valid = legal_mask.gather(1, order)
            count = min(max_candidates, width)
            ids[:, :count] = torch.where(
                compact_valid[:, :count],
                compact_ids[:, :count],
                torch.full_like(compact_ids[:, :count], -1),
            )
            valid[:, :count] = compact_valid[:, :count]
            scores[:, :count] = torch.where(
                compact_valid[:, :count],
                torch.zeros_like(compact_ids[:, :count], dtype=torch.float32),
                torch.full_like(compact_ids[:, :count], -torch.inf, dtype=torch.float32),
            )
        return EventCandidates(ids, scores, valid)
    future_end = np.asarray(bank.future_end)
    for batch_index in range(batch):
        query_weekday = int(weekday[batch_index].item())
        query_slot = int(slot[batch_index].item())
        radius = 1 if candidate_protocol == "relaxed_calendar" else 0
        collected: list[int] = []
        seen: set[int] = set()
        for slot_offset in range(-radius, radius + 1):
            candidate_slot = query_slot + slot_offset
            if not 0 <= candidate_slot < slots_per_day:
                continue
            calendar_ids = np.asarray(
                bank.calendar.lookup(query_weekday, candidate_slot),
                dtype=np.int64,
            )
            for event_id in calendar_ids.tolist():
                event_id = int(event_id)
                if event_id in seen or future_end[event_id] >= int(context_start[batch_index].item()):
                    continue
                seen.add(event_id)
                collected.append(event_id)
        legal = np.asarray(collected, dtype=np.int64)
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
    """Load the forecast-tail contexts as [B, R, T, N, C]."""
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    ends = np.asarray(bank.context_end)[safe_ids]
    starts = ends - int(context_length) + 1
    indices = starts[..., None] + np.arange(context_length, dtype=np.int64)
    raw = np.asarray(series.values[indices], dtype=np.float32)
    observed = np.asarray(series.observed[indices], dtype=bool)
    mean = scaler.mean[None, None, None, ...]
    std = scaler.std[None, None, None, ...]
    model_values = (raw - mean) / (std + scaler.eps)
    model_values = np.where(observed, model_values, 0.0).astype(np.float32)
    return torch.from_numpy(model_values).to(device), torch.from_numpy(observed).to(device)


def candidate_contexts_for_nodes(
    bank: Any,
    event_ids: torch.Tensor,
    node_ids: torch.Tensor,
    series: Any,
    scaler: Any,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load only target-node contexts as ``[B,N,T,K,C]``."""
    if event_ids.shape != node_ids.shape or event_ids.ndim != 3:
        raise ValueError("event_ids and node_ids must be [B,N,K]")
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    target_nodes = node_ids.cpu().numpy().astype(np.int64, copy=False)
    ends = np.asarray(bank.context_end)[safe_ids]
    starts = ends - int(context_length) + 1
    indices = starts[..., None] + np.arange(context_length, dtype=np.int64)
    raw = np.asarray(series.values[indices, target_nodes[..., None], :], dtype=np.float32)
    observed = np.asarray(series.observed[indices, target_nodes[..., None], :], dtype=bool)
    mean = np.asarray(scaler.mean, dtype=np.float32)[target_nodes][..., None, :]
    std = np.asarray(scaler.std, dtype=np.float32)[target_nodes][..., None, :]
    model_values = (raw - mean) / (std + scaler.eps)
    model_values = np.where(observed, model_values, 0.0).astype(np.float32)
    return (
        torch.from_numpy(model_values.transpose(0, 1, 3, 2, 4)).to(device),
        torch.from_numpy(observed.transpose(0, 1, 3, 2, 4)).to(device),
    )


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


def event_candidate_futures_for_nodes(
    bank: Any,
    event_ids: torch.Tensor,
    event_valid: torch.Tensor,
    node_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load target-node futures as ``[B,H,N,K,C]`` without all-node expansion."""
    if event_ids.shape != event_valid.shape or event_ids.shape != node_ids.shape:
        raise ValueError("event_ids, event_valid, and node_ids must be [B,N,K]")
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    target_nodes = node_ids.cpu().numpy().astype(np.int64, copy=False)
    values = np.asarray(bank.future_values[safe_ids, :, target_nodes, :], dtype=np.float32)
    observed = np.asarray(
        bank.future_masks[safe_ids, :, target_nodes, :], dtype=np.uint8
    ).astype(bool)
    future = torch.from_numpy(values.transpose(0, 3, 1, 2, 4)).to(device)
    valid = torch.from_numpy(observed.transpose(0, 3, 1, 2, 4)).to(device)
    valid = valid & event_valid[:, None, :, :, None].to(device=device)
    return future, valid


def offset_decay_aggregation(
    candidates: NodeCandidates,
    query_context: torch.Tensor,
    query_observed: torch.Tensor,
    bank: Any,
    series: Any,
    scaler: Any,
    context_length: int,
    device: torch.device,
) -> AggregationOutput:
    """Aggregate learned candidates in the zero-parameter OffsetDecay coordinate."""
    if query_context.ndim != 4 or query_observed.shape != query_context.shape:
        raise ValueError("query context and mask must be [B, T, N, C]")
    batch, time, nodes, channels = query_context.shape
    if time != context_length:
        raise ValueError("query context time axis must match context_length")
    if candidates.event_ids.shape[:2] != (batch, nodes):
        raise ValueError("candidate event ids must align with query batch and nodes")
    top_k = candidates.event_ids.shape[-1]
    if candidates.weights.shape != (batch, nodes, top_k):
        raise ValueError("candidate weights must be [B, N, K]")

    query_statistics = estimate_local_trend(
        query_context,
        query_observed,
        context_length,
        mode="offset",
    )
    node_ids = torch.arange(nodes, device=device).view(1, nodes, 1).expand(
        batch, nodes, top_k
    )
    candidate_context, candidate_context_observed = candidate_contexts_for_nodes(
        bank,
        candidates.event_ids,
        node_ids,
        series,
        scaler,
        context_length,
        device,
    )
    candidate_statistics = estimate_local_trend(
        candidate_context,
        candidate_context_observed,
        context_length,
        mode="offset",
    )
    candidate_levels = candidate_statistics.level
    candidate_level_valid = candidate_statistics.valid & candidates.valid.unsqueeze(-1)

    selected_future, selected_future_valid = event_candidate_futures_for_nodes(
        bank,
        candidates.event_ids,
        candidates.valid,
        node_ids,
        device,
    )
    horizon = selected_future.shape[1]

    query_level = query_statistics.level[:, None, :, None, :]
    candidate_level = candidate_levels[:, None, :, :, :]
    aligned = query_level + selected_future - candidate_level
    decay = torch.linspace(
        1.0,
        0.0,
        horizon,
        dtype=selected_future.dtype,
        device=device,
    ).view(1, horizon, 1, 1, 1)
    offset_decay = selected_future + decay * (aligned - selected_future)
    valid = (
        selected_future_valid
        & query_statistics.valid[:, None, :, None, :]
        & candidate_level_valid[:, None, :, :, :]
    )
    offset_decay = torch.where(valid, offset_decay, torch.zeros_like(offset_decay))

    effective = candidates.weights[:, None, :, :, None] * valid.to(
        candidates.weights.dtype
    )
    denominator = effective.sum(dim=3)
    prediction = (effective * offset_decay).sum(dim=3) / denominator.clamp_min(1.0e-8)
    prediction_valid = denominator > 0
    prediction = torch.where(prediction_valid, prediction, torch.zeros_like(prediction))
    difference = offset_decay - prediction.unsqueeze(3)
    variance = (effective * difference.square()).sum(dim=3) / denominator.clamp_min(1.0e-8)
    variance = torch.where(prediction_valid, variance, torch.zeros_like(variance))
    return AggregationOutput(
        prediction=prediction,
        variance=variance,
        valid=prediction_valid,
        candidate_futures=offset_decay,
        candidate_masks=valid,
    )


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

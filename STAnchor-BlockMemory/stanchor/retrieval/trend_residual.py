"""Mask-aware trend-residual transforms and retrieval diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalTrendStatistics:
    """Local statistics with shape ``[..., N, C]``."""

    level: torch.Tensor
    slope: torch.Tensor
    scale: torch.Tensor
    valid: torch.Tensor


def _validate_series(values: torch.Tensor, observed: torch.Tensor, name: str) -> None:
    if values.ndim < 3 or observed.shape != values.shape:
        raise ValueError(f"{name} and its observed mask must be [..., T, N, C]")
    if values.shape[-3] <= 0 or values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError(f"{name} dimensions must be positive")


def _time_values(length: int, reference: torch.Tensor) -> torch.Tensor:
    shape = [1] * reference.ndim
    shape[-3] = length
    return torch.arange(length, dtype=reference.dtype, device=reference.device).view(shape)


def _validate_statistics(values: torch.Tensor, statistics: LocalTrendStatistics) -> None:
    expected = values.shape[:-3] + values.shape[-2:]
    for name, tensor in (
        ("level", statistics.level),
        ("slope", statistics.slope),
        ("scale", statistics.scale),
        ("valid", statistics.valid),
    ):
        if tensor.shape != expected:
            raise ValueError(f"statistics.{name} has shape {tensor.shape}, expected {expected}")


def estimate_local_trend(
    values: torch.Tensor,
    observed: torch.Tensor,
    trend_length: int,
    mode: str = "trend",
    eps: float = 1.0e-6,
) -> LocalTrendStatistics:
    """Estimate endpoint level, linear slope and local difference scale.

    The time axis is always ``-3``, so both ``[B,T,N,C]`` and
    ``[B,R,T,N,C]`` are accepted. Only the visible tail is used.
    """
    _validate_series(values, observed, "values")
    if mode not in {"offset", "trend"}:
        raise ValueError("mode must be offset or trend")
    if trend_length <= 0 or trend_length > values.shape[-3]:
        raise ValueError("trend_length must fit the context time axis")
    if eps <= 0:
        raise ValueError("eps must be positive")

    tail = values.narrow(-3, values.shape[-3] - trend_length, trend_length)
    visible = observed.narrow(-3, observed.shape[-3] - trend_length, trend_length).bool()
    visible = visible & torch.isfinite(tail)
    time = _time_values(trend_length, tail)
    count = visible.sum(dim=-3)
    safe_count = count.clamp_min(1)
    visible_float = visible.to(tail.dtype)

    mean_time = (visible_float * time).sum(dim=-3) / safe_count
    mean_value = torch.where(visible, tail, torch.zeros_like(tail)).sum(dim=-3) / safe_count
    centered_time = time - mean_time.unsqueeze(-3)
    centered_value = tail - mean_value.unsqueeze(-3)
    time_variance = (visible_float * centered_time.square()).sum(dim=-3)
    covariance = (visible_float * centered_time * centered_value).sum(dim=-3)
    enough_for_slope = (count >= 2) & (time_variance > eps)
    slope = torch.where(
        enough_for_slope,
        covariance / time_variance.clamp_min(eps),
        torch.zeros_like(covariance),
    )
    if mode == "offset":
        slope = torch.zeros_like(slope)

    endpoint = torch.as_tensor(trend_length - 1, dtype=tail.dtype, device=tail.device)
    fitted_endpoint = mean_value + (endpoint - mean_time) * slope
    endpoint_visible = visible.select(-3, trend_length - 1)
    endpoint_value = tail.select(-3, trend_length - 1)
    level = torch.where(endpoint_visible, endpoint_value, fitted_endpoint)
    statistics_valid = count > 0
    level = torch.where(statistics_valid, level, torch.zeros_like(level))

    adjacent_visible = visible[..., 1:, :, :] & visible[..., :-1, :, :]
    adjacent_count = adjacent_visible.sum(dim=-3)
    differences = tail[..., 1:, :, :] - tail[..., :-1, :, :]
    difference_energy = torch.where(
        adjacent_visible,
        differences.square(),
        torch.zeros_like(differences),
    ).sum(dim=-3)
    difference_scale = torch.sqrt(difference_energy / adjacent_count.clamp_min(1) + eps)

    historical_baseline = level.unsqueeze(-3) + (
        time - endpoint
    ) * slope.unsqueeze(-3)
    detrended = tail - historical_baseline
    detrended_energy = torch.where(
        visible,
        detrended.square(),
        torch.zeros_like(detrended),
    ).sum(dim=-3)
    fallback_scale = torch.sqrt(detrended_energy / safe_count + eps)
    scale = torch.where(adjacent_count > 0, difference_scale, fallback_scale)
    scale = torch.where(statistics_valid, scale.clamp_min(eps**0.5), torch.ones_like(scale))

    return LocalTrendStatistics(
        level=level,
        slope=slope,
        scale=scale,
        valid=statistics_valid,
    )


def residualize_context(
    values: torch.Tensor,
    observed: torch.Tensor,
    statistics: LocalTrendStatistics,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express visible historical values relative to the fitted endpoint."""
    _validate_series(values, observed, "values")
    _validate_statistics(values, statistics)
    length = values.shape[-3]
    time = _time_values(length, values)
    endpoint = torch.as_tensor(length - 1, dtype=values.dtype, device=values.device)
    baseline = statistics.level.unsqueeze(-3) + (
        time - endpoint
    ) * statistics.slope.unsqueeze(-3)
    valid = observed.bool() & torch.isfinite(values) & statistics.valid.unsqueeze(-3)
    residual = (values - baseline) / statistics.scale.unsqueeze(-3).clamp_min(eps)
    return torch.where(valid, residual, torch.zeros_like(residual)), valid


def residualize_future(
    future: torch.Tensor,
    observed: torch.Tensor,
    statistics: LocalTrendStatistics,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express future values in the context-local trend coordinate."""
    _validate_series(future, observed, "future")
    _validate_statistics(future, statistics)
    horizon = future.shape[-3]
    steps = _time_values(horizon, future) + 1.0
    baseline = statistics.level.unsqueeze(-3) + steps * statistics.slope.unsqueeze(-3)
    valid = observed.bool() & torch.isfinite(future) & statistics.valid.unsqueeze(-3)
    residual = (future - baseline) / statistics.scale.unsqueeze(-3).clamp_min(eps)
    return torch.where(valid, residual, torch.zeros_like(residual)), valid


def reconstruct_future(
    residual: torch.Tensor,
    observed: torch.Tensor,
    statistics: LocalTrendStatistics,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a residual future into the supplied query-local coordinate."""
    _validate_series(residual, observed, "residual")
    _validate_statistics(residual, statistics)
    horizon = residual.shape[-3]
    steps = _time_values(horizon, residual) + 1.0
    baseline = statistics.level.unsqueeze(-3) + steps * statistics.slope.unsqueeze(-3)
    valid = observed.bool() & torch.isfinite(residual) & statistics.valid.unsqueeze(-3)
    reconstructed = baseline + residual * statistics.scale.unsqueeze(-3)
    return torch.where(valid, reconstructed, torch.zeros_like(reconstructed)), valid


def masked_pearson_candidate_scores(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return node-wise history Pearson scores as ``[B,N,R]``."""
    _validate_series(query, query_observed, "query")
    _validate_series(candidates, candidate_observed, "candidates")
    if query.ndim != 4 or candidates.ndim != 5:
        raise ValueError("query must be [B,T,N,C] and candidates [B,R,T,N,C]")
    batch, candidate_count, time, nodes, channels = candidates.shape
    if query.shape != (batch, time, nodes, channels):
        raise ValueError("query and candidate context dimensions do not align")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B,R]")

    query_expanded = query[:, None]
    common = (
        query_observed[:, None].bool()
        & candidate_observed.bool()
        & torch.isfinite(query_expanded)
        & torch.isfinite(candidates)
    )
    count = common.sum(dim=(2, 4))
    safe_count = count.clamp_min(1)
    query_mean = torch.where(common, query_expanded, torch.zeros_like(candidates)).sum(
        dim=(2, 4)
    ) / safe_count
    candidate_mean = torch.where(common, candidates, torch.zeros_like(candidates)).sum(
        dim=(2, 4)
    ) / safe_count
    centered_query = query_expanded - query_mean[:, :, None, :, None]
    centered_candidates = candidates - candidate_mean[:, :, None, :, None]
    covariance = torch.where(
        common,
        centered_query * centered_candidates,
        torch.zeros_like(candidates),
    ).sum(dim=(2, 4))
    query_energy = torch.where(
        common,
        centered_query.square(),
        torch.zeros_like(candidates),
    ).sum(dim=(2, 4))
    candidate_energy = torch.where(
        common,
        centered_candidates.square(),
        torch.zeros_like(candidates),
    ).sum(dim=(2, 4))
    denominator = torch.sqrt(query_energy * candidate_energy)
    score_valid = (
        (count >= 2)
        & (denominator > eps)
        & event_valid[:, :, None].bool()
    )
    scores = (covariance / denominator.clamp_min(eps)).clamp(-1.0, 1.0)
    scores = scores.permute(0, 2, 1).contiguous()
    score_valid = score_valid.permute(0, 2, 1).contiguous()
    return scores.masked_fill(~score_valid, -torch.inf), score_valid


def masked_future_l1_scores(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return node-wise future residual distances as ``[B,N,R]``."""
    _validate_series(query, query_observed, "query")
    _validate_series(candidates, candidate_observed, "candidates")
    if query.ndim != 4 or candidates.ndim != 5:
        raise ValueError("query must be [B,H,N,C] and candidates [B,R,H,N,C]")
    batch, candidate_count, horizon, nodes, channels = candidates.shape
    if query.shape != (batch, horizon, nodes, channels):
        raise ValueError("query and candidate future dimensions do not align")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B,R]")

    common = (
        query_observed[:, None].bool()
        & candidate_observed.bool()
        & event_valid[:, :, None, None, None].bool()
    )
    count = common.sum(dim=(2, 4))
    absolute = (query[:, None] - candidates).abs()
    distance = torch.where(common, absolute, torch.zeros_like(absolute)).sum(dim=(2, 4))
    valid = count > 0
    distance = (distance / count.clamp_min(1)).permute(0, 2, 1).contiguous()
    valid = valid.permute(0, 2, 1).contiguous()
    return distance.masked_fill(~valid, torch.inf), valid


def match_selected_event_positions(
    event_ids: torch.Tensor,
    selected_event_ids: torch.Tensor,
    selected_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global selected event ids back to the local event-pool axis."""
    if event_ids.ndim != 2 or selected_event_ids.ndim != 3:
        raise ValueError("event_ids must be [B,R] and selected_event_ids [B,N,K]")
    if selected_event_ids.shape != selected_valid.shape:
        raise ValueError("selected_valid must match selected_event_ids")
    if selected_event_ids.shape[0] != event_ids.shape[0]:
        raise ValueError("batch dimensions do not align")
    matches = selected_event_ids[..., None] == event_ids[:, None, None, :]
    found = matches.any(dim=-1)
    requested = selected_valid.bool() & (selected_event_ids >= 0)
    if bool((requested & ~found).any()):
        raise ValueError("selected event is not present in the event candidate pool")
    positions = matches.to(torch.int64).argmax(dim=-1)
    valid = requested & found
    return positions, valid


def softmax_topk_weights(
    scores: torch.Tensor,
    valid: torch.Tensor,
    top_k: int,
    temperature: float,
    largest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select Top-K candidates and normalize their score weights."""
    if scores.ndim != 3 or valid.shape != scores.shape:
        raise ValueError("scores and valid must be [B,N,R]")
    if top_k <= 0 or top_k > scores.shape[-1]:
        raise ValueError("top_k must fit the candidate axis")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    fill = -torch.inf if largest else torch.inf
    masked_scores = scores.masked_fill(~valid.bool(), fill)
    top_scores, selected = torch.topk(masked_scores, top_k, dim=-1, largest=largest)
    selected_valid = valid.gather(-1, selected) & torch.isfinite(top_scores)
    logits = top_scores / temperature if largest else -top_scores / temperature
    stable = logits.masked_fill(~selected_valid, -torch.inf)
    maximum = stable.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.where(
        selected_valid,
        torch.exp(logits - maximum),
        torch.zeros_like(logits),
    )
    weights = exponent / exponent.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return selected, selected_valid, weights


def weighted_candidate_mean(
    candidates: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate ``[B,H,N,K,C]`` candidates with per-node weights."""
    if candidates.ndim != 5 or valid.shape != candidates.shape:
        raise ValueError("candidates and valid must be [B,H,N,K,C]")
    batch, _, nodes, top_k, _ = candidates.shape
    if weights.shape != (batch, nodes, top_k):
        raise ValueError("weights must be [B,N,K]")
    effective = weights[:, None, :, :, None] * valid.to(candidates.dtype)
    denominator = effective.sum(dim=3)
    prediction = (effective * candidates).sum(dim=3) / denominator.clamp_min(1.0e-8)
    prediction_valid = denominator > 0
    return torch.where(prediction_valid, prediction, torch.zeros_like(prediction)), prediction_valid


def masked_spearman_rank_correlation(
    first_scores: torch.Tensor,
    second_scores: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ordinal Spearman correlation over the final candidate axis."""
    if first_scores.ndim != 3 or second_scores.shape != first_scores.shape:
        raise ValueError("score tensors must both be [B,N,R]")
    if valid.shape != first_scores.shape:
        raise ValueError("valid must match score tensors")
    common = valid.bool() & torch.isfinite(first_scores) & torch.isfinite(second_scores)

    def ordinal_rank(scores: torch.Tensor) -> torch.Tensor:
        filled = scores.masked_fill(~common, -torch.inf)
        order = filled.argsort(dim=-1)
        return order.argsort(dim=-1).to(scores.dtype)

    first_rank = ordinal_rank(first_scores)
    second_rank = ordinal_rank(second_scores)
    count = common.sum(dim=-1)
    safe_count = count.clamp_min(1)
    first_mean = torch.where(common, first_rank, torch.zeros_like(first_rank)).sum(
        dim=-1
    ) / safe_count
    second_mean = torch.where(common, second_rank, torch.zeros_like(second_rank)).sum(
        dim=-1
    ) / safe_count
    first_centered = first_rank - first_mean.unsqueeze(-1)
    second_centered = second_rank - second_mean.unsqueeze(-1)
    covariance = torch.where(
        common,
        first_centered * second_centered,
        torch.zeros_like(first_centered),
    ).sum(dim=-1)
    first_energy = torch.where(
        common,
        first_centered.square(),
        torch.zeros_like(first_centered),
    ).sum(dim=-1)
    second_energy = torch.where(
        common,
        second_centered.square(),
        torch.zeros_like(second_centered),
    ).sum(dim=-1)
    denominator = torch.sqrt(first_energy * second_energy)
    correlation_valid = (count >= 2) & (denominator > eps)
    correlation = covariance / denominator.clamp_min(eps)
    return torch.where(correlation_valid, correlation, torch.zeros_like(correlation)), correlation_valid

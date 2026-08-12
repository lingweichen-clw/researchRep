"""Zero-training diagnostics for future-relation teacher metrics.

This module deliberately keeps query future on the offline evaluation side.
It never changes a key, a Bank file, or a downstream model.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.diagnostics.retrieval_visualization import (
    _candidate_node_keys,
    alignment_statistics,
    build_diagnostic_event_candidates,
    future_neighbor_recall_at_k,
    node_key_distances,
)
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.pretrainer import build_validation_loader
from stanchor.engine.target import _validate_bank
from stanchor.losses.pretraining import (
    _endpoint_level_from_context,
    _masked_softmax,
    anchor_mean_normalize_distances,
    build_future_increment,
    build_offset_decay_signature,
    build_future_relation_targets,
)
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.strategies import candidate_contexts, event_candidate_futures
from stanchor.utils import resolve_device, save_json


def candidate_distance(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
    clip_delta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked candidate distances as ``[B, N, R]``.

    Query is ``[B,H,N,C]`` and candidates are ``[B,R,H,N,C]``.  ``clip_delta``
    implements the offline clipped-L1 stress test and does not alter stored
    future values.
    """
    if query.ndim != 4 or candidates.ndim != 5:
        raise ValueError("query/candidates must be [B,H,N,C]/[B,R,H,N,C]")
    if query_observed.shape != query.shape or candidate_observed.shape != candidates.shape:
        raise ValueError("observed masks must align with values")
    batch, candidate_count, horizon, nodes, channels = candidates.shape
    if query.shape != (batch, horizon, nodes, channels):
        raise ValueError("query and candidates do not align")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B,R]")
    if clip_delta is not None and clip_delta <= 0:
        raise ValueError("clip_delta must be positive")

    common = query_observed[:, None].bool() & candidate_observed.bool()
    common &= torch.isfinite(query[:, None]) & torch.isfinite(candidates)
    error = (query[:, None] - candidates).abs()
    if clip_delta is not None:
        error = error.clamp_max(float(clip_delta))
    count = common.sum(dim=(2, 4))  # [B,R,N]
    total = torch.where(common, error, torch.zeros_like(error)).sum(dim=(2, 4))
    valid = (count > 0) & event_valid[:, :, None].bool()
    distance = (total / count.clamp_min(1)).permute(0, 2, 1).contiguous()
    valid = valid.permute(0, 2, 1).contiguous()
    return torch.where(valid, distance, torch.zeros_like(distance)), valid


def normalize_candidate_distance(
    distance: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Apply the existing per-anchor mean normalization to ``[B,N,R]``."""
    normalized = anchor_mean_normalize_distances(
        distance.permute(0, 2, 1),
        valid.permute(0, 2, 1),
    ).permute(0, 2, 1)
    return normalized


def symmetric_geometric_mean_normalize_distances(
    distances: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize square pair distances by both endpoint scales.

    The last three axes are ``[E,E,N]``.  For every event-node anchor ``i``,
    ``mu_i`` is its mean valid distance to the other events.  A valid pair is
    normalized as ``d_ij / sqrt((mu_i + eps) * (mu_j + eps))``.
    """
    if distances.ndim < 3 or distances.shape[-3] != distances.shape[-2]:
        raise ValueError("distances must end with square [E,E,N] axes")
    if valid.shape != distances.shape:
        raise ValueError("valid must align with distances")
    if eps <= 0:
        raise ValueError("eps must be positive")

    transposed_valid = valid.transpose(-3, -2).bool()
    finite_valid = valid.bool() & transposed_valid
    finite_valid &= torch.isfinite(distances) & torch.isfinite(
        distances.transpose(-3, -2)
    )
    symmetric_distance = 0.5 * (
        distances + distances.transpose(-3, -2)
    )
    count = finite_valid.sum(dim=-2)  # [...,E,N]
    total = torch.where(
        finite_valid,
        symmetric_distance,
        torch.zeros_like(symmetric_distance),
    ).sum(dim=-2)
    mean_distance = total / count.clamp_min(1)
    denominator = torch.sqrt(
        (mean_distance.unsqueeze(-2) + eps)
        * (mean_distance.unsqueeze(-3) + eps)
    )
    scale_valid = count > 0
    normalized_valid = finite_valid
    normalized_valid &= scale_valid.unsqueeze(-2) & scale_valid.unsqueeze(-3)
    normalized = symmetric_distance / denominator.clamp_min(eps)
    return torch.where(
        normalized_valid,
        normalized,
        torch.zeros_like(normalized),
    ), normalized_valid


def _event_set_pairwise_distance(
    events: torch.Tensor,
    observed: torch.Tensor,
    event_valid: torch.Tensor,
    chunk_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise masked MAE for ``[B,E,H,N,C]`` event sets."""
    if events.ndim != 5 or observed.shape != events.shape:
        raise ValueError("events and observed must be [B,E,H,N,C]")
    batch, event_count, _, nodes, _ = events.shape
    if event_valid.shape != (batch, event_count):
        raise ValueError("event_valid must be [B,E]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    distances = torch.zeros(
        (batch, event_count, event_count, nodes),
        dtype=events.dtype,
        device=events.device,
    )
    pair_valid = torch.zeros_like(distances, dtype=torch.bool)
    right_values = events[:, None]  # [B,1,E,H,N,C]
    right_observed = observed[:, None].bool()
    for start in range(0, event_count, chunk_size):
        end = min(start + chunk_size, event_count)
        left_values = events[:, start:end, None]
        left_observed = observed[:, start:end, None].bool()
        common = left_observed & right_observed
        common &= torch.isfinite(left_values) & torch.isfinite(right_values)
        error = (left_values - right_values).abs()
        point_count = common.sum(dim=(3, 5))  # [B,L,E,N]
        total = torch.where(common, error, torch.zeros_like(error)).sum(dim=(3, 5))
        valid_chunk = point_count > 0
        valid_chunk &= event_valid[:, start:end, None, None]
        valid_chunk &= event_valid[:, None, :, None]
        distances[:, start:end] = torch.where(
            valid_chunk,
            total / point_count.clamp_min(1),
            torch.zeros_like(total),
        )
        pair_valid[:, start:end] = valid_chunk

    non_self = ~torch.eye(event_count, dtype=torch.bool, device=events.device)
    pair_valid &= non_self.view(1, event_count, event_count, 1)
    pair_valid = pair_valid & pair_valid.transpose(1, 2)
    distances = 0.5 * (distances + distances.transpose(1, 2))
    return torch.where(pair_valid, distances, torch.zeros_like(distances)), pair_valid


def symmetric_candidate_distance(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
    eps: float = 1.0e-6,
    chunk_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SymNorm query-candidate distances as ``[B,N,R]``.

    Each query and its ``R`` Bank candidates form one common event set.  Both
    query scale ``mu_i`` and candidate scale ``mu_j`` are computed on that same
    set before selecting the query-to-candidate row.
    """
    if query.ndim != 4 or candidates.ndim != 5:
        raise ValueError("query/candidates must be [B,H,N,C]/[B,R,H,N,C]")
    if query_observed.shape != query.shape or candidate_observed.shape != candidates.shape:
        raise ValueError("observed masks must align with values")
    batch, candidate_count, horizon, nodes, channels = candidates.shape
    if query.shape != (batch, horizon, nodes, channels):
        raise ValueError("query and candidates do not align")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B,R]")

    values = torch.cat((query.unsqueeze(1), candidates), dim=1)
    observed = torch.cat((query_observed.unsqueeze(1), candidate_observed), dim=1)
    common_event_valid = torch.cat(
        (
            torch.ones((batch, 1), dtype=torch.bool, device=query.device),
            event_valid.bool(),
        ),
        dim=1,
    )
    pair_distance, pair_valid = _event_set_pairwise_distance(
        values,
        observed,
        common_event_valid,
        chunk_size=chunk_size,
    )
    normalized, normalized_valid = symmetric_geometric_mean_normalize_distances(
        pair_distance,
        pair_valid,
        eps=eps,
    )
    query_candidate = normalized[:, 0, 1:].permute(0, 2, 1).contiguous()
    query_candidate_valid = normalized_valid[:, 0, 1:].permute(0, 2, 1).contiguous()
    return query_candidate, query_candidate_valid


def distance_distribution(
    distance: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Convert ``[B,N,R]`` distances to masked teacher probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = (-distance / temperature).masked_fill(~valid.bool(), -torch.inf)
    has_candidate = valid.any(dim=-1)
    safe_logits = torch.where(has_candidate.unsqueeze(-1), logits, torch.zeros_like(logits))
    return torch.softmax(safe_logits, dim=-1) * valid.to(distance.dtype)


def effective_support_from_distance(
    distance: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return ``K_eff=1/sum(q^2)`` for each ``[B,N]`` anchor."""
    probabilities = distance_distribution(distance, valid, temperature)
    support = 1.0 / probabilities.square().sum(dim=-1).clamp_min(1.0e-12)
    return torch.where(valid.any(dim=-1), support, torch.zeros_like(support))


def topk_indices(
    distance: torch.Tensor,
    valid: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return smallest-distance indices and eligibility as ``[B,N,K]``."""
    if k <= 0 or k > distance.shape[-1]:
        raise ValueError("k must be in [1,R]")
    infinity = torch.full_like(distance, torch.inf)
    order = torch.topk(
        torch.where(valid.bool(), distance, infinity),
        k,
        dim=-1,
        largest=False,
    ).indices
    eligible = valid.sum(dim=-1) >= k
    return order, eligible


def topk_jaccard(clean: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
    """Return Jaccard overlap of two unique Top-K supports."""
    if clean.shape != perturbed.shape or clean.ndim < 2:
        raise ValueError("clean and perturbed must have matching [...,K] shapes")
    overlap = (
        clean.unsqueeze(-1) == perturbed.unsqueeze(-2)
    ).any(dim=(-1, -2)).to(torch.float32)
    intersection = (
        clean.unsqueeze(-1) == perturbed.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1).to(torch.float32)
    k = clean.shape[-1]
    union = (2.0 * k - intersection).clamp_min(1.0)
    return intersection / union


def near_tie_collision_rate(
    od_distance: torch.Tensor,
    increment_distance: torch.Tensor,
    valid: torch.Tensor,
    od_tolerance: float = 0.05,
    increment_gap: float = 0.25,
) -> tuple[float, int]:
    """Measure OD near-ties whose increment distances disagree.

    Distances must already be anchor-mean normalized and have shape ``[B,N,R]``.
    The returned integer is the number of colliding candidate pairs; the
    denominator (all OD near-tie pairs) is included in the companion counter
    from ``collision_pair_counts`` in the runner.
    """
    if od_distance.shape != increment_distance.shape or valid.shape != od_distance.shape:
        raise ValueError("distance and valid tensors must align")
    if od_tolerance <= 0 or increment_gap <= 0:
        raise ValueError("collision thresholds must be positive")
    flat_od = od_distance.reshape(-1, od_distance.shape[-1])
    flat_inc = increment_distance.reshape(-1, increment_distance.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1]).bool()
    collision_count = 0
    near_tie_count = 0
    for od_row, inc_row, valid_row in zip(flat_od, flat_inc, flat_valid):
        pair = valid_row[:, None] & valid_row[None, :]
        upper = torch.triu(torch.ones_like(pair, dtype=torch.bool), diagonal=1)
        near = pair & upper & (od_row[:, None] - od_row[None, :]).abs().le(od_tolerance)
        separated = (inc_row[:, None] - inc_row[None, :]).abs().ge(increment_gap)
        near_tie_count += int(near.sum().item())
        collision_count += int((near & separated).sum().item())
    rate = collision_count / max(near_tie_count, 1)
    return float(rate), collision_count


def collision_pair_counts(
    od_distance: torch.Tensor,
    increment_distance: torch.Tensor,
    valid: torch.Tensor,
    od_tolerance: float = 0.05,
    increment_gap: float = 0.25,
) -> tuple[int, int]:
    """Return ``(collisions, all_near_ties)`` for reporting denominators."""
    if od_distance.shape != increment_distance.shape or valid.shape != od_distance.shape:
        raise ValueError("distance and valid tensors must align")
    flat_od = od_distance.reshape(-1, od_distance.shape[-1])
    flat_inc = increment_distance.reshape(-1, increment_distance.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1]).bool()
    collisions = 0
    near_ties = 0
    for od_row, inc_row, valid_row in zip(flat_od, flat_inc, flat_valid):
        pair = valid_row[:, None] & valid_row[None, :]
        upper = torch.triu(torch.ones_like(pair, dtype=torch.bool), diagonal=1)
        near = pair & upper & (od_row[:, None] - od_row[None, :]).abs().le(od_tolerance)
        separated = (inc_row[:, None] - inc_row[None, :]).abs().ge(increment_gap)
        near_ties += int(near.sum().item())
        collisions += int((near & separated).sum().item())
    return collisions, near_ties


def distribution_asymmetry(
    probabilities: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    """Measure asymmetry for square in-batch distributions ``[B,B,N]``."""
    if probabilities.ndim != 3 or valid.shape != probabilities.shape:
        raise ValueError("probabilities and valid must be [B,B,N]")
    pair_valid = valid.bool() & valid.transpose(0, 1).bool()
    upper = torch.triu(
        torch.ones((probabilities.shape[0], probabilities.shape[0]), dtype=torch.bool, device=probabilities.device),
        diagonal=1,
    ).unsqueeze(-1)
    pair_valid &= upper
    if not bool(pair_valid.any()):
        raise ValueError("distribution asymmetry requires at least one valid symmetric pair")
    difference = (probabilities - probabilities.transpose(0, 1)).abs()
    count = pair_valid.sum()
    return {
        "mean_absolute_asymmetry": float(difference.masked_select(pair_valid).mean().detach().cpu()),
        "valid_pair_count": int(count.item()),
    }


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> float:
    selected = values.masked_select(valid.bool())
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().mean().detach().cpu())


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def estimate_clip_delta(
    bank: MemoryBank,
    sample_size: int = 200_000,
    seed: int = 42,
) -> float:
    """Estimate a fixed source-bank 95th percentile OD pair residual."""
    rng = np.random.default_rng(seed)
    events = int(bank.manifest.num_events)
    horizon = int(bank.manifest.horizon)
    nodes = int(bank.manifest.num_nodes)
    channels = int(bank.manifest.channels)
    decay = np.linspace(1.0, 0.0, horizon, dtype=np.float32)
    values: list[np.ndarray] = []
    collected = 0
    attempts = 0
    while collected < sample_size and attempts < 20:
        count = max(10_000, min(sample_size - collected, 200_000))
        left_event = rng.integers(0, events, size=count)
        right_event = rng.integers(0, events, size=count)
        horizon_id = rng.integers(0, horizon, size=count)
        node_id = rng.integers(0, nodes, size=count)
        channel_id = rng.integers(0, channels, size=count)
        left_value = np.asarray(bank.future_values[left_event, horizon_id, node_id, channel_id])
        right_value = np.asarray(bank.future_values[right_event, horizon_id, node_id, channel_id])
        left_alpha = np.asarray(
            bank.level_features[left_event, node_id, 2 * channels + channel_id]
        )
        right_alpha = np.asarray(
            bank.level_features[right_event, node_id, 2 * channels + channel_id]
        )
        left_valid = np.asarray(bank.future_masks[left_event, horizon_id, node_id, channel_id]).astype(bool)
        right_valid = np.asarray(bank.future_masks[right_event, horizon_id, node_id, channel_id]).astype(bool)
        finite = left_valid & right_valid & np.isfinite(left_value) & np.isfinite(right_value)
        residual = np.abs(
            (left_value - decay[horizon_id] * left_alpha)
            - (right_value - decay[horizon_id] * right_alpha)
        )
        if finite.any():
            values.append(residual[finite])
            collected += int(finite.sum())
        attempts += 1
    if not values:
        raise ValueError("unable to estimate source clip delta from Bank futures")
    return float(np.quantile(np.concatenate(values)[:sample_size], 0.95))


def _candidate_signatures(
    bank: MemoryBank,
    events: Any,
    candidate_future: torch.Tensor,
    candidate_valid: torch.Tensor,
    data: Any,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate OD signature, increment, endpoint, and validity."""
    batch, candidate_count, horizon, nodes, channels = candidate_future.shape
    contexts, context_valid = candidate_contexts(
        bank,
        events.event_ids,
        data.series,
        data.scaler,
        context_length,
        device,
    )
    flat_future = candidate_future.reshape(batch * candidate_count, horizon, nodes, channels)
    flat_future_valid = candidate_valid.reshape(batch * candidate_count, horizon, nodes, channels)
    flat_context = contexts.reshape(batch * candidate_count, context_length, nodes, channels)
    flat_context_valid = context_valid.reshape(batch * candidate_count, context_length, nodes, channels)
    signature, signature_valid = build_offset_decay_signature(
        flat_future,
        flat_future_valid,
        flat_context,
        flat_context_valid,
    )
    endpoint, endpoint_valid = _endpoint_level_from_context(flat_context, flat_context_valid)
    increment, increment_valid = build_future_increment(
        flat_future,
        flat_future_valid,
        endpoint,
        endpoint_valid,
    )
    return (
        signature.reshape(batch, candidate_count, horizon, nodes, channels),
        signature_valid.reshape(batch, candidate_count, horizon, nodes, channels),
        increment.reshape(batch, candidate_count, horizon, nodes, channels),
        increment_valid.reshape(batch, candidate_count, horizon, nodes, channels),
    )


def _normalize_mix(
    od_distance: torch.Tensor,
    od_valid: torch.Tensor,
    increment_distance: torch.Tensor,
    increment_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = od_valid.bool() & increment_valid.bool()
    od = normalize_candidate_distance(od_distance, valid)
    increment = normalize_candidate_distance(increment_distance, valid)
    return 0.5 * od + 0.5 * increment, valid


def _perturb_candidate_future(
    candidate_future: torch.Tensor,
    candidate_valid: torch.Tensor,
    event_ids: torch.Tensor,
    delta: float,
    rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inject one deterministic mid-horizon outlier per selected series."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("perturbation rate must be in [0,1]")
    batch, candidate_count, horizon, nodes, channels = candidate_future.shape
    safe_ids = event_ids.clamp_min(0)
    node_ids = torch.arange(nodes, device=candidate_future.device).view(1, 1, nodes)
    selected = ((safe_ids.unsqueeze(-1) + 17 * node_ids) % 1000) < int(rate * 1000)
    selected = selected & event_ids.ge(0).unsqueeze(-1)
    horizon_id = horizon // 2
    selected = selected & candidate_valid[:, :, horizon_id].all(dim=-1)
    sign = torch.where(
        ((safe_ids.unsqueeze(-1) + node_ids) % 2) == 0,
        torch.ones_like(selected, dtype=candidate_future.dtype),
        -torch.ones_like(selected, dtype=candidate_future.dtype),
    )
    perturbed = candidate_future.clone()
    perturbation = (sign * (3.0 * float(delta))).unsqueeze(2).unsqueeze(-1)
    mask = selected.unsqueeze(2).unsqueeze(-1)
    perturbed = torch.where(mask, perturbed + perturbation, perturbed)
    return perturbed, selected


def _offset_decay_payloads(
    candidate_future: torch.Tensor,
    candidate_valid: torch.Tensor,
    candidate_contexts_tensor: torch.Tensor,
    candidate_context_valid: torch.Tensor,
    query_context: torch.Tensor,
    query_context_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build all candidate node payloads as ``[B,H,N,R,C]``."""
    batch, candidate_count, horizon, nodes, channels = candidate_future.shape
    query_endpoint, query_endpoint_valid = _endpoint_level_from_context(
        query_context,
        query_context_valid,
    )
    flat_context = candidate_contexts_tensor.reshape(
        batch * candidate_count,
        candidate_contexts_tensor.shape[2],
        nodes,
        channels,
    )
    flat_valid = candidate_context_valid.reshape_as(flat_context)
    flat_endpoint, flat_endpoint_valid = _endpoint_level_from_context(flat_context, flat_valid)
    candidate_endpoint = flat_endpoint.reshape(batch, candidate_count, nodes, channels)
    candidate_endpoint_valid = flat_endpoint_valid.reshape(batch, candidate_count, nodes, channels)
    decay = torch.linspace(
        1.0,
        0.0,
        horizon,
        dtype=candidate_future.dtype,
        device=candidate_future.device,
    ).view(1, horizon, 1, 1, 1)
    future = candidate_future.permute(0, 2, 3, 1, 4).contiguous()
    query_level = query_endpoint.unsqueeze(1).unsqueeze(3)
    candidate_level = candidate_endpoint.permute(0, 2, 1, 3).unsqueeze(1)
    payload = future + decay * (query_level - candidate_level)
    valid = candidate_valid.permute(0, 2, 3, 1, 4).contiguous()
    valid &= query_endpoint_valid.unsqueeze(1).unsqueeze(3)
    valid &= candidate_endpoint_valid.permute(0, 2, 1, 3).unsqueeze(1)
    return torch.where(valid, payload, torch.zeros_like(payload)), valid


def _aggregate_uniform_topk(
    payload: torch.Tensor,
    payload_valid: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate selected payloads; payload is ``[B,H,N,R,C]``."""
    batch, horizon, nodes, _, channels = payload.shape
    k = indices.shape[-1]
    gather = indices.unsqueeze(1).unsqueeze(-1).expand(batch, horizon, nodes, k, channels)
    selected = payload.gather(3, gather)
    selected_valid = payload_valid.gather(3, gather)
    weights = selected_valid.to(payload.dtype)
    denominator = weights.sum(dim=3)
    prediction = (selected * weights).sum(dim=3) / denominator.clamp_min(1.0)
    valid = denominator > 0
    return torch.where(valid, prediction, torch.zeros_like(prediction)), valid


@torch.no_grad()
def _batch_relation_asymmetry(
    model: Any,
    batch: dict[str, torch.Tensor],
    graph: Any,
    teacher_temperature: float,
    student_temperature: float,
) -> dict[str, float | int]:
    """Compare AnchorMean and SymNorm teachers on one source validation batch."""
    encoding = model.encode_clean(
        batch["retrieval_x"],
        batch["retrieval_observed"],
        batch["retrieval_weekday"],
        batch["retrieval_slot"],
        graph,
    )
    raw_targets = build_future_relation_targets(
        future_model=batch["y"],
        context_statistics=encoding.statistics,
        future_observed=batch["y_observed"],
        context_start=batch["context_start"],
        future_end=batch["future_end"],
        teacher_temperature=teacher_temperature,
        relation_teacher_mode="offset_decay",
        forecast_context=batch["x"],
        forecast_context_observed=batch["x_observed"],
        relation_distance_normalization="none",
    )
    anchor_distance = anchor_mean_normalize_distances(
        raw_targets.future_distance,
        raw_targets.candidate_mask,
    )
    symnorm_distance, symnorm_mask = symmetric_geometric_mean_normalize_distances(
        raw_targets.future_distance,
        raw_targets.candidate_mask,
    )
    anchor_teacher_distribution = _masked_softmax(
        -anchor_distance / teacher_temperature,
        raw_targets.candidate_mask,
        raw_targets.valid_anchors,
    )
    symnorm_valid_anchors = symnorm_mask.sum(dim=1) >= 2
    symnorm_teacher_distribution = _masked_softmax(
        -symnorm_distance / teacher_temperature,
        symnorm_mask,
        symnorm_valid_anchors,
    )
    keys = functional.normalize(encoding.retrieval.node_keys, dim=-1)
    student_logits = torch.einsum("ind,jnd->ijn", keys, keys) / student_temperature
    student_distribution = _masked_softmax(
        student_logits,
        raw_targets.candidate_mask,
        raw_targets.valid_anchors,
    )
    pair_valid = raw_targets.candidate_mask & raw_targets.candidate_mask.transpose(0, 1)
    symnorm_pair_valid = symnorm_mask & symnorm_mask.transpose(0, 1)
    anchor_teacher_asym = distribution_asymmetry(
        anchor_teacher_distribution,
        pair_valid,
    )
    symnorm_teacher_asym = distribution_asymmetry(
        symnorm_teacher_distribution,
        symnorm_pair_valid,
    )
    student_asym = distribution_asymmetry(student_distribution, pair_valid)
    anchor_teacher_logits = -anchor_distance / teacher_temperature
    symnorm_teacher_logits = -symnorm_distance / teacher_temperature
    pair_upper = torch.triu(
        torch.ones_like(pair_valid, dtype=torch.bool), diagonal=1
    )
    logit_valid = pair_valid & pair_upper
    symnorm_logit_valid = symnorm_pair_valid & pair_upper
    anchor_distance_asym = _masked_mean(
        (anchor_distance - anchor_distance.transpose(0, 1)).abs(),
        logit_valid,
    )
    symnorm_distance_asym = _masked_mean(
        (symnorm_distance - symnorm_distance.transpose(0, 1)).abs(),
        symnorm_logit_valid,
    )
    anchor_teacher_logit_asym = _masked_mean(
        (anchor_teacher_logits - anchor_teacher_logits.transpose(0, 1)).abs(),
        logit_valid,
    )
    symnorm_teacher_logit_asym = _masked_mean(
        (symnorm_teacher_logits - symnorm_teacher_logits.transpose(0, 1)).abs(),
        symnorm_logit_valid,
    )
    student_logit_asym = _masked_mean(
        (student_logits - student_logits.transpose(0, 1)).abs(),
        logit_valid,
    )
    anchor_teacher_support = 1.0 / anchor_teacher_distribution.square().sum(dim=1).clamp_min(1.0e-12)
    symnorm_teacher_support = 1.0 / symnorm_teacher_distribution.square().sum(dim=1).clamp_min(1.0e-12)
    student_support = 1.0 / student_distribution.square().sum(dim=1).clamp_min(1.0e-12)
    valid_anchor = raw_targets.valid_anchors
    return {
        "anchor_mean_teacher_probability_asymmetry": anchor_teacher_asym["mean_absolute_asymmetry"],
        "symnorm_teacher_probability_asymmetry": symnorm_teacher_asym["mean_absolute_asymmetry"],
        "student_probability_asymmetry": student_asym["mean_absolute_asymmetry"],
        "anchor_mean_distance_asymmetry": anchor_distance_asym,
        "symnorm_distance_asymmetry": symnorm_distance_asym,
        "anchor_mean_teacher_logit_asymmetry": anchor_teacher_logit_asym,
        "symnorm_teacher_logit_asymmetry": symnorm_teacher_logit_asym,
        "student_logit_asymmetry": student_logit_asym,
        "anchor_mean_teacher_effective_support": _masked_mean(
            anchor_teacher_support,
            valid_anchor,
        ),
        "symnorm_teacher_effective_support": _masked_mean(
            symnorm_teacher_support,
            symnorm_valid_anchors,
        ),
        "student_effective_support": _masked_mean(student_support, valid_anchor),
        "valid_pair_count": anchor_teacher_asym["valid_pair_count"],
        "valid_anchor_count": int(valid_anchor.sum().item()),
    }


def _metric_accumulator(horizon: int) -> dict[str, ForecastMetricAccumulator]:
    return {
        name: ForecastMetricAccumulator(horizon)
        for name in (
            "od_top1",
            "od_top5",
            "symnorm_od_top1",
            "symnorm_od_top5",
            "clipped_od_top1",
            "clipped_od_top5",
            "od_increment_top1",
            "od_increment_top5",
        )
    }


@torch.no_grad()
def run_teacher_metric_diagnostic(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    split: str,
    output_dir: str | Path,
    candidate_protocol: str = "relaxed_calendar",
    max_batches: int | None = None,
    clip_delta: float | None = None,
    perturb_rate: float = 0.1,
) -> dict[str, Any]:
    """Run the fixed-bank, no-training teacher metric diagnostic."""
    if split != "val":
        raise ValueError("teacher metric diagnostic is restricted to validation")
    if candidate_protocol != "relaxed_calendar":
        raise ValueError("the first diagnostic is fixed to relaxed_calendar")
    started = time.perf_counter()
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    dataset: Dataset = data.val
    loader = DataLoader(
        dataset,
        batch_size=config.target.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    checkpoint_path = resolve_project_path(checkpoint_path)
    bank_path = resolve_project_path(bank_path)
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_pretrained_model(
        config,
        checkpoint_path,
        data.series.slots_per_day,
        device,
    )
    model.eval()

    accumulators = _metric_accumulator(config.data.horizon)
    alignment_distances: dict[str, list[np.ndarray]] = {
        "od": [],
        "symnorm_od": [],
        "clipped_od": [],
        "od_increment": [],
    }
    alignment_keys: dict[str, list[np.ndarray]] = {name: [] for name in alignment_distances}
    recall_values: dict[str, list[np.ndarray]] = {name: [] for name in alignment_distances}
    candidate_counts: list[np.ndarray] = []
    valid_point_counts: list[np.ndarray] = []
    support_values: dict[str, list[np.ndarray]] = {name: [] for name in alignment_distances}
    perturb_jaccard: dict[str, list[np.ndarray]] = {name: [] for name in alignment_distances}
    perturb_tv: dict[str, list[np.ndarray]] = {name: [] for name in alignment_distances}
    collision_total = 0
    near_tie_total = 0
    asymmetry_values: list[dict[str, float | int]] = []
    asymmetry_batches = 0
    asymmetry_skipped_batches = 0
    query_count = 0
    batch_count = 0

    with MemoryBank(bank_path) as bank:
        _validate_bank(bank, model, graph_cpu, data.scaler.state_dict())
        effective_clip_delta = (
            float(clip_delta)
            if clip_delta is not None
            else estimate_clip_delta(bank)
        )
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = {name: value.to(device) for name, value in raw_batch.items() if torch.is_tensor(value)}
            encoding = model.encode_clean(
                batch["retrieval_x"],
                batch["retrieval_observed"],
                batch["retrieval_weekday"],
                batch["retrieval_slot"],
                graph,
            )
            events = build_diagnostic_event_candidates(
                bank,
                batch["query_weekday"],
                batch["query_slot"],
                batch["context_start"],
                config.bank.event_top_r,
                device,
                candidate_protocol,
            )
            candidate_counts.append(events.valid.sum(dim=-1).detach().cpu().numpy())
            candidate_keys = _candidate_node_keys(bank, events.event_ids, device)
            key_distance, key_valid = node_key_distances(
                encoding.retrieval.node_keys,
                candidate_keys,
                events.valid,
            )
            event_future, event_future_valid = event_candidate_futures(
                bank,
                events.event_ids,
                events.valid,
                device,
            )
            candidate_future = event_future.permute(0, 3, 1, 2, 4).contiguous()
            candidate_future_valid = event_future_valid.permute(0, 3, 1, 2, 4).contiguous()
            candidate_context_tensor, candidate_context_valid = candidate_contexts(
                bank,
                events.event_ids,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            query_signature, query_signature_valid = build_offset_decay_signature(
                batch["y"],
                batch["y_observed"],
                batch["x"],
                batch["x_observed"],
            )
            candidate_signature, candidate_signature_valid, candidate_increment, candidate_increment_valid = _candidate_signatures(
                bank,
                events,
                candidate_future,
                candidate_future_valid,
                data,
                config.data.context_length,
                device,
            )
            query_endpoint, query_endpoint_valid = _endpoint_level_from_context(
                batch["x"], batch["x_observed"]
            )
            query_increment, query_increment_valid = build_future_increment(
                batch["y"],
                batch["y_observed"],
                query_endpoint,
                query_endpoint_valid,
            )
            od_distance, od_valid = candidate_distance(
                query_signature,
                query_signature_valid,
                candidate_signature,
                candidate_signature_valid,
                events.valid,
            )
            clipped_distance, clipped_valid = candidate_distance(
                query_signature,
                query_signature_valid,
                candidate_signature,
                candidate_signature_valid,
                events.valid,
                clip_delta=effective_clip_delta,
            )
            increment_distance, increment_valid = candidate_distance(
                query_increment,
                query_increment_valid,
                candidate_increment,
                candidate_increment_valid,
                events.valid,
            )
            mix_distance, mix_valid = _normalize_mix(
                od_distance,
                od_valid,
                increment_distance,
                increment_valid,
            )
            symnorm_distance, symnorm_valid = symmetric_candidate_distance(
                query_signature,
                query_signature_valid,
                candidate_signature,
                candidate_signature_valid,
                events.valid,
            )
            od_distance = normalize_candidate_distance(od_distance, od_valid)
            clipped_distance = normalize_candidate_distance(clipped_distance, clipped_valid)
            method_distances = {
                "od": (od_distance, od_valid),
                "symnorm_od": (symnorm_distance, symnorm_valid),
                "clipped_od": (clipped_distance, clipped_valid),
                "od_increment": (mix_distance, mix_valid),
            }

            payload, payload_valid = _offset_decay_payloads(
                candidate_future,
                candidate_future_valid,
                candidate_context_tensor,
                candidate_context_valid,
                batch["x"],
                batch["x_observed"],
            )
            target_physical = data.scaler.inverse_transform_torch(batch["y"])
            target_valid = batch["y_observed"].bool()
            for name, (distance, valid) in method_distances.items():
                top1, top1_eligible = topk_indices(distance, valid, 1)
                top5, top5_eligible = topk_indices(distance, valid, min(5, distance.shape[-1]))
                pred_top1, pred_top1_valid = _aggregate_uniform_topk(
                    payload, payload_valid, top1
                )
                pred_top5, pred_top5_valid = _aggregate_uniform_topk(
                    payload, payload_valid, top5
                )
                pred_top1_physical = data.scaler.inverse_transform_torch(pred_top1)
                pred_top5_physical = data.scaler.inverse_transform_torch(pred_top5)
                accumulators[f"{name}_top1"].update(
                    pred_top1_physical,
                    target_physical,
                    target_valid & pred_top1_valid,
                )
                accumulators[f"{name}_top5"].update(
                    pred_top5_physical,
                    target_physical,
                    target_valid & pred_top5_valid,
                )
                alignment_distances[name].append(
                    distance.masked_select(valid & key_valid).detach().cpu().numpy()
                )
                alignment_keys[name].append(
                    key_distance.masked_select(valid & key_valid).detach().cpu().numpy()
                )
                recall, eligible = future_neighbor_recall_at_k(
                    key_distance,
                    distance,
                    valid & key_valid,
                    k=min(5, distance.shape[-1]),
                )
                recall_values[name].append(recall.masked_select(eligible).detach().cpu().numpy())
                support_values[name].append(
                    effective_support_from_distance(distance, valid, config.pretrain.relation_teacher_temperature)
                    .masked_select(valid.any(dim=-1))
                    .detach()
                    .cpu()
                    .numpy()
                )

            point_count = (
                query_signature_valid[:, None].bool() & candidate_signature_valid.bool()
            ).sum(dim=(2, 4)).permute(0, 2, 1)
            valid_point_counts.append(point_count.masked_select(od_valid).detach().cpu().numpy())
            collisions, near_ties = collision_pair_counts(
                od_distance,
                normalize_candidate_distance(increment_distance, mix_valid),
                mix_valid,
            )
            collision_total += collisions
            near_tie_total += near_ties

            perturbed_future, perturb_mask = _perturb_candidate_future(
                candidate_future,
                candidate_future_valid,
                events.event_ids,
                effective_clip_delta,
                perturb_rate,
            )
            perturbed_signature, perturbed_signature_valid, perturbed_increment, perturbed_increment_valid = _candidate_signatures(
                bank,
                events,
                perturbed_future,
                candidate_future_valid,
                data,
                config.data.context_length,
                device,
            )
            perturbed_od, perturbed_od_valid = candidate_distance(
                query_signature,
                query_signature_valid,
                perturbed_signature,
                perturbed_signature_valid,
                events.valid,
            )
            perturbed_clip, perturbed_clip_valid = candidate_distance(
                query_signature,
                query_signature_valid,
                perturbed_signature,
                perturbed_signature_valid,
                events.valid,
                clip_delta=effective_clip_delta,
            )
            perturbed_inc, perturbed_inc_valid = candidate_distance(
                query_increment,
                query_increment_valid,
                perturbed_increment,
                perturbed_increment_valid,
                events.valid,
            )
            perturbed_mix, perturbed_mix_valid = _normalize_mix(
                perturbed_od,
                perturbed_od_valid,
                perturbed_inc,
                perturbed_inc_valid,
            )
            perturbed_symnorm, perturbed_symnorm_valid = symmetric_candidate_distance(
                query_signature,
                query_signature_valid,
                perturbed_signature,
                perturbed_signature_valid,
                events.valid,
            )
            perturbed_methods = {
                "od": (normalize_candidate_distance(perturbed_od, perturbed_od_valid), perturbed_od_valid),
                "symnorm_od": (perturbed_symnorm, perturbed_symnorm_valid),
                "clipped_od": (normalize_candidate_distance(perturbed_clip, perturbed_clip_valid), perturbed_clip_valid),
                "od_increment": (perturbed_mix, perturbed_mix_valid),
            }
            for name, (distance, valid) in method_distances.items():
                clean_top5, clean_eligible = topk_indices(distance, valid, min(5, distance.shape[-1]))
                perturbed_distance, perturbed_valid = perturbed_methods[name]
                perturbed_top5, perturbed_eligible = topk_indices(
                    perturbed_distance,
                    perturbed_valid,
                    min(5, distance.shape[-1]),
                )
                eligible = clean_eligible & perturbed_eligible
                perturb_jaccard[name].append(
                    topk_jaccard(clean_top5, perturbed_top5).masked_select(eligible).detach().cpu().numpy()
                )
                clean_prob = distance_distribution(distance, valid, config.pretrain.relation_teacher_temperature)
                perturbed_prob = distance_distribution(
                    perturbed_distance,
                    perturbed_valid,
                    config.pretrain.relation_teacher_temperature,
                )
                perturb_tv[name].append(
                    (0.5 * (clean_prob - perturbed_prob).abs().sum(dim=-1))
                    .masked_select(eligible)
                    .detach()
                    .cpu()
                    .numpy()
                )

            query_count += int(batch["y"].shape[0])
            batch_count += 1
            if batch_count == 1 or batch_count % 10 == 0:
                print(
                    f"[teacher-diagnostic/{candidate_protocol}] processed "
                    f"{query_count}/{len(dataset)} validation queries ({batch_count} batches)",
                    flush=True,
                )

        relation_loader = build_validation_loader(
            dataset,
            config.pretrain.batch_size,
            config.data.num_workers,
            config.runtime.seed,
        )
        for relation_batch_index, raw_relation_batch in enumerate(relation_loader):
            if max_batches is not None and relation_batch_index >= max_batches:
                break
            relation_batch = {
                name: value.to(device)
                for name, value in raw_relation_batch.items()
                if torch.is_tensor(value)
            }
            try:
                asymmetry = _batch_relation_asymmetry(
                    model,
                    relation_batch,
                    graph,
                    config.pretrain.relation_teacher_temperature,
                    config.pretrain.relation_student_temperature,
                )
            except ValueError as error:
                if "valid symmetric pair" not in str(error):
                    raise
                asymmetry_skipped_batches += 1
                continue
            asymmetry_values.append(asymmetry)
            asymmetry_batches += 1

    if not asymmetry_values:
        raise ValueError("no source-validation batch contained a valid symmetric relation pair")

    results: dict[str, Any] = {
        "schema_version": 2,
        "diagnostic": "e5a_zero_training_teacher_metric",
        "dataset": str(config.data.raw_path),
        "split": split,
        "complete_validation": max_batches is None,
        "queries": query_count,
        "batches": batch_count,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "bank": str(bank_path),
        "candidate_protocol": candidate_protocol,
        "distance_normalization": {
            "od": {
                "name": "AnchorMean",
                "computation": "d_ij / (mu_i + eps)",
                "candidate_scale_source": "query-to-bank-candidates",
            },
            "symnorm_od": {
                "name": "SymmetricGeometricMeanNormalization",
                "computation": "d_ij / sqrt((mu_i + eps) * (mu_j + eps))",
                "candidate_scale_source": "pairwise OD-MAE on each query plus its shared Bank candidate event set",
            },
        },
        "candidate_pool": _summary(np.concatenate(candidate_counts)),
        "clip_delta": effective_clip_delta,
        "perturbation": {
            "rate": perturb_rate,
            "horizon": config.data.horizon // 2,
            "magnitude": 3.0 * effective_clip_delta,
            "query_future_used_for_ranking": False,
            "query_future_used_for_offline_teacher_metrics": True,
        },
        "valid_point_count": _summary(np.concatenate(valid_point_counts)),
        "collision": {
            "near_tie_pairs": near_tie_total,
            "collision_pairs": collision_total,
            "collision_rate": collision_total / max(near_tie_total, 1),
        },
        "teacher_support": {
            name: _summary(np.concatenate(values)) for name, values in support_values.items()
        },
        "perturbation_stability": {
            name: {
                "top5_jaccard": _summary(np.concatenate(perturb_jaccard[name])),
                "total_variation": _summary(np.concatenate(perturb_tv[name])),
            }
            for name in perturb_jaccard
        },
        "alignment": {},
        "memory_metrics": {name: accumulator.compute() for name, accumulator in accumulators.items()},
        "teacher_student_asymmetry": {
            "processed_batches": asymmetry_batches,
            "skipped_batches": asymmetry_skipped_batches,
            "metrics": {
                key: _summary([float(item[key]) for item in asymmetry_values])
                for key in asymmetry_values[0]
                if key != "valid_pair_count" and key != "valid_anchor_count"
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    for name in alignment_distances:
        alignment_key_array = np.concatenate(alignment_keys[name])
        distances = np.concatenate(alignment_distances[name])
        results["alignment"][name] = alignment_statistics(
            alignment_key_array,
            distances,
            np.ones_like(distances, dtype=bool),
        )
        recalls = np.concatenate(recall_values[name])
        results["alignment"][name]["future_neighbor_recall_at_5"] = float(recalls.mean()) if recalls.size else 0.0
        results["alignment"][name]["recall_at_5_eligible_anchors"] = int(recalls.size)

    output_path = resolve_project_path(output_dir)
    save_json(output_path / "metrics.json", results)
    with (output_path / "valid_point_count.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("count", "frequency"))
        counts = np.concatenate(valid_point_counts).astype(np.int64)
        for count in range(config.data.horizon + 1):
            writer.writerow((count, int((counts == count).sum())))
    return results

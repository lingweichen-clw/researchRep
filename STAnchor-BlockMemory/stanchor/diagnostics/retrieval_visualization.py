"""Teacher-aligned diagnostics for learned historical retrieval keys."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, default_collate

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.data.normalization import normalize_future_with_context, normalize_window
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.losses.pretraining import (
    anchor_mean_normalize_distances,
    build_offset_decay_signature,
)
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import AggregationOutput, EventCandidates, TwoStageRetriever
from stanchor.retrieval.strategies import (
    candidate_contexts,
    event_candidate_futures,
    offset_decay_aggregation,
)
from stanchor.utils import resolve_device, save_json


SUPPORTED_VERSIONS = {"e2", "e3", "e5a"}
SUPPORTED_CANDIDATE_PROTOCOLS = {
    "exact_calendar",
    "relaxed_calendar",
    "broad_causal",
}


def validate_aligned_bank_axes(
    pretrained_bank: Any,
    random_bank: Any,
) -> dict[str, Any]:
    """Require both Banks to expose the same event axis and non-key payload."""
    manifest_fields = ("num_events", "num_nodes", "horizon", "channels")
    for field in manifest_fields:
        left = getattr(pretrained_bank.manifest, field)
        right = getattr(random_bank.manifest, field)
        if left != right:
            raise ValueError(f"Bank manifest mismatch for {field}: {left} != {right}")
    axis_fields = (
        "sample_id",
        "weekday",
        "slot",
        "context_start",
        "context_end",
        "future_end",
        "future_masks",
        "future_values",
    )
    for field in axis_fields:
        if not np.array_equal(np.asarray(getattr(pretrained_bank, field)), np.asarray(getattr(random_bank, field))):
            raise ValueError(f"pretrained/random Bank mismatch for {field}")
    return {
        "same_event_axis": True,
        "same_future_payload": True,
        "checked_fields": list(axis_fields),
        "canonical_payload": "pretrained_bank",
    }


def build_teacher_aligned_signature(
    version: str,
    future: torch.Tensor,
    future_observed: torch.Tensor,
    context: torch.Tensor,
    context_observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the future representation used by one version's pretraining teacher."""
    version = version.lower()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"version must be one of {sorted(SUPPORTED_VERSIONS)}")
    if future.ndim != 4 or future_observed.shape != future.shape:
        raise ValueError("future and future_observed must be [B, H, N, C]")
    if context.ndim != 4 or context_observed.shape != context.shape:
        raise ValueError("context and context_observed must be [B, T, N, C]")
    if future.shape[0] != context.shape[0] or future.shape[2:] != context.shape[2:]:
        raise ValueError("future and context batch/node/channel dimensions must align")

    if version in {"e2", "e3"}:
        context_statistics = normalize_window(context, context_observed)
        signature = normalize_future_with_context(future, context_statistics)
        valid = future_observed.bool() & torch.isfinite(signature)
        return torch.where(valid, signature, torch.zeros_like(signature)), valid
    return build_offset_decay_signature(
        future,
        future_observed,
        context,
        context_observed,
    )


def masked_candidate_future_mae(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare query and candidate signatures, returning distances as [B, N, R]."""
    if query.ndim != 4 or query_observed.shape != query.shape:
        raise ValueError("query and query_observed must be [B, H, N, C]")
    if candidates.ndim != 5 or candidate_observed.shape != candidates.shape:
        raise ValueError("candidates and candidate_observed must be [B, R, H, N, C]")
    batch, candidate_count, horizon, nodes, channels = candidates.shape
    if query.shape != (batch, horizon, nodes, channels):
        raise ValueError("query and candidate future dimensions must align")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B, R]")

    common = query_observed[:, None].bool() & candidate_observed.bool()
    common &= torch.isfinite(query[:, None]) & torch.isfinite(candidates)
    count = common.sum(dim=(2, 4))  # [B, R, N]
    absolute = (query[:, None] - candidates).abs()
    total = torch.where(common, absolute, torch.zeros_like(absolute)).sum(dim=(2, 4))
    valid = (count > 0) & event_valid[:, :, None].bool()
    distance = total / count.clamp_min(1)
    distance = distance.permute(0, 2, 1).contiguous()
    valid = valid.permute(0, 2, 1).contiguous()
    return torch.where(valid, distance, torch.zeros_like(distance)), valid


def _symmetric_candidate_distance(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
    eps: float = 1.0e-6,
    chunk_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize query-candidate MAE by query and candidate event scales."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    values = torch.cat((query.unsqueeze(1), candidates), dim=1)
    observed = torch.cat((query_observed.unsqueeze(1), candidate_observed), dim=1)
    batch, event_count, _, nodes, _ = values.shape
    all_event_valid = torch.cat(
        (
            torch.ones((batch, 1), dtype=torch.bool, device=query.device),
            event_valid.bool(),
        ),
        dim=1,
    )
    pair_distance = torch.zeros(
        (batch, event_count, event_count, nodes),
        dtype=query.dtype,
        device=query.device,
    )
    pair_valid = torch.zeros_like(pair_distance, dtype=torch.bool)
    right = values[:, None]
    right_observed = observed[:, None].bool()
    for start in range(0, event_count, chunk_size):
        end = min(start + chunk_size, event_count)
        left = values[:, start:end, None]
        common = observed[:, start:end, None].bool() & right_observed
        common &= torch.isfinite(left) & torch.isfinite(right)
        counts = common.sum(dim=(3, 5))  # [B, chunk, E, N]
        totals = torch.where(
            common,
            (left - right).abs(),
            torch.zeros_like(left),
        ).sum(dim=(3, 5))
        valid_chunk = (counts > 0) & all_event_valid[:, start:end, None, None]
        valid_chunk &= all_event_valid[:, None, :, None]
        pair_distance[:, start:end] = torch.where(
            valid_chunk,
            totals / counts.clamp_min(1),
            torch.zeros_like(totals),
        )
        pair_valid[:, start:end] = valid_chunk
    pair_valid &= ~torch.eye(event_count, dtype=torch.bool, device=query.device).view(
        1, event_count, event_count, 1
    )
    scale_count = pair_valid.sum(dim=2)
    scale = torch.where(
        pair_valid,
        pair_distance,
        torch.zeros_like(pair_distance),
    ).sum(dim=2) / scale_count.clamp_min(1).to(pair_distance.dtype)
    denominator = torch.sqrt(
        (scale.unsqueeze(2) + eps) * (scale.unsqueeze(1) + eps)
    )
    normalized = pair_distance / denominator.clamp_min(eps)
    valid = pair_valid & (scale_count > 0).unsqueeze(2) & (scale_count > 0).unsqueeze(1)
    return (
        torch.where(valid[:, 0, 1:], normalized[:, 0, 1:], torch.zeros_like(normalized[:, 0, 1:]))
        .permute(0, 2, 1)
        .contiguous(),
        valid[:, 0, 1:].permute(0, 2, 1).contiguous(),
    )


def teacher_candidate_distances(
    query: torch.Tensor,
    query_observed: torch.Tensor,
    candidates: torch.Tensor,
    candidate_observed: torch.Tensor,
    event_valid: torch.Tensor,
    normalization: str,
    symmetric_chunk_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute candidate future distances using the configured teacher metric."""
    if normalization == "symmetric_geometric_mean":
        return _symmetric_candidate_distance(
            query,
            query_observed,
            candidates,
            candidate_observed,
            event_valid,
            chunk_size=symmetric_chunk_size,
        )
    distance, valid = masked_candidate_future_mae(
        query,
        query_observed,
        candidates,
        candidate_observed,
        event_valid,
    )
    if normalization == "anchor_mean":
        distance = anchor_mean_normalize_distances(
            distance.permute(0, 2, 1),
            valid.permute(0, 2, 1),
        ).permute(0, 2, 1)
    elif normalization != "none":
        raise ValueError(f"unknown teacher distance normalization: {normalization}")
    return distance, valid


def node_key_distances(
    query_keys: torch.Tensor,
    candidate_keys: torch.Tensor,
    event_valid: torch.Tensor,
    profile_dim: int | None = None,
    profile_weight: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-minus-cosine node-key distances as [B, N, R]."""
    if query_keys.ndim != 3:
        raise ValueError("query_keys must be [B, N, D]")
    if candidate_keys.ndim != 4:
        raise ValueError("candidate_keys must be [B, R, N, D]")
    batch, candidate_count, nodes, dimension = candidate_keys.shape
    if query_keys.shape != (batch, nodes, dimension):
        raise ValueError("query and candidate key dimensions must align")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B, R]")

    if (profile_dim is None) != (profile_weight is None):
        raise ValueError("profile_dim and profile_weight must be provided together")
    if profile_weight is None:
        query = torch.nn.functional.normalize(query_keys.float(), dim=-1)
        candidates = torch.nn.functional.normalize(candidate_keys.float(), dim=-1)
        cosine = torch.einsum("bnd,brnd->bnr", query, candidates)
    else:
        if profile_dim is None or profile_dim <= 0 or profile_dim >= dimension:
            raise ValueError("profile_dim must split the profile and latent key dimensions")
        if not 0.0 <= profile_weight <= 1.0:
            raise ValueError("profile_weight must be in [0, 1]")
        query_profile = torch.nn.functional.normalize(
            query_keys[..., :profile_dim].float(), dim=-1
        )
        candidate_profile = torch.nn.functional.normalize(
            candidate_keys[..., :profile_dim].float(), dim=-1
        )
        query_latent = torch.nn.functional.normalize(
            query_keys[..., profile_dim:].float(), dim=-1
        )
        candidate_latent = torch.nn.functional.normalize(
            candidate_keys[..., profile_dim:].float(), dim=-1
        )
        profile_cosine = torch.einsum(
            "bnd,brnd->bnr", query_profile, candidate_profile
        )
        latent_cosine = torch.einsum(
            "bnd,brnd->bnr", query_latent, candidate_latent
        )
        cosine = profile_weight * profile_cosine + (1.0 - profile_weight) * latent_cosine
    cosine = cosine.clamp(-1.0, 1.0)
    valid = event_valid[:, None, :].expand(batch, nodes, candidate_count).bool()
    distance = (1.0 - cosine).clamp(0.0, 2.0)
    return torch.where(valid, distance, torch.zeros_like(distance)), valid


def memory_mae_by_anchor(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average absolute error across horizon/channel while preserving [B, N]."""
    if prediction.ndim != 4 or target.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError("prediction, target, and valid must be [B, H, N, C]")
    mask = valid.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    count = mask.sum(dim=(1, 3))
    total = torch.where(mask, (prediction - target).abs(), torch.zeros_like(prediction)).sum(
        dim=(1, 3)
    )
    anchor_valid = count > 0
    mae = total / count.clamp_min(1)
    return torch.where(anchor_valid, mae, torch.zeros_like(mae)), anchor_valid


def complete_anchor_mask(valid: torch.Tensor) -> torch.Tensor:
    """Return [B,N] anchors with an observed value at every horizon/channel."""
    if valid.ndim != 4:
        raise ValueError("valid must be [B, H, N, C]")
    return valid.bool().all(dim=(1, 3))


def future_neighbor_recall_at_k(
    key_distance: torch.Tensor,
    future_distance: torch.Tensor,
    valid: torch.Tensor,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-anchor future-neighbor Recall@K and its non-trivial eligibility mask."""
    if key_distance.ndim != 3 or future_distance.shape != key_distance.shape:
        raise ValueError("key_distance and future_distance must be [B, N, R]")
    if valid.shape != key_distance.shape:
        raise ValueError("valid must match the distance tensors")
    if k <= 0 or k > key_distance.shape[-1]:
        raise ValueError("k must be in [1, R]")

    finite_valid = (
        valid.bool()
        & torch.isfinite(key_distance)
        & torch.isfinite(future_distance)
    )
    eligible = finite_valid.sum(dim=-1) > k
    infinity = torch.full_like(key_distance, torch.inf)
    key_order = torch.topk(
        torch.where(finite_valid, key_distance, infinity),
        k,
        dim=-1,
        largest=False,
    ).indices
    future_order = torch.topk(
        torch.where(finite_valid, future_distance, infinity),
        k,
        dim=-1,
        largest=False,
    ).indices
    overlap = (key_order.unsqueeze(-1) == future_order.unsqueeze(-2)).any(dim=-1)
    recall = overlap.to(key_distance.dtype).sum(dim=-1) / float(k)
    return torch.where(eligible, recall, torch.zeros_like(recall)), eligible


def alignment_statistics(
    key_distance: np.ndarray,
    future_distance: np.ndarray,
    valid: np.ndarray,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Summarize global rank alignment and equal-frequency distance bins."""
    key = np.asarray(key_distance, dtype=np.float64).reshape(-1)
    future = np.asarray(future_distance, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if key.shape != future.shape or key.shape != mask.shape:
        raise ValueError("key_distance, future_distance, and valid must align")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    mask &= np.isfinite(key) & np.isfinite(future)
    key = key[mask]
    future = future[mask]
    if key.size < 2:
        raise ValueError("at least two valid distance pairs are required")

    statistic = float(spearmanr(key, future).statistic)
    order = np.argsort(key, kind="stable")
    distance_bins = []
    for index, selected in enumerate(np.array_split(order, num_bins), start=1):
        if selected.size == 0:
            continue
        key_values = key[selected]
        future_values = future[selected]
        distance_bins.append(
            {
                "bin": index,
                "count": int(selected.size),
                "key_distance_min": float(key_values.min()),
                "key_distance_max": float(key_values.max()),
                "key_distance_mean": float(key_values.mean()),
                "future_distance_mean": float(future_values.mean()),
                "future_distance_median": float(np.median(future_values)),
            }
        )
    return {
        "valid_pairs": int(key.size),
        "spearman": statistic,
        "distance_bins": distance_bins,
    }


def select_quantile_cases(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select strong, representative, and failure cases by fixed gain quantiles."""
    if not records:
        raise ValueError("records must not be empty")
    quantiles = {
        "strong_win": 0.9,
        "representative": 0.5,
        "failure": 0.1,
    }
    gains = np.asarray([float(record["mae_gain"]) for record in records], dtype=np.float64)
    if not np.isfinite(gains).all():
        raise ValueError("case gains must be finite")

    chosen: dict[str, Any] = {}
    used: set[tuple[int, int]] = set()
    for name, quantile in quantiles.items():
        target = float(np.quantile(gains, quantile))
        order = sorted(
            range(len(records)),
            key=lambda index: (
                abs(gains[index] - target),
                int(records[index]["sample_id"]),
                int(records[index]["node_id"]),
            ),
        )
        selected_index = next(
            index
            for index in order
            if (int(records[index]["sample_id"]), int(records[index]["node_id"]))
            not in used
        )
        selected = dict(records[selected_index])
        selected["target_quantile"] = quantile
        selected["target_gain"] = target
        chosen[name] = selected
        used.add((int(selected["sample_id"]), int(selected["node_id"])))
    chosen["selection_rule"] = {
        "score": "random_memory_mae_minus_pretrained_memory_mae",
        "quantiles": quantiles,
        "tie_break": "smallest_absolute_gap_then_sample_id_then_node_id",
        "manual_selection": False,
    }
    return chosen


def future_information_boundary() -> dict[str, Any]:
    """Return auditable metadata describing where query future is permitted."""
    return {
        "evaluation_split": "validation_only",
        "ranking_inputs": [
            "query_history",
            "query_calendar",
            "causal_bank_metadata",
            "historical_bank_keys",
            "historical_bank_levels",
        ],
        "query_future_used_for_ranking": False,
        "query_future_use": "post-ranking metrics, deterministic case selection, and plots only",
        "deployment_available_inputs_only_before_ranking": True,
    }


def build_diagnostic_event_candidates(
    bank: MemoryBank,
    weekday: torch.Tensor,
    slot: torch.Tensor,
    context_start: torch.Tensor,
    max_candidates: int,
    device: torch.device,
    protocol: str = "exact_calendar",
) -> EventCandidates:
    """Build a shared, causal candidate pool for protocol attribution.

    ``broad_causal`` uses deterministic chronological quantile sampling when the
    causal Bank is larger than ``max_candidates``. The sampling is independent
    of either model's key, so pretrained/random compare the same event axis.
    """
    protocol = protocol.lower()
    if protocol not in SUPPORTED_CANDIDATE_PROTOCOLS:
        raise ValueError(
            f"candidate protocol must be one of {sorted(SUPPORTED_CANDIDATE_PROTOCOLS)}"
        )
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if weekday.ndim != 1 or slot.shape != weekday.shape or context_start.shape != weekday.shape:
        raise ValueError("weekday, slot, and context_start must be [B]")

    batch = int(weekday.shape[0])
    ids = torch.full((batch, max_candidates), -1, dtype=torch.long, device=device)
    scores = torch.full((batch, max_candidates), -torch.inf, dtype=torch.float32, device=device)
    valid = torch.zeros((batch, max_candidates), dtype=torch.bool, device=device)
    future_end = np.asarray(bank.future_end)
    slots_per_day = int(bank.manifest.slots_per_day)

    for batch_index in range(batch):
        query_weekday = int(weekday[batch_index].item())
        query_slot = int(slot[batch_index].item())
        query_start = int(context_start[batch_index].item())
        if protocol == "broad_causal":
            legal = np.flatnonzero(future_end < query_start).astype(np.int64)
            if legal.size > max_candidates:
                positions = np.rint(
                    np.linspace(0, legal.size - 1, max_candidates)
                ).astype(np.int64)
                legal = legal[positions]
        else:
            radius = 0 if protocol == "exact_calendar" else 1
            collected: list[int] = []
            seen: set[int] = set()
            for offset in range(-radius, radius + 1):
                candidate_slot = query_slot + offset
                if not 0 <= candidate_slot < slots_per_day:
                    continue
                calendar_ids = np.asarray(
                    bank.calendar.lookup(query_weekday, candidate_slot),
                    dtype=np.int64,
                )
                for event_id in calendar_ids.tolist():
                    event_id = int(event_id)
                    if event_id in seen or future_end[event_id] >= query_start:
                        continue
                    seen.add(event_id)
                    collected.append(event_id)
            legal = np.asarray(collected, dtype=np.int64)
            if legal.size > max_candidates:
                raise ValueError(
                    f"{protocol} candidate pool ({legal.size}) exceeds max_candidates={max_candidates}"
                )
        if legal.size == 0:
            continue
        count = int(legal.size)
        ids[batch_index, :count] = torch.from_numpy(legal).to(device)
        scores[batch_index, :count] = 0.0
        valid[batch_index, :count] = True
    return EventCandidates(ids, scores, valid)


def _candidate_node_keys(
    bank: MemoryBank,
    event_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    return torch.from_numpy(
        np.asarray(bank.node_keys[safe_ids], dtype=np.float32)
    ).to(device)  # [B, R, N, D]


def _candidate_teacher_signatures(
    version: str,
    bank: MemoryBank,
    events: EventCandidates,
    candidate_future: torch.Tensor,
    candidate_observed: torch.Tensor,
    data: Any,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build candidate signatures as [B, R, H, N, C]."""
    batch, candidate_count, horizon, nodes, channels = candidate_future.shape
    if version in {"e2", "e3"}:
        safe_ids = events.event_ids.clamp_min(0).cpu().numpy()
        levels = torch.from_numpy(
            np.asarray(bank.level_features[safe_ids], dtype=np.float32)
        ).to(device)
        context_mean = levels[..., :channels]
        context_std = levels[..., channels : 2 * channels]
        signature = (
            candidate_future - context_mean[:, :, None]
        ) / (context_std[:, :, None] + 1.0e-6)
        valid = candidate_observed.bool() & torch.isfinite(signature)
        return torch.where(valid, signature, torch.zeros_like(signature)), valid

    contexts, context_valid = candidate_contexts(
        bank,
        events.event_ids,
        data.series,
        data.scaler,
        context_length,
        device,
    )
    signature, valid = build_teacher_aligned_signature(
        version,
        candidate_future.reshape(batch * candidate_count, horizon, nodes, channels),
        candidate_observed.reshape(batch * candidate_count, horizon, nodes, channels),
        contexts.reshape(batch * candidate_count, context_length, nodes, channels),
        context_valid.reshape(batch * candidate_count, context_length, nodes, channels),
    )
    return (
        signature.reshape(batch, candidate_count, horizon, nodes, channels),
        valid.reshape(batch, candidate_count, horizon, nodes, channels),
    )


def _array_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _physical_node_series(
    values: torch.Tensor,
    node_id: int,
    scaler: Any,
) -> np.ndarray:
    """Convert a tensor ending in channel C for one node to physical units."""
    array = values.detach().float().cpu().numpy()
    mean = np.asarray(scaler.mean[node_id], dtype=np.float32)
    std = np.asarray(scaler.std[node_id], dtype=np.float32)
    return array * std + mean


def _collect_case_payloads(
    selected: dict[str, Any],
    version: str,
    dataset: Dataset,
    data: Any,
    graph: Any,
    config: ExperimentConfig,
    pretrained_model: Any,
    random_model: Any,
    pretrained_retriever: TwoStageRetriever,
    random_retriever: TwoStageRetriever,
    pretrained_bank: MemoryBank,
    random_bank: MemoryBank,
    device: torch.device,
    candidate_protocol: str,
) -> dict[str, Any]:
    case_names = ("strong_win", "representative", "failure")
    index_by_sample = {
        int(sample_id): index
        for index, sample_id in enumerate(dataset.context_end_indices.tolist())
    }
    samples = [dataset[index_by_sample[int(selected[name]["sample_id"])]] for name in case_names]
    batch = default_collate(samples)
    graph_device = graph.to(device)
    with torch.no_grad():
        pretrained_encoding = pretrained_model.encode_clean(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            batch["retrieval_weekday"].to(device),
            batch["retrieval_slot"].to(device),
            graph_device,
        )
        random_encoding = random_model.encode_clean(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            batch["retrieval_weekday"].to(device),
            batch["retrieval_slot"].to(device),
            graph_device,
        )
        events = build_diagnostic_event_candidates(
            pretrained_bank,
            batch["query_weekday"].to(device),
            batch["query_slot"].to(device),
            batch["context_start"].to(device),
            config.bank.event_top_r,
            device,
            candidate_protocol,
        )
        pretrained_candidates = pretrained_retriever.rerank_nodes(
            pretrained_encoding.retrieval.node_keys,
            pretrained_encoding.statistics.level_features,
            events,
        )
        random_candidates = random_retriever.rerank_nodes(
            random_encoding.retrieval.node_keys,
            random_encoding.statistics.level_features,
            events,
        )
        pretrained_raw = pretrained_retriever.aggregate(pretrained_candidates)
        random_raw = random_retriever.aggregate(random_candidates)
        if version == "e5a":
            pretrained_deployed = offset_decay_aggregation(
                pretrained_candidates,
                batch["x"].to(device),
                batch["x_observed"].to(device),
                pretrained_bank,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            random_deployed = offset_decay_aggregation(
                random_candidates,
                batch["x"].to(device),
                batch["x_observed"].to(device),
                random_bank,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
        else:
            pretrained_deployed = pretrained_raw
            random_deployed = random_raw

    payloads: dict[str, Any] = {}
    for batch_index, case_name in enumerate(case_names):
        node_id = int(selected[case_name]["node_id"])
        channel_id = 0
        truth = _physical_node_series(
            batch["y"][batch_index, :, node_id, channel_id],
            node_id,
            data.scaler,
        )
        history = _physical_node_series(
            batch["x"][batch_index, :, node_id, channel_id],
            node_id,
            data.scaler,
        )

        def aggregation_payload(
            candidates: Any,
            deployed: AggregationOutput,
        ) -> tuple[list[float], list[list[float]], list[int], list[int], list[float], list[float]]:
            memory = _physical_node_series(
                deployed.prediction[batch_index, :, node_id, channel_id],
                node_id,
                data.scaler,
            )
            candidate_tensor = deployed.candidate_futures[
                batch_index, :, node_id, :, channel_id
            ]  # [H, K]
            candidate_physical = _physical_node_series(
                candidate_tensor,
                node_id,
                data.scaler,
            ).T
            event_ids = candidates.event_ids[batch_index, node_id].detach().cpu().tolist()
            sample_ids = [int(pretrained_bank.sample_id[event_id]) for event_id in event_ids]
            return (
                memory.tolist(),
                candidate_physical.tolist(),
                [int(value) for value in event_ids],
                sample_ids,
                candidates.weights[batch_index, node_id].detach().float().cpu().tolist(),
                candidates.shape_scores[batch_index, node_id].detach().float().cpu().tolist(),
            )

        (
            pretrained_memory,
            pretrained_futures,
            pretrained_event_ids,
            pretrained_sample_ids,
            pretrained_weights,
            pretrained_scores,
        ) = aggregation_payload(pretrained_candidates, pretrained_deployed)
        (
            random_memory,
            random_futures,
            random_event_ids,
            random_sample_ids,
            random_weights,
            random_scores,
        ) = aggregation_payload(random_candidates, random_deployed)
        case = dict(selected[case_name])
        case.update(
            {
                "timestamp": np.datetime_as_string(
                    np.datetime64(int(batch["timestamp_ns"][batch_index]), "ns")
                ),
                "query_history": history.tolist(),
                "query_future": truth.tolist(),
                "pretrained_memory": pretrained_memory,
                "random_memory": random_memory,
                "pretrained_candidate_futures": pretrained_futures,
                "random_candidate_futures": random_futures,
                "pretrained_event_ids": pretrained_event_ids,
                "random_event_ids": random_event_ids,
                "pretrained_candidate_sample_ids": pretrained_sample_ids,
                "random_candidate_sample_ids": random_sample_ids,
                "pretrained_weights": pretrained_weights,
                "random_weights": random_weights,
                "pretrained_key_cosine_scores": pretrained_scores,
                "random_key_cosine_scores": random_scores,
            }
        )
        if version == "e5a":
            case["pretrained_raw_memory"] = _physical_node_series(
                pretrained_raw.prediction[batch_index, :, node_id, channel_id],
                node_id,
                data.scaler,
            ).tolist()
            case["pretrained_offset_decay_memory"] = pretrained_memory
        payloads[case_name] = case
    payloads["selection_rule"] = selected["selection_rule"]
    return payloads


def _write_alignment_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "selector",
                "bin",
                "count",
                "key_distance_min",
                "key_distance_max",
                "key_distance_mean",
                "future_distance_mean",
                "future_distance_median",
            ),
        )
        writer.writeheader()
        for selector in ("pretrained", "random"):
            for item in result["alignment"][selector]["distance_bins"]:
                writer.writerow({"selector": selector, **item})


@torch.no_grad()
def run_retrieval_visualization(
    version: str,
    config: ExperimentConfig | None,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    random_checkpoint_path: str | Path,
    random_bank_path: str | Path,
    split: str,
    output_dir: str | Path,
    max_batches: int | None = None,
    candidate_protocol: str = "exact_calendar",
    profile_weight_override: float | None = None,
) -> dict[str, Any]:
    """Run the leakage-safe E2/E3/E5A validation visualization experiment."""
    if split != "val":
        raise ValueError("retrieval visualization is restricted to the validation split")
    version = version.lower()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"version must be one of {sorted(SUPPORTED_VERSIONS)}")
    candidate_protocol = candidate_protocol.lower()
    if candidate_protocol not in SUPPORTED_CANDIDATE_PROTOCOLS:
        raise ValueError(
            f"candidate protocol must be one of {sorted(SUPPORTED_CANDIDATE_PROTOCOLS)}"
        )
    if config is None:
        raise ValueError("config is required")
    if profile_weight_override is not None:
        if config.model.profile_dim <= 0:
            raise ValueError("profile_weight_override requires a profile-enabled model")
        if not 0.0 <= profile_weight_override <= 1.0:
            raise ValueError("profile_weight_override must be in [0, 1]")
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
    random_checkpoint_path = resolve_project_path(random_checkpoint_path)
    random_bank_path = resolve_project_path(random_bank_path)
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pretrained_model, pretrained_checkpoint = load_pretrained_model(
        config,
        checkpoint_path,
        data.series.slots_per_day,
        device,
    )
    random_model, random_checkpoint = load_pretrained_model(
        config,
        random_checkpoint_path,
        data.series.slots_per_day,
        device,
    )
    pretrained_model.eval()
    random_model.eval()

    pretrained_key_chunks: list[np.ndarray] = []
    random_key_chunks: list[np.ndarray] = []
    future_distance_chunks: list[np.ndarray] = []
    pretrained_recall_chunks: list[np.ndarray] = []
    random_recall_chunks: list[np.ndarray] = []
    candidate_count_chunks: list[np.ndarray] = []
    case_records: list[dict[str, Any]] = []
    payload_case_records: list[dict[str, Any]] = []
    query_count = 0
    batch_count = 0
    metric_names = ["pretrained_memory", "random_memory"]
    if version == "e5a":
        metric_names.extend(("pretrained_raw_memory", "random_raw_memory"))
    metrics = {
        name: ForecastMetricAccumulator(config.data.horizon) for name in metric_names
    }

    with MemoryBank(bank_path) as pretrained_bank, MemoryBank(random_bank_path) as random_bank:
        bank_alignment = validate_aligned_bank_axes(pretrained_bank, random_bank)
        _validate_bank(pretrained_bank, pretrained_model, graph_cpu, data.scaler.state_dict())
        _validate_bank(random_bank, random_model, graph_cpu, data.scaler.state_dict())
        pretrained_retriever = TwoStageRetriever(
            pretrained_bank,
            config.bank.event_top_r,
            config.bank.node_top_k,
            config.bank.level_weight,
            config.bank.level_temperature,
            config.bank.search_temperature,
            device,
            profile_weight_override=profile_weight_override,
        )
        random_retriever = TwoStageRetriever(
            random_bank,
            config.bank.event_top_r,
            config.bank.node_top_k,
            config.bank.level_weight,
            config.bank.level_temperature,
            config.bank.search_temperature,
            device,
            profile_weight_override=profile_weight_override,
        )

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            pretrained_encoding = pretrained_model.encode_clean(
                batch["retrieval_x"].to(device),
                batch["retrieval_observed"].to(device),
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            random_encoding = random_model.encode_clean(
                batch["retrieval_x"].to(device),
                batch["retrieval_observed"].to(device),
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            events = build_diagnostic_event_candidates(
                pretrained_bank,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
                config.bank.event_top_r,
                device,
                candidate_protocol,
            )
            candidate_counts = events.valid.sum(dim=-1)
            candidate_count_chunks.append(candidate_counts.detach().cpu().numpy())

            pretrained_candidate_keys = _candidate_node_keys(
                pretrained_bank, events.event_ids, device
            )
            random_candidate_keys = _candidate_node_keys(
                random_bank, events.event_ids, device
            )
            pretrained_key_distance, pretrained_key_valid = node_key_distances(
                pretrained_encoding.retrieval.node_keys,
                pretrained_candidate_keys,
                events.valid,
                profile_dim=(config.model.profile_dim if profile_weight_override is not None else None),
                profile_weight=profile_weight_override,
            )
            random_key_distance, random_key_valid = node_key_distances(
                random_encoding.retrieval.node_keys,
                random_candidate_keys,
                events.valid,
                profile_dim=(config.model.profile_dim if profile_weight_override is not None else None),
                profile_weight=profile_weight_override,
            )

            event_future, event_future_valid = event_candidate_futures(
                pretrained_bank,
                events.event_ids,
                events.valid,
                device,
            )
            candidate_future = event_future.permute(0, 3, 1, 2, 4).contiguous()
            candidate_future_valid = event_future_valid.permute(0, 3, 1, 2, 4).contiguous()
            if version in {"e2", "e3"}:
                query_context = batch["retrieval_x"].to(device)
                query_context_valid = batch["retrieval_observed"].to(device)
            else:
                query_context = batch["x"].to(device)
                query_context_valid = batch["x_observed"].to(device)
            query_signature, query_signature_valid = build_teacher_aligned_signature(
                version,
                batch["y"].to(device),
                batch["y_observed"].to(device),
                query_context,
                query_context_valid,
            )
            candidate_signature, candidate_signature_valid = _candidate_teacher_signatures(
                version,
                pretrained_bank,
                events,
                candidate_future,
                candidate_future_valid,
                data,
                config.data.context_length,
                device,
            )
            normalization = (
                config.pretrain.relation_distance_normalization
                if version == "e5a"
                else "none"
            )
            future_distance, future_valid = teacher_candidate_distances(
                query_signature,
                query_signature_valid,
                candidate_signature,
                candidate_signature_valid,
                events.valid,
                normalization,
            )
            alignment_valid = future_valid & pretrained_key_valid & random_key_valid
            pretrained_key_chunks.append(
                pretrained_key_distance.masked_select(alignment_valid).detach().cpu().numpy()
            )
            random_key_chunks.append(
                random_key_distance.masked_select(alignment_valid).detach().cpu().numpy()
            )
            future_distance_chunks.append(
                future_distance.masked_select(alignment_valid).detach().cpu().numpy()
            )
            pretrained_recall, recall_eligible = future_neighbor_recall_at_k(
                pretrained_key_distance,
                future_distance,
                alignment_valid,
                k=5,
            )
            random_recall, random_recall_eligible = future_neighbor_recall_at_k(
                random_key_distance,
                future_distance,
                alignment_valid,
                k=5,
            )
            joint_recall_eligible = recall_eligible & random_recall_eligible
            pretrained_recall_chunks.append(
                pretrained_recall.masked_select(joint_recall_eligible).detach().cpu().numpy()
            )
            random_recall_chunks.append(
                random_recall.masked_select(joint_recall_eligible).detach().cpu().numpy()
            )

            pretrained_candidates = pretrained_retriever.rerank_nodes(
                pretrained_encoding.retrieval.node_keys,
                pretrained_encoding.statistics.level_features,
                events,
            )
            random_candidates = random_retriever.rerank_nodes(
                random_encoding.retrieval.node_keys,
                random_encoding.statistics.level_features,
                events,
            )
            pretrained_raw = pretrained_retriever.aggregate(pretrained_candidates)
            random_raw = random_retriever.aggregate(random_candidates)
            if version == "e5a":
                pretrained_deployed = offset_decay_aggregation(
                    pretrained_candidates,
                    batch["x"].to(device),
                    batch["x_observed"].to(device),
                    pretrained_bank,
                    data.series,
                    data.scaler,
                    config.data.context_length,
                    device,
                )
                random_deployed = offset_decay_aggregation(
                    random_candidates,
                    batch["x"].to(device),
                    batch["x_observed"].to(device),
                    random_bank,
                    data.series,
                    data.scaler,
                    config.data.context_length,
                    device,
                )
                aggregations = {
                    "pretrained_memory": pretrained_deployed,
                    "random_memory": random_deployed,
                    "pretrained_raw_memory": pretrained_raw,
                    "random_raw_memory": random_raw,
                }
            else:
                pretrained_deployed = pretrained_raw
                random_deployed = random_raw
                aggregations = {
                    "pretrained_memory": pretrained_deployed,
                    "random_memory": random_deployed,
                }
            target = batch["y"].to(device)
            target_valid = batch["y_observed"].to(device).bool()
            common_metric_valid = target_valid.clone()
            for aggregation in aggregations.values():
                common_metric_valid &= aggregation.valid
            target_physical = data.scaler.inverse_transform_torch(target)
            physical_predictions: dict[str, torch.Tensor] = {}
            for name, aggregation in aggregations.items():
                prediction_physical = data.scaler.inverse_transform_torch(aggregation.prediction)
                physical_predictions[name] = prediction_physical
                metrics[name].update(prediction_physical, target_physical, common_metric_valid)

            pretrained_anchor_mae, pretrained_anchor_valid = memory_mae_by_anchor(
                physical_predictions["pretrained_memory"],
                target_physical,
                common_metric_valid,
            )
            random_anchor_mae, random_anchor_valid = memory_mae_by_anchor(
                physical_predictions["random_memory"],
                target_physical,
                common_metric_valid,
            )
            case_valid = (
                pretrained_anchor_valid
                & random_anchor_valid
                & complete_anchor_mask(common_metric_valid)
            )
            gains = random_anchor_mae - pretrained_anchor_mae
            for local_query, node_id in case_valid.nonzero(as_tuple=False).detach().cpu().tolist():
                case_records.append(
                    {
                        "sample_id": int(batch["sample_id"][local_query]),
                        "node_id": int(node_id),
                        "mae_gain": float(gains[local_query, node_id].detach().cpu()),
                        "pretrained_mae": float(
                            pretrained_anchor_mae[local_query, node_id].detach().cpu()
                        ),
                        "random_mae": float(
                            random_anchor_mae[local_query, node_id].detach().cpu()
                        ),
                    }
                )
            if version == "e5a":
                pretrained_raw_anchor_mae, pretrained_raw_anchor_valid = memory_mae_by_anchor(
                    physical_predictions["pretrained_raw_memory"],
                    target_physical,
                    common_metric_valid,
                )
                payload_valid = (
                    pretrained_anchor_valid
                    & pretrained_raw_anchor_valid
                    & complete_anchor_mask(common_metric_valid)
                )
                payload_gain = pretrained_raw_anchor_mae - pretrained_anchor_mae
                for local_query, node_id in payload_valid.nonzero(
                    as_tuple=False
                ).detach().cpu().tolist():
                    payload_case_records.append(
                        {
                            "sample_id": int(batch["sample_id"][local_query]),
                            "node_id": int(node_id),
                            "mae_gain": float(
                                payload_gain[local_query, node_id].detach().cpu()
                            ),
                            "raw_mae": float(
                                pretrained_raw_anchor_mae[
                                    local_query, node_id
                                ].detach().cpu()
                            ),
                            "offset_decay_mae": float(
                                pretrained_anchor_mae[
                                    local_query, node_id
                                ].detach().cpu()
                            ),
                        }
                    )
            query_count += int(target.shape[0])
            batch_count += 1
            if batch_count == 1 or batch_count % 10 == 0:
                print(
                    f"[{version}/{candidate_protocol}] processed "
                    f"{query_count}/{len(dataset)} validation queries "
                    f"({batch_count} batches)",
                    flush=True,
                )

        if batch_count == 0:
            raise ValueError("no validation batches were processed")
        pretrained_keys = np.concatenate(pretrained_key_chunks)
        random_keys = np.concatenate(random_key_chunks)
        future_distances = np.concatenate(future_distance_chunks)
        shared_valid = np.ones_like(future_distances, dtype=bool)
        pretrained_alignment = alignment_statistics(
            pretrained_keys, future_distances, shared_valid
        )
        random_alignment = alignment_statistics(random_keys, future_distances, shared_valid)
        pretrained_recall_values = np.concatenate(pretrained_recall_chunks)
        random_recall_values = np.concatenate(random_recall_chunks)
        pretrained_alignment["future_neighbor_recall_at_5"] = float(
            pretrained_recall_values.mean()
        )
        pretrained_alignment["recall_at_5_eligible_anchors"] = int(
            pretrained_recall_values.size
        )
        random_alignment["future_neighbor_recall_at_5"] = float(
            random_recall_values.mean()
        )
        random_alignment["recall_at_5_eligible_anchors"] = int(
            random_recall_values.size
        )
        selected = select_quantile_cases(case_records)
        cases = _collect_case_payloads(
            selected,
            version,
            dataset,
            data,
            graph_cpu,
            config,
            pretrained_model,
            random_model,
            pretrained_retriever,
            random_retriever,
            pretrained_bank,
            random_bank,
            device,
            candidate_protocol,
        )
        payload_selected: dict[str, Any] | None = None
        if version == "e5a":
            payload_selected = select_quantile_cases(payload_case_records)
            payload_selected["selection_rule"]["score"] = (
                "rawfuture_memory_mae_minus_offset_decay_memory_mae"
            )
            cases["offset_decay_payload_cases"] = _collect_case_payloads(
                payload_selected,
                version,
                dataset,
                data,
                graph_cpu,
                config,
                pretrained_model,
                random_model,
                pretrained_retriever,
                random_retriever,
                pretrained_bank,
                random_bank,
                device,
                candidate_protocol,
            )

        candidate_counts = np.concatenate(candidate_count_chunks).astype(np.float64)
        positive_candidate_counts = candidate_counts[candidate_counts > 0]
        if positive_candidate_counts.size == 0:
            raise ValueError("candidate protocol produced no causal candidates")
        top_k_saturation = np.minimum(
            positive_candidate_counts,
            float(config.bank.node_top_k),
        ) / positive_candidate_counts
        result: dict[str, Any] = {
            "schema_version": 1,
            "version": version,
            "dataset": pretrained_bank.manifest.dataset_name,
            "split": split,
            "complete_validation": max_batches is None,
            "queries": query_count,
            "batches": batch_count,
            "nodes": pretrained_bank.manifest.num_nodes,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": pretrained_checkpoint.get("epoch"),
            "random_checkpoint": str(random_checkpoint_path),
            "random_checkpoint_epoch": random_checkpoint.get("epoch"),
            "bank": str(bank_path),
            "random_bank": str(random_bank_path),
            "bank_alignment": bank_alignment,
            "candidate_protocol": {
                "name": candidate_protocol,
                "same_weekday": candidate_protocol != "broad_causal",
                "slot_radius": 1 if candidate_protocol == "relaxed_calendar" else 0,
                "strict_causal": True,
                "shared_pretrained_random_event_axis": True,
                "broad_sampling": (
                    "chronological_quantiles_up_to_event_top_r"
                    if candidate_protocol == "broad_causal"
                    else "all_legal_events"
                ),
            },
            "candidate_pool": {
                **_array_summary(candidate_counts),
                "top_k_saturation_mean": float(top_k_saturation.mean()),
                "fully_saturated_fraction": float(
                    (positive_candidate_counts <= config.bank.node_top_k).mean()
                ),
            },
            "teacher_signature": {
                "name": (
                    "ContextNormalizedFutureSignature"
                    if version in {"e2", "e3"}
                    else "DeploymentAlignedOffsetDecaySignature"
                ),
                "context_steps": (
                    config.data.encoder_context_length
                    if version in {"e2", "e3"}
                    else config.data.context_length
                ),
                "anchor_mean_distance_normalization": version == "e5a",
            },
            "future_information_boundary": future_information_boundary(),
            "alignment": {
                "pretrained": pretrained_alignment,
                "random": random_alignment,
                "delta_pretrained_minus_random": {
                    "spearman": pretrained_alignment["spearman"]
                    - random_alignment["spearman"],
                    "future_neighbor_recall_at_5": pretrained_alignment[
                        "future_neighbor_recall_at_5"
                    ]
                    - random_alignment["future_neighbor_recall_at_5"],
                },
            },
            "memory_metrics": {name: accumulator.compute() for name, accumulator in metrics.items()},
            "case_selection": selected,
            "offset_decay_payload_case_selection": payload_selected,
            "config": {
                "event_top_r": config.bank.event_top_r,
                "node_top_k": config.bank.node_top_k,
                "level_weight": config.bank.level_weight,
                "search_temperature": config.bank.search_temperature,
                "profile_weight_manifest": config.model.profile_weight,
                "profile_weight_override": profile_weight_override,
                "profile_weight_effective": (
                    config.model.profile_weight
                    if profile_weight_override is None
                    else profile_weight_override
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }

    result_path = output_path / "metrics.json"
    cases_path = output_path / "cases.json"
    bins_path = output_path / "alignment_bins.csv"
    save_json(result_path, result)
    save_json(cases_path, cases)
    _write_alignment_csv(result, bins_path)
    figure_paths = render_visualization_figures(result, cases, output_path)
    result["outputs"] = {
        "metrics": str(result_path),
        "cases": str(cases_path),
        "alignment_bins": str(bins_path),
        "figures": [str(path) for path in figure_paths],
    }
    save_json(result_path, result)
    return result


def _plot_alignment(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pretrained = result["alignment"]["pretrained"]
    random = result["alignment"]["random"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    colors = {"Pretrained": "#C43C39", "Random": "#4C78A8"}
    for label, values in (("Pretrained", pretrained), ("Random", random)):
        bins = values["distance_bins"]
        x = [item["bin"] for item in bins]
        y = [item["future_distance_mean"] for item in bins]
        axes[0].plot(x, y, marker="o", linewidth=2.2, label=label, color=colors[label])
    axes[0].set_xlabel("Key-distance decile (near to far)")
    axes[0].set_ylabel("Mean teacher-aligned future distance")
    axes[0].set_xticks(range(1, 11))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    labels = ["Spearman", "Future-Neighbor\nRecall@5"]
    pretrained_values = [
        pretrained["spearman"],
        pretrained["future_neighbor_recall_at_5"],
    ]
    random_values = [random["spearman"], random["future_neighbor_recall_at_5"]]
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    bars_pretrained = axes[1].bar(
        positions - width / 2,
        pretrained_values,
        width,
        color=colors["Pretrained"],
        label="Pretrained",
    )
    bars_random = axes[1].bar(
        positions + width / 2,
        random_values,
        width,
        color=colors["Random"],
        label="Random",
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Score")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].bar_label(bars_pretrained, fmt="%.3f", padding=3, fontsize=9)
    axes[1].bar_label(bars_random, fmt="%.3f", padding=3, fontsize=9)
    figure.suptitle(f"{result['version'].upper()} Key-Future Alignment", fontsize=14)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _plot_cases(cases: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_names = ("strong_win", "representative", "failure")
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), constrained_layout=True)
    colors = {"Pretrained": "#C43C39", "Random": "#4C78A8", "Truth": "#111111"}
    for row, case_name in enumerate(case_names):
        case = cases[case_name]
        truth = np.asarray(case["query_future"], dtype=np.float64)
        for column, label in enumerate(("Pretrained", "Random")):
            axis = axes[row, column]
            prefix = label.lower()
            candidate_values = case[f"{prefix}_candidate_futures"]
            for candidate in candidate_values:
                axis.plot(candidate, color=colors[label], alpha=0.22, linewidth=1.0)
            axis.plot(
                case[f"{prefix}_memory"],
                color=colors[label],
                linewidth=2.4,
                marker="o",
                label=f"{label} memory",
            )
            axis.plot(truth, color=colors["Truth"], linewidth=2.6, marker="s", label="True future")
            axis.set_title(
                f"{case_name.replace('_', ' ').title()} | {label} | "
                f"MAE={case[f'{prefix}_mae']:.3f}"
            )
            axis.set_xlabel("Forecast step")
            axis.set_ylabel("Traffic speed")
            axis.grid(axis="y", alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
        row_values = [
            truth,
            np.asarray(case["pretrained_memory"]),
            np.asarray(case["random_memory"]),
        ]
        row_values.extend(np.asarray(value) for value in case["pretrained_candidate_futures"])
        row_values.extend(np.asarray(value) for value in case["random_candidate_futures"])
        lower = min(float(value.min()) for value in row_values)
        upper = max(float(value.max()) for value in row_values)
        margin = max((upper - lower) * 0.08, 0.5)
        for column in range(2):
            axes[row, column].set_ylim(lower - margin, upper + margin)
    figure.suptitle("Deterministic Top-5 Retrieval Cases", fontsize=14)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _plot_offset_decay_cases(cases: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload_cases = cases.get("offset_decay_payload_cases", cases)
    case_names = ("strong_win", "representative", "failure")
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 10.0), constrained_layout=True)
    for axis, case_name in zip(axes, case_names):
        case = payload_cases[case_name]
        truth = np.asarray(case["query_future"], dtype=np.float64)
        raw_memory = np.asarray(case["pretrained_raw_memory"], dtype=np.float64)
        offset_memory = np.asarray(
            case["pretrained_offset_decay_memory"], dtype=np.float64
        )
        raw_mae = float(np.mean(np.abs(raw_memory - truth)))
        offset_mae = float(np.mean(np.abs(offset_memory - truth)))
        axis.plot(truth, color="#111111", linewidth=2.6, marker="s", label="True future")
        axis.plot(
            raw_memory,
            color="#59A14F",
            linewidth=2.2,
            marker="o",
            label="RawFuture memory",
        )
        axis.plot(
            offset_memory,
            color="#C43C39",
            linewidth=2.2,
            marker="o",
            label="OffsetDecay memory",
        )
        axis.set_title(
            f"OffsetDecay {case_name.replace('_', ' ').title()} | "
            f"Raw MAE={raw_mae:.3f} | OD MAE={offset_mae:.3f}"
        )
        axis.set_xlabel("Forecast step")
        axis.set_ylabel("Traffic speed")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("E5A Payload Alignment on Identical Retrieved Candidates", fontsize=14)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def render_visualization_figures(
    result: dict[str, Any],
    cases: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Render the fixed paper-facing figures and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "key_future_alignment.png",
        output_dir / "deterministic_top5_cases.png",
    ]
    _plot_alignment(result, paths[0])
    _plot_cases(cases, paths[1])
    if str(result["version"]).lower() == "e5a":
        offset_path = output_dir / "offset_decay_payload_cases.png"
        _plot_offset_decay_cases(cases, offset_path)
        paths.append(offset_path)
    return paths

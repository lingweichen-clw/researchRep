"""Teacher-aligned diagnostics for learned historical retrieval keys."""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader, Dataset, default_collate

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
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
    raw_l1_node_candidates,
)
from stanchor.utils import resolve_device, save_json


CURRENT_VISUALIZATION_VERSION = "hn_offset_decay_v2"
SUPPORTED_VERSIONS = {CURRENT_VISUALIZATION_VERSION}
SUPPORTED_CANDIDATE_PROTOCOLS = {
    "exact_calendar",
    "relaxed_calendar",
    "relaxed_calendar_diverse",
    "weekday_radius1_overlap",
    "broad_causal",
    "pretrain_broad_causal",
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
    """Build the current HN-OffsetDecay v2 future representation."""
    version = version.lower()
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"version must be one of {sorted(SUPPORTED_VERSIONS)}")
    if future.ndim != 4 or future_observed.shape != future.shape:
        raise ValueError("future and future_observed must be [B, H, N, C]")
    if context.ndim != 4 or context_observed.shape != context.shape:
        raise ValueError("context and context_observed must be [B, T, N, C]")
    if future.shape[0] != context.shape[0] or future.shape[2:] != context.shape[2:]:
        raise ValueError("future and context batch/node/channel dimensions must align")

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


def anchor_wise_ranking_metrics(
    key_distance: np.ndarray,
    teacher_distance: np.ndarray,
    valid: np.ndarray,
    ndcg_k: int = 5,
    teacher_temperature: float = 0.1,
) -> dict[str, Any]:
    """Evaluate local candidate orderings independently for every anchor.

    ``key_distance``, ``teacher_distance``, and ``valid`` are ``[B, N, R]``.
    An anchor is a fixed ``(query, node)``; its candidate count can vary through
    padding.  Anchors with fewer than two finite candidates are excluded from
    all ranking metrics so that one-candidate pools do not create trivial wins.
    Kendall's statistic uses only strict candidate pairs, excluding ties from
    the denominator.  The returned ``*_values`` arrays are intended for plots
    and uncertainty summaries; scalar fields are JSON-safe after conversion.
    """
    key = np.asarray(key_distance, dtype=np.float64)
    teacher = np.asarray(teacher_distance, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if key.ndim != 3 or teacher.shape != key.shape or mask.shape != key.shape:
        raise ValueError("ranking distances and valid mask must be [B, N, R] and aligned")
    if ndcg_k <= 0:
        raise ValueError("ndcg_k must be positive")
    if teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")

    spearman_values: list[float] = []
    kendall_values: list[float] = []
    recall_at_1_values: list[float] = []
    ndcg_values: list[float] = []
    recall_at_5_values: list[float] = []
    candidate_counts: list[int] = []

    def _kendall_strict(left: np.ndarray, right: np.ndarray) -> float:
        concordant = 0
        discordant = 0
        for left_index in range(left.size - 1):
            left_delta = left[left_index] - left[left_index + 1 :]
            right_delta = right[left_index] - right[left_index + 1 :]
            strict = (left_delta != 0.0) & (right_delta != 0.0)
            if not np.any(strict):
                continue
            signs_match = np.sign(left_delta[strict]) == np.sign(right_delta[strict])
            concordant += int(signs_match.sum())
            discordant += int((~signs_match).sum())
        denominator = concordant + discordant
        return 0.0 if denominator == 0 else (concordant - discordant) / denominator

    def _ndcg(left: np.ndarray, right: np.ndarray) -> float:
        cutoff = min(ndcg_k, left.size)
        relevance = np.exp(-right / teacher_temperature)
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
        key_order = np.argsort(left, kind="stable")[:cutoff]
        ideal_order = np.argsort(right, kind="stable")[:cutoff]
        dcg = float(np.sum((2.0**relevance[key_order] - 1.0) * discounts))
        ideal = float(np.sum((2.0**relevance[ideal_order] - 1.0) * discounts))
        return 0.0 if ideal <= 0.0 else dcg / ideal

    for batch_index in range(key.shape[0]):
        for node_index in range(key.shape[1]):
            anchor_mask = mask[batch_index, node_index]
            anchor_mask &= np.isfinite(key[batch_index, node_index])
            anchor_mask &= np.isfinite(teacher[batch_index, node_index])
            candidate_key = key[batch_index, node_index][anchor_mask]
            candidate_teacher = teacher[batch_index, node_index][anchor_mask]
            candidate_count = int(candidate_key.size)
            if candidate_count < 2:
                continue
            candidate_counts.append(candidate_count)

            key_ranks = rankdata(candidate_key, method="average")
            teacher_ranks = rankdata(candidate_teacher, method="average")
            if np.ptp(key_ranks) == 0.0 or np.ptp(teacher_ranks) == 0.0:
                spearman = 0.0
            else:
                spearman = float(np.corrcoef(key_ranks, teacher_ranks)[0, 1])
            spearman_values.append(spearman)
            kendall_values.append(_kendall_strict(candidate_key, candidate_teacher))

            key_top1 = int(np.argsort(candidate_key, kind="stable")[0])
            teacher_top1 = int(np.argsort(candidate_teacher, kind="stable")[0])
            recall_at_1_values.append(float(key_top1 == teacher_top1))
            ndcg_values.append(_ndcg(candidate_key, candidate_teacher))

            if candidate_count > ndcg_k:
                top_k = ndcg_k
                key_top = set(np.argsort(candidate_key, kind="stable")[:top_k].tolist())
                teacher_top = set(np.argsort(candidate_teacher, kind="stable")[:top_k].tolist())
                recall_at_5_values.append(len(key_top & teacher_top) / float(top_k))

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def _std(values: list[float]) -> float:
        return float(np.std(values)) if values else 0.0

    counts = np.asarray(candidate_counts, dtype=np.float64)
    return {
        "spearman_mean": _mean(spearman_values),
        "spearman_std": _std(spearman_values),
        "spearman_eligible_anchors": len(spearman_values),
        "kendall_mean": _mean(kendall_values),
        "kendall_std": _std(kendall_values),
        "kendall_eligible_anchors": len(kendall_values),
        "recall_at_1_mean": _mean(recall_at_1_values),
        "recall_at_1_std": _std(recall_at_1_values),
        "recall_at_1_eligible_anchors": len(recall_at_1_values),
        "ndcg_at_5_mean": _mean(ndcg_values),
        "ndcg_at_5_std": _std(ndcg_values),
        "ndcg_at_5_eligible_anchors": len(ndcg_values),
        "recall_at_5_mean": _mean(recall_at_5_values),
        "recall_at_5_std": _std(recall_at_5_values),
        "recall_at_5_eligible_anchors": len(recall_at_5_values),
        "candidate_count_mean": float(counts.mean()) if counts.size else 0.0,
        "candidate_count_min": int(counts.min()) if counts.size else 0,
        "candidate_count_max": int(counts.max()) if counts.size else 0,
        "random_recall_at_1_expected": (
            float(np.mean(1.0 / counts)) if counts.size else 0.0
        ),
        "random_recall_at_5_expected": (
            float(np.mean(ndcg_k / counts[counts > ndcg_k]))
            if np.any(counts > ndcg_k)
            else 0.0
        ),
        "spearman_values": np.asarray(spearman_values, dtype=np.float64),
        "kendall_values": np.asarray(kendall_values, dtype=np.float64),
        "recall_at_1_values": np.asarray(recall_at_1_values, dtype=np.float64),
        "ndcg_at_5_values": np.asarray(ndcg_values, dtype=np.float64),
        "recall_at_5_values": np.asarray(recall_at_5_values, dtype=np.float64),
    }


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
    protocol: str = "weekday_radius1_overlap",
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
    if protocol == "pretrain_broad_causal":
        protocol = "broad_causal"
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
                positions = np.rint(np.linspace(0, legal.size - 1, max_candidates)).astype(np.int64)
                legal = legal[positions]
        elif protocol == "weekday_radius1_overlap":
            query_context_end = query_start + int(bank.manifest.context_length) - 1
            collected: list[int] = []
            seen: set[int] = set()
            for weekday_offset in (-1, 0, 1):
                candidate_weekday = (query_weekday + weekday_offset) % 7
                for event_id in np.asarray(
                    bank.calendar.lookup(candidate_weekday, query_slot), dtype=np.int64
                ).tolist():
                    event_id = int(event_id)
                    if event_id in seen or future_end[event_id] > query_context_end:
                        continue
                    seen.add(event_id)
                    collected.append(event_id)
            legal = np.asarray(collected, dtype=np.int64)
        else:
            radius = 0 if protocol == "exact_calendar" else 1
            collected: list[int] = []
            seen: set[int] = set()
            for offset in range(-radius, radius + 1):
                candidate_slot = query_slot + offset
                if not 0 <= candidate_slot < slots_per_day:
                    continue
                for event_id in np.asarray(
                    bank.calendar.lookup(query_weekday, candidate_slot), dtype=np.int64
                ).tolist():
                    event_id = int(event_id)
                    if event_id in seen or future_end[event_id] >= query_start:
                        continue
                    seen.add(event_id)
                    collected.append(event_id)
            legal = np.asarray(collected, dtype=np.int64)
            if protocol == "relaxed_calendar_diverse":
                min_gap = int(getattr(bank.manifest, "context_length", 0)) + int(
                    getattr(bank.manifest, "horizon", 0)
                )
                min_gap = max(min_gap, 1)
                selected: list[int] = []
                for event_id in legal.tolist():
                    if all(
                        abs(int(bank.context_end[event_id]) - int(bank.context_end[other])) >= min_gap
                        for other in selected
                    ):
                        selected.append(event_id)
                legal = np.asarray(selected, dtype=np.int64)
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
    signature_chunks: list[torch.Tensor] = []
    valid_chunks: list[torch.Tensor] = []
    chunk_size = 8
    for start in range(0, candidate_count, chunk_size):
        stop = min(start + chunk_size, candidate_count)
        contexts, context_valid = candidate_contexts(
            bank,
            events.event_ids[:, start:stop],
            data.series,
            data.scaler,
            context_length,
            device,
        )
        chunk_signature, chunk_valid = build_teacher_aligned_signature(
            CURRENT_VISUALIZATION_VERSION,
            candidate_future[:, start:stop].reshape(
                batch * (stop - start), horizon, nodes, channels
            ),
            candidate_observed[:, start:stop].reshape(
                batch * (stop - start), horizon, nodes, channels
            ),
            contexts.reshape(
                batch * (stop - start), context_length, nodes, channels
            ),
            context_valid.reshape(
                batch * (stop - start), context_length, nodes, channels
            ),
        )
        signature_chunks.append(
            chunk_signature.reshape(batch, stop - start, horizon, nodes, channels)
        )
        valid_chunks.append(
            chunk_valid.reshape(batch, stop - start, horizon, nodes, channels)
        )
    return torch.cat(signature_chunks, dim=1), torch.cat(valid_chunks, dim=1)


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


def _ranking_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop NumPy per-anchor arrays before writing ranking metrics to JSON."""
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def _write_ranking_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("selector", "metric", "mean", "std", "eligible_anchors")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for selector in ("pretrained", "random"):
            metrics = result["ranking"][selector]
            for metric, label in (
                ("spearman", "spearman_mean"),
                ("kendall", "kendall_mean"),
                ("recall_at_1", "recall_at_1_mean"),
                ("ndcg_at_5", "ndcg_at_5_mean"),
                ("recall_at_5", "recall_at_5_mean"),
            ):
                writer.writerow(
                    {
                        "selector": selector,
                        "metric": metric,
                        "mean": metrics[label],
                        "std": metrics.get(label.replace("_mean", "_std"), 0.0),
                        "eligible_anchors": metrics.get(
                            label.replace("_mean", "_eligible_anchors"), 0
                        ),
                    }
                )


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
        raw_l1_candidates, _, _ = raw_l1_node_candidates(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            pretrained_bank,
            events,
            data.series,
            data.scaler,
            config.data.encoder_context_length,
            config.bank.node_top_k,
            device,
        )
        raw_l1_memory = pretrained_retriever.aggregate(raw_l1_candidates)
        pretrained_raw = pretrained_retriever.aggregate(pretrained_candidates)
        random_raw = random_retriever.aggregate(random_candidates)
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
            sample_ids = [int(pretrained_bank.sample_id[event_id]) if event_id >= 0 else -1 for event_id in event_ids]
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
        (
            raw_l1_prediction,
            raw_l1_futures,
            raw_l1_event_ids,
            raw_l1_sample_ids,
            raw_l1_weights,
            raw_l1_scores,
        ) = aggregation_payload(raw_l1_candidates, raw_l1_memory)
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
                "raw_l1_memory": raw_l1_prediction,
                "raw_l1_candidate_futures": raw_l1_futures,
                "raw_l1_event_ids": raw_l1_event_ids,
                "raw_l1_candidate_sample_ids": raw_l1_sample_ids,
                "raw_l1_weights": raw_l1_weights,
                "raw_l1_context_l1_scores": raw_l1_scores,
            }
        )
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
    candidate_protocol: str = "weekday_radius1_overlap",
    profile_weight_override: float | None = None,
    node_top_k_override: int | None = None,
) -> dict[str, Any]:
    """Run the leakage-safe HN-OffsetDecay v2 validation visualization experiment."""
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
    if node_top_k_override is not None:
        if node_top_k_override <= 0:
            raise ValueError("node_top_k_override must be positive")
        config = replace(config, bank=replace(config.bank, node_top_k=node_top_k_override))
        config.validate()
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
    pretrained_anchor_key_chunks: list[np.ndarray] = []
    random_anchor_key_chunks: list[np.ndarray] = []
    anchor_future_distance_chunks: list[np.ndarray] = []
    anchor_valid_chunks: list[np.ndarray] = []
    pretrained_recall_chunks: list[np.ndarray] = []
    random_recall_chunks: list[np.ndarray] = []
    candidate_count_chunks: list[np.ndarray] = []
    case_records: list[dict[str, Any]] = []
    payload_case_records: list[dict[str, Any]] = []
    query_count = 0
    batch_count = 0
    metric_names = [
        "pretrained_memory",
        "random_memory",
        "raw_l1_memory",
        "pretrained_raw_memory",
        "random_raw_memory",
    ]
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
        )
        random_retriever = TwoStageRetriever(
            random_bank,
            config.bank.event_top_r,
            config.bank.node_top_k,
            config.bank.level_weight,
            config.bank.level_temperature,
            config.bank.search_temperature,
            device,
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
            query_signature, query_signature_valid = build_teacher_aligned_signature(
                version,
                batch["y"].to(device),
                batch["y_observed"].to(device),
                batch["x"].to(device),
                batch["x_observed"].to(device),
            )
            candidate_signature, candidate_signature_valid = _candidate_teacher_signatures(
                pretrained_bank,
                events,
                candidate_future,
                candidate_future_valid,
                data,
                config.data.context_length,
                device,
            )
            normalization = config.pretrain.relation_distance_normalization
            future_distance, future_valid = teacher_candidate_distances(
                query_signature,
                query_signature_valid,
                candidate_signature,
                candidate_signature_valid,
                events.valid,
                normalization,
            )
            alignment_valid = future_valid & pretrained_key_valid & random_key_valid
            pretrained_anchor_key_chunks.append(
                pretrained_key_distance.detach().cpu().numpy()
            )
            random_anchor_key_chunks.append(
                random_key_distance.detach().cpu().numpy()
            )
            anchor_future_distance_chunks.append(
                future_distance.detach().cpu().numpy()
            )
            anchor_valid_chunks.append(alignment_valid.detach().cpu().numpy())
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
            raw_l1_candidates, _, _ = raw_l1_node_candidates(
                batch["retrieval_x"].to(device),
                batch["retrieval_observed"].to(device),
                pretrained_bank,
                events,
                data.series,
                data.scaler,
                config.data.encoder_context_length,
                config.bank.node_top_k,
                device,
            )
            raw_l1_memory = pretrained_retriever.aggregate(raw_l1_candidates)
            pretrained_raw = pretrained_retriever.aggregate(pretrained_candidates)
            random_raw = random_retriever.aggregate(random_candidates)
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
                "raw_l1_memory": raw_l1_memory,
                "pretrained_raw_memory": pretrained_raw,
                "random_raw_memory": random_raw,
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
            raw_l1_anchor_mae, raw_l1_anchor_valid = memory_mae_by_anchor(
                physical_predictions["raw_l1_memory"],
                target_physical,
                common_metric_valid,
            )
            case_valid = (
                pretrained_anchor_valid
                & random_anchor_valid
                & raw_l1_anchor_valid
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
                        "raw_l1_mae": float(
                            raw_l1_anchor_mae[local_query, node_id].detach().cpu()
                        ),
                    }
                )
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
                            pretrained_raw_anchor_mae[local_query, node_id]
                            .detach()
                            .cpu()
                        ),
                        "offset_decay_mae": float(
                            pretrained_anchor_mae[local_query, node_id]
                            .detach()
                            .cpu()
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
        anchor_key_pretrained = np.concatenate(pretrained_anchor_key_chunks, axis=0)
        anchor_key_random = np.concatenate(random_anchor_key_chunks, axis=0)
        anchor_future = np.concatenate(anchor_future_distance_chunks, axis=0)
        anchor_valid = np.concatenate(anchor_valid_chunks, axis=0)
        pretrained_ranking = anchor_wise_ranking_metrics(
            anchor_key_pretrained,
            anchor_future,
            anchor_valid,
            ndcg_k=5,
            teacher_temperature=config.pretrain.relation_teacher_temperature,
        )
        random_ranking = anchor_wise_ranking_metrics(
            anchor_key_random,
            anchor_future,
            anchor_valid,
            ndcg_k=5,
            teacher_temperature=config.pretrain.relation_teacher_temperature,
        )
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
        ranking_payload = {
            "pretrained": _ranking_summary(pretrained_ranking),
            "random": _ranking_summary(random_ranking),
            "delta_pretrained_minus_random": {
                metric: pretrained_ranking[f"{metric}_mean"]
                - random_ranking[f"{metric}_mean"]
                for metric in ("spearman", "kendall", "recall_at_1", "ndcg_at_5", "recall_at_5")
            },
        }
        selected = select_quantile_cases(case_records)
        cases = _collect_case_payloads(
            selected,
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
        payload_selected = select_quantile_cases(payload_case_records)
        payload_selected["selection_rule"]["score"] = (
            "rawfuture_memory_mae_minus_offset_decay_memory_mae"
        )
        cases["offset_decay_payload_cases"] = _collect_case_payloads(
            payload_selected,
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
                "same_weekday": candidate_protocol
                not in {"broad_causal", "pretrain_broad_causal"},
                "slot_radius": (
                    1
                    if candidate_protocol in {"relaxed_calendar", "relaxed_calendar_diverse"}
                    else 0
                ),
                "weekday_radius": 1 if candidate_protocol == "weekday_radius1_overlap" else 0,
                "allow_context_overlap": candidate_protocol == "weekday_radius1_overlap",
                "causal_boundary": (
                    "query_context_end"
                    if candidate_protocol == "weekday_radius1_overlap"
                    else "query_context_start"
                ),
                "deduplicate_overlapping_windows": candidate_protocol == "relaxed_calendar_diverse",
                "min_event_gap_steps": (
                    int(pretrained_bank.manifest.context_length)
                    + int(pretrained_bank.manifest.horizon)
                    if candidate_protocol == "relaxed_calendar_diverse"
                    else None
                ),
                "strict_causal": True,
                "shared_pretrained_random_event_axis": True,
                "broad_sampling": (
                    "chronological_quantiles_up_to_event_top_r"
                    if candidate_protocol in {"broad_causal", "pretrain_broad_causal"}
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
                "name": "DeploymentAlignedOffsetDecaySignature",
                "context_steps": config.data.context_length,
                "distance_normalization": config.pretrain.relation_distance_normalization,
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
            "ranking": ranking_payload,
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
    ranking_path = output_path / "ranking_metrics.csv"
    save_json(result_path, result)
    save_json(cases_path, cases)
    _write_alignment_csv(result, bins_path)
    _write_ranking_csv(result, ranking_path)
    figure_paths = render_visualization_figures(result, cases, output_path)
    result["outputs"] = {
        "metrics": str(result_path),
        "cases": str(cases_path),
        "alignment_bins": str(bins_path),
        "ranking_metrics": str(ranking_path),
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
    methods = (
        ("Learned", "pretrained", "#C43C39"),
        ("Raw-L1", "raw_l1", "#E69F00"),
        ("Matched Random", "random", "#4C78A8"),
    )
    figure, axes = plt.subplots(3, 3, figsize=(17.0, 10.0), constrained_layout=True)
    top_k = len(cases["strong_win"]["pretrained_candidate_futures"])
    for row, case_name in enumerate(case_names):
        case = cases[case_name]
        truth = np.asarray(case["query_future"], dtype=np.float64)
        for column, (label, prefix, color) in enumerate(methods):
            axis = axes[row, column]
            candidate_values = np.asarray(case[f"{prefix}_candidate_futures"])
            for candidate in candidate_values:
                axis.plot(candidate, color=color, alpha=0.20, linewidth=1.0)
            axis.plot(
                case[f"{prefix}_memory"],
                color=color,
                linewidth=2.4,
                marker="o",
                label=f"{label} memory",
            )
            axis.plot(truth, color="#111111", linewidth=2.6, marker="s", label="True future")
            axis.set_title(
                f"{case_name.replace('_', ' ').title()} | {label} | "
                f"MAE={np.mean(np.abs(np.asarray(case[f'{prefix}_memory']) - truth)):.3f}"
            )
            axis.set_xlabel("Forecast step")
            axis.set_ylabel("Traffic speed")
            axis.grid(axis="y", alpha=0.2)
            axis.legend(frameon=False, fontsize=8)
        row_values = [
            truth,
            np.asarray(case["pretrained_memory"]),
            np.asarray(case["random_memory"]),
            np.asarray(case["raw_l1_memory"]),
        ]
        row_values.extend(np.asarray(value) for value in case["pretrained_candidate_futures"])
        row_values.extend(np.asarray(value) for value in case["raw_l1_candidate_futures"])
        row_values.extend(np.asarray(value) for value in case["random_candidate_futures"])
        lower = min(float(value.min()) for value in row_values)
        upper = max(float(value.max()) for value in row_values)
        margin = max((upper - lower) * 0.08, 0.5)
        for column in range(3):
            axes[row, column].set_ylim(lower - margin, upper + margin)
    figure.suptitle(
        f"Deterministic Top-{top_k} Retrieval Cases on a Shared Legal Pool",
        fontsize=14,
    )
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
    figure.suptitle("OffsetDecay Payload Alignment on Identical Retrieved Candidates", fontsize=14)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _plot_top5_error_profiles(cases: dict[str, Any], output_path: Path) -> None:
    """Show candidate error by retrieved rank so close trajectories remain distinguishable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_names = ("strong_win", "representative", "failure")
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), constrained_layout=True)
    methods = (
        ("Learned", "pretrained", "#C43C39"),
        ("Raw-L1", "raw_l1", "#E69F00"),
        ("Matched Random", "random", "#4C78A8"),
    )
    colors = {"Pretrained": "#C43C39", "Random": "#4C78A8"}
    for axis, case_name in zip(axes, case_names):
        case = cases[case_name]
        truth = np.asarray(case["query_future"], dtype=np.float64)
        profiles: dict[str, np.ndarray] = {}
        for label, prefix, color in methods:
            candidates = np.asarray(case[f"{prefix}_candidate_futures"], dtype=np.float64)
            profiles[label] = np.mean(np.abs(candidates - truth[None, :]), axis=1)
            ranks = np.arange(1, profiles[label].size + 1)
            axis.plot(
                ranks,
                profiles[label],
                marker="o",
                linewidth=2.0,
                color=color,
                label=f"{label} candidate MAE",
            )
            memory = np.asarray(case[f"{prefix}_memory"], dtype=np.float64)
            axis.axhline(
                np.mean(np.abs(memory - truth)),
                color=color,
                linestyle="--",
                linewidth=1.1,
                alpha=0.75,
                label=f"{label} memory MAE",
            )
        finite = np.concatenate(tuple(values[np.isfinite(values)] for values in profiles.values()))
        if finite.size:
            span = float(finite.max() - finite.min())
            margin = max(span * 0.15, 1.0e-3)
            axis.set_ylim(float(finite.min()) - margin, float(finite.max()) + margin)
        axis.set_title(f"{case_name.replace('_', ' ').title()} | lower is better")
        axis.set_xlabel("Retrieved candidate rank")
        axis.set_ylabel("Future MAE")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(f"Top-{profiles['Learned'].size} Candidate Error Profiles", fontsize=14)
    figure.savefig(output_path, dpi=220, facecolor="white")
    plt.close(figure)


def _plot_ranking_metrics(result: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranking = result["ranking"]
    labels = ["Anchor Spearman", "Anchor Kendall", "Recall@1", "NDCG@5", "Recall@5\nsecondary"]
    keys = ("spearman_mean", "kendall_mean", "recall_at_1_mean", "ndcg_at_5_mean", "recall_at_5_mean")
    pretrained = [float(ranking["pretrained"][key]) for key in keys]
    random = [float(ranking["random"][key]) for key in keys]
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.34
    figure, axis = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    bars_pretrained = axis.bar(
        positions - width / 2, pretrained, width, color="#C43C39", label="Joint v2"
    )
    bars_random = axis.bar(
        positions + width / 2, random, width, color="#4C78A8", label="Matched random"
    )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Anchor-wise score")
    axis.set_title(
        f"Local candidate ordering | protocol={result['candidate_protocol']['name']}"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.bar_label(bars_pretrained, fmt="%.3f", padding=3, fontsize=8)
    axis.bar_label(bars_random, fmt="%.3f", padding=3, fontsize=8)
    expected = float(ranking["pretrained"].get("random_recall_at_1_expected", 0.0))
    if expected > 0.0:
        axis.axhline(
            expected,
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            label=f"Random Recall@1 expectation={expected:.3f}",
        )
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
        output_dir / "top5_error_profiles.png",
    ]
    _plot_alignment(result, paths[0])
    _plot_cases(cases, paths[1])
    _plot_top5_error_profiles(cases, paths[2])
    if "ranking" in result:
        ranking_path = output_dir / "ranking_metrics.png"
        _plot_ranking_metrics(result, ranking_path)
        paths.append(ranking_path)
    offset_path = output_dir / "offset_decay_payload_cases.png"
    _plot_offset_decay_cases(cases, offset_path)
    paths.append(offset_path)
    return paths

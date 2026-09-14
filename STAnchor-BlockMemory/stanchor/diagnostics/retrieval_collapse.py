"""Frozen-encoder diagnostics for retrieval collapse and input reliance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import default_collate

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.data.graph import GraphData
from stanchor.data.normalization import normalize_window
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.models.retrieval_head import RetrievalOutput
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.retrieval.strategies import (
    event_candidate_futures,
    offset_decay_aggregation,
    offset_only_aggregation,
)
from stanchor.utils import resolve_device, save_json


@dataclass(frozen=True)
class ComponentEncoding:
    """Encoder and retrieval outputs for explicitly supplied input components."""

    hidden: torch.Tensor  # [B, P, N, D]
    retrieval: RetrievalOutput


def encode_components(
    model: Any,
    *,
    normalized: torch.Tensor,
    level_features: torch.Tensor,
    level_valid: torch.Tensor,
    weekday: torch.Tensor,
    slot: torch.Tensor,
    observed: torch.Tensor,
    graph: GraphData,
) -> ComponentEncoding:
    """Encode separated shape, level, and calendar components without future input."""
    if normalized.ndim != 4 or observed.shape != normalized.shape:
        raise ValueError("normalized and observed must be aligned [B,T,N,C] tensors")
    if weekday.shape != normalized.shape[:2] or slot.shape != normalized.shape[:2]:
        raise ValueError("weekday and slot must align with [B,T]")
    if level_features.shape[:2] != (normalized.shape[0], normalized.shape[2]):
        raise ValueError("level_features must align with [B,N]")
    if level_valid.shape != (normalized.shape[0], normalized.shape[2], 1):
        raise ValueError("level_valid must be [B,N,1]")

    tokens = model.embedding(
        normalized,
        level_features,
        level_valid,
        weekday,
        slot,
    )
    hidden = model.encoder(tokens, graph)
    if model.dynamics_adapter is not None:
        dynamics = model.dynamics_adapter(
            hidden,
            normalized,
            observed.bool(),
            graph,
        )
        hidden = dynamics.hidden
    return ComponentEncoding(hidden=hidden, retrieval=model.retrieval_head(hidden))


def reverse_temporal_patches(values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reverse the patch axis while retaining order inside every patch."""
    if values.ndim != 4:
        raise ValueError("values must be [B,T,N,C]")
    if patch_size <= 0 or values.shape[1] % patch_size != 0:
        raise ValueError("patch_size must be positive and divide the time dimension")
    batch, time, nodes, channels = values.shape
    patches = time // patch_size
    return values.reshape(batch, patches, patch_size, nodes, channels).flip(1).reshape_as(values)


def shift_calendar(
    weekday: torch.Tensor,
    slot: torch.Tensor,
    *,
    slots_per_day: int,
    shift_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Circularly shift a complete weekly calendar while preserving chronology."""
    if weekday.ndim != 2 or slot.shape != weekday.shape:
        raise ValueError("weekday and slot must be aligned [B,T] tensors")
    if slots_per_day <= 0:
        raise ValueError("slots_per_day must be positive")
    if bool((weekday < 0).any()) or bool((weekday >= 7).any()):
        raise ValueError("weekday ids must be in [0,6]")
    if bool((slot < 0).any()) or bool((slot >= slots_per_day).any()):
        raise ValueError("slot ids are outside slots_per_day")
    weekly_slots = 7 * slots_per_day
    absolute = weekday.to(torch.long) * slots_per_day + slot.to(torch.long)
    shifted = torch.remainder(absolute + int(shift_slots), weekly_slots)
    return shifted // slots_per_day, shifted % slots_per_day


def pooling_effective_patch_count(weights: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Return entropy-based effective patch counts as [B,N]."""
    if weights.ndim != 3:
        raise ValueError("weights must be [B,P,N]")
    probabilities = weights.float().clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)
    return entropy.exp()


def mean_cosine_change(reference: torch.Tensor, changed: torch.Tensor) -> float:
    """Return mean one-minus-cosine distance over all leading positions."""
    if reference.shape != changed.shape or reference.ndim < 2:
        raise ValueError("reference and changed must share a shape ending in features")
    cosine = functional.cosine_similarity(reference.float(), changed.float(), dim=-1)
    return float((1.0 - cosine).mean().detach().cpu())


def topk_jaccard(reference_ids: torch.Tensor, changed_ids: torch.Tensor) -> torch.Tensor:
    """Set Jaccard for aligned [B,N,K] event ids, ignoring negative padding."""
    if reference_ids.shape != changed_ids.shape or reference_ids.ndim != 3:
        raise ValueError("reference_ids and changed_ids must be aligned [B,N,K]")
    batch, nodes, _ = reference_ids.shape
    result = torch.zeros((batch, nodes), dtype=torch.float32, device=reference_ids.device)
    for batch_index in range(batch):
        for node_index in range(nodes):
            left = set(
                int(value)
                for value in reference_ids[batch_index, node_index].detach().cpu().tolist()
                if int(value) >= 0
            )
            right = set(
                int(value)
                for value in changed_ids[batch_index, node_index].detach().cpu().tolist()
                if int(value) >= 0
            )
            union = left | right
            result[batch_index, node_index] = 1.0 if not union else len(left & right) / len(union)
    return result


def select_topk_event_ids(
    key_distance: torch.Tensor,
    event_ids: torch.Tensor,
    event_valid: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Select nearest candidate event ids independently for every query node."""
    if key_distance.ndim != 3:
        raise ValueError("key_distance must be [B,N,R]")
    batch, nodes, candidates = key_distance.shape
    if event_ids.shape != (batch, candidates) or event_valid.shape != event_ids.shape:
        raise ValueError("event_ids and event_valid must be aligned [B,R]")
    if k <= 0 or k > candidates:
        raise ValueError("k must be in [1,R]")
    # ``expand`` returns a shared view; clone before applying the finite-distance mask.
    valid = event_valid[:, None, :].expand(batch, nodes, candidates).clone()
    valid &= torch.isfinite(key_distance)
    masked = torch.where(valid, key_distance, torch.full_like(key_distance, torch.inf))
    local = torch.topk(masked, k, dim=-1, largest=False).indices
    expanded_ids = event_ids[:, None, :].expand(batch, nodes, candidates)
    selected = expanded_ids.gather(-1, local)
    selected_valid = valid.gather(-1, local)
    return torch.where(selected_valid, selected, torch.full_like(selected, -1))


def weekly_donor_pairs(
    context_end_indices: np.ndarray,
    *,
    weekly_steps: int,
    max_queries: int | None = None,
) -> np.ndarray:
    """Pair each selected dataset row with the same slot one week away."""
    ends = np.asarray(context_end_indices, dtype=np.int64)
    if ends.ndim != 1 or ends.size == 0:
        raise ValueError("context_end_indices must be a non-empty vector")
    if weekly_steps <= 0:
        raise ValueError("weekly_steps must be positive")
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive when provided")
    row_by_end = {int(end): row for row, end in enumerate(ends.tolist())}
    pairs: list[tuple[int, int]] = []
    for row, end in enumerate(ends.tolist()):
        donor = row_by_end.get(int(end - weekly_steps))
        if donor is None:
            donor = row_by_end.get(int(end + weekly_steps))
        if donor is not None and donor != row:
            pairs.append((row, donor))
    if not pairs:
        raise ValueError("validation split contains no same-calendar weekly donor pairs")
    if max_queries is not None and len(pairs) > max_queries:
        positions = np.rint(np.linspace(0, len(pairs) - 1, max_queries)).astype(np.int64)
        pairs = [pairs[int(position)] for position in positions]
    return np.asarray(pairs, dtype=np.int64)


def key_geometry_summary(
    keys: np.ndarray | torch.Tensor,
    *,
    pair_samples: int = 50_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Summarize norm, anisotropy, and centered effective rank of a key sample."""
    array = (
        keys.detach().float().cpu().numpy()
        if isinstance(keys, torch.Tensor)
        else np.asarray(keys, dtype=np.float32)
    )
    if array.ndim < 2:
        raise ValueError("keys must have one or more sample axes and one feature axis")
    flat = array.reshape(-1, array.shape[-1]).astype(np.float64, copy=False)
    finite = np.isfinite(flat).all(axis=1)
    flat = flat[finite]
    if flat.shape[0] < 2:
        raise ValueError("at least two finite keys are required")
    if pair_samples <= 0:
        raise ValueError("pair_samples must be positive")

    norms = np.linalg.norm(flat, axis=1)
    normalized = flat / np.maximum(norms[:, None], 1.0e-12)
    centered = flat - flat.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigen_sum = float(eigenvalues.sum())
    squared_sum = float(np.square(eigenvalues).sum())
    participation = 0.0 if squared_sum <= 1.0e-24 else eigen_sum**2 / squared_sum

    rng = np.random.default_rng(seed)
    maximum_pairs = flat.shape[0] * (flat.shape[0] - 1) // 2
    if pair_samples >= maximum_pairs and flat.shape[0] <= 4_096:
        left, right = np.triu_indices(flat.shape[0], k=1)
    else:
        left = rng.integers(0, flat.shape[0], size=pair_samples)
        right = rng.integers(0, flat.shape[0] - 1, size=pair_samples)
        right = right + (right >= left)
    cosine = np.sum(normalized[left] * normalized[right], axis=1)
    return {
        "sample_count": int(flat.shape[0]),
        "dimension": int(flat.shape[1]),
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
        "centered_variance": float(eigen_sum),
        "effective_rank_participation": float(participation),
        "pair_cosine_mean": float(cosine.mean()),
        "pair_cosine_std": float(cosine.std()),
        "pair_cosine_p95": float(np.quantile(cosine, 0.95)),
        "pair_cosine_p99": float(np.quantile(cosine, 0.99)),
    }


def _metric_array_summary(values: list[np.ndarray]) -> dict[str, float | int]:
    array = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std()) if array.size else 0.0,
    }


def _sample_bank_node_keys(
    bank: MemoryBank,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    if sample_count <= 1:
        raise ValueError("geometry sample_count must exceed one")
    total = bank.manifest.num_events * bank.manifest.num_nodes
    count = min(sample_count, total)
    rng = np.random.default_rng(seed)
    flat_ids = rng.choice(total, size=count, replace=False)
    event_ids = flat_ids // bank.manifest.num_nodes
    node_ids = flat_ids % bank.manifest.num_nodes
    return np.asarray(bank.node_keys[event_ids, node_ids], dtype=np.float32)


def _aggregation_for_version(version: str) -> tuple[Any, str]:
    """Return the forecast payload matching the frozen teacher version."""
    version = version.lower()
    if version == "hn_offset_only_v1":
        return offset_only_aggregation, "offset_only"
    if version == "hn_offset_decay_v2":
        return offset_decay_aggregation, "offset_decay"
    raise ValueError("version must be hn_offset_decay_v2 or hn_offset_only_v1")


@torch.inference_mode()
def run_retrieval_collapse_diagnostic(
    *,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    output_dir: str | Path,
    version: str = "hn_offset_decay_v2",
    candidate_protocol: str = "weekday_radius1_overlap",
    max_queries: int | None = 1_024,
    batch_size: int = 4,
    geometry_samples: int = 40_000,
    random_bank_path: str | Path | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Run frozen query interventions against one fixed historical Bank."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive when provided")
    if geometry_samples <= 1:
        raise ValueError("geometry_samples must exceed one")

    # Imports stay local to avoid coupling model code to visualization-only helpers.
    from stanchor.diagnostics.retrieval_visualization import (
        _candidate_node_keys,
        _candidate_teacher_signatures,
        anchor_wise_ranking_metrics,
        build_diagnostic_event_candidates,
        build_teacher_aligned_signature,
        future_information_boundary,
        node_key_distances,
        teacher_candidate_distances,
    )

    started = time.perf_counter()
    aggregation_fn, candidate_payload = _aggregation_for_version(version)
    device = torch.device(device_override) if device_override else resolve_device(config.runtime.device)
    if device.type == "cuda":
        # PyTorch 2.11 on Windows rejects a ``torch.device`` argument here.
        torch.cuda.reset_peak_memory_stats()
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    dataset = data.val
    donor_pairs = weekly_donor_pairs(
        dataset.context_end_indices,
        weekly_steps=7 * data.series.slots_per_day,
        max_queries=max_queries,
    )
    checkpoint = resolve_project_path(checkpoint_path)
    bank_source = resolve_project_path(bank_path)
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model, checkpoint_payload = load_pretrained_model(
        config,
        checkpoint,
        data.series.slots_per_day,
        device,
    )
    model.eval()

    conditions = ("original", "shape_swap", "patch_reverse", "level_swap", "calendar_shift")
    hidden_change: dict[str, list[float]] = {name: [] for name in conditions[1:]}
    key_change: dict[str, list[float]] = {name: [] for name in conditions[1:]}
    topk_overlap: dict[str, list[np.ndarray]] = {name: [] for name in conditions[1:]}
    effective_patches: dict[str, list[np.ndarray]] = {name: [] for name in conditions}
    ranking_values: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            "spearman": [],
            "kendall": [],
            "recall_at_1": [],
            "ndcg_at_5": [],
            "recall_at_5": [],
        }
        for name in conditions
    }
    forecasting = {
        name: ForecastMetricAccumulator(config.data.horizon) for name in conditions
    }
    calendar_shift_slots = 3 * data.series.slots_per_day + data.series.slots_per_day // 2
    processed = 0

    with MemoryBank(bank_source) as bank:
        if bank.manifest.num_nodes != data.series.num_nodes:
            raise ValueError("Bank and validation data node counts do not match")
        if bank.manifest.retrieval_dim != config.model.retrieval_dim:
            raise ValueError("Bank and model retrieval dimensions do not match")
        trained_geometry = key_geometry_summary(
            _sample_bank_node_keys(bank, sample_count=geometry_samples, seed=config.runtime.seed),
            seed=config.runtime.seed,
        )
        random_geometry = None
        if random_bank_path is not None:
            with MemoryBank(resolve_project_path(random_bank_path)) as random_bank:
                random_geometry = key_geometry_summary(
                    _sample_bank_node_keys(
                        random_bank,
                        sample_count=geometry_samples,
                        seed=config.runtime.seed,
                    ),
                    seed=config.runtime.seed,
                )

        retriever = TwoStageRetriever(
            bank,
            config.bank.event_top_r,
            config.bank.node_top_k,
            config.bank.level_weight,
            config.bank.level_temperature,
            config.bank.search_temperature,
            device,
        )
        for start in range(0, donor_pairs.shape[0], batch_size):
            pair_chunk = donor_pairs[start : start + batch_size]
            query_batch = default_collate([dataset[int(row)] for row in pair_chunk[:, 0]])
            donor_batch = default_collate([dataset[int(row)] for row in pair_chunk[:, 1]])
            if not torch.equal(query_batch["query_weekday"], donor_batch["query_weekday"]):
                raise ValueError("weekly donors must preserve query weekday")
            if not torch.equal(query_batch["query_slot"], donor_batch["query_slot"]):
                raise ValueError("weekly donors must preserve query slot")

            query_x = query_batch["retrieval_x"].to(device)
            query_observed = query_batch["retrieval_observed"].to(device).bool()
            donor_x = donor_batch["retrieval_x"].to(device)
            donor_observed = donor_batch["retrieval_observed"].to(device).bool()
            weekday = query_batch["retrieval_weekday"].to(device)
            slot = query_batch["retrieval_slot"].to(device)
            query_statistics = normalize_window(query_x, query_observed)
            donor_statistics = normalize_window(donor_x, donor_observed)
            shifted_weekday, shifted_slot = shift_calendar(
                weekday,
                slot,
                slots_per_day=data.series.slots_per_day,
                shift_slots=calendar_shift_slots,
            )
            patch_size = config.model.patch_size
            encodings = {
                "original": encode_components(
                    model,
                    normalized=query_statistics.normalized,
                    level_features=query_statistics.level_features,
                    level_valid=query_statistics.level_valid,
                    weekday=weekday,
                    slot=slot,
                    observed=query_observed,
                    graph=graph,
                ),
                "shape_swap": encode_components(
                    model,
                    normalized=donor_statistics.normalized,
                    level_features=query_statistics.level_features,
                    level_valid=query_statistics.level_valid,
                    weekday=weekday,
                    slot=slot,
                    observed=donor_observed,
                    graph=graph,
                ),
                "patch_reverse": encode_components(
                    model,
                    normalized=reverse_temporal_patches(
                        query_statistics.normalized, patch_size
                    ),
                    level_features=query_statistics.level_features,
                    level_valid=query_statistics.level_valid,
                    weekday=weekday,
                    slot=slot,
                    observed=reverse_temporal_patches(query_observed, patch_size),
                    graph=graph,
                ),
                "level_swap": encode_components(
                    model,
                    normalized=query_statistics.normalized,
                    level_features=donor_statistics.level_features,
                    level_valid=donor_statistics.level_valid,
                    weekday=weekday,
                    slot=slot,
                    observed=query_observed,
                    graph=graph,
                ),
                "calendar_shift": encode_components(
                    model,
                    normalized=query_statistics.normalized,
                    level_features=query_statistics.level_features,
                    level_valid=query_statistics.level_valid,
                    weekday=shifted_weekday,
                    slot=shifted_slot,
                    observed=query_observed,
                    graph=graph,
                ),
            }
            original = encodings["original"]
            for name, encoding in encodings.items():
                effective_patches[name].append(
                    pooling_effective_patch_count(
                        encoding.retrieval.pooling_weights
                    ).detach().cpu().numpy()
                )
                if name != "original":
                    hidden_change[name].append(
                        mean_cosine_change(original.hidden, encoding.hidden)
                    )
                    key_change[name].append(
                        mean_cosine_change(
                            original.retrieval.node_keys,
                            encoding.retrieval.node_keys,
                        )
                    )

            events = build_diagnostic_event_candidates(
                bank,
                query_batch["query_weekday"].to(device),
                query_batch["query_slot"].to(device),
                query_batch["context_start"].to(device),
                config.bank.event_top_r,
                device,
                candidate_protocol,
            )
            candidate_keys = _candidate_node_keys(bank, events.event_ids, device)
            event_future, event_future_valid = event_candidate_futures(
                bank, events.event_ids, events.valid, device
            )
            candidate_future = event_future.permute(0, 3, 1, 2, 4).contiguous()
            candidate_future_valid = event_future_valid.permute(0, 3, 1, 2, 4).contiguous()
            query_signature, query_signature_valid = build_teacher_aligned_signature(
                version,
                query_batch["y"].to(device),
                query_batch["y_observed"].to(device),
                query_batch["x"].to(device),
                query_batch["x_observed"].to(device),
            )
            candidate_signature, candidate_signature_valid = _candidate_teacher_signatures(
                version,
                bank,
                events,
                candidate_future,
                candidate_future_valid,
                data,
                config.data.context_length,
                device,
            )
            teacher_distance, teacher_valid = teacher_candidate_distances(
                query_signature,
                query_signature_valid,
                candidate_signature,
                candidate_signature_valid,
                events.valid,
                config.pretrain.relation_distance_normalization,
            )

            top_ids: dict[str, torch.Tensor] = {}
            aggregations = {}
            for name, encoding in encodings.items():
                key_distance, key_valid = node_key_distances(
                    encoding.retrieval.node_keys,
                    candidate_keys,
                    events.valid,
                )
                common_valid = teacher_valid & key_valid
                ranking = anchor_wise_ranking_metrics(
                    key_distance.detach().cpu().numpy(),
                    teacher_distance.detach().cpu().numpy(),
                    common_valid.detach().cpu().numpy(),
                    teacher_temperature=config.pretrain.relation_teacher_temperature,
                )
                for metric_name, output_name in (
                    ("spearman", "spearman_values"),
                    ("kendall", "kendall_values"),
                    ("recall_at_1", "recall_at_1_values"),
                    ("ndcg_at_5", "ndcg_at_5_values"),
                    ("recall_at_5", "recall_at_5_values"),
                ):
                    ranking_values[name][metric_name].append(ranking[output_name])
                top_ids[name] = select_topk_event_ids(
                    key_distance,
                    events.event_ids,
                    events.valid,
                    k=config.bank.node_top_k,
                )
                node_candidates = retriever.rerank_nodes(
                    encoding.retrieval.node_keys,
                    query_statistics.level_features,
                    events,
                )
                aggregations[name] = aggregation_fn(
                    node_candidates,
                    query_batch["x"].to(device),
                    query_batch["x_observed"].to(device),
                    bank,
                    data.series,
                    data.scaler,
                    config.data.context_length,
                    device,
                )

            for name in conditions[1:]:
                topk_overlap[name].append(
                    topk_jaccard(top_ids["original"], top_ids[name]).detach().cpu().numpy()
                )

            target = query_batch["y"].to(device)
            target_valid = query_batch["y_observed"].to(device).bool()
            shared_valid = target_valid.clone()
            for aggregation in aggregations.values():
                shared_valid &= aggregation.valid
            target_physical = data.scaler.inverse_transform_torch(target)
            for name, aggregation in aggregations.items():
                forecasting[name].update(
                    data.scaler.inverse_transform_torch(aggregation.prediction),
                    target_physical,
                    shared_valid,
                )
            processed += int(query_x.shape[0])
            print(
                f"[retrieval-collapse] processed {processed}/{donor_pairs.shape[0]} queries",
                flush=True,
            )

    ranking_result = {
        condition: {
            metric: _metric_array_summary(chunks)
            for metric, chunks in metric_chunks.items()
        }
        for condition, metric_chunks in ranking_values.items()
    }
    sensitivity = {
        name: {
            "hidden_cosine_change": {
                "mean": float(np.mean(hidden_change[name])),
                "std": float(np.std(hidden_change[name])),
            },
            "key_cosine_change": {
                "mean": float(np.mean(key_change[name])),
                "std": float(np.std(key_change[name])),
            },
            "topk_jaccard": _metric_array_summary(topk_overlap[name]),
        }
        for name in conditions[1:]
    }
    result: dict[str, Any] = {
        "diagnostic": "retrieval_context_reliance",
        "version": version,
        "candidate_protocol": candidate_protocol,
        "candidate_payload": candidate_payload,
        "query_count": processed,
        "donor_rule": "same calendar position exactly seven days away",
        "interventions": {
            "shape_swap": "donor normalized history plus original level and calendar",
            "patch_reverse": "reverse 24 patches while preserving values inside each patch",
            "level_swap": "donor mean/std/last/slope plus original shape and calendar",
            "calendar_shift": f"circular weekly shift of {calendar_shift_slots} slots",
        },
        "future_information_boundary": future_information_boundary(),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "epoch": checkpoint_payload.get("epoch"),
        },
        "bank": str(bank_source.resolve()),
        "geometry": {
            "trained": trained_geometry,
            "random": random_geometry,
        },
        "sensitivity": sensitivity,
        "pooling_effective_patches": {
            name: _metric_array_summary(chunks) for name, chunks in effective_patches.items()
        },
        "future_ranking": ranking_result,
        "memory_forecasting": {
            name: accumulator.compute() for name, accumulator in forecasting.items()
        },
        "runtime": {
            "device": str(device),
            "batch_size": batch_size,
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_cuda_memory_gb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            ),
        },
    }
    save_json(output_path / "collapse_diagnostic.json", result)
    return result

"""E5 trend-residual retrieval diagnostics with explicit future boundaries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig
from stanchor.diagnostics.retrieval import _method_result, _summary
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.retrieval.strategies import (
    candidate_contexts,
    event_candidate_futures,
    select_candidate_set_by_node,
)
from stanchor.retrieval.trend_residual import (
    LocalTrendStatistics,
    estimate_local_trend,
    masked_future_l1_scores,
    masked_pearson_candidate_scores,
    masked_spearman_rank_correlation,
    match_selected_event_positions,
    reconstruct_future,
    residualize_context,
    residualize_future,
    softmax_topk_weights,
    weighted_candidate_mean,
)
from stanchor.utils import resolve_device


DEPLOYABLE_METHODS = (
    "learned_raw_topk",
    "learned_offset_topk",
    "learned_offset_decay_topk",
    "learned_trend_topk",
    "fixed_offset_topk",
    "fixed_trend_topk",
)
DIAGNOSTIC_ONLY_METHODS = ("future_oracle_trend_top1",)
METHOD_NAMES = DEPLOYABLE_METHODS + DIAGNOSTIC_ONLY_METHODS


def _expand_query_statistics(
    statistics: LocalTrendStatistics,
    candidate_count: int,
) -> LocalTrendStatistics:
    def expand(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(-1, candidate_count, -1, -1)

    return LocalTrendStatistics(
        level=expand(statistics.level),
        slope=expand(statistics.slope),
        scale=expand(statistics.scale),
        valid=expand(statistics.valid),
    )


def _unit_scale_statistics(
    statistics: LocalTrendStatistics,
) -> LocalTrendStatistics:
    """Keep level/trend coordinates while disabling unstable scale transfer."""
    return LocalTrendStatistics(
        level=statistics.level,
        slope=statistics.slope,
        scale=torch.ones_like(statistics.scale),
        valid=statistics.valid,
    )


def _horizon_decay_blend(
    raw: torch.Tensor,
    aligned: torch.Tensor,
) -> torch.Tensor:
    """Linearly decay query-level alignment from one to zero over the horizon."""
    if raw.shape != aligned.shape or raw.ndim != 4:
        raise ValueError("raw and aligned predictions must share [B,H,N,C]")
    horizon = raw.shape[1]
    decay = torch.linspace(
        1.0,
        0.0,
        horizon,
        dtype=raw.dtype,
        device=raw.device,
    ).view(1, horizon, 1, 1)
    return raw + decay * (aligned - raw)


def _select_and_aggregate(
    candidate_values: torch.Tensor,
    candidate_valid: torch.Tensor,
    selected: torch.Tensor,
    selected_valid: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_values, selected_value_valid = select_candidate_set_by_node(
        candidate_values,
        candidate_valid,
        selected,
        selected_valid,
    )
    return weighted_candidate_mean(selected_values, selected_value_valid, weights)


def assemble_trend_residual_methods(
    *,
    query_context: torch.Tensor,
    query_observed: torch.Tensor,
    target: torch.Tensor,
    target_observed: torch.Tensor,
    candidate_contexts: torch.Tensor,
    candidate_context_observed: torch.Tensor,
    candidate_futures: torch.Tensor,
    candidate_future_observed: torch.Tensor,
    event_ids: torch.Tensor,
    event_valid: torch.Tensor,
    learned_event_ids: torch.Tensor,
    learned_valid: torch.Tensor,
    learned_weights: torch.Tensor,
    trend_length: int,
    fixed_top_k: int,
    temperature: float,
) -> dict[str, Any]:
    """Build comparable raw, residual and oracle predictions for one batch."""
    if query_context.ndim != 4 or target.ndim != 4:
        raise ValueError("query_context and target must be [B,T/H,N,C]")
    if candidate_contexts.ndim != 5:
        raise ValueError("candidate_contexts must be [B,R,T,N,C]")
    if candidate_futures.ndim != 5:
        raise ValueError("candidate_futures must be [B,H,N,R,C]")
    batch, candidate_count, _, nodes, channels = candidate_contexts.shape
    if candidate_futures.shape[0] != batch or candidate_futures.shape[2:] != (
        nodes,
        candidate_count,
        channels,
    ):
        raise ValueError("candidate future axes must be [B,H,N,R,C]")

    query_offset_estimate = estimate_local_trend(
        query_context,
        query_observed,
        trend_length,
        mode="offset",
    )
    query_trend_estimate = estimate_local_trend(
        query_context,
        query_observed,
        trend_length,
        mode="trend",
    )
    candidate_offset_estimate = estimate_local_trend(
        candidate_contexts,
        candidate_context_observed,
        trend_length,
        mode="offset",
    )
    candidate_trend_estimate = estimate_local_trend(
        candidate_contexts,
        candidate_context_observed,
        trend_length,
        mode="trend",
    )
    query_offset_stats = _unit_scale_statistics(query_offset_estimate)
    query_trend_stats = _unit_scale_statistics(query_trend_estimate)
    candidate_offset_stats = _unit_scale_statistics(candidate_offset_estimate)
    candidate_trend_stats = _unit_scale_statistics(candidate_trend_estimate)

    # [B,H,N,R,C] -> [B,R,H,N,C], aligning the event axis with context stats.
    event_future = candidate_futures.permute(0, 3, 1, 2, 4).contiguous()
    event_future_valid = candidate_future_observed.permute(0, 3, 1, 2, 4).contiguous()
    offset_future_residual, offset_future_valid = residualize_future(
        event_future,
        event_future_valid,
        candidate_offset_stats,
    )
    trend_future_residual, trend_future_valid = residualize_future(
        event_future,
        event_future_valid,
        candidate_trend_stats,
    )
    reconstructed_offset, reconstructed_offset_valid = reconstruct_future(
        offset_future_residual,
        offset_future_valid,
        _expand_query_statistics(query_offset_stats, candidate_count),
    )
    reconstructed_trend, reconstructed_trend_valid = reconstruct_future(
        trend_future_residual,
        trend_future_valid,
        _expand_query_statistics(query_trend_stats, candidate_count),
    )

    raw_pool = candidate_futures
    raw_pool_valid = candidate_future_observed.bool()
    offset_pool = reconstructed_offset.permute(0, 2, 3, 1, 4).contiguous()
    offset_pool_valid = reconstructed_offset_valid.permute(0, 2, 3, 1, 4).contiguous()
    trend_pool = reconstructed_trend.permute(0, 2, 3, 1, 4).contiguous()
    trend_pool_valid = reconstructed_trend_valid.permute(0, 2, 3, 1, 4).contiguous()

    learned_positions, learned_position_valid = match_selected_event_positions(
        event_ids,
        learned_event_ids,
        learned_valid,
    )
    learned_raw, learned_raw_valid = _select_and_aggregate(
        raw_pool,
        raw_pool_valid,
        learned_positions,
        learned_position_valid,
        learned_weights,
    )
    learned_offset, learned_offset_valid = _select_and_aggregate(
        offset_pool,
        offset_pool_valid,
        learned_positions,
        learned_position_valid,
        learned_weights,
    )
    learned_offset_decay = _horizon_decay_blend(learned_raw, learned_offset)
    learned_offset_decay_valid = learned_raw_valid & learned_offset_valid
    learned_trend, learned_trend_valid = _select_and_aggregate(
        trend_pool,
        trend_pool_valid,
        learned_positions,
        learned_position_valid,
        learned_weights,
    )

    query_offset_context, query_offset_context_valid = residualize_context(
        query_context,
        query_observed,
        query_offset_stats,
    )
    candidate_offset_context, candidate_offset_context_valid = residualize_context(
        candidate_contexts,
        candidate_context_observed,
        candidate_offset_stats,
    )
    offset_history_scores, offset_history_valid = masked_pearson_candidate_scores(
        query_offset_context,
        query_offset_context_valid,
        candidate_offset_context,
        candidate_offset_context_valid,
        event_valid,
    )
    offset_selected, offset_selected_valid, offset_weights = softmax_topk_weights(
        offset_history_scores,
        offset_history_valid,
        fixed_top_k,
        temperature,
        largest=True,
    )
    fixed_offset, fixed_offset_valid = _select_and_aggregate(
        offset_pool,
        offset_pool_valid,
        offset_selected,
        offset_selected_valid,
        offset_weights,
    )

    query_trend_context, query_trend_context_valid = residualize_context(
        query_context,
        query_observed,
        query_trend_stats,
    )
    candidate_trend_context, candidate_trend_context_valid = residualize_context(
        candidate_contexts,
        candidate_context_observed,
        candidate_trend_stats,
    )
    trend_history_scores, trend_history_valid = masked_pearson_candidate_scores(
        query_trend_context,
        query_trend_context_valid,
        candidate_trend_context,
        candidate_trend_context_valid,
        event_valid,
    )
    trend_selected, trend_selected_valid, trend_weights = softmax_topk_weights(
        trend_history_scores,
        trend_history_valid,
        fixed_top_k,
        temperature,
        largest=True,
    )
    fixed_trend, fixed_trend_valid = _select_and_aggregate(
        trend_pool,
        trend_pool_valid,
        trend_selected,
        trend_selected_valid,
        trend_weights,
    )

    query_offset_future, query_offset_future_valid = residualize_future(
        target,
        target_observed,
        query_offset_stats,
    )
    offset_future_scores, offset_future_score_valid = masked_future_l1_scores(
        query_offset_future,
        query_offset_future_valid,
        offset_future_residual,
        offset_future_valid,
        event_valid,
    )
    query_trend_future, query_trend_future_valid = residualize_future(
        target,
        target_observed,
        query_trend_stats,
    )
    trend_future_scores, trend_future_score_valid = masked_future_l1_scores(
        query_trend_future,
        query_trend_future_valid,
        trend_future_residual,
        trend_future_valid,
        event_valid,
    )
    oracle_selected, oracle_selected_valid, oracle_weights = softmax_topk_weights(
        trend_future_scores,
        trend_future_score_valid,
        top_k=1,
        temperature=temperature,
        largest=False,
    )
    oracle_trend, oracle_trend_valid = _select_and_aggregate(
        trend_pool,
        trend_pool_valid,
        oracle_selected,
        oracle_selected_valid,
        oracle_weights,
    )

    offset_rank, offset_rank_valid = masked_spearman_rank_correlation(
        offset_history_scores,
        -offset_future_scores,
        offset_history_valid & offset_future_score_valid,
    )
    trend_rank, trend_rank_valid = masked_spearman_rank_correlation(
        trend_history_scores,
        -trend_future_scores,
        trend_history_valid & trend_future_score_valid,
    )

    return {
        "predictions": {
            "learned_raw_topk": learned_raw,
            "learned_offset_topk": learned_offset,
            "learned_offset_decay_topk": learned_offset_decay,
            "learned_trend_topk": learned_trend,
            "fixed_offset_topk": fixed_offset,
            "fixed_trend_topk": fixed_trend,
            "future_oracle_trend_top1": oracle_trend,
        },
        "valid": {
            "learned_raw_topk": learned_raw_valid,
            "learned_offset_topk": learned_offset_valid,
            "learned_offset_decay_topk": learned_offset_decay_valid,
            "learned_trend_topk": learned_trend_valid,
            "fixed_offset_topk": fixed_offset_valid,
            "fixed_trend_topk": fixed_trend_valid,
            "future_oracle_trend_top1": oracle_trend_valid,
        },
        "rank_correlation": {
            "offset": offset_rank,
            "offset_valid": offset_rank_valid,
            "trend": trend_rank,
            "trend_valid": trend_rank_valid,
        },
        "deployable_methods": DEPLOYABLE_METHODS,
        "diagnostic_only_methods": DIAGNOSTIC_ONLY_METHODS,
        "future_information_boundary": "oracle_and_rank_diagnostics_only",
        "payload_scale_mode": "unit",
        "offset_decay": "linear_1_to_0",
    }


def _optional_summary(values: list[torch.Tensor]) -> dict[str, float | int]:
    nonempty = [value for value in values if value.numel() > 0]
    if not nonempty:
        return {"count": 0}
    return _summary(nonempty)


@torch.no_grad()
def diagnose_trend_residual_value(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    split: str = "val",
    trend_length: int = 12,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run T0 with a fixed E3 encoder, causal event pool and no confidence."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if trend_length <= 0 or trend_length > config.data.context_length:
        raise ValueError("trend_length must fit data.context_length")
    started = time.perf_counter()
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    model, checkpoint = load_pretrained_model(
        config,
        checkpoint_path,
        data.series.slots_per_day,
        device,
    )
    model.eval()
    dataset: Dataset = getattr(data, split)
    loader = DataLoader(
        dataset,
        batch_size=config.target.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    accumulators = {
        name: ForecastMetricAccumulator(config.data.horizon) for name in METHOD_NAMES
    }
    individual_valid_counts = {name: 0 for name in METHOD_NAMES}
    target_count = 0
    common_valid_count = 0
    query_count = 0
    legal_candidate_counts: list[torch.Tensor] = []
    offset_rank_values: list[torch.Tensor] = []
    trend_rank_values: list[torch.Tensor] = []

    with MemoryBank(bank_path) as bank:
        _validate_bank(bank, model, graph_cpu, data.scaler.state_dict())
        retriever = TwoStageRetriever(
            bank,
            config.bank.event_top_r,
            config.bank.node_top_k,
            config.bank.level_weight,
            config.bank.level_temperature,
            config.bank.search_temperature,
            device,
        )
        bank_future_end = np.asarray(bank.future_end)
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            query_context = batch["x"].to(device)
            query_observed = batch["x_observed"].to(device).bool()
            target = batch["y"].to(device)
            target_observed = batch["y_observed"].to(device).bool()
            encoding = model.encode_clean(
                batch["retrieval_x"].to(device),
                batch["retrieval_observed"].to(device),
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            events = retriever.search_events(
                encoding.retrieval.event_keys,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
            )
            learned_candidates = retriever.rerank_nodes(
                encoding.retrieval.node_keys,
                encoding.statistics.level_features,
                events,
            )

            legal_counts = []
            for weekday, slot, context_start in zip(
                batch["query_weekday"].tolist(),
                batch["query_slot"].tolist(),
                batch["context_start"].tolist(),
            ):
                ids = bank.calendar.lookup(int(weekday), int(slot))
                legal_counts.append(int((bank_future_end[ids] < int(context_start)).sum()))
            legal_count_tensor = torch.tensor(legal_counts, device=device)
            if bool((legal_count_tensor > config.bank.event_top_r).any()):
                raise ValueError(
                    "event_top_r truncates the legal calendar pool; increase it for exact T0"
                )

            contexts, contexts_observed = candidate_contexts(
                bank,
                events.event_ids,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            futures, futures_observed = event_candidate_futures(
                bank,
                events.event_ids,
                events.valid,
                device,
            )
            assembled = assemble_trend_residual_methods(
                query_context=query_context,
                query_observed=query_observed,
                target=target,
                target_observed=target_observed,
                candidate_contexts=contexts,
                candidate_context_observed=contexts_observed,
                candidate_futures=futures,
                candidate_future_observed=futures_observed,
                event_ids=events.event_ids,
                event_valid=events.valid,
                learned_event_ids=learned_candidates.event_ids,
                learned_valid=learned_candidates.valid,
                learned_weights=learned_candidates.weights,
                trend_length=trend_length,
                fixed_top_k=config.bank.node_top_k,
                temperature=config.bank.search_temperature,
            )

            common_valid = target_observed.clone()
            for name in DEPLOYABLE_METHODS:
                common_valid &= assembled["valid"][name]
            for name in METHOD_NAMES:
                individual_valid = target_observed & assembled["valid"][name]
                individual_valid_counts[name] += int(individual_valid.sum().item())
            target_count += int(target_observed.sum().item())
            common_valid_count += int(common_valid.sum().item())
            target_physical = data.scaler.inverse_transform_torch(target)
            for name in METHOD_NAMES:
                prediction_physical = data.scaler.inverse_transform_torch(
                    assembled["predictions"][name]
                )
                evaluation_mask = common_valid & assembled["valid"][name]
                accumulators[name].update(prediction_physical, target_physical, evaluation_mask)

            rank = assembled["rank_correlation"]
            offset_rank_values.append(rank["offset"][rank["offset_valid"]])
            trend_rank_values.append(rank["trend"][rank["trend_valid"]])
            legal_candidate_counts.append(legal_count_tensor)
            query_count += int(query_context.shape[0])

        if query_count == 0:
            raise ValueError("diagnostic processed zero queries")
        methods = {
            name: _method_result(accumulators[name], individual_valid_counts[name], target_count)
            for name in METHOD_NAMES
        }
        raw_mae = float(methods["learned_raw_topk"]["mae"])
        oracle_mae = float(methods["future_oracle_trend_top1"]["mae"])
        gains = {}
        for name in DEPLOYABLE_METHODS:
            method_mae = float(methods[name]["mae"])
            gain = raw_mae - method_mae
            gains[name] = {
                "absolute_mae_gain_vs_learned_raw": gain,
                "relative_mae_gain_percent_vs_learned_raw": 100.0 * gain / raw_mae,
                "mae_gap_to_future_oracle_trend_top1": method_mae - oracle_mae,
            }

        return {
            "schema_version": 1,
            "experiment": "E5_T0_trend_residual",
            "dataset": bank.manifest.dataset_name,
            "split": split,
            "queries": query_count,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "bank": str(Path(bank_path).resolve()),
            "bank_events": bank.manifest.num_events,
            "trend_length": trend_length,
            "payload_scale_mode": "unit",
            "offset_decay": "linear_1_to_0",
            "common_evaluation_coverage": common_valid_count / max(target_count, 1),
            "methods": methods,
            "gains": gains,
            "selector": {
                "legal_candidate_count": _summary(legal_candidate_counts),
                "past_future_rank_correlation": {
                    "offset": _optional_summary(offset_rank_values),
                    "trend": _optional_summary(trend_rank_values),
                },
            },
            "deployable_methods": DEPLOYABLE_METHODS,
            "diagnostic_only_methods": DIAGNOSTIC_ONLY_METHODS,
            "future_information_boundary": "oracle_and_rank_diagnostics_only",
            "confidence_used": False,
            "config": {
                "event_top_r": config.bank.event_top_r,
                "node_top_k": config.bank.node_top_k,
                "level_weight": config.bank.level_weight,
                "search_temperature": config.bank.search_temperature,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }

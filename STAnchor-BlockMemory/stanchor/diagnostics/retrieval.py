"""Retrieval-value diagnostics and transparent non-parametric baselines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.retrieval.strategies import (
    candidate_contexts as _candidate_contexts,
    event_candidate_futures as _event_futures,
    masked_candidate_mean,
    raw_l1_candidate_scores,
    select_candidate_set_by_node,
)
from stanchor.utils import resolve_device


def effective_support_size(weights: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return 1 / sum(w^2) for each node-level candidate distribution."""
    if weights.ndim != 3 or valid.shape != weights.shape:
        raise ValueError("weights and valid must be [B, N, K]")
    masked = torch.where(valid.bool(), weights, torch.zeros_like(weights))
    total = masked.sum(dim=-1, keepdim=True)
    normalized = masked / total.clamp_min(1.0e-8)
    support = normalized.square().sum(dim=-1).clamp_min(1.0e-8).reciprocal()
    return torch.where(total.squeeze(-1) > 0, support, torch.zeros_like(support))


def select_oracle_future(
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    target: torch.Tensor,
    target_observed: torch.Tensor,
    event_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one historical event per node using the query's true future."""
    if candidates.ndim != 5 or candidate_valid.shape != candidates.shape:
        raise ValueError("candidates and candidate_valid must be [B, H, N, R, C]")
    batch, horizon, nodes, candidate_count, channels = candidates.shape
    if target.shape != (batch, horizon, nodes, channels) or target_observed.shape != target.shape:
        raise ValueError("target and target_observed must be [B, H, N, C]")
    if event_valid.shape != (batch, candidate_count):
        raise ValueError("event_valid must be [B, R]")

    common = (
        candidate_valid.bool()
        & target_observed.unsqueeze(3).bool()
        & event_valid[:, None, None, :, None].bool()
    )
    count = common.sum(dim=(1, 4))  # [B, N, R]
    absolute = (candidates - target.unsqueeze(3)).abs()
    error = torch.where(common, absolute, torch.zeros_like(absolute)).sum(dim=(1, 4))
    candidate_mae = (error / count.clamp_min(1)).masked_fill(count == 0, torch.inf)
    selected = candidate_mae.argmin(dim=-1)  # [B, N]
    has_candidate = torch.isfinite(candidate_mae).any(dim=-1)

    gather_index = selected[:, None, :, None, None].expand(
        batch,
        horizon,
        nodes,
        1,
        channels,
    )
    prediction = candidates.gather(3, gather_index).squeeze(3)
    prediction_valid = candidate_valid.gather(3, gather_index).squeeze(3)
    prediction_valid = prediction_valid & has_candidate[:, None, :, None]
    prediction = torch.where(prediction_valid, prediction, torch.zeros_like(prediction))
    return prediction, prediction_valid, selected


def _select_candidate_by_node(
    candidates: torch.Tensor,
    candidate_valid: torch.Tensor,
    selected: torch.Tensor,
    selected_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, horizon, nodes, _, channels = candidates.shape
    if selected.shape != (batch, nodes) or selected_valid.shape != selected.shape:
        raise ValueError("selected and selected_valid must be [B, N]")
    gather_index = selected[:, None, :, None, None].expand(batch, horizon, nodes, 1, channels)
    prediction = candidates.gather(3, gather_index).squeeze(3)
    valid = candidate_valid.gather(3, gather_index).squeeze(3)
    valid = valid & selected_valid[:, None, :, None]
    return torch.where(valid, prediction, torch.zeros_like(prediction)), valid


def _summary(values: list[torch.Tensor]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty diagnostic")
    merged = torch.cat([value.detach().double().cpu().reshape(-1) for value in values])
    if merged.numel() == 0:
        raise ValueError("cannot summarize zero valid diagnostic values")
    quantiles = torch.quantile(merged, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    return {
        "count": int(merged.numel()),
        "mean": float(merged.mean()),
        "std": float(merged.std(unbiased=False)),
        "min": float(merged.min()),
        "q10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q90": float(quantiles[2]),
        "max": float(merged.max()),
    }


def _method_result(
    accumulator: ForecastMetricAccumulator,
    valid_count: int,
    target_count: int,
) -> dict[str, Any]:
    result = accumulator.compute()
    result["coverage"] = valid_count / max(target_count, 1)
    result["valid_values"] = valid_count
    return result


@torch.no_grad()
def diagnose_retrieval_value(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    split: str = "val",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Compare learned retrieval against simple and oracle historical baselines."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
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
    method_names = (
        "weekly_mean",
        "raw_l1_top1",
        "raw_l1_topk",
        "learned_top1",
        "learned_uniform_topk",
        "learned_topk",
        "oracle_top1",
    )
    accumulators = {
        name: ForecastMetricAccumulator(config.data.horizon) for name in method_names
    }
    individual_valid_counts = {name: 0 for name in method_names}
    common_valid_count = 0
    target_count = 0
    query_count = 0
    candidate_counts: list[torch.Tensor] = []
    node_candidate_counts: list[torch.Tensor] = []
    top1_weights: list[torch.Tensor] = []
    effective_support: list[torch.Tensor] = []

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
            x = batch["x"].to(device)
            x_observed = batch["x_observed"].to(device)
            target = batch["y"].to(device)
            target_observed = batch["y_observed"].to(device).bool()
            encoding = model.encode_clean(
                x,
                x_observed,
                batch["weekday"].to(device),
                batch["slot"].to(device),
                graph,
            )
            events = retriever.search_events(
                encoding.retrieval.event_keys,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
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
                    "event_top_r truncates the legal calendar pool; increase it for an exact diagnostic"
                )
            candidates = retriever.rerank_nodes(
                encoding.retrieval.node_keys,
                encoding.statistics.level_features,
                events,
            )
            aggregation = retriever.aggregate(candidates)

            event_future, event_future_valid = _event_futures(
                bank,
                events.event_ids,
                events.valid,
                device,
            )
            weekly_prediction, weekly_valid = masked_candidate_mean(
                event_future,
                event_future_valid,
            )
            oracle_prediction, oracle_valid, _ = select_oracle_future(
                event_future,
                event_future_valid,
                target,
                target_observed,
                events.valid,
            )

            context_values, context_observed = _candidate_contexts(
                bank,
                events.event_ids,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            raw_scores, raw_score_valid = raw_l1_candidate_scores(
                x,
                x_observed,
                context_values,
                context_observed,
                events.valid,
            )
            raw_selected = raw_scores.argmin(dim=-1)
            raw_prediction, raw_valid = _select_candidate_by_node(
                event_future,
                event_future_valid,
                raw_selected,
                raw_score_valid.any(dim=-1),
            )
            raw_topk_scores, raw_topk_selected = torch.topk(
                raw_scores,
                config.bank.node_top_k,
                dim=-1,
                largest=False,
            )
            raw_topk_selected_valid = torch.isfinite(raw_topk_scores)
            raw_topk_futures, raw_topk_future_valid = select_candidate_set_by_node(
                event_future,
                event_future_valid,
                raw_topk_selected,
                raw_topk_selected_valid,
            )
            raw_topk_prediction, raw_topk_valid = masked_candidate_mean(
                raw_topk_futures,
                raw_topk_future_valid,
            )

            learned_top1_prediction = aggregation.candidate_futures[:, :, :, 0, :]
            learned_top1_valid = aggregation.candidate_masks[:, :, :, 0, :]
            learned_uniform_prediction, learned_uniform_valid = masked_candidate_mean(
                aggregation.candidate_futures,
                aggregation.candidate_masks,
            )
            predictions = {
                "weekly_mean": weekly_prediction,
                "raw_l1_top1": raw_prediction,
                "raw_l1_topk": raw_topk_prediction,
                "learned_top1": learned_top1_prediction,
                "learned_uniform_topk": learned_uniform_prediction,
                "learned_topk": aggregation.prediction,
                "oracle_top1": oracle_prediction,
            }
            method_valid = {
                "weekly_mean": weekly_valid,
                "raw_l1_top1": raw_valid,
                "raw_l1_topk": raw_topk_valid,
                "learned_top1": learned_top1_valid,
                "learned_uniform_topk": learned_uniform_valid,
                "learned_topk": aggregation.valid,
                "oracle_top1": oracle_valid,
            }
            common_valid = target_observed.clone()
            for name in method_names:
                individual_valid = target_observed & method_valid[name]
                individual_valid_counts[name] += int(individual_valid.sum().item())
                common_valid &= method_valid[name]
            target_count += int(target_observed.sum().item())
            common_valid_count += int(common_valid.sum().item())
            target_physical = data.scaler.inverse_transform_torch(target)
            for name in method_names:
                prediction_physical = data.scaler.inverse_transform_torch(predictions[name])
                accumulators[name].update(prediction_physical, target_physical, common_valid)

            candidate_counts.append(legal_count_tensor)
            node_valid = candidates.valid.any(dim=-1)
            node_candidate_counts.append(candidates.valid.sum(dim=-1)[node_valid])
            top1_weights.append(candidates.weights[..., 0][node_valid])
            support = effective_support_size(candidates.weights, candidates.valid)
            effective_support.append(support[node_valid])
            query_count += int(x.shape[0])

        methods = {
            name: _method_result(
                accumulators[name],
                individual_valid_counts[name],
                target_count,
            )
            for name in method_names
        }
        weekly_mae = float(methods["weekly_mean"]["mae"])
        gains = {}
        for name in method_names:
            method_mae = float(methods[name]["mae"])
            absolute = weekly_mae - method_mae
            gains[name] = {
                "absolute_mae_gain": absolute,
                "relative_mae_gain_percent": 100.0 * absolute / weekly_mae,
            }

        learned_top1_mae = float(methods["learned_top1"]["mae"])
        oracle_top1_mae = float(methods["oracle_top1"]["mae"])
        return {
            "schema_version": 1,
            "dataset": bank.manifest.dataset_name,
            "split": split,
            "queries": query_count,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "bank": str(Path(bank_path).resolve()),
            "bank_events": bank.manifest.num_events,
            "common_evaluation_coverage": common_valid_count / max(target_count, 1),
            "methods": methods,
            "gains_vs_weekly_mean": gains,
            "selector": {
                "raw_legal_candidate_count": _summary(candidate_counts),
                "valid_node_topk_count": _summary(node_candidate_counts),
                "top1_weight": _summary(top1_weights),
                "effective_support_k": _summary(effective_support),
                "learned_top1_minus_oracle_top1_mae": learned_top1_mae
                - oracle_top1_mae,
            },
            "config": {
                "event_top_r": config.bank.event_top_r,
                "node_top_k": config.bank.node_top_k,
                "level_weight": config.bank.level_weight,
                "level_temperature": config.bank.level_temperature,
                "search_temperature": config.bank.search_temperature,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }

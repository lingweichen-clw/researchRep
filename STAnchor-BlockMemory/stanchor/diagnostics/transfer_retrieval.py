"""Frozen-encoder transfer diagnostics for an explicit candidate protocol."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig
from stanchor.diagnostics.retrieval import select_oracle_future
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.retrieval.strategies import (
    calendar_event_candidates,
    candidate_contexts,
    event_candidate_futures,
    masked_candidate_mean,
    raw_l1_candidate_scores,
    select_candidate_set_by_node,
)
from stanchor.utils import resolve_device


def _select_node(
    candidates: torch.Tensor,
    valid: torch.Tensor,
    selected: torch.Tensor,
    selected_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, horizon, nodes, _, channels = candidates.shape
    index = selected[:, None, :, None, None].expand(batch, horizon, nodes, 1, channels)
    prediction = candidates.gather(3, index).squeeze(3)
    output_valid = valid.gather(3, index).squeeze(3)
    output_valid &= selected_valid[:, None, :, None]
    return torch.where(output_valid, prediction, torch.zeros_like(prediction)), output_valid


@torch.no_grad()
def diagnose_transfer_retrieval(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    split: str = "val",
    candidate_protocol: str = "weekday_radius1_overlap",
    max_batches: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    model, checkpoint = load_pretrained_model(
        config, checkpoint_path, data.series.slots_per_day, device
    )
    model.eval()
    loader = DataLoader(
        getattr(data, split),
        batch_size=config.target.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    names = ("learned_topk", "learned_uniform_topk", "raw_l1_topk", "oracle_top1")
    accumulators = {name: ForecastMetricAccumulator(config.data.horizon) for name in names}
    valid_counts = {name: 0 for name in names}
    target_count = 0
    query_count = 0
    legal_counts: list[torch.Tensor] = []
    node_counts: list[torch.Tensor] = []
    offset_counts = {"-1": 0, "0": 0, "+1": 0, "other": 0}

    with MemoryBank(
        bank_path,
        expected_schema_version=(2 if model.model_config.profile_dim > 0 else 1),
    ) as bank:
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
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = batch["x"].to(device)
            x_observed = batch["x_observed"].to(device).bool()
            target = batch["y"].to(device)
            target_observed = batch["y_observed"].to(device).bool()
            encoding = model.encode_clean(
                batch["retrieval_x"].to(device),
                batch["retrieval_observed"].to(device),
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            events = calendar_event_candidates(
                bank,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
                config.bank.event_top_r,
                device,
                candidate_protocol=candidate_protocol,
            )
            legal_counts.append(events.valid.sum(dim=1).cpu())
            query_weekday = batch["query_weekday"].cpu().numpy()
            event_ids = events.event_ids.cpu().numpy()
            event_valid = events.valid.cpu().numpy()
            bank_weekday = np.asarray(bank.weekday)
            for row in range(event_ids.shape[0]):
                for col in np.flatnonzero(event_valid[row]):
                    delta = int(bank_weekday[event_ids[row, col]]) - int(query_weekday[row])
                    if delta in (-6, 6):
                        delta = 1 if delta == -6 else -1
                    key = str(delta) if delta in (-1, 0, 1) else "other"
                    if key == "0":
                        offset_counts["0"] += 1
                    elif key == "-1":
                        offset_counts["-1"] += 1
                    elif key == "+1":
                        offset_counts["+1"] += 1
                    else:
                        offset_counts["other"] += 1

            candidates = retriever.rerank_nodes(
                encoding.retrieval.node_keys,
                encoding.statistics.level_features,
                events,
            )
            aggregation = retriever.aggregate(candidates)
            event_future, event_future_valid = event_candidate_futures(
                bank, events.event_ids, events.valid, device
            )
            oracle_prediction, oracle_valid, _ = select_oracle_future(
                event_future, event_future_valid, target, target_observed, events.valid
            )
            contexts, contexts_valid = candidate_contexts(
                bank, events.event_ids, data.series, data.scaler,
                config.data.context_length, device
            )
            raw_scores, raw_valid = raw_l1_candidate_scores(
                x, x_observed, contexts, contexts_valid, events.valid
            )
            raw_scores, raw_selected = torch.topk(
                raw_scores, config.bank.node_top_k, dim=-1, largest=False
            )
            raw_selected_valid = torch.isfinite(raw_scores)
            raw_futures, raw_future_valid = select_candidate_set_by_node(
                event_future, event_future_valid, raw_selected, raw_selected_valid
            )
            raw_prediction, raw_prediction_valid = masked_candidate_mean(
                raw_futures, raw_future_valid
            )
            predictions = {
                "learned_topk": aggregation.prediction,
                "learned_uniform_topk": masked_candidate_mean(
                    aggregation.candidate_futures, aggregation.candidate_masks
                )[0],
                "raw_l1_topk": raw_prediction,
                "oracle_top1": oracle_prediction,
            }
            method_valid = {
                "learned_topk": aggregation.valid,
                "learned_uniform_topk": masked_candidate_mean(
                    aggregation.candidate_futures, aggregation.candidate_masks
                )[1],
                "raw_l1_topk": raw_prediction_valid,
                "oracle_top1": oracle_valid,
            }
            common_valid = target_observed.clone()
            for name in names:
                individual = target_observed & method_valid[name]
                valid_counts[name] += int(individual.sum())
                common_valid &= method_valid[name]
            target_count += int(target_observed.sum())
            target_physical = data.scaler.inverse_transform_torch(target)
            for name in names:
                prediction_physical = data.scaler.inverse_transform_torch(predictions[name])
                accumulators[name].update(prediction_physical, target_physical, common_valid)
            node_counts.append(candidates.valid.sum(dim=-1).flatten().cpu())
            query_count += int(x.shape[0])

    results: dict[str, Any] = {}
    for name in names:
        value = accumulators[name].compute()
        value["coverage"] = valid_counts[name] / max(target_count, 1)
        results[name] = value
    legal = torch.cat(legal_counts).double() if legal_counts else torch.zeros(0)
    nodes = torch.cat(node_counts).double() if node_counts else torch.zeros(0)
    return {
        "schema_version": 1,
        "dataset": Path(bank_path).name,
        "split": split,
        "candidate_protocol": candidate_protocol,
        "queries": query_count,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "bank": str(Path(bank_path).resolve()),
        "methods": results,
        "candidate_pool": {
            "mean": float(legal.mean()) if legal.numel() else 0.0,
            "median": float(legal.median()) if legal.numel() else 0.0,
            "min": float(legal.min()) if legal.numel() else 0.0,
            "max": float(legal.max()) if legal.numel() else 0.0,
            "node_topk_valid_mean": float(nodes.mean()) if nodes.numel() else 0.0,
            "weekday_offset_counts": offset_counts,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


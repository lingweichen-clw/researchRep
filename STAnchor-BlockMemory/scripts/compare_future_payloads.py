"""Compare raw, fixed-offset, and OffsetDecay candidate future payloads.

This is a validation-only diagnostic.  It keeps the learned selector, legal
candidate pool, node-level Top-K, and candidate weights fixed, and changes
only the future payload supplied to the weighted memory aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.bank.storage import MemoryBank
from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.retrieval_visualization import (
    build_diagnostic_event_candidates,
)
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import NodeCandidates, TwoStageRetriever
from stanchor.retrieval.strategies import (
    candidate_contexts_for_nodes,
    event_candidate_futures_for_nodes,
    offset_decay_aggregation,
)
from stanchor.retrieval.trend_residual import estimate_local_trend
from stanchor.utils import resolve_device, save_json


def _aggregate_payload(
    candidates: NodeCandidates,
    candidate_futures: torch.Tensor,
    candidate_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted candidate aggregation for [B,H,N,K,C] payload tensors."""
    effective = candidates.weights[:, None, :, :, None] * candidate_valid.to(
        candidates.weights.dtype
    )
    denominator = effective.sum(dim=3)
    prediction = (effective * candidate_futures).sum(dim=3) / denominator.clamp_min(
        1.0e-8
    )
    valid = denominator > 0
    return torch.where(valid, prediction, torch.zeros_like(prediction)), valid


@torch.no_grad()
def _fixed_offset_payload(
    candidates: NodeCandidates,
    query_context: torch.Tensor,
    query_observed: torch.Tensor,
    bank: MemoryBank,
    series,
    scaler,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build raw and constant-offset payloads using the same valid mask."""
    batch, time, nodes, channels = query_context.shape
    node_ids = torch.arange(nodes, device=device).view(1, nodes, 1).expand(
        batch, nodes, candidates.event_ids.shape[-1]
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
    query_statistics = estimate_local_trend(
        query_context, query_observed, context_length, mode="offset"
    )
    candidate_statistics = estimate_local_trend(
        candidate_context,
        candidate_context_observed,
        context_length,
        mode="offset",
    )
    selected_future, selected_future_valid = event_candidate_futures_for_nodes(
        bank,
        candidates.event_ids,
        candidates.valid,
        node_ids,
        device,
    )
    query_level = query_statistics.level[:, None, :, None, :]
    candidate_level = candidate_statistics.level[:, None, :, :, :]
    level_offset = query_level - candidate_level
    valid = (
        selected_future_valid
        & query_statistics.valid[:, None, :, None, :]
        & candidate_statistics.valid[:, None, :, :, :]
    )
    valid = valid & candidates.valid[:, None, :, :, None]
    raw = torch.where(valid, selected_future, torch.zeros_like(selected_future))
    fixed_offset = torch.where(valid, selected_future + level_offset, torch.zeros_like(selected_future))
    return raw, fixed_offset, valid


@torch.inference_mode()
def _run(config, checkpoint_path, bank_path, split, output_path, protocol, max_batches):
    """Run the matched payload comparison strictly in inference mode.

    The retrieval encoder processes 288 history steps over all nodes.  This
    diagnostic never backpropagates, so retaining autograd activations here
    only inflates the CUDA peak without changing any metric or ranking.
    """
    started = time.perf_counter()
    device = resolve_device(config.runtime.device)
    if device.type == "cuda":
        # PyTorch 2.11 on Windows accepts the current device implicitly here;
        # passing a torch.device object raises Invalid device argument.
        torch.cuda.reset_peak_memory_stats()
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
    metrics = {
        name: ForecastMetricAccumulator(config.data.horizon)
        for name in ("raw", "offset_only", "offset_decay")
    }
    offset_sum = 0.0
    offset_count = 0
    query_count = 0
    batch_count = 0

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
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            retrieval_encoding = model.encode_clean(
                batch["retrieval_x"].to(device),
                batch["retrieval_observed"].to(device),
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            events = build_diagnostic_event_candidates(
                bank,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
                config.bank.event_top_r,
                device,
                protocol,
            )
            candidates = retriever.rerank_nodes(
                retrieval_encoding.retrieval.node_keys,
                retrieval_encoding.statistics.level_features,
                events,
            )
            query_context = batch["x"].to(device)
            query_observed = batch["x_observed"].to(device)
            raw, offset_only, common_valid = _fixed_offset_payload(
                candidates,
                query_context,
                query_observed,
                bank,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            offset_decay = offset_decay_aggregation(
                candidates,
                query_context,
                query_observed,
                bank,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            # All three use the same validity intersection.  This prevents a
            # missing endpoint level from giving one payload extra metric mass.
            raw_prediction, raw_valid = _aggregate_payload(candidates, raw, common_valid)
            offset_prediction, offset_valid = _aggregate_payload(
                candidates, offset_only, common_valid
            )
            od_prediction, od_valid = _aggregate_payload(
                candidates,
                offset_decay.candidate_futures,
                common_valid & offset_decay.candidate_masks,
            )
            shared_valid = (
                batch["y_observed"].to(device).bool()
                & raw_valid
                & offset_valid
                & od_valid
            )
            target_physical = data.scaler.inverse_transform_torch(batch["y"].to(device))
            predictions = {
                "raw": data.scaler.inverse_transform_torch(raw_prediction),
                "offset_only": data.scaler.inverse_transform_torch(offset_prediction),
                "offset_decay": data.scaler.inverse_transform_torch(od_prediction),
            }
            for name, prediction in predictions.items():
                metrics[name].update(prediction, target_physical, shared_valid)
            offset_values = (offset_only - raw).abs().masked_select(common_valid)
            offset_sum += float(offset_values.sum().cpu())
            offset_count += int(offset_values.numel())
            query_count += int(query_context.shape[0])
            batch_count += 1
            if batch_count == 1 or batch_count % 10 == 0:
                print(
                    f"[{protocol}] processed {query_count}/{len(getattr(data, split))} "
                    f"queries ({batch_count} batches)",
                    flush=True,
                )

    if batch_count == 0:
        raise ValueError("no batches were processed")
    result = {
        "schema_version": 1,
        "diagnostic": "matched_future_payload_comparison",
        "dataset": bank.manifest.dataset_name,
        "split": split,
        "complete_validation": max_batches is None,
        "queries": query_count,
        "batches": batch_count,
        "candidate_protocol": protocol,
        "event_top_r": config.bank.event_top_r,
        "node_top_k": config.bank.node_top_k,
        "checkpoint": str(resolve_project_path(checkpoint_path)),
        "bank": str(resolve_project_path(bank_path)),
        "payload_definitions": {
            "raw": "Y_raw",
            "offset_only": "Y_raw + (level_query - level_candidate) for every horizon",
            "offset_decay": "Y_raw + lambda_h * (level_query - level_candidate), lambda: 1 -> 0",
        },
        "metric_mask": "shared target and payload-valid positions across all three variants",
        "mean_absolute_offset_normalized_units": offset_sum / max(offset_count, 1),
        "metrics": {name: accumulator.compute() for name, accumulator in metrics.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_mb": (
            float(torch.cuda.max_memory_allocated() / (1024.0**2))
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_mb": (
            float(torch.cuda.max_memory_reserved() / (1024.0**2))
            if device.type == "cuda"
            else None
        ),
        "future_information_boundary": (
            "Candidate selection uses query history and bank keys only; query future is used "
            "only for validation metrics."
        ),
    }
    save_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--candidate-protocol", default="weekday_radius1_overlap")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    result = _run(
        config,
        args.checkpoint,
        args.bank,
        args.split,
        resolve_project_path(args.output),
        args.candidate_protocol,
        args.max_batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

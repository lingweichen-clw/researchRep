"""Validation-only counterfactual upper-bound diagnostics.

This module intentionally uses the current validation target to construct
oracle policies.  The outputs are mechanism diagnostics only; they are not
deployable predictions and must not be used as test evidence.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig
from stanchor.engine.common import build_data_and_graph, load_checkpoint, load_pretrained_model
from stanchor.engine.target import (
    _validate_bank,
    build_downstream_model,
    checkpoint_bank_level_weight,
    checkpoint_candidate_protocol,
    checkpoint_downstream_mode,
    retrieve_for_downstream_mode,
)
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.utils import resolve_device


def _metric_store(horizon: int, names: list[str]) -> dict[str, ForecastMetricAccumulator]:
    return {name: ForecastMetricAccumulator(horizon) for name in names}


def _update(store: dict[str, ForecastMetricAccumulator], predictions: dict[str, torch.Tensor], target: torch.Tensor, observed: torch.Tensor) -> None:
    for name, prediction in predictions.items():
        store[name].update(prediction, target, observed)


def _fixed_aggregation_predictions(
    candidate_futures: torch.Tensor,
    candidate_masks: torch.Tensor,
    weights: torch.Tensor,
    base: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build validation-only fixed aggregation alternatives without targets."""
    valid = candidate_masks.bool()
    effective = weights[:, None, :, :, None].to(candidate_futures.dtype) * valid.to(candidate_futures.dtype)
    denominator = effective.sum(dim=3).clamp_min(1.0e-8)
    weighted_mean = (effective * candidate_futures).sum(dim=3) / denominator
    valid_location = valid.any(dim=3)

    top1_idx = weights.argmax(dim=-1)
    top1_gather = top1_idx[:, None, :, None, None].expand(-1, candidate_futures.shape[1], -1, 1, candidate_futures.shape[-1])
    top1 = candidate_futures.gather(3, top1_gather).squeeze(3)
    top1_valid = valid.gather(3, top1_gather).squeeze(3)

    topk = min(3, candidate_futures.shape[3])
    topk_idx = torch.topk(weights, topk, dim=-1).indices
    topk_gather = topk_idx[:, None, :, :, None].expand(-1, candidate_futures.shape[1], -1, -1, candidate_futures.shape[-1])
    topk_values = candidate_futures.gather(3, topk_gather)
    topk_valid = valid.gather(3, topk_gather)
    topk_weights = topk_idx.new_zeros(topk_idx.shape, dtype=candidate_futures.dtype)
    topk_weights = torch.gather(weights, -1, topk_idx)
    topk_effective = topk_weights[:, None, :, :, None] * topk_valid.to(candidate_futures.dtype)
    top3 = (topk_effective * topk_values).sum(dim=3) / topk_effective.sum(dim=3).clamp_min(1.0e-8)

    rank = torch.argsort(weights, dim=-1, descending=True)
    trim_k = max(1, candidate_futures.shape[3] - 1)
    trim_idx = rank[..., :trim_k]
    trim_gather = trim_idx[:, None, :, :, None].expand(-1, candidate_futures.shape[1], -1, -1, candidate_futures.shape[-1])
    trim_values = candidate_futures.gather(3, trim_gather)
    trim_valid = valid.gather(3, trim_gather)
    trim_weights = torch.gather(weights, -1, trim_idx)
    trim_effective = trim_weights[:, None, :, :, None] * trim_valid.to(candidate_futures.dtype)
    trimmed = (trim_effective * trim_values).sum(dim=3) / trim_effective.sum(dim=3).clamp_min(1.0e-8)

    delta = candidate_futures - base.unsqueeze(3)
    direction = torch.sign(delta.mean(dim=-1))
    positive_effective = effective * (direction >= 0).unsqueeze(-1).to(candidate_futures.dtype)
    negative_effective = effective * (direction < 0).unsqueeze(-1).to(candidate_futures.dtype)
    positive_sum = positive_effective.sum(dim=3)
    negative_sum = negative_effective.sum(dim=3)
    positive_mean = (positive_effective * candidate_futures).sum(dim=3) / positive_sum.clamp_min(1.0e-8)
    negative_mean = (negative_effective * candidate_futures).sum(dim=3) / negative_sum.clamp_min(1.0e-8)
    sign_cluster = torch.where(positive_sum >= negative_sum, positive_mean, negative_mean)
    sign_valid = (positive_sum > 0) | (negative_sum > 0)
    return {
        "memory_weighted_mean": torch.where(valid_location, weighted_mean, base),
        "memory_top1": torch.where(top1_valid, top1, base),
        "memory_top3": torch.where(topk_valid.any(dim=3), top3, base),
        "memory_trimmed": torch.where(trim_valid.any(dim=3), trimmed, base),
        "memory_sign_cluster": torch.where(sign_valid, sign_cluster, base),
    }


def _oracle_candidate_predictions(
    candidate_futures: torch.Tensor,
    candidate_masks: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-location best candidate and error-weighted candidate aggregate.

    Shapes:
      candidate_futures: [B,H,N,K,C]
      candidate_masks:   [B,H,N,K,C]
      target:            [B,H,N,C]
    The target is used only for validation-only upper-bound analysis.
    """
    if candidate_futures.ndim != 5 or candidate_masks.shape != candidate_futures.shape:
        raise ValueError("candidate futures and masks must be [B,H,N,K,C]")
    if target.shape != candidate_futures.shape[:3] + candidate_futures.shape[-1:]:
        raise ValueError("target shape does not match candidate futures")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = candidate_masks.bool().all(dim=-1)
    errors = (candidate_futures - target.unsqueeze(3)).abs().mean(dim=-1)
    errors = errors.masked_fill(~valid, torch.inf)
    best_index = errors.argmin(dim=-1)
    best = candidate_futures.gather(
        3,
        best_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, candidate_futures.shape[-1]),
    ).squeeze(3)
    finite = torch.isfinite(errors)
    logits = (-errors / temperature).masked_fill(~finite, -torch.inf)
    max_value = logits.amax(dim=-1, keepdim=True)
    max_value = torch.where(torch.isfinite(max_value), max_value, torch.zeros_like(max_value))
    weights = torch.where(finite, torch.exp(logits - max_value), torch.zeros_like(logits))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    aggregate = (weights.unsqueeze(-1) * candidate_futures).sum(dim=3)
    valid_location = finite.any(dim=-1, keepdim=True)
    best = torch.where(valid_location, best, torch.zeros_like(best))
    aggregate = torch.where(valid_location, aggregate, torch.zeros_like(aggregate))
    return best, aggregate


@torch.no_grad()
def diagnose_counterfactual_checkpoint(
    config: ExperimentConfig,
    pretrained_checkpoint: str | Path,
    downstream_checkpoint: str | Path,
    bank_path: str | Path,
    split: str = "val",
    max_batches: int | None = None,
    candidate_protocol: str | None = None,
    fixed_alphas: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5),
    oracle_alpha_grid: tuple[float, ...] | None = None,
    oracle_candidate_temperature: float = 0.05,
) -> dict[str, Any]:
    """Run base/current/fixed/oracle candidate and fusion comparisons."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if not fixed_alphas:
        raise ValueError("fixed_alphas must not be empty")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in fixed_alphas):
        raise ValueError("fixed alphas must be in [0,1]")
    if oracle_alpha_grid is None:
        oracle_alpha_grid = tuple(index / 20.0 for index in range(21))
    if not oracle_alpha_grid:
        raise ValueError("oracle_alpha_grid must not be empty")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in oracle_alpha_grid):
        raise ValueError("oracle alpha grid must be in [0,1]")

    started = time.perf_counter()
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    pretrained, _ = load_pretrained_model(
        config, pretrained_checkpoint, data.series.slots_per_day, device
    )
    checkpoint = load_checkpoint(downstream_checkpoint, device)
    mode = checkpoint_downstream_mode(checkpoint)
    level_weight = checkpoint_bank_level_weight(checkpoint, default=config.bank.level_weight)
    saved_protocol = checkpoint_candidate_protocol(checkpoint)
    protocol = saved_protocol if candidate_protocol is None else checkpoint_candidate_protocol(checkpoint, expected=candidate_protocol)
    config = replace(
        config,
        bank=replace(config.bank, level_weight=level_weight),
        target=replace(config.target, downstream_mode=mode, candidate_protocol=protocol),
    )
    downstream = build_downstream_model(config, graph).to(device)
    downstream.load_state_dict(checkpoint["downstream_state_dict"], strict=True)
    pretrained.eval()
    downstream.eval()
    dataset = getattr(data, split)
    loader = DataLoader(
        dataset,
        batch_size=config.target.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    names = ["base", "current_learned_gate", "oracle_binary_gate", "oracle_continuous_alpha", "oracle_candidate_top1", "oracle_candidate_error_weighted", "memory_weighted_mean", "memory_top1", "memory_top3", "memory_trimmed", "memory_sign_cluster"]
    names += [f"fixed_alpha_{alpha:.2f}" for alpha in fixed_alphas]
    stores = _metric_store(config.data.horizon, names)
    helpful_count = 0
    location_count = 0
    aggregation_helpful = {name: 0 for name in ["memory_weighted_mean", "memory_top1", "memory_top3", "memory_trimmed", "memory_sign_cluster"]}
    aggregation_valid = {name: 0 for name in aggregation_helpful}
    oracle_alpha_sum = None
    oracle_alpha_count = None
    batches = 0
    queries = 0
    dataset_name = "unknown"

    with MemoryBank(
        bank_path,
        expected_schema_version=(2 if pretrained.model_config.profile_dim > 0 else 1),
    ) as bank:
        dataset_name = bank.manifest.dataset_name
        _validate_bank(bank, pretrained, graph_cpu, data.scaler.state_dict())
        if checkpoint.get("bank_manifest") != bank.manifest.to_dict():
            raise ValueError("diagnostic bank differs from the bank used for downstream training")
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
            observed_x = batch["x_observed"].to(device)
            candidates, aggregation = retrieve_for_downstream_mode(
                mode,
                pretrained,
                retriever,
                bank,
                data,
                graph,
                batch,
                x,
                observed_x,
                device,
                candidate_protocol=protocol,
            )
            output = downstream(x, candidates, aggregation)
            target = batch["y"].to(device)
            observed = batch["y_observed"].to(device)
            base = data.scaler.inverse_transform_torch(output.base_prediction)
            memory = data.scaler.inverse_transform_torch(output.memory_prediction)
            current = data.scaler.inverse_transform_torch(output.final_prediction)
            target_physical = data.scaler.inverse_transform_torch(target)
            # Per-node/horizon oracle decision uses mean absolute channel error.
            valid = observed.bool() & output.memory_valid.expand_as(observed)
            base_error = (base - target_physical).abs()
            memory_error = (memory - target_physical).abs()
            gain = base_error - memory_error
            gain_node = gain.mean(dim=-1, keepdim=True)
            binary = (gain_node > 0.0) & output.memory_valid
            oracle_binary = base + binary.to(base.dtype) * (memory - base)
            alpha_candidates = []
            alpha_errors = []
            for alpha in oracle_alpha_grid:
                candidate = base + float(alpha) * (memory - base)
                alpha_candidates.append(candidate)
                alpha_errors.append((candidate - target_physical).abs().mean(dim=-1, keepdim=True))
            alpha_errors_tensor = torch.cat(alpha_errors, dim=-1)
            best_alpha_index = alpha_errors_tensor.argmin(dim=-1)
            grid = torch.tensor(oracle_alpha_grid, device=device, dtype=base.dtype)
            best_alpha = grid[best_alpha_index]
            oracle_continuous = base + best_alpha.unsqueeze(-1) * (memory - base)
            oracle_continuous = torch.where(output.memory_valid, oracle_continuous, base)
            candidate_top1, candidate_error_weighted = _oracle_candidate_predictions(
                aggregation.candidate_futures * (torch.as_tensor(data.scaler.std, dtype=aggregation.candidate_futures.dtype, device=device)[None, None, :, None, :] + data.scaler.eps) + torch.as_tensor(data.scaler.mean, dtype=aggregation.candidate_futures.dtype, device=device)[None, None, :, None, :],
                aggregation.candidate_masks,
                target_physical,
                oracle_candidate_temperature,
            )
            candidate_valid = aggregation.candidate_masks.bool().all(dim=-1).any(dim=3, keepdim=True)
            candidate_top1 = torch.where(candidate_valid, candidate_top1, base)
            candidate_error_weighted = torch.where(candidate_valid, candidate_error_weighted, base)
            fixed_aggregations = _fixed_aggregation_predictions(
                aggregation.candidate_futures * (torch.as_tensor(data.scaler.std, dtype=aggregation.candidate_futures.dtype, device=device)[None, None, :, None, :] + data.scaler.eps)
                + torch.as_tensor(data.scaler.mean, dtype=aggregation.candidate_futures.dtype, device=device)[None, None, :, None, :],
                aggregation.candidate_masks,
                candidates.weights,
                base,
            )
            current_candidates = {
                "base": base,
                "current_learned_gate": current,
                "oracle_binary_gate": oracle_binary,
                "oracle_continuous_alpha": oracle_continuous,
                "oracle_candidate_top1": candidate_top1,
                "oracle_candidate_error_weighted": candidate_error_weighted,
                **fixed_aggregations,
            }
            for alpha in fixed_alphas:
                fixed = base + float(alpha) * (memory - base)
                current_candidates[f"fixed_alpha_{alpha:.2f}"] = torch.where(
                    output.memory_valid, fixed, base
                )
            _update(stores, current_candidates, target_physical, observed)
            for aggregation_name, aggregation_prediction in fixed_aggregations.items():
                aggregation_error = (aggregation_prediction - target_physical).abs().mean(dim=-1)
                base_error_scalar = (base - target_physical).abs().mean(dim=-1)
                aggregation_mask = observed.bool().all(dim=-1) & output.memory_valid.squeeze(-1)
                aggregation_helpful[aggregation_name] += int((aggregation_error < base_error_scalar).masked_select(aggregation_mask).sum().item())
                aggregation_valid[aggregation_name] += int(aggregation_mask.sum().item())
            location_valid = valid.any(dim=-1, keepdim=True)
            helpful_count += int((gain_node > 0.0).masked_select(location_valid).sum().item())
            location_count += int(location_valid.sum().item())
            alpha_valid = location_valid & output.memory_valid
            alpha_sum = torch.where(alpha_valid.squeeze(-1), best_alpha, torch.zeros_like(best_alpha)).sum(dim=(0, 2)).detach().cpu()
            alpha_num = alpha_valid.squeeze(-1).sum(dim=(0, 2)).detach().cpu()
            if oracle_alpha_sum is None:
                oracle_alpha_sum = alpha_sum
                oracle_alpha_count = alpha_num
            else:
                oracle_alpha_sum += alpha_sum
                oracle_alpha_count += alpha_num
            batches += 1
            queries += int(x.shape[0])
    if batches == 0:
        raise ValueError("counterfactual diagnostic processed no batches")
    result = {
        "schema_version": 1,
        "diagnostic": "validation_only_counterfactual_upper_bound",
        "dataset": dataset_name,
        "downstream_mode": mode,
        "candidate_protocol": protocol,
        "level_weight": config.bank.level_weight,
        "split": split,
        "queries": queries,
        "batches": batches,
        "fixed_alphas": list(fixed_alphas),
        "oracle_alpha_grid": list(oracle_alpha_grid),
        "methods": {name: stores[name].compute() for name in names},
        "memory_helpful_prevalence": helpful_count / max(location_count, 1),
        "fixed_aggregation_helpful_rate": {
            name: aggregation_helpful[name] / max(aggregation_valid[name], 1)
            for name in aggregation_helpful
        },
        "oracle_alpha_mean_by_horizon": (
            (oracle_alpha_sum / oracle_alpha_count.clamp_min(1)).tolist()
            if oracle_alpha_sum is not None and oracle_alpha_count is not None
            else []
        ),
        "checkpoint": str(Path(downstream_checkpoint).resolve()),
        "pretrained_checkpoint": str(Path(pretrained_checkpoint).resolve()),
        "bank": str(Path(bank_path).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "future_information_boundary": "target is used only for validation-only oracle policies; no oracle output is deployable",
    }
    return result








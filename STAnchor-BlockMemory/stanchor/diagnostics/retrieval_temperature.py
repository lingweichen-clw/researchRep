"""Zero-training sweep of deployed Top-K retrieval aggregation weights."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.diagnostics.retrieval_visualization import (
    build_diagnostic_event_candidates,
    validate_aligned_bank_axes,
)
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import AggregationOutput, TwoStageRetriever
from stanchor.retrieval.strategies import offset_decay_aggregation
from stanchor.utils import resolve_device, save_json


DEFAULT_TEMPERATURES = (0.20, 0.10, 0.05, 0.02)


def retrieval_weights(
    scores: torch.Tensor,
    valid: torch.Tensor,
    setting: float | str,
) -> torch.Tensor:
    """Convert fixed Top-K scores to node-level aggregation weights."""
    if scores.ndim != 3 or valid.shape != scores.shape:
        raise ValueError("scores and valid must be [B, N, K]")
    valid = valid.bool() & torch.isfinite(scores)
    if setting == "uniform":
        weights = valid.to(scores.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    if setting == "hard_top1":
        stable = scores.masked_fill(~valid, -torch.inf)
        selected = stable.argmax(dim=-1, keepdim=True)
        has_candidate = valid.any(dim=-1, keepdim=True)
        weights = torch.zeros_like(scores).scatter(-1, selected, 1.0)
        return weights * has_candidate.to(weights.dtype)
    if isinstance(setting, bool) or not isinstance(setting, (float, int)):
        raise ValueError("setting must be uniform, hard_top1, or a positive temperature")
    temperature = float(setting)
    if temperature <= 0:
        raise ValueError("retrieval temperature must be positive")
    logits = scores / temperature
    stable = logits.masked_fill(~valid, -torch.inf)
    maximum = stable.amax(dim=-1, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponent = torch.where(valid, torch.exp(logits - maximum), torch.zeros_like(logits))
    return exponent / exponent.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def aggregate_weighted_candidates(
    candidate_futures: torch.Tensor,
    candidate_masks: torch.Tensor,
    weights: torch.Tensor,
) -> AggregationOutput:
    """Aggregate fixed candidate payloads with point-wise mask renormalization."""
    if candidate_futures.ndim != 5 or candidate_masks.shape != candidate_futures.shape:
        raise ValueError("candidate futures and masks must be [B, H, N, K, C]")
    batch, _, nodes, top_k, _ = candidate_futures.shape
    if weights.shape != (batch, nodes, top_k):
        raise ValueError("weights must be [B, N, K]")
    mask = candidate_masks.bool() & torch.isfinite(candidate_futures)
    base_weights = weights[:, None, :, :, None]
    effective = base_weights * mask.to(base_weights.dtype)
    denominator = effective.sum(dim=3)
    prediction = (effective * candidate_futures).sum(dim=3) / denominator.clamp_min(1.0e-8)
    valid = denominator > 0
    prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    difference = candidate_futures - prediction.unsqueeze(3)
    variance = (effective * difference.square()).sum(dim=3) / denominator.clamp_min(1.0e-8)
    variance = torch.where(valid, variance, torch.zeros_like(variance))
    return AggregationOutput(
        prediction=prediction,
        variance=variance,
        valid=valid,
        candidate_futures=candidate_futures,
        candidate_masks=mask,
    )


def common_setting_masks(
    target_valid: torch.Tensor,
    output_valid: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Build a pretrained/random common mask independently for each setting."""
    if target_valid.ndim != 4:
        raise ValueError("target_valid must be [B, H, N, C]")
    selectors = tuple(output_valid)
    if not selectors:
        raise ValueError("at least one selector is required")
    setting_names = tuple(output_valid[selectors[0]])
    if not setting_names:
        raise ValueError("at least one weight setting is required")
    expected_names = set(setting_names)
    for selector in selectors:
        if set(output_valid[selector]) != expected_names:
            raise ValueError("selectors must expose identical weight settings")
    result: dict[str, torch.Tensor] = {}
    for name in setting_names:
        common = target_valid.bool().clone()
        for selector in selectors:
            valid = output_valid[selector][name]
            if valid.shape != target_valid.shape:
                raise ValueError("output validity masks must match target_valid")
            common &= valid.bool()
        result[name] = common
    return result


def _setting_name(setting: float | str) -> str:
    if setting == "uniform":
        return "uniform_top5"
    if setting == "hard_top1":
        return "hard_top1"
    return f"temperature_{float(setting):.2f}"


def _settings(temperatures: Sequence[float]) -> tuple[float | str, ...]:
    values = tuple(float(value) for value in temperatures)
    if not values or any(value <= 0 for value in values):
        raise ValueError("temperatures must contain positive values")
    if len(set(values)) != len(values):
        raise ValueError("temperatures must be unique")
    return ("uniform", *values, "hard_top1")


@dataclass
class _WeightAccumulator:
    top1_sum: float = 0.0
    support_sum: float = 0.0
    count: int = 0

    def update(self, weights: torch.Tensor, valid: torch.Tensor) -> None:
        node_valid = valid.bool().any(dim=-1)
        if not bool(node_valid.any()):
            return
        masked = torch.where(valid.bool(), weights, torch.zeros_like(weights))
        normalized = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        support = normalized.square().sum(dim=-1).clamp_min(1.0e-8).reciprocal()
        self.top1_sum += float(normalized[..., 0].masked_select(node_valid).sum().cpu())
        self.support_sum += float(support.masked_select(node_valid).sum().cpu())
        self.count += int(node_valid.sum().item())

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("no valid candidate weights were accumulated")
        return {
            "valid_query_nodes": self.count,
            "mean_top1_weight": self.top1_sum / self.count,
            "mean_effective_support": self.support_sum / self.count,
        }


def _write_summary_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for selector, settings in result["results"].items():
        for name, values in settings.items():
            metrics = values["metrics"]
            rows.append(
                {
                    "selector": selector,
                    "setting": name,
                    "temperature": values["temperature"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "mean_top1_weight": values["weights"]["mean_top1_weight"],
                    "mean_effective_support": values["weights"]["mean_effective_support"],
                    "individual_coverage": values["individual_coverage"],
                    "common_coverage": result["common_evaluation_coverage"][name],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_horizon_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    frequency = int(result["frequency_minutes"])
    for selector, settings in result["results"].items():
        for name, values in settings.items():
            metrics = values["metrics"]
            for index, (mae, rmse, mape) in enumerate(
                zip(
                    metrics["horizon_mae"],
                    metrics["horizon_rmse"],
                    metrics["horizon_mape"],
                )
            ):
                rows.append(
                    {
                        "selector": selector,
                        "setting": name,
                        "horizon_step": index + 1,
                        "horizon_minutes": (index + 1) * frequency,
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def run_retrieval_temperature_sweep(
    config: ExperimentConfig | None,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    random_checkpoint_path: str | Path,
    random_bank_path: str | Path,
    output_dir: str | Path,
    split: str = "val",
    candidate_protocol: str = "relaxed_calendar",
    temperatures: Sequence[float] = DEFAULT_TEMPERATURES,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate fixed E5A Top-K candidates under different weight temperatures."""
    if split != "val":
        raise ValueError("retrieval-temperature sweep is restricted to validation")
    if candidate_protocol != "relaxed_calendar":
        raise ValueError("the formal E5A temperature sweep requires relaxed_calendar")
    if config is None:
        raise ValueError("config is required")
    started = time.perf_counter()
    settings = _settings(temperatures)
    setting_names = tuple(_setting_name(setting) for setting in settings)
    if len(set(setting_names)) != len(setting_names):
        raise ValueError("temperature formatting produced duplicate setting names")

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

    selectors = ("pretrained", "random")
    accumulators = {
        selector: {
            name: ForecastMetricAccumulator(config.data.horizon)
            for name in setting_names
        }
        for selector in selectors
    }
    weight_accumulators = {
        selector: {name: _WeightAccumulator() for name in setting_names}
        for selector in selectors
    }
    individual_valid_counts = {
        selector: {name: 0 for name in setting_names}
        for selector in selectors
    }
    target_count = 0
    common_valid_counts = {name: 0 for name in setting_names}
    query_count = 0
    batch_count = 0

    with MemoryBank(bank_path) as pretrained_bank, MemoryBank(random_bank_path) as random_bank:
        bank_alignment = validate_aligned_bank_axes(pretrained_bank, random_bank)
        _validate_bank(pretrained_bank, pretrained_model, graph_cpu, data.scaler.state_dict())
        _validate_bank(random_bank, random_model, graph_cpu, data.scaler.state_dict())
        retrievers = {
            "pretrained": TwoStageRetriever(
                pretrained_bank,
                config.bank.event_top_r,
                config.bank.node_top_k,
                config.bank.level_weight,
                config.bank.level_temperature,
                config.bank.search_temperature,
                device,
            ),
            "random": TwoStageRetriever(
                random_bank,
                config.bank.event_top_r,
                config.bank.node_top_k,
                config.bank.level_weight,
                config.bank.level_temperature,
                config.bank.search_temperature,
                device,
            ),
        }
        models = {"pretrained": pretrained_model, "random": random_model}
        banks = {"pretrained": pretrained_bank, "random": random_bank}

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            events = build_diagnostic_event_candidates(
                pretrained_bank,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
                config.bank.event_top_r,
                device,
                candidate_protocol,
            )
            outputs: dict[str, dict[str, AggregationOutput]] = {}
            for selector in selectors:
                encoding = models[selector].encode_clean(
                    batch["retrieval_x"].to(device),
                    batch["retrieval_observed"].to(device),
                    batch["retrieval_weekday"].to(device),
                    batch["retrieval_slot"].to(device),
                    graph,
                )
                candidates = retrievers[selector].rerank_nodes(
                    encoding.retrieval.node_keys,
                    encoding.statistics.level_features,
                    events,
                )
                payload = offset_decay_aggregation(
                    candidates,
                    batch["x"].to(device),
                    batch["x_observed"].to(device),
                    banks[selector],
                    data.series,
                    data.scaler,
                    config.data.context_length,
                    device,
                )
                outputs[selector] = {}
                for setting, name in zip(settings, setting_names):
                    weights = retrieval_weights(
                        candidates.total_scores,
                        candidates.valid,
                        setting,
                    )
                    weight_accumulators[selector][name].update(weights, candidates.valid)
                    outputs[selector][name] = aggregate_weighted_candidates(
                        payload.candidate_futures,
                        payload.candidate_masks,
                        weights,
                    )

            target = batch["y"].to(device)
            target_valid = batch["y_observed"].to(device).bool()
            output_valid = {
                selector: {
                    name: outputs[selector][name].valid for name in setting_names
                }
                for selector in selectors
            }
            common_valid = common_setting_masks(target_valid, output_valid)
            for selector in selectors:
                for name in setting_names:
                    output = outputs[selector][name]
                    individual_valid_counts[selector][name] += int(
                        (target_valid & output.valid).sum().item()
                    )
            target_count += int(target_valid.sum().item())
            for name in setting_names:
                common_valid_counts[name] += int(common_valid[name].sum().item())
            target_physical = data.scaler.inverse_transform_torch(target)
            for selector in selectors:
                for name in setting_names:
                    prediction_physical = data.scaler.inverse_transform_torch(
                        outputs[selector][name].prediction
                    )
                    accumulators[selector][name].update(
                        prediction_physical,
                        target_physical,
                        common_valid[name],
                    )
            batch_size = int(target.shape[0])
            query_count += batch_size
            batch_count += 1
            if batch_count == 1 or batch_count % 10 == 0:
                print(
                    f"[e5a-temperature/{candidate_protocol}] processed "
                    f"{query_count}/{len(dataset)} validation queries "
                    f"({batch_count} batches)",
                    flush=True,
                )

        if batch_count == 0:
            raise ValueError("no validation batches were processed")
        common_coverage = {
            name: common_valid_counts[name] / max(target_count, 1)
            for name in setting_names
        }
        results: dict[str, dict[str, Any]] = {}
        for selector in selectors:
            results[selector] = {}
            for setting, name in zip(settings, setting_names):
                results[selector][name] = {
                    "temperature": float(setting) if isinstance(setting, float) else None,
                    "kind": (
                        "softmax_temperature"
                        if isinstance(setting, float)
                        else str(setting)
                    ),
                    "metrics": accumulators[selector][name].compute(),
                    "weights": weight_accumulators[selector][name].compute(),
                    "individual_coverage": individual_valid_counts[selector][name]
                    / max(target_count, 1),
                    "common_coverage": common_coverage[name],
                }

        comparisons: dict[str, Any] = {}
        baseline_name = _setting_name(0.10)
        for name in setting_names:
            pretrained_metrics = results["pretrained"][name]["metrics"]
            random_metrics = results["random"][name]["metrics"]
            baseline_metrics = results["pretrained"][baseline_name]["metrics"]
            comparisons[name] = {
                "pretrained_gain_over_random": {
                    metric: float(random_metrics[metric]) - float(pretrained_metrics[metric])
                    for metric in ("mae", "rmse", "mape")
                },
                "pretrained_gain_over_temperature_0.10": {
                    metric: float(baseline_metrics[metric]) - float(pretrained_metrics[metric])
                    for metric in ("mae", "rmse", "mape")
                },
            }
        result: dict[str, Any] = {
            "schema_version": 1,
            "diagnostic": "e5a_deployed_retrieval_weight_temperature_sweep",
            "dataset": pretrained_bank.manifest.dataset_name,
            "split": split,
            "complete_validation": max_batches is None,
            "queries": query_count,
            "batches": batch_count,
            "frequency_minutes": config.data.frequency_minutes,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": pretrained_checkpoint.get("epoch"),
            "random_checkpoint": str(random_checkpoint_path),
            "random_checkpoint_epoch": random_checkpoint.get("epoch"),
            "bank": str(bank_path),
            "random_bank": str(random_bank_path),
            "bank_alignment": bank_alignment,
            "candidate_protocol": candidate_protocol,
            "fixed_contract": {
                "selector_candidates_reused_across_settings": True,
                "top_k": config.bank.node_top_k,
                "payload": "OffsetDecay",
                "query_future_used_for_ranking": False,
                "query_future_used_for_weighting": False,
                "query_future_used_for_offline_metrics_only": True,
            },
            "baseline_setting": baseline_name,
            "common_evaluation_coverage": common_coverage,
            "common_valid_values": common_valid_counts,
            "target_valid_values": target_count,
            "results": results,
            "comparisons": comparisons,
            "elapsed_seconds": time.perf_counter() - started,
        }

    metrics_path = output_path / "metrics.json"
    summary_path = output_path / "summary.csv"
    horizon_path = output_path / "horizon_metrics.csv"
    _write_summary_csv(result, summary_path)
    _write_horizon_csv(result, horizon_path)
    result["outputs"] = {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "horizon_metrics": str(horizon_path),
    }
    save_json(metrics_path, result)
    return result

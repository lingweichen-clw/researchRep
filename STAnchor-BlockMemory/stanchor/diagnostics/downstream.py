"""Downstream branch and confidence diagnostic utilities."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig
from stanchor.engine.common import (
    build_data_and_graph,
    load_checkpoint,
    load_pretrained_model,
)
from stanchor.engine.target import (
    _validate_bank,
    build_downstream_model,
    checkpoint_bank_level_weight,
    checkpoint_candidate_protocol,
    checkpoint_downstream_mode,
    retrieve_for_downstream_mode,
)
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.modes import LEARNED_TOPK_ERROR_AWARE
from stanchor.losses.downstream import build_blend_target, build_huber_risk_target
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.utils import resolve_device


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    """Summarize one finite diagnostic vector."""
    vector = _finite_vector(values, "values").astype(np.float64, copy=False)
    quantiles = np.quantile(vector, [0.1, 0.5, 0.9])
    return {
        "count": int(vector.size),
        "mean": float(vector.mean()),
        "std": float(vector.std()),
        "min": float(vector.min()),
        "q10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q90": float(quantiles[2]),
        "max": float(vector.max()),
    }


def binary_confidence_metrics(
    confidence: np.ndarray,
    helpful: np.ndarray,
) -> dict[str, float | int | None]:
    """Return tie-aware AUROC/AUPRC and probability calibration metrics."""
    scores = _finite_vector(confidence, "confidence").astype(np.float64, copy=False)
    labels = np.asarray(helpful).reshape(-1).astype(bool, copy=False)
    if labels.size != scores.size:
        raise ValueError("confidence and helpful must have the same number of values")
    if bool(((scores < 0.0) | (scores > 1.0)).any()):
        raise ValueError("confidence must be in [0, 1]")

    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    prevalence = positives / labels.size
    auroc: float | None = None
    auprc: float | None = None
    if positives > 0 and negatives > 0:
        order = np.argsort(scores, kind="mergesort")
        ordered_scores = scores[order]
        ordered_labels = labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1]
        ends = np.r_[starts[1:], ordered_scores.size]
        negative_before = 0
        concordant = 0.0
        for start, end in zip(starts, ends):
            group = ordered_labels[start:end]
            group_positive = int(group.sum())
            group_negative = int(group.size - group_positive)
            concordant += group_positive * negative_before
            concordant += 0.5 * group_positive * group_negative
            negative_before += group_negative
        auroc = concordant / (positives * negatives)

    if positives > 0:
        order = np.argsort(-scores, kind="mergesort")
        ordered_scores = scores[order]
        ordered_labels = labels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1]
        ends = np.r_[starts[1:], ordered_scores.size]
        true_positive = 0
        predicted_positive = 0
        area = 0.0
        for start, end in zip(starts, ends):
            group_positive = int(ordered_labels[start:end].sum())
            true_positive += group_positive
            predicted_positive += int(end - start)
            precision = true_positive / predicted_positive
            area += precision * (group_positive / positives)
        auprc = area

    squared = np.square(scores - labels.astype(np.float64))
    ece_bins = 10
    bin_ids = np.minimum((scores * ece_bins).astype(np.int64), ece_bins - 1)
    ece = 0.0
    for bin_index in range(ece_bins):
        in_bin = bin_ids == bin_index
        if not bool(in_bin.any()):
            continue
        bin_confidence = float(scores[in_bin].mean())
        bin_accuracy = float(labels[in_bin].mean())
        ece += float(in_bin.mean()) * abs(bin_accuracy - bin_confidence)
    return {
        "count": int(labels.size),
        "positives": positives,
        "negatives": negatives,
        "prevalence": float(prevalence),
        "auroc": None if auroc is None else float(auroc),
        "auprc": None if auprc is None else float(auprc),
        "brier": float(squared.mean()),
        "ece": ece,
        "ece_bins": ece_bins,
        "constant_brier": float(prevalence * (1.0 - prevalence)),
    }


def confidence_quartile_gains(
    confidence: np.ndarray,
    base_error: np.ndarray,
    memory_error: np.ndarray,
) -> list[dict[str, Any]]:
    """Split positions by confidence rank and report direct memory value."""
    scores = _finite_vector(confidence, "confidence").astype(np.float64, copy=False)
    base = _finite_vector(base_error, "base_error").astype(np.float64, copy=False)
    memory = _finite_vector(memory_error, "memory_error").astype(np.float64, copy=False)
    if scores.size != base.size or scores.size != memory.size:
        raise ValueError("confidence and error vectors must have identical sizes")
    if bool(((scores < 0.0) | (scores > 1.0)).any()):
        raise ValueError("confidence must be in [0, 1]")

    ordered = np.argsort(scores, kind="mergesort")
    groups: list[dict[str, Any]] = []
    for group_index, indices in enumerate(np.array_split(ordered, 4), start=1):
        if indices.size == 0:
            continue
        group_base = base[indices]
        group_memory = memory[indices]
        base_mae = float(group_base.mean())
        memory_mae = float(group_memory.mean())
        absolute_gain = base_mae - memory_mae
        groups.append(
            {
                "quartile": group_index,
                "count": int(indices.size),
                "confidence_mean": float(scores[indices].mean()),
                "confidence_min": float(scores[indices].min()),
                "confidence_max": float(scores[indices].max()),
                "base_mae": base_mae,
                "memory_mae": memory_mae,
                "absolute_memory_gain": absolute_gain,
                "relative_memory_gain_percent": 100.0 * absolute_gain / max(base_mae, 1.0e-12),
                "helpful_rate": float((group_memory < group_base).mean()),
            }
        )
    return groups


def error_aware_diagnostic_metrics(
    predicted_risk: np.ndarray,
    true_risk: np.ndarray,
    fusion_weight: np.ndarray,
    blend_target: np.ndarray,
    blend_valid: np.ndarray,
    contributions: np.ndarray,
) -> dict[str, Any]:
    """Summarize error-aware signals without assigning probability semantics."""
    predicted = _finite_vector(predicted_risk, "predicted_risk").astype(np.float64)
    actual = _finite_vector(true_risk, "true_risk").astype(np.float64)
    weight = np.asarray(fusion_weight).reshape(-1).astype(np.float64)
    target = np.asarray(blend_target).reshape(-1).astype(np.float64)
    valid = np.asarray(blend_valid).reshape(-1).astype(bool)
    if predicted.size != actual.size:
        raise ValueError("predicted and true risk must align")
    if weight.size != target.size or valid.size != target.size:
        raise ValueError("fusion weight, blend target, and validity must align")
    contribution_array = np.asarray(contributions, dtype=np.float64)
    if contribution_array.ndim != 2 or contribution_array.shape[0] != weight.size:
        raise ValueError("contributions must be [positions, features]")
    if not np.isfinite(weight).all() or not np.isfinite(target).all():
        raise ValueError("fusion or blend values contain NaN or Inf")
    if not np.isfinite(contribution_array).all():
        raise ValueError("contributions contain NaN or Inf")
    residual_sum = float(np.square(actual - predicted).sum())
    total_sum = float(np.square(actual - actual.mean()).sum())
    risk_r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else None
    if np.ptp(predicted) == 0.0 or np.ptp(actual) == 0.0:
        risk_spearman = None
    else:
        risk_spearman = float(spearmanr(predicted, actual).statistic)
        if not np.isfinite(risk_spearman):
            risk_spearman = None
    blend_mae = (
        float(np.abs(weight[valid] - target[valid]).mean())
        if bool(valid.any())
        else None
    )
    return {
        "risk_mae": float(np.abs(predicted - actual).mean()),
        "risk_spearman": risk_spearman,
        "risk_r2": risk_r2,
        "blend_target_mae": blend_mae,
        "blend_valid_count": int(valid.sum()),
        "contribution_distributions": (
            [
                distribution_summary(contribution_array[:, index])
                for index in range(contribution_array.shape[1])
            ]
            if contribution_array.shape[0] > 0
            else None
        ),
    }


class DownstreamDiagnosticAccumulator:
    """Accumulate branch metrics and confidence semantics without retaining predictions."""

    def __init__(self, horizon: int, confidence_is_probability: bool = True) -> None:
        self.confidence_is_probability = confidence_is_probability
        self.branch_metrics = {
            "base": ForecastMetricAccumulator(horizon),
            "base_memory_common": ForecastMetricAccumulator(horizon),
            "memory": ForecastMetricAccumulator(horizon),
            "final": ForecastMetricAccumulator(horizon),
        }
        self.observed_count = 0
        self.memory_valid_count = 0
        self._confidence: list[np.ndarray] = []
        self._fusion_weight: list[np.ndarray] = []
        self._base_error: list[np.ndarray] = []
        self._memory_error: list[np.ndarray] = []
        self._fusion_horizon_sum = np.zeros(horizon, dtype=np.float64)
        self._fusion_horizon_count = np.zeros(horizon, dtype=np.int64)
        self._predicted_risk: list[np.ndarray] = []
        self._true_risk: list[np.ndarray] = []
        self._blend_target: list[np.ndarray] = []
        self._blend_valid: list[np.ndarray] = []
        self._additive_contributions: list[np.ndarray] = []

    @torch.no_grad()
    def update(
        self,
        base: torch.Tensor,
        memory: torch.Tensor,
        final: torch.Tensor,
        target: torch.Tensor,
        observed: torch.Tensor,
        memory_valid: torch.Tensor,
        confidence: torch.Tensor,
        fusion_weight: torch.Tensor,
        predicted_risk: torch.Tensor | None = None,
        true_risk: torch.Tensor | None = None,
        blend_target: torch.Tensor | None = None,
        blend_valid: torch.Tensor | None = None,
        additive_contributions: torch.Tensor | None = None,
    ) -> None:
        if base.shape != target.shape or memory.shape != target.shape or final.shape != target.shape:
            raise ValueError("base, memory, final, and target must have identical shapes")
        if observed.shape != target.shape:
            raise ValueError("observed must match target")
        expected_node_shape = target.shape[:-1] + (1,)
        if memory_valid.shape != expected_node_shape:
            raise ValueError("memory_valid must be [B, H, N, 1]")
        if confidence.shape != expected_node_shape or fusion_weight.shape != expected_node_shape:
            raise ValueError("confidence and fusion_weight must be [B, H, N, 1]")
        if not all(bool(torch.isfinite(value).all()) for value in (base, memory, final, target)):
            raise ValueError("diagnostic predictions and target must be finite")

        observed_bool = observed.bool()
        memory_channel_valid = observed_bool & memory_valid.bool().expand_as(observed_bool)
        self.branch_metrics["base"].update(base, target, observed_bool)
        self.branch_metrics["final"].update(final, target, observed_bool)
        self.branch_metrics["base_memory_common"].update(
            base,
            target,
            memory_channel_valid,
        )
        self.branch_metrics["memory"].update(memory, target, memory_channel_valid)
        self.observed_count += int(observed_bool.sum().item())
        self.memory_valid_count += int(memory_channel_valid.sum().item())

        node_valid = observed_bool.all(dim=-1) & memory_valid.squeeze(-1).bool()
        if predicted_risk is not None:
            if true_risk is None or blend_target is None or blend_valid is None or additive_contributions is None:
                raise ValueError("error-aware diagnostic tensors must be provided together")
            if predicted_risk.shape != expected_node_shape or true_risk.shape != expected_node_shape:
                raise ValueError("risk diagnostics must be [B,H,N,1]")
            if blend_target.shape != expected_node_shape or blend_valid.shape != expected_node_shape:
                raise ValueError("blend diagnostics must be [B,H,N,1]")
            if additive_contributions.ndim != 4 or additive_contributions.shape[:3] != target.shape[:3]:
                raise ValueError("additive_contributions must be [B,H,N,F]")
            risk_valid = observed_bool.any(dim=-1)
            self._predicted_risk.append(predicted_risk.squeeze(-1)[risk_valid].float().cpu().numpy())
            self._true_risk.append(true_risk.squeeze(-1)[risk_valid].float().cpu().numpy())
            if bool(node_valid.any()):
                self._blend_target.append(blend_target.squeeze(-1)[node_valid].float().cpu().numpy())
                self._blend_valid.append(blend_valid.squeeze(-1)[node_valid].cpu().numpy())
                self._additive_contributions.append(additive_contributions[node_valid].float().cpu().numpy())
        if not bool(node_valid.any()):
            return
        base_error = (base - target).abs().mean(dim=-1)
        memory_error = (memory - target).abs().mean(dim=-1)
        confidence_node = confidence.squeeze(-1)
        fusion_node = fusion_weight.squeeze(-1)
        self._confidence.append(confidence_node[node_valid].float().cpu().numpy())
        self._fusion_weight.append(fusion_node[node_valid].float().cpu().numpy())
        self._base_error.append(base_error[node_valid].float().cpu().numpy())
        self._memory_error.append(memory_error[node_valid].float().cpu().numpy())
        for horizon_index in range(target.shape[1]):
            valid_horizon = node_valid[:, horizon_index]
            self._fusion_horizon_sum[horizon_index] += float(
                fusion_node[:, horizon_index][valid_horizon].sum().cpu()
            )
            self._fusion_horizon_count[horizon_index] += int(valid_horizon.sum().item())

    def compute(self) -> dict[str, Any]:
        branches = {
            name: (accumulator.compute() if accumulator.count > 0 else None)
            for name, accumulator in self.branch_metrics.items()
        }
        base_mae = float(branches["base"]["mae"])
        final_mae = float(branches["final"]["mae"])
        common_branch = branches["base_memory_common"]
        memory_branch = branches["memory"]
        result: dict[str, Any] = {
            "branches": branches,
            "memory_coverage": self.memory_valid_count / max(self.observed_count, 1),
            "gains": {
                "final_vs_base_absolute_mae": base_mae - final_mae,
                "final_vs_base_percent": 100.0 * (base_mae - final_mae) / base_mae,
                "memory_vs_base_common_absolute_mae": (
                    None
                    if common_branch is None or memory_branch is None
                    else float(common_branch["mae"] - memory_branch["mae"])
                ),
                "memory_vs_base_common_percent": (
                    None
                    if common_branch is None or memory_branch is None
                    else 100.0
                    * float(common_branch["mae"] - memory_branch["mae"])
                    / float(common_branch["mae"])
                ),
            },
        }
        if not self._confidence:
            result.update(
                {
                    "confidence_quality": None,
                    "confidence_distribution": None,
                    "fusion_weight_distribution": None,
                    "fusion_weight_horizon_mean": None,
                    "confidence_quartile_memory_gain": None,
                }
            )
            if self._predicted_risk:
                result["error_aware_quality"] = error_aware_diagnostic_metrics(
                    predicted_risk=np.concatenate(self._predicted_risk),
                    true_risk=np.concatenate(self._true_risk),
                    fusion_weight=np.empty(0, dtype=np.float64),
                    blend_target=np.empty(0, dtype=np.float64),
                    blend_valid=np.empty(0, dtype=bool),
                    contributions=np.empty((0, 9), dtype=np.float64),
                )
            else:
                result["error_aware_quality"] = None
            return result

        confidence = np.concatenate(self._confidence)
        fusion_weight = np.concatenate(self._fusion_weight)
        base_error = np.concatenate(self._base_error)
        memory_error = np.concatenate(self._memory_error)
        helpful = memory_error < base_error
        horizon_mean = np.divide(
            self._fusion_horizon_sum,
            np.maximum(self._fusion_horizon_count, 1),
        )
        ranking = binary_confidence_metrics(confidence, helpful)
        ranking = {
            key: value
            for key, value in ranking.items()
            if key not in {"brier", "ece", "ece_bins", "constant_brier"}
        }
        result.update(
            {
                "confidence_quality": (
                    binary_confidence_metrics(confidence, helpful)
                    if self.confidence_is_probability
                    else None
                ),
                "fusion_helpfulness_ranking": ranking,
                "confidence_distribution": distribution_summary(confidence),
                "fusion_weight_distribution": distribution_summary(fusion_weight),
                "fusion_weight_horizon_mean": horizon_mean.tolist(),
                "confidence_quartile_memory_gain": confidence_quartile_gains(
                    confidence,
                    base_error,
                    memory_error,
                ),
            }
        )
        if self._predicted_risk:
            result["error_aware_quality"] = error_aware_diagnostic_metrics(
                predicted_risk=np.concatenate(self._predicted_risk),
                true_risk=np.concatenate(self._true_risk),
                fusion_weight=fusion_weight,
                blend_target=np.concatenate(self._blend_target),
                blend_valid=np.concatenate(self._blend_valid),
                contributions=np.concatenate(self._additive_contributions),
            )
        else:
            result["error_aware_quality"] = None
        return result


@torch.no_grad()
def diagnose_downstream_checkpoint(
    config: ExperimentConfig,
    pretrained_checkpoint: str | Path,
    downstream_checkpoint: str | Path,
    bank_path: str | Path,
    split: str = "val",
    max_batches: int | None = None,
    candidate_protocol: str | None = None,
) -> dict[str, Any]:
    """Evaluate base, memory, final, and confidence branches of a trained model."""
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    started = time.perf_counter()
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    pretrained, _ = load_pretrained_model(
        config,
        pretrained_checkpoint,
        data.series.slots_per_day,
        device,
    )
    checkpoint = load_checkpoint(downstream_checkpoint, device)
    mode = checkpoint_downstream_mode(checkpoint)
    level_weight = checkpoint_bank_level_weight(
        checkpoint,
        default=config.bank.level_weight,
    )
    saved_candidate_protocol = checkpoint_candidate_protocol(checkpoint)
    if candidate_protocol is not None:
        candidate_protocol = checkpoint_candidate_protocol(
            checkpoint,
            expected=candidate_protocol,
        )
    else:
        candidate_protocol = saved_candidate_protocol
    config = replace(
        config,
        bank=replace(config.bank, level_weight=level_weight),
        target=replace(
            config.target,
            downstream_mode=mode,
            candidate_protocol=candidate_protocol,
        ),
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
    accumulator = DownstreamDiagnosticAccumulator(
        config.data.horizon,
        confidence_is_probability=mode != LEARNED_TOPK_ERROR_AWARE,
    )
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
                candidate_protocol=candidate_protocol,
            )
            output = downstream(x, candidates, aggregation)
            target = batch["y"].to(device)
            predicted_risk = true_risk = blend_target = blend_valid = contributions = None
            if mode == LEARNED_TOPK_ERROR_AWARE:
                predicted_risk = output.predicted_base_risk
                contributions = output.additive_contributions
                if predicted_risk is None or contributions is None:
                    raise ValueError("error-aware checkpoint did not return diagnostic tensors")
                true_risk, _ = build_huber_risk_target(
                    output.base_prediction,
                    target,
                    batch["y_observed"].to(device),
                )
                blend_target, blend_valid = build_blend_target(
                    output.base_prediction,
                    output.memory_prediction,
                    target,
                    batch["y_observed"].to(device),
                    output.memory_valid,
                    minimum_direction_norm=config.target.blend_minimum_direction_norm,
                )
            accumulator.update(
                data.scaler.inverse_transform_torch(output.base_prediction),
                data.scaler.inverse_transform_torch(output.memory_prediction),
                data.scaler.inverse_transform_torch(output.final_prediction),
                data.scaler.inverse_transform_torch(target),
                batch["y_observed"].to(device),
                output.memory_valid,
                output.confidence,
                output.fusion_weight,
                predicted_risk=predicted_risk,
                true_risk=true_risk,
                blend_target=blend_target,
                blend_valid=blend_valid,
                additive_contributions=contributions,
            )
            batches += 1
            queries += int(x.shape[0])
    if batches == 0:
        raise ValueError("downstream diagnostic processed no batches")
    result = accumulator.compute()
    result.update(
        {
            "schema_version": 2,
            "dataset": dataset_name,
            "downstream_mode": mode,
            "candidate_protocol": candidate_protocol,
            "level_weight": config.bank.level_weight,
            "split": split,
            "queries": queries,
            "batches": batches,
            "checkpoint": str(Path(downstream_checkpoint).resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "pretrained_checkpoint": str(Path(pretrained_checkpoint).resolve()),
            "bank": str(Path(bank_path).resolve()),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return result

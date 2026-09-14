"""Future-surprise stratification for frozen retrieval diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import default_collate

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.metrics import ForecastMetricAccumulator
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.utils import array_sha256, resolve_device, save_json


def persistence_surprise(
    context_physical: torch.Tensor,
    context_observed: torch.Tensor,
    future_physical: torch.Tensor,
    future_observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return persistence-baseline MAE and validity for every ``[B,N]`` anchor.

    The endpoint is the last observed context value when available.  If the
    final context step is missing, the visible context mean is used instead.
    All inputs and the returned surprise are in physical data units.
    """
    if context_physical.ndim != 4 or future_physical.ndim != 4:
        raise ValueError("context and future must be [B,T,N,C] tensors")
    if context_observed.shape != context_physical.shape:
        raise ValueError("context_observed must match context_physical")
    if future_observed.shape != future_physical.shape:
        raise ValueError("future_observed must match future_physical")
    if context_physical.shape[0] != future_physical.shape[0]:
        raise ValueError("context and future batch dimensions must match")
    if context_physical.shape[2:] != future_physical.shape[2:]:
        raise ValueError("context and future node/channel dimensions must match")

    context_valid = context_observed.bool() & torch.isfinite(context_physical)
    future_valid = future_observed.bool() & torch.isfinite(future_physical)
    visible_count = context_valid.sum(dim=1)
    visible_sum = torch.where(
        context_valid,
        context_physical,
        torch.zeros_like(context_physical),
    ).sum(dim=1)
    visible_mean = visible_sum / visible_count.clamp_min(1).to(context_physical.dtype)
    last_valid = context_valid[:, -1]
    endpoint = torch.where(last_valid, context_physical[:, -1], visible_mean)
    endpoint_valid = last_valid | (visible_count > 0)

    common = future_valid & endpoint_valid[:, None]
    count = common.sum(dim=(1, 3))
    absolute = (future_physical - endpoint[:, None]).abs()
    total = torch.where(common, absolute, torch.zeros_like(absolute)).sum(dim=(1, 3))
    valid = count > 0
    surprise = total / count.clamp_min(1).to(total.dtype)
    surprise = torch.where(valid, surprise, torch.full_like(surprise, torch.nan))
    return surprise, valid


def surprise_thresholds(
    surprise: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute shared q80/q95 thresholds over finite valid anchors."""
    values = np.asarray(
        surprise.detach().cpu().numpy() if isinstance(surprise, torch.Tensor) else surprise,
        dtype=np.float64,
    )
    mask = np.asarray(
        valid.detach().cpu().numpy() if isinstance(valid, torch.Tensor) else valid,
        dtype=bool,
    )
    if values.shape != mask.shape:
        raise ValueError("surprise and valid must have identical shapes")
    selected = values[mask & np.isfinite(values)]
    if selected.size == 0:
        raise ValueError("no finite surprise values are available")
    return {
        "q80": float(np.quantile(selected, 0.80)),
        "q95": float(np.quantile(selected, 0.95)),
    }


def surprise_strata_masks(
    surprise: torch.Tensor | np.ndarray,
    valid: torch.Tensor | np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    """Build disjoint ordinary/top20 masks and a nested top5 mask."""
    values = torch.as_tensor(surprise)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    if values.shape != mask.shape:
        raise ValueError("surprise and valid must have identical shapes")
    if "q80" not in thresholds or "q95" not in thresholds:
        raise ValueError("thresholds must contain q80 and q95")
    finite = mask & torch.isfinite(values)
    q80 = float(thresholds["q80"])
    q95 = float(thresholds["q95"])
    if q95 < q80:
        raise ValueError("q95 must not be below q80")
    return {
        "all": finite,
        "ordinary_80": finite & (values < q80),
        "surprise_top20": finite & (values >= q80),
        "surprise_top5": finite & (values >= q95),
    }


def stratified_forecast_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    strata: Mapping[str, torch.Tensor | np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Compute physical-unit forecast metrics for shared ``[B,N]`` strata."""
    if prediction.ndim != 4 or target.shape != prediction.shape or observed.shape != target.shape:
        raise ValueError("prediction, target, and observed must be aligned [B,H,N,C]")
    result: dict[str, dict[str, Any]] = {}
    for name, anchor_mask_source in strata.items():
        anchor_mask = torch.as_tensor(
            anchor_mask_source,
            dtype=torch.bool,
            device=prediction.device,
        )
        if anchor_mask.shape != (prediction.shape[0], prediction.shape[2]):
            raise ValueError("every stratum mask must be [B,N]")
        selected = observed.bool() & anchor_mask[:, None, :, None]
        accumulator = ForecastMetricAccumulator(prediction.shape[1])
        accumulator.update(prediction, target, selected)
        if accumulator.count == 0:
            result[name] = {
                "anchor_count": int(anchor_mask.sum().item()),
                "count": 0,
                "mae": None,
                "rmse": None,
                "mape": None,
                "horizon_mae": [],
                "horizon_rmse": [],
                "horizon_mape": [],
            }
            continue
        metrics = accumulator.compute()
        result[name] = {
            "anchor_count": int(anchor_mask.sum().item()),
            "count": int(accumulator.count),
            **metrics,
        }
    return result


def aligned_anchor_ranking_metrics(
    key_distance: np.ndarray,
    teacher_distance: np.ndarray,
    valid: np.ndarray,
    *,
    k: int = 5,
    chunk_size: int = 4096,
) -> dict[str, np.ndarray]:
    """Return Spearman and Recall@K without losing ``[B,N]`` alignment."""
    key = np.asarray(key_distance, dtype=np.float64)
    teacher = np.asarray(teacher_distance, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if key.ndim != 3 or teacher.shape != key.shape or mask.shape != key.shape:
        raise ValueError("ranking distances and valid must be aligned [B,N,R] arrays")
    if k <= 0 or k > key.shape[-1]:
        raise ValueError("k must be in [1,R]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    shape = key.shape[:2]
    flat_key = key.reshape(-1, key.shape[-1])
    flat_teacher = teacher.reshape(-1, teacher.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1])
    spearman = np.full(flat_key.shape[0], np.nan, dtype=np.float64)
    recall = np.full(flat_key.shape[0], np.nan, dtype=np.float64)
    counts_output = np.zeros(flat_key.shape[0], dtype=np.int64)

    for start in range(0, flat_key.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_key.shape[0])
        left = flat_key[start:stop]
        right = flat_teacher[start:stop]
        finite = flat_mask[start:stop] & np.isfinite(left) & np.isfinite(right)
        counts = finite.sum(axis=1)
        counts_output[start:stop] = counts
        eligible = counts >= 2
        if np.any(eligible):
            eligible_indices = np.flatnonzero(eligible)
            selected_left = left[eligible]
            selected_right = right[eligible]
            selected_finite = finite[eligible]
            selected_counts = counts[eligible]
            left_rank = rankdata(
                np.where(selected_finite, selected_left, np.nan),
                axis=1,
                method="average",
                nan_policy="omit",
            )
            right_rank = rankdata(
                np.where(selected_finite, selected_right, np.nan),
                axis=1,
                method="average",
                nan_policy="omit",
            )
            left_rank = np.where(selected_finite, left_rank, 0.0)
            right_rank = np.where(selected_finite, right_rank, 0.0)
            left_center = left_rank - left_rank.sum(axis=1, keepdims=True) / selected_counts[:, None]
            right_center = right_rank - right_rank.sum(axis=1, keepdims=True) / selected_counts[:, None]
            covariance = (left_center * right_center * selected_finite).sum(axis=1)
            denominator = np.sqrt(
                (np.square(left_center) * selected_finite).sum(axis=1)
                * (np.square(right_center) * selected_finite).sum(axis=1)
            )
            values = np.divide(
                covariance,
                denominator,
                out=np.zeros_like(covariance),
                where=denominator > 0,
            )
            spearman[start + eligible_indices] = values

        recall_eligible = counts > k
        if np.any(recall_eligible):
            recall_indices = np.flatnonzero(recall_eligible)
            selected_finite = finite[recall_eligible]
            key_order = np.argsort(
                np.where(selected_finite, left[recall_eligible], np.inf),
                axis=1,
                kind="stable",
            )[:, :k]
            teacher_order = np.argsort(
                np.where(selected_finite, right[recall_eligible], np.inf),
                axis=1,
                kind="stable",
            )[:, :k]
            overlap = (key_order[:, :, None] == teacher_order[:, None, :]).any(axis=2).sum(axis=1)
            recall[start + recall_indices] = overlap.astype(np.float64) / float(k)

    return {
        "spearman": spearman.reshape(shape),
        "recall_at_5": recall.reshape(shape),
        "candidate_count": counts_output.reshape(shape),
    }


def stratified_scalar_summary(
    values: np.ndarray | torch.Tensor,
    strata: Mapping[str, np.ndarray | torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    """Summarize aligned scalar values within each ``[B,N]`` stratum."""
    array = np.asarray(
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values,
        dtype=np.float64,
    )
    result: dict[str, dict[str, float | int]] = {}
    for name, mask_source in strata.items():
        mask = np.asarray(
            mask_source.detach().cpu().numpy()
            if isinstance(mask_source, torch.Tensor)
            else mask_source,
            dtype=bool,
        )
        if mask.shape != array.shape:
            raise ValueError("values and every stratum mask must have identical shapes")
        selected = array[mask & np.isfinite(array)]
        result[name] = {
            "count": int(selected.size),
            "mean": float(selected.mean()) if selected.size else 0.0,
            "std": float(selected.std()) if selected.size else 0.0,
        }
    return result


@dataclass(frozen=True)
class _FrozenVersionOutput:
    version: str
    checkpoint: str
    checkpoint_epoch: int | None
    bank: str
    encoder_fingerprint: str
    sample_id: np.ndarray
    context_end: np.ndarray
    event_ids: np.ndarray
    event_valid: np.ndarray
    surprise: torch.Tensor
    surprise_valid: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    target_observed: torch.Tensor
    aggregation_valid: torch.Tensor
    spearman: np.ndarray
    recall_at_5: np.ndarray
    candidate_count: np.ndarray
    elapsed_seconds: float
    peak_cuda_memory_gb: float


def _assert_matched_configs(
    offset_only: ExperimentConfig,
    offset_decay: ExperimentConfig,
) -> None:
    """Reject comparisons that differ beyond the intended teacher version."""
    if offset_only.data != offset_decay.data:
        raise ValueError("Offset-only and OffsetDecay data configs must match exactly")
    if offset_only.model != offset_decay.model:
        raise ValueError("Offset-only and OffsetDecay model configs must match exactly")
    if offset_only.runtime.seed != offset_decay.runtime.seed:
        raise ValueError("Offset-only and OffsetDecay seeds must match")
    bank_fields = (
        "memory_fraction",
        "event_top_r",
        "node_top_k",
        "level_weight",
        "level_temperature",
        "search_temperature",
        "key_dtype",
    )
    for name in bank_fields:
        if getattr(offset_only.bank, name) != getattr(offset_decay.bank, name):
            raise ValueError(f"Offset-only and OffsetDecay bank.{name} must match")
    if offset_only.pretrain.relation_teacher_mode != "offset_only":
        raise ValueError("the Offset-only config must use relation_teacher_mode=offset_only")
    if offset_decay.pretrain.relation_teacher_mode != "offset_decay":
        raise ValueError("the OffsetDecay config must use relation_teacher_mode=offset_decay")
    if (
        offset_only.pretrain.relation_distance_normalization
        != offset_decay.pretrain.relation_distance_normalization
    ):
        raise ValueError("teacher distance normalization must match")


@torch.inference_mode()
def _run_frozen_version(
    *,
    version: str,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    data: Any,
    graph_cpu: Any,
    query_rows: np.ndarray,
    candidate_protocol: str,
    batch_size: int,
    device: torch.device,
) -> _FrozenVersionOutput:
    """Run one original-only model/Bank pair on fixed validation rows."""
    from stanchor.diagnostics.retrieval_collapse import _aggregation_for_version
    from stanchor.diagnostics.retrieval_visualization import (
        _candidate_node_keys,
        _candidate_teacher_signatures,
        build_diagnostic_event_candidates,
        build_teacher_aligned_signature,
        node_key_distances,
        teacher_candidate_distances,
    )
    from stanchor.engine.target import _validate_bank
    from stanchor.retrieval.strategies import event_candidate_futures

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    graph = graph_cpu.to(device)
    checkpoint_source = resolve_project_path(checkpoint_path)
    bank_source = resolve_project_path(bank_path)
    model, checkpoint_payload = load_pretrained_model(
        config,
        checkpoint_source,
        data.series.slots_per_day,
        device,
    )
    model.eval()
    aggregation_fn, _ = _aggregation_for_version(version)

    sample_ids: list[np.ndarray] = []
    context_ends: list[np.ndarray] = []
    event_ids_output: list[np.ndarray] = []
    event_valid_output: list[np.ndarray] = []
    surprises: list[torch.Tensor] = []
    surprise_validity: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []
    aggregation_masks: list[torch.Tensor] = []
    spearman_values: list[np.ndarray] = []
    recall_values: list[np.ndarray] = []
    candidate_counts: list[np.ndarray] = []

    with MemoryBank(bank_source) as bank:
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
        fingerprint = str(bank.manifest.encoder_fingerprint)
        processed = 0
        for start in range(0, query_rows.size, batch_size):
            rows = query_rows[start : start + batch_size]
            query_batch = default_collate([data.val[int(row)] for row in rows])
            retrieval_x = query_batch["retrieval_x"].to(device)
            retrieval_observed = query_batch["retrieval_observed"].to(device).bool()
            encoding = model.encode_clean(
                retrieval_x,
                retrieval_observed,
                query_batch["retrieval_weekday"].to(device),
                query_batch["retrieval_slot"].to(device),
                graph,
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
                bank,
                events.event_ids,
                events.valid,
                device,
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
            key_distance, key_valid = node_key_distances(
                encoding.retrieval.node_keys,
                candidate_keys,
                events.valid,
            )
            ranking = aligned_anchor_ranking_metrics(
                key_distance.detach().cpu().numpy(),
                teacher_distance.detach().cpu().numpy(),
                (key_valid & teacher_valid).detach().cpu().numpy(),
                k=5,
            )

            node_candidates = retriever.rerank_nodes(
                encoding.retrieval.node_keys,
                encoding.statistics.level_features,
                events,
            )
            aggregation = aggregation_fn(
                node_candidates,
                query_batch["x"].to(device),
                query_batch["x_observed"].to(device),
                bank,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            context_physical = data.scaler.inverse_transform_torch(
                query_batch["x"].to(device)
            )
            target_physical = data.scaler.inverse_transform_torch(
                query_batch["y"].to(device)
            )
            prediction_physical = data.scaler.inverse_transform_torch(
                aggregation.prediction
            )
            surprise, surprise_valid = persistence_surprise(
                context_physical,
                query_batch["x_observed"].to(device),
                target_physical,
                query_batch["y_observed"].to(device),
            )

            sample_ids.append(query_batch["sample_id"].cpu().numpy())
            context_ends.append(query_batch["context_end"].cpu().numpy())
            event_ids_output.append(events.event_ids.detach().cpu().numpy())
            event_valid_output.append(events.valid.detach().cpu().numpy())
            surprises.append(surprise.cpu())
            surprise_validity.append(surprise_valid.cpu())
            predictions.append(prediction_physical.cpu())
            targets.append(target_physical.cpu())
            target_masks.append(query_batch["y_observed"].bool().cpu())
            aggregation_masks.append(aggregation.valid.cpu())
            spearman_values.append(ranking["spearman"])
            recall_values.append(ranking["recall_at_5"])
            candidate_counts.append(ranking["candidate_count"])

            processed += int(rows.size)
            if processed == query_rows.size or processed % 128 == 0:
                print(
                    f"[retrieval-surprise:{version}] processed {processed}/{query_rows.size} queries",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = float(torch.cuda.max_memory_allocated(device) / 1024**3)
        else:
            peak_memory = 0.0

    output = _FrozenVersionOutput(
        version=version,
        checkpoint=str(checkpoint_source.resolve()),
        checkpoint_epoch=(
            int(checkpoint_payload["epoch"])
            if checkpoint_payload.get("epoch") is not None
            else None
        ),
        bank=str(bank_source.resolve()),
        encoder_fingerprint=fingerprint,
        sample_id=np.concatenate(sample_ids),
        context_end=np.concatenate(context_ends),
        event_ids=np.concatenate(event_ids_output),
        event_valid=np.concatenate(event_valid_output),
        surprise=torch.cat(surprises),
        surprise_valid=torch.cat(surprise_validity),
        prediction=torch.cat(predictions),
        target=torch.cat(targets),
        target_observed=torch.cat(target_masks),
        aggregation_valid=torch.cat(aggregation_masks),
        spearman=np.concatenate(spearman_values),
        recall_at_5=np.concatenate(recall_values),
        candidate_count=np.concatenate(candidate_counts),
        elapsed_seconds=float(time.perf_counter() - started),
        peak_cuda_memory_gb=peak_memory,
    )
    del model, graph
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _per_anchor_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> np.ndarray:
    mask = valid.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    count = mask.sum(dim=(1, 3))
    total = torch.where(
        mask,
        (prediction - target).abs(),
        torch.zeros_like(prediction),
    ).sum(dim=(1, 3))
    values = total / count.clamp_min(1).to(total.dtype)
    values = torch.where(count > 0, values, torch.full_like(values, torch.nan))
    return values.cpu().numpy()


def _tail_degradation(
    forecast: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    ordinary = float(forecast["ordinary_80"]["mae"])
    output: dict[str, dict[str, float]] = {}
    for name in ("surprise_top20", "surprise_top5"):
        value = float(forecast[name]["mae"])
        output[name] = {
            "mae_minus_ordinary": value - ordinary,
            "mae_relative_increase_percent": 100.0 * (value / ordinary - 1.0),
        }
    return output


def _save_comparison_plot(
    path: Path,
    versions: Mapping[str, Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    strata = ("all", "ordinary_80", "surprise_top20", "surprise_top5")
    labels = ("All", "Ordinary 80%", "Surprise top 20%", "Surprise top 5%")
    x = np.arange(len(strata), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    for offset, (name, color) in enumerate(
        (("offset_only", "#2878B5"), ("offset_decay", "#D95319"))
    ):
        values = [float(versions[name]["memory_forecasting"][key]["mae"]) for key in strata]
        positions = x + (offset - 0.5) * width
        bars = axis.bar(positions, values, width=width, label=name, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Memory MAE (physical units)")
    axis.set_title("Frozen retrieval by Future Surprise")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def run_retrieval_surprise_comparison(
    *,
    offset_only_config: ExperimentConfig,
    offset_only_checkpoint: str | Path,
    offset_only_bank: str | Path,
    offset_decay_config: ExperimentConfig,
    offset_decay_checkpoint: str | Path,
    offset_decay_bank: str | Path,
    output_dir: str | Path,
    candidate_protocol: str = "weekday_radius1_overlap",
    max_queries: int = 1024,
    batch_size: int = 4,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Compare matched frozen retrieval models under persistence surprise."""
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _assert_matched_configs(offset_only_config, offset_decay_config)

    from stanchor.diagnostics.retrieval_collapse import weekly_donor_pairs
    from stanchor.diagnostics.retrieval_visualization import future_information_boundary

    started = time.perf_counter()
    device = (
        torch.device(device_override)
        if device_override is not None
        else resolve_device(offset_only_config.runtime.device)
    )
    data, graph_cpu = build_data_and_graph(offset_only_config)
    query_rows = weekly_donor_pairs(
        data.val.context_end_indices,
        weekly_steps=7 * data.series.slots_per_day,
        max_queries=max_queries,
    )[:, 0]
    if np.unique(query_rows).size != query_rows.size:
        raise ValueError("fixed query selection contains duplicate validation rows")

    offset_only = _run_frozen_version(
        version="hn_offset_only_v1",
        config=offset_only_config,
        checkpoint_path=offset_only_checkpoint,
        bank_path=offset_only_bank,
        data=data,
        graph_cpu=graph_cpu,
        query_rows=query_rows,
        candidate_protocol=candidate_protocol,
        batch_size=batch_size,
        device=device,
    )
    offset_decay = _run_frozen_version(
        version="hn_offset_decay_v2",
        config=offset_decay_config,
        checkpoint_path=offset_decay_checkpoint,
        bank_path=offset_decay_bank,
        data=data,
        graph_cpu=graph_cpu,
        query_rows=query_rows,
        candidate_protocol=candidate_protocol,
        batch_size=batch_size,
        device=device,
    )

    if not np.array_equal(offset_only.sample_id, offset_decay.sample_id):
        raise ValueError("the two versions did not evaluate identical sample ids")
    if not np.array_equal(offset_only.context_end, offset_decay.context_end):
        raise ValueError("the two versions did not evaluate identical context ends")
    if not np.array_equal(offset_only.event_ids, offset_decay.event_ids):
        raise ValueError("the two versions did not use identical candidate event ids")
    if not np.array_equal(offset_only.event_valid, offset_decay.event_valid):
        raise ValueError("the two versions did not use identical candidate validity")
    if not torch.equal(offset_only.target_observed, offset_decay.target_observed):
        raise ValueError("the two versions have different target masks")
    if not torch.allclose(offset_only.target, offset_decay.target, equal_nan=True):
        raise ValueError("the two versions have different physical targets")
    if not torch.equal(offset_only.surprise_valid, offset_decay.surprise_valid):
        raise ValueError("the two versions have different surprise validity")
    if not torch.allclose(offset_only.surprise, offset_decay.surprise, equal_nan=True):
        raise ValueError("the two versions have different surprise values")

    shared_forecast_valid = (
        offset_only.target_observed
        & offset_only.aggregation_valid
        & offset_decay.aggregation_valid
    )
    shared_anchor_valid = offset_only.surprise_valid & shared_forecast_valid.any(dim=(1, 3))
    thresholds = surprise_thresholds(offset_only.surprise, shared_anchor_valid)
    strata = surprise_strata_masks(offset_only.surprise, shared_anchor_valid, thresholds)

    outputs = {
        "offset_only": offset_only,
        "offset_decay": offset_decay,
    }
    version_results: dict[str, dict[str, Any]] = {}
    per_anchor_mae: dict[str, np.ndarray] = {}
    for name, output in outputs.items():
        forecast = stratified_forecast_metrics(
            output.prediction,
            output.target,
            shared_forecast_valid,
            strata,
        )
        per_anchor_mae[name] = _per_anchor_mae(
            output.prediction,
            output.target,
            shared_forecast_valid,
        )
        version_results[name] = {
            "version": output.version,
            "checkpoint": {
                "path": output.checkpoint,
                "epoch": output.checkpoint_epoch,
            },
            "bank": {
                "path": output.bank,
                "encoder_fingerprint": output.encoder_fingerprint,
            },
            "memory_forecasting": forecast,
            "memory_anchor_mae": stratified_scalar_summary(per_anchor_mae[name], strata),
            "future_ranking": {
                "spearman": stratified_scalar_summary(output.spearman, strata),
                "recall_at_5": stratified_scalar_summary(output.recall_at_5, strata),
            },
            "tail_degradation": _tail_degradation(forecast),
            "runtime": {
                "elapsed_seconds": output.elapsed_seconds,
                "peak_cuda_memory_gb": output.peak_cuda_memory_gb,
            },
        }

    paired_delta = per_anchor_mae["offset_only"] - per_anchor_mae["offset_decay"]
    comparison_by_stratum: dict[str, dict[str, float]] = {}
    paired_summary = stratified_scalar_summary(paired_delta, strata)
    for stratum in strata:
        only_mae = float(version_results["offset_only"]["memory_forecasting"][stratum]["mae"])
        decay_mae = float(version_results["offset_decay"]["memory_forecasting"][stratum]["mae"])
        comparison_by_stratum[stratum] = {
            "observation_weighted_mae_offset_only_minus_offset_decay": only_mae - decay_mae,
            "anchor_weighted_mae_offset_only_minus_offset_decay": float(
                paired_summary[stratum]["mean"]
            ),
            "paired_anchor_count": int(paired_summary[stratum]["count"]),
        }

    only_tail = version_results["offset_only"]["tail_degradation"]
    decay_tail = version_results["offset_decay"]["tail_degradation"]
    only_worse_both = all(
        comparison_by_stratum[name]["observation_weighted_mae_offset_only_minus_offset_decay"] > 0
        and only_tail[name]["mae_relative_increase_percent"]
        > decay_tail[name]["mae_relative_increase_percent"]
        for name in ("surprise_top20", "surprise_top5")
    )
    only_not_worse_both = all(
        comparison_by_stratum[name]["observation_weighted_mae_offset_only_minus_offset_decay"] <= 0
        for name in ("surprise_top20", "surprise_top5")
    )
    if only_worse_both:
        descriptive_outcome = "offset_only_has_consistent_tail_warning_on_point_estimates"
    elif only_not_worse_both:
        descriptive_outcome = "offset_only_not_worse_in_both_tail_strata_on_point_estimates"
    else:
        descriptive_outcome = "mixed_tail_evidence"

    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "surprise_anchor_values.npz",
        sample_id=offset_only.sample_id,
        context_end=offset_only.context_end,
        surprise=offset_only.surprise.numpy(),
        surprise_valid=shared_anchor_valid.numpy(),
        offset_only_spearman=offset_only.spearman,
        offset_decay_spearman=offset_decay.spearman,
        offset_only_recall_at_5=offset_only.recall_at_5,
        offset_decay_recall_at_5=offset_decay.recall_at_5,
        offset_only_memory_mae=per_anchor_mae["offset_only"],
        offset_decay_memory_mae=per_anchor_mae["offset_decay"],
    )

    valid_surprise = offset_only.surprise[shared_anchor_valid].numpy()
    result: dict[str, Any] = {
        "diagnostic": "future_surprise_stratified_retrieval",
        "query_count": int(offset_only.sample_id.size),
        "node_count": int(offset_only.surprise.shape[1]),
        "candidate_protocol": candidate_protocol,
        "surprise_definition": {
            "baseline": "repeat the last observed 12-step context value; use visible context mean if the final value is missing",
            "metric": "masked mean absolute future deviation in physical units",
            "thresholds": thresholds,
            "distribution": {
                "count": int(valid_surprise.size),
                "mean": float(valid_surprise.mean()),
                "std": float(valid_surprise.std()),
                "median": float(np.median(valid_surprise)),
                "q80": thresholds["q80"],
                "q95": thresholds["q95"],
                "max": float(valid_surprise.max()),
            },
        },
        "strata": {
            name: {"anchor_count": int(mask.sum().item())}
            for name, mask in strata.items()
        },
        "matching_audit": {
            "same_data_config": True,
            "same_model_config": True,
            "same_seed": True,
            "same_query_ids": True,
            "same_context_ends": True,
            "same_candidate_event_ids": True,
            "same_target_values_and_masks": True,
            "same_surprise_values_and_thresholds": True,
            "sample_id_sha256": array_sha256(offset_only.sample_id),
            "candidate_event_id_sha256": array_sha256(offset_only.event_ids),
        },
        "future_information_boundary": future_information_boundary(),
        "versions": version_results,
        "comparison": {
            "mae_delta_sign": "negative means Offset-only is better",
            "by_stratum": comparison_by_stratum,
            "descriptive_outcome": descriptive_outcome,
            "scope": "point estimates from one fixed seed and validation query set; not a significance claim",
        },
        "runtime": {
            "device": str(device),
            "batch_size": batch_size,
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_cuda_memory_gb": max(
                offset_only.peak_cuda_memory_gb,
                offset_decay.peak_cuda_memory_gb,
            ),
        },
        "artifacts": {
            "anchor_values": "surprise_anchor_values.npz",
            "comparison_plot": "surprise_stratified_memory_mae.png",
        },
    }
    _save_comparison_plot(
        output_path / "surprise_stratified_memory_mae.png",
        version_results,
    )
    save_json(output_path / "surprise_diagnostic.json", result)
    return result

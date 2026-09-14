"""Frozen 2-Key x 2-payload attribution for retrieval-memory forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import default_collate

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.diagnostics.retrieval_surprise import (
    _assert_matched_configs,
    _per_anchor_mae,
    persistence_surprise,
    stratified_forecast_metrics,
    stratified_scalar_summary,
    surprise_strata_masks,
    surprise_thresholds,
)
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.retrieval.strategies import offset_decay_aggregation, offset_only_aggregation
from stanchor.utils import array_sha256, resolve_device, save_json


KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY = "key_offset_only__payload_offset_only"
KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY = "key_offset_only__payload_offset_decay"
KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY = "key_offset_decay__payload_offset_only"
KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY = "key_offset_decay__payload_offset_decay"

NON_KEY_BANK_FILES = (
    "calendar_event_ids.npy",
    "calendar_offsets.npy",
    "context_end.npy",
    "context_start.npy",
    "future_end.npy",
    "future_masks.npy",
    "future_values.npy",
    "level_features.npy",
    "sample_id.npy",
    "slot.npy",
    "weekday.npy",
)


def _file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_non_key_bank_files(
    offset_only_bank: str | Path,
    offset_decay_bank: str | Path,
) -> dict[str, Any]:
    """Require byte-identical Bank content for every non-Key array."""
    left_root = Path(offset_only_bank)
    right_root = Path(offset_decay_bank)
    result: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    for name in NON_KEY_BANK_FILES:
        left = left_root / name
        right = right_root / name
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"required non-Key Bank file is missing: {name}")
        left_hash = _file_sha256(left)
        right_hash = _file_sha256(right)
        identical = left_hash == right_hash and left.stat().st_size == right.stat().st_size
        result[name] = {
            "identical": identical,
            "sha256": left_hash if identical else None,
            "offset_only_sha256": left_hash,
            "offset_decay_sha256": right_hash,
            "bytes": int(left.stat().st_size),
        }
        if not identical:
            mismatches.append(name)
    if mismatches:
        raise ValueError(f"non-Key Bank files differ: {', '.join(mismatches)}")
    return {"all_identical": True, "files": result}


def factorial_mae_effects(
    cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Compute payload, Key, and difference-in-differences MAE effects."""
    required = (
        KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY,
        KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY,
        KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY,
        KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY,
    )
    missing = [name for name in required if name not in cells]
    if missing:
        raise ValueError(f"factorial cells are missing: {missing}")
    strata = tuple(cells[required[0]].keys())
    if any(tuple(cells[name].keys()) != strata for name in required[1:]):
        raise ValueError("all factorial cells must expose the same ordered strata")

    result: dict[str, dict[str, float]] = {}
    for stratum in strata:
        oo_oo = float(cells[KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY][stratum]["mae"])
        oo_od = float(cells[KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY][stratum]["mae"])
        od_oo = float(cells[KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY][stratum]["mae"])
        od_od = float(cells[KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY][stratum]["mae"])
        payload_under_oo_key = oo_oo - oo_od
        payload_under_od_key = od_oo - od_od
        key_under_oo_payload = oo_oo - od_oo
        key_under_od_payload = oo_od - od_od
        result[stratum] = {
            "payload_effect_under_offset_only_key": payload_under_oo_key,
            "payload_effect_under_offset_decay_key": payload_under_od_key,
            "key_effect_under_offset_only_payload": key_under_oo_payload,
            "key_effect_under_offset_decay_payload": key_under_od_payload,
            "interaction_difference_in_differences": (
                payload_under_oo_key - payload_under_od_key
            ),
            "average_payload_main_effect": 0.5 * (
                payload_under_oo_key + payload_under_od_key
            ),
            "average_key_main_effect": 0.5 * (
                key_under_oo_payload + key_under_od_payload
            ),
            "matched_diagonal_total_effect": oo_oo - od_od,
        }
    return result


def query_block_bootstrap_contrast(
    paired_delta: np.ndarray,
    strata: Mapping[str, np.ndarray | torch.Tensor],
    *,
    resamples: int = 5000,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Bootstrap paired anchor effects by resampling complete query rows."""
    delta = np.asarray(paired_delta, dtype=np.float64)
    if delta.ndim != 2:
        raise ValueError("paired_delta must be [Q,N]")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    result: dict[str, dict[str, Any]] = {}
    for offset, (name, mask_source) in enumerate(strata.items()):
        mask = np.asarray(
            mask_source.detach().cpu().numpy()
            if isinstance(mask_source, torch.Tensor)
            else mask_source,
            dtype=bool,
        )
        if mask.shape != delta.shape:
            raise ValueError("every stratum mask must match paired_delta")
        finite = mask & np.isfinite(delta)
        anchor_count = int(finite.sum())
        if anchor_count == 0:
            raise ValueError(f"stratum {name} has no finite paired effects")
        query_sum = np.where(finite, delta, 0.0).sum(axis=1)
        query_count = finite.sum(axis=1)
        query_used = query_count > 0
        query_mean = query_sum[query_used] / query_count[query_used]
        generator = np.random.default_rng(seed + offset)
        samples = np.empty(resamples, dtype=np.float64)
        for sample_index in range(resamples):
            selected = generator.integers(0, delta.shape[0], size=delta.shape[0])
            denominator = int(query_count[selected].sum())
            samples[sample_index] = query_sum[selected].sum() / denominator
        result[name] = {
            "anchor_count": anchor_count,
            "query_count": int(query_used.sum()),
            "anchor_weighted_mean": float(delta[finite].mean()),
            "query_left_worse_fraction": float((query_mean > 0).mean()),
            "query_block_bootstrap_ci95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
            "resamples": int(resamples),
            "seed": int(seed + offset),
        }
    return result


def topk_jaccard_aligned(
    left_ids: np.ndarray,
    left_valid: np.ndarray,
    right_ids: np.ndarray,
    right_valid: np.ndarray,
) -> np.ndarray:
    """Return set Jaccard while preserving each ``[Q,N]`` anchor."""
    left = np.asarray(left_ids)
    right = np.asarray(right_ids)
    left_mask = np.asarray(left_valid, dtype=bool)
    right_mask = np.asarray(right_valid, dtype=bool)
    if left.ndim != 3 or right.shape != left.shape:
        raise ValueError("left_ids and right_ids must be aligned [Q,N,K]")
    if left_mask.shape != left.shape or right_mask.shape != right.shape:
        raise ValueError("Top-K validity masks must match their ids")
    output = np.full(left.shape[:2], np.nan, dtype=np.float64)
    for query_index in range(left.shape[0]):
        for node_index in range(left.shape[1]):
            left_set = set(left[query_index, node_index][left_mask[query_index, node_index]].tolist())
            right_set = set(right[query_index, node_index][right_mask[query_index, node_index]].tolist())
            union = left_set | right_set
            if union:
                output[query_index, node_index] = len(left_set & right_set) / len(union)
    return output


@dataclass(frozen=True)
class _KeySystemOutput:
    key_name: str
    checkpoint: str
    checkpoint_epoch: int | None
    bank: str
    encoder_fingerprint: str
    sample_id: np.ndarray
    context_end: np.ndarray
    event_ids: np.ndarray
    event_valid: np.ndarray
    selected_event_ids: np.ndarray
    selected_valid: np.ndarray
    surprise: torch.Tensor
    surprise_valid: torch.Tensor
    predictions: dict[str, torch.Tensor]
    aggregation_valid: dict[str, torch.Tensor]
    target: torch.Tensor
    target_observed: torch.Tensor
    elapsed_seconds: float
    peak_cuda_memory_gb: float


@torch.inference_mode()
def _run_key_system(
    *,
    key_name: str,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    data: Any,
    graph_cpu: Any,
    query_rows: np.ndarray,
    candidate_protocol: str,
    batch_size: int,
    device: torch.device,
) -> _KeySystemOutput:
    from stanchor.diagnostics.retrieval_visualization import build_diagnostic_event_candidates
    from stanchor.engine.target import _validate_bank

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

    sample_ids: list[np.ndarray] = []
    context_ends: list[np.ndarray] = []
    event_ids_output: list[np.ndarray] = []
    event_valid_output: list[np.ndarray] = []
    selected_ids_output: list[np.ndarray] = []
    selected_valid_output: list[np.ndarray] = []
    surprises: list[torch.Tensor] = []
    surprise_validity: list[torch.Tensor] = []
    predictions: dict[str, list[torch.Tensor]] = {
        "payload_offset_only": [],
        "payload_offset_decay": [],
    }
    aggregation_masks: dict[str, list[torch.Tensor]] = {
        "payload_offset_only": [],
        "payload_offset_decay": [],
    }
    targets: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []

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
            encoding = model.encode_clean(
                query_batch["retrieval_x"].to(device),
                query_batch["retrieval_observed"].to(device).bool(),
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
            node_candidates = retriever.rerank_nodes(
                encoding.retrieval.node_keys,
                encoding.statistics.level_features,
                events,
            )
            aggregations = {
                "payload_offset_only": offset_only_aggregation(
                    node_candidates,
                    query_batch["x"].to(device),
                    query_batch["x_observed"].to(device),
                    bank,
                    data.series,
                    data.scaler,
                    config.data.context_length,
                    device,
                ),
                "payload_offset_decay": offset_decay_aggregation(
                    node_candidates,
                    query_batch["x"].to(device),
                    query_batch["x_observed"].to(device),
                    bank,
                    data.series,
                    data.scaler,
                    config.data.context_length,
                    device,
                ),
            }
            if not torch.equal(
                aggregations["payload_offset_only"].valid,
                aggregations["payload_offset_decay"].valid,
            ):
                raise ValueError("payload swap changed aggregation validity within one Key row")

            context_physical = data.scaler.inverse_transform_torch(
                query_batch["x"].to(device)
            )
            target_physical = data.scaler.inverse_transform_torch(
                query_batch["y"].to(device)
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
            selected_ids_output.append(node_candidates.event_ids.detach().cpu().numpy())
            selected_valid_output.append(node_candidates.valid.detach().cpu().numpy())
            surprises.append(surprise.cpu())
            surprise_validity.append(surprise_valid.cpu())
            targets.append(target_physical.cpu())
            target_masks.append(query_batch["y_observed"].bool().cpu())
            for payload_name, aggregation in aggregations.items():
                predictions[payload_name].append(
                    data.scaler.inverse_transform_torch(aggregation.prediction).cpu()
                )
                aggregation_masks[payload_name].append(aggregation.valid.cpu())

            processed += int(rows.size)
            if processed == query_rows.size or processed % 128 == 0:
                print(
                    f"[key-payload-factorial:{key_name}] processed {processed}/{query_rows.size} queries",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = float(torch.cuda.max_memory_allocated(device) / 1024**3)
        else:
            peak_memory = 0.0

    output = _KeySystemOutput(
        key_name=key_name,
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
        selected_event_ids=np.concatenate(selected_ids_output),
        selected_valid=np.concatenate(selected_valid_output),
        surprise=torch.cat(surprises),
        surprise_valid=torch.cat(surprise_validity),
        predictions={name: torch.cat(chunks) for name, chunks in predictions.items()},
        aggregation_valid={
            name: torch.cat(chunks) for name, chunks in aggregation_masks.items()
        },
        target=torch.cat(targets),
        target_observed=torch.cat(target_masks),
        elapsed_seconds=float(time.perf_counter() - started),
        peak_cuda_memory_gb=peak_memory,
    )
    del model, graph
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _save_factorial_plot(
    path: Path,
    cell_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    strata = ("all", "ordinary_80", "surprise_top20", "surprise_top5")
    labels = ("All", "Ordinary 80%", "Surprise top 20%", "Surprise top 5%")
    styles = (
        (KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY, "K_OO x P_OO", "#2878B5"),
        (KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY, "K_OO x P_OD", "#69A9D4"),
        (KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY, "K_OD x P_OO", "#E69F67"),
        (KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY, "K_OD x P_OD", "#D95319"),
    )
    x = np.arange(len(strata), dtype=np.float64)
    width = 0.19
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    for index, (cell, label, color) in enumerate(styles):
        values = [float(cell_metrics[cell][name]["mae"]) for name in strata]
        positions = x + (index - 1.5) * width
        bars = axis.bar(positions, values, width=width, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Memory MAE (physical units)")
    axis.set_title("2-Key x 2-Payload frozen factorial attribution")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def run_key_payload_factorial(
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
    bootstrap_resamples: int = 5000,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Run the complete frozen Key-by-payload factorial on shared queries."""
    if max_queries <= 0 or batch_size <= 0 or bootstrap_resamples <= 0:
        raise ValueError("query, batch, and bootstrap counts must be positive")
    _assert_matched_configs(offset_only_config, offset_decay_config)
    from stanchor.diagnostics.retrieval_collapse import weekly_donor_pairs
    from stanchor.diagnostics.retrieval_visualization import future_information_boundary

    started = time.perf_counter()
    only_bank_path = resolve_project_path(offset_only_bank)
    decay_bank_path = resolve_project_path(offset_decay_bank)
    non_key_audit = audit_non_key_bank_files(only_bank_path, decay_bank_path)
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

    only_key = _run_key_system(
        key_name="key_offset_only",
        config=offset_only_config,
        checkpoint_path=offset_only_checkpoint,
        bank_path=only_bank_path,
        data=data,
        graph_cpu=graph_cpu,
        query_rows=query_rows,
        candidate_protocol=candidate_protocol,
        batch_size=batch_size,
        device=device,
    )
    decay_key = _run_key_system(
        key_name="key_offset_decay",
        config=offset_decay_config,
        checkpoint_path=offset_decay_checkpoint,
        bank_path=decay_bank_path,
        data=data,
        graph_cpu=graph_cpu,
        query_rows=query_rows,
        candidate_protocol=candidate_protocol,
        batch_size=batch_size,
        device=device,
    )

    if not np.array_equal(only_key.sample_id, decay_key.sample_id):
        raise ValueError("Key systems did not evaluate identical sample ids")
    if not np.array_equal(only_key.context_end, decay_key.context_end):
        raise ValueError("Key systems did not evaluate identical context ends")
    if not np.array_equal(only_key.event_ids, decay_key.event_ids):
        raise ValueError("Key systems did not use identical event candidate ids")
    if not np.array_equal(only_key.event_valid, decay_key.event_valid):
        raise ValueError("Key systems did not use identical event candidate validity")
    if not torch.equal(only_key.target_observed, decay_key.target_observed):
        raise ValueError("Key systems have different target masks")
    if not torch.allclose(only_key.target, decay_key.target, equal_nan=True):
        raise ValueError("Key systems have different physical targets")
    if not torch.equal(only_key.surprise_valid, decay_key.surprise_valid):
        raise ValueError("Key systems have different surprise validity")
    if not torch.allclose(only_key.surprise, decay_key.surprise, equal_nan=True):
        raise ValueError("Key systems have different surprise values")

    cell_predictions = {
        KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY: only_key.predictions["payload_offset_only"],
        KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY: only_key.predictions["payload_offset_decay"],
        KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY: decay_key.predictions["payload_offset_only"],
        KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY: decay_key.predictions["payload_offset_decay"],
    }
    cell_validity = {
        KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY: only_key.aggregation_valid["payload_offset_only"],
        KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY: only_key.aggregation_valid["payload_offset_decay"],
        KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY: decay_key.aggregation_valid["payload_offset_only"],
        KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY: decay_key.aggregation_valid["payload_offset_decay"],
    }
    shared_valid = only_key.target_observed.clone()
    for cell_valid in cell_validity.values():
        shared_valid &= cell_valid
    shared_anchor_valid = only_key.surprise_valid & shared_valid.any(dim=(1, 3))
    thresholds = surprise_thresholds(only_key.surprise, shared_anchor_valid)
    strata = surprise_strata_masks(only_key.surprise, shared_anchor_valid, thresholds)

    cell_metrics: dict[str, dict[str, Any]] = {}
    anchor_mae: dict[str, np.ndarray] = {}
    for cell, prediction in cell_predictions.items():
        cell_metrics[cell] = stratified_forecast_metrics(
            prediction,
            only_key.target,
            shared_valid,
            strata,
        )
        anchor_mae[cell] = _per_anchor_mae(prediction, only_key.target, shared_valid)

    effects = factorial_mae_effects(cell_metrics)
    contrast_arrays = {
        "payload_effect_under_offset_only_key": (
            anchor_mae[KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY]
            - anchor_mae[KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY]
        ),
        "payload_effect_under_offset_decay_key": (
            anchor_mae[KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY]
            - anchor_mae[KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY]
        ),
        "key_effect_under_offset_only_payload": (
            anchor_mae[KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY]
            - anchor_mae[KEY_OFFSET_DECAY_PAYLOAD_OFFSET_ONLY]
        ),
        "key_effect_under_offset_decay_payload": (
            anchor_mae[KEY_OFFSET_ONLY_PAYLOAD_OFFSET_DECAY]
            - anchor_mae[KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY]
        ),
    }
    contrast_arrays["interaction_difference_in_differences"] = (
        contrast_arrays["payload_effect_under_offset_only_key"]
        - contrast_arrays["payload_effect_under_offset_decay_key"]
    )
    contrast_arrays["average_payload_main_effect"] = 0.5 * (
        contrast_arrays["payload_effect_under_offset_only_key"]
        + contrast_arrays["payload_effect_under_offset_decay_key"]
    )
    contrast_arrays["average_key_main_effect"] = 0.5 * (
        contrast_arrays["key_effect_under_offset_only_payload"]
        + contrast_arrays["key_effect_under_offset_decay_payload"]
    )
    contrast_arrays["matched_diagonal_total_effect"] = (
        anchor_mae[KEY_OFFSET_ONLY_PAYLOAD_OFFSET_ONLY]
        - anchor_mae[KEY_OFFSET_DECAY_PAYLOAD_OFFSET_DECAY]
    )
    bootstrap = {
        name: query_block_bootstrap_contrast(
            values,
            strata,
            resamples=bootstrap_resamples,
            seed=offset_only_config.runtime.seed + 100 * index,
        )
        for index, (name, values) in enumerate(contrast_arrays.items())
    }

    topk_jaccard = topk_jaccard_aligned(
        only_key.selected_event_ids,
        only_key.selected_valid,
        decay_key.selected_event_ids,
        decay_key.selected_valid,
    )
    valid_surprise = only_key.surprise[shared_anchor_valid].numpy()
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "factorial_anchor_values.npz",
        sample_id=only_key.sample_id,
        context_end=only_key.context_end,
        surprise=only_key.surprise.numpy(),
        surprise_valid=shared_anchor_valid.numpy(),
        topk_jaccard=topk_jaccard,
        **{f"mae__{name}": values for name, values in anchor_mae.items()},
        **{f"contrast__{name}": values for name, values in contrast_arrays.items()},
    )

    result: dict[str, Any] = {
        "diagnostic": "key_payload_2x2_factorial",
        "query_count": int(only_key.sample_id.size),
        "node_count": int(only_key.surprise.shape[1]),
        "candidate_protocol": candidate_protocol,
        "factor_definitions": {
            "key": "checkpoint encoder plus fingerprint-matched query and Bank keys; determines Top-K and weights",
            "payload_offset_only": "historical_future + (query_endpoint_level - candidate_endpoint_level) at every horizon",
            "payload_offset_decay": "historical_future + lambda_h * (query_endpoint_level - candidate_endpoint_level), lambda linearly decays from 1 to 0",
            "payload_effect_sign": "positive means OffsetDecay payload reduces MAE",
            "key_effect_sign": "positive means OffsetDecay Key system reduces MAE",
        },
        "surprise": {
            "thresholds": thresholds,
            "count": int(valid_surprise.size),
            "mean": float(valid_surprise.mean()),
            "median": float(np.median(valid_surprise)),
            "q80": thresholds["q80"],
            "q95": thresholds["q95"],
        },
        "strata": {
            name: {"anchor_count": int(mask.sum().item())}
            for name, mask in strata.items()
        },
        "matching_audit": {
            "same_data_model_seed_and_retrieval_hyperparameters": True,
            "checkpoint_bank_fingerprints_validated": True,
            "non_key_bank_files": non_key_audit,
            "same_query_ids": True,
            "same_context_ends": True,
            "same_event_candidate_pool": True,
            "same_target_and_surprise": True,
            "same_validity_within_each_key_row": True,
            "sample_id_sha256": array_sha256(only_key.sample_id),
            "event_candidate_id_sha256": array_sha256(only_key.event_ids),
        },
        "key_systems": {
            "key_offset_only": {
                "checkpoint": {"path": only_key.checkpoint, "epoch": only_key.checkpoint_epoch},
                "bank": {"path": only_key.bank, "encoder_fingerprint": only_key.encoder_fingerprint},
            },
            "key_offset_decay": {
                "checkpoint": {"path": decay_key.checkpoint, "epoch": decay_key.checkpoint_epoch},
                "bank": {"path": decay_key.bank, "encoder_fingerprint": decay_key.encoder_fingerprint},
            },
        },
        "topk_selection_overlap": {
            "jaccard": stratified_scalar_summary(topk_jaccard, strata),
            "definition": "intersection size divided by union size for the two Key systems' Top-K event sets",
        },
        "cells": cell_metrics,
        "factorial_mae_effects": effects,
        "paired_query_block_bootstrap": bootstrap,
        "future_information_boundary": future_information_boundary(),
        "runtime": {
            "device": str(device),
            "batch_size": int(batch_size),
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_cuda_memory_gb": max(
                only_key.peak_cuda_memory_gb,
                decay_key.peak_cuda_memory_gb,
            ),
            "per_key_seconds": {
                "key_offset_only": only_key.elapsed_seconds,
                "key_offset_decay": decay_key.elapsed_seconds,
            },
        },
        "artifacts": {
            "anchor_values": "factorial_anchor_values.npz",
            "comparison_plot": "key_payload_factorial_memory_mae.png",
        },
    }
    _save_factorial_plot(
        output_path / "key_payload_factorial_memory_mae.png",
        cell_metrics,
    )
    save_json(output_path / "key_payload_factorial.json", result)
    return result

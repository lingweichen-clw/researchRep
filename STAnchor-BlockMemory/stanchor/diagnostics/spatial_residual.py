"""Spatial diagnostics for frozen-backbone residuals and candidate utility.

The query future is used only as an offline validation label. It never enters
the backbone, retrieval encoder, candidate selection, or deployable features.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import (
    _validate_bank,
    build_downstream_model,
    load_frozen_base_backbone,
    retrieve_for_downstream_mode,
)
from stanchor.models.downstream import BASE_ONLY, LEARNED_TOPK_ERROR_AWARE
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.utils import resolve_device


def target_degree_matched_nonedges(
    edge_index: np.ndarray,
    num_nodes: int,
    seed: int,
) -> np.ndarray:
    """Sample one non-edge per physical edge while preserving target degrees."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must be [2,E]")
    adjacency = np.zeros((num_nodes, num_nodes), dtype=bool)
    adjacency[edges[0], edges[1]] = True
    np.fill_diagonal(adjacency, True)
    rng = np.random.default_rng(seed)
    result: list[tuple[int, int]] = []
    for target in range(num_nodes):
        degree = int(np.sum((edges[0] == target) & (edges[1] != target)))
        if degree == 0:
            continue
        legal = np.flatnonzero(~adjacency[target])
        if legal.size < degree:
            raise ValueError(
                f"node {target} has {degree} edges but only {legal.size} non-edges"
            )
        for source in rng.choice(legal, size=degree, replace=False):
            result.append((target, int(source)))
    return np.asarray(result, dtype=np.int64).T


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "positive_fraction": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "positive_fraction": float((finite > 0.0).mean()),
    }


def _masked_column_pearson(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return one temporal Pearson coefficient per pair/column."""
    valid_float = valid.astype(np.float64, copy=False)
    count = valid_float.sum(axis=0)
    safe_count = np.maximum(count, 1.0)
    left_safe = np.where(valid, left, 0.0).astype(np.float64, copy=False)
    right_safe = np.where(valid, right, 0.0).astype(np.float64, copy=False)
    left_mean = left_safe.sum(axis=0) / safe_count
    right_mean = right_safe.sum(axis=0) / safe_count
    left_centered = np.where(valid, left_safe - left_mean, 0.0)
    right_centered = np.where(valid, right_safe - right_mean, 0.0)
    numerator = (left_centered * right_centered).sum(axis=0)
    denominator = np.sqrt(
        (left_centered.square() if hasattr(left_centered, "square") else left_centered**2).sum(axis=0)
        * (right_centered.square() if hasattr(right_centered, "square") else right_centered**2).sum(axis=0)
    )
    result = np.full(count.shape, np.nan, dtype=np.float64)
    usable = (count >= 3) & (denominator > 1.0e-12)
    result[usable] = numerator[usable] / denominator[usable]
    return result


def _rank_nodes(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ranked = np.full(values.shape, np.nan, dtype=np.float64)
    for node in range(values.shape[1]):
        node_valid = valid[:, node] & np.isfinite(values[:, node])
        if int(node_valid.sum()) >= 3:
            ranked[node_valid, node] = rankdata(values[node_valid, node])
    return ranked


def _pair_correlations(
    values: np.ndarray,
    valid: np.ndarray,
    pairs: np.ndarray,
    ranked: bool = False,
) -> np.ndarray:
    target, source = pairs
    pair_valid = valid[:, target] & valid[:, source]
    if ranked:
        values = _rank_nodes(values, valid)
        pair_valid &= np.isfinite(values[:, target]) & np.isfinite(values[:, source])
    return _masked_column_pearson(
        values[:, target], values[:, source], pair_valid
    )


def _center_by_sample(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid, values, 0.0)
    count = valid.sum(axis=1, keepdims=True).clip(min=1)
    mean = safe.sum(axis=1, keepdims=True) / count
    return np.where(valid, values - mean, np.nan)


def _moran_by_sample(
    centered: np.ndarray,
    valid: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    target, source = pairs
    pair_valid = valid[:, target] & valid[:, source]
    left = np.where(pair_valid, centered[:, target], 0.0)
    right = np.where(pair_valid, centered[:, source], 0.0)
    numerator = (left * right).sum(axis=1)
    pair_count = pair_valid.sum(axis=1)
    node_count = valid.sum(axis=1)
    denominator = np.where(valid, centered, 0.0)
    denominator = (denominator**2).sum(axis=1)
    result = np.full(values_shape := numerator.shape, np.nan, dtype=np.float64)
    usable = (pair_count > 0) & (node_count > 1) & (denominator > 1.0e-12)
    result[usable] = (
        node_count[usable]
        / pair_count[usable]
        * numerator[usable]
        / denominator[usable]
    )
    return result.reshape(values_shape)


def _correlation_block(
    values: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
    nonedges: np.ndarray,
) -> dict[str, Any]:
    centered = _center_by_sample(values, valid)
    blocks = {}
    for name, pair_index in (("edge", edges), ("matched_nonedge", nonedges)):
        blocks[name] = {
            "pearson": _finite_summary(
                _pair_correlations(values, valid, pair_index)
            ),
            "spearman": _finite_summary(
                _pair_correlations(values, valid, pair_index, ranked=True)
            ),
            "centered_pearson": _finite_summary(
                _pair_correlations(centered, valid, pair_index)
            ),
            "moran_i": _finite_summary(
                _moran_by_sample(centered, valid, pair_index)
            ),
        }
    for metric in ("pearson", "spearman", "centered_pearson", "moran_i"):
        edge_mean = blocks["edge"][metric]["mean"]
        nonedge_mean = blocks["matched_nonedge"][metric]["mean"]
        blocks[f"{metric}_mean_excess"] = (
            None
            if edge_mean is None or nonedge_mean is None
            else float(edge_mean - nonedge_mean)
        )
    return blocks


def _binary_pair_metrics(
    helpful: np.ndarray,
    valid: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, Any]:
    target, source = pairs
    pair_valid = valid[:, target] & valid[:, source]
    left = helpful[:, target]
    right = helpful[:, source]
    count = int(pair_valid.sum())
    if count == 0:
        return {
            "pair_count": 0,
            "agreement": None,
            "joint_helpful": None,
            "conditional_helpful": None,
            "conditional_lift": None,
            "phi": _finite_summary(np.asarray([], dtype=np.float64)),
        }
    agreement = float(((left == right) & pair_valid).sum() / count)
    joint = float((left & right & pair_valid).sum() / count)
    source_positive = int((right & pair_valid).sum())
    conditional = (
        float((left & right & pair_valid).sum() / source_positive)
        if source_positive
        else None
    )
    target_prevalence = float((left & pair_valid).sum() / count)
    lift = (
        conditional / target_prevalence
        if conditional is not None and target_prevalence > 0.0
        else None
    )
    return {
        "pair_count": count,
        "agreement": agreement,
        "joint_helpful": joint,
        "conditional_helpful": conditional,
        "conditional_lift": lift,
        "phi": _finite_summary(
            _pair_correlations(
                helpful.astype(np.float32), valid, pairs
            )
        ),
    }


def spatial_residual_metrics(
    residuals: np.ndarray,
    residual_valid: np.ndarray,
    helpful: np.ndarray,
    helpful_valid: np.ndarray,
    edge_index: np.ndarray,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute physical-edge versus degree-matched non-edge diagnostics.

    Arrays are ``[samples,horizon,nodes]``. Residuals should be in normalized
    model space so high-variance sensors do not dominate cross-node statistics.
    """
    if residuals.ndim != 3 or residual_valid.shape != residuals.shape:
        raise ValueError("residual arrays must be aligned [S,H,N]")
    if helpful.shape != residuals.shape or helpful_valid.shape != residuals.shape:
        raise ValueError("helpfulness arrays must align with residuals")
    num_nodes = residuals.shape[-1]
    edges = np.asarray(edge_index, dtype=np.int64)
    keep = edges[0] != edges[1]
    edges = np.unique(edges[:, keep], axis=1)
    nonedges = target_degree_matched_nonedges(edges, num_nodes, seed)

    horizon_results = []
    for horizon in range(residuals.shape[1]):
        residual_block = _correlation_block(
            residuals[:, horizon], residual_valid[:, horizon], edges, nonedges
        )
        helpful_edge = _binary_pair_metrics(
            helpful[:, horizon], helpful_valid[:, horizon], edges
        )
        helpful_nonedge = _binary_pair_metrics(
            helpful[:, horizon], helpful_valid[:, horizon], nonedges
        )
        horizon_results.append(
            {
                "horizon_index": horizon + 1,
                "base_residual": residual_block,
                "candidate_helpfulness": {
                    "prevalence": float(
                        helpful[:, horizon][helpful_valid[:, horizon]].mean()
                    ),
                    "edge": helpful_edge,
                    "matched_nonedge": helpful_nonedge,
                    "agreement_excess": float(
                        helpful_edge["agreement"] - helpful_nonedge["agreement"]
                    ),
                    "phi_mean_excess": float(
                        helpful_edge["phi"]["mean"]
                        - helpful_nonedge["phi"]["mean"]
                    ),
                },
            }
        )

    flat_residuals = residuals.reshape(-1, num_nodes)
    flat_residual_valid = residual_valid.reshape(-1, num_nodes)
    flat_helpful = helpful.reshape(-1, num_nodes)
    flat_helpful_valid = helpful_valid.reshape(-1, num_nodes)
    overall_edge_help = _binary_pair_metrics(
        flat_helpful, flat_helpful_valid, edges
    )
    overall_nonedge_help = _binary_pair_metrics(
        flat_helpful, flat_helpful_valid, nonedges
    )
    return {
        "num_samples": int(residuals.shape[0]),
        "num_horizons": int(residuals.shape[1]),
        "num_nodes": int(num_nodes),
        "physical_directed_edges": int(edges.shape[1]),
        "matched_directed_nonedges": int(nonedges.shape[1]),
        "overall": {
            "base_residual": _correlation_block(
                flat_residuals,
                flat_residual_valid,
                edges,
                nonedges,
            ),
            "candidate_helpfulness": {
                "prevalence": float(flat_helpful[flat_helpful_valid].mean()),
                "edge": overall_edge_help,
                "matched_nonedge": overall_nonedge_help,
                "agreement_excess": float(
                    overall_edge_help["agreement"]
                    - overall_nonedge_help["agreement"]
                ),
                "phi_mean_excess": float(
                    overall_edge_help["phi"]["mean"]
                    - overall_nonedge_help["phi"]["mean"]
                ),
            },
        },
        "horizons": horizon_results,
    }


@torch.no_grad()
def diagnose_spatial_residuals(
    config: ExperimentConfig,
    pretrained_checkpoint: str | Path,
    base_checkpoint: str | Path,
    bank_path: str | Path,
    split: str = "val",
    batch_size: int | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Collect frozen-base residuals and candidate utility, then diagnose them."""
    if split not in {"val", "test"}:
        raise ValueError("spatial residual diagnosis is restricted to val or test")
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    pretrained, pretrained_state = load_pretrained_model(
        config,
        pretrained_checkpoint,
        data.series.slots_per_day,
        device,
    )
    pretrained.eval()

    base_target = replace(config.target, downstream_mode=BASE_ONLY)
    base_config = replace(config, target=base_target)
    downstream = build_downstream_model(base_config, graph).to(device)
    provenance = load_frozen_base_backbone(downstream, base_checkpoint, device)
    downstream.backbone.eval()

    loader = DataLoader(
        getattr(data, split),
        batch_size=batch_size or config.target.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    residuals = []
    residual_valid = []
    helpful = []
    helpful_valid = []
    processed_batches = 0

    with MemoryBank(
        bank_path,
        expected_schema_version=(2 if pretrained.model_config.profile_dim > 0 else 1),
    ) as bank:
        _validate_bank(bank, pretrained, graph_cpu, data.scaler.state_dict())
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
            target = batch["y"].to(device)
            observed = batch["y_observed"].to(device).bool()
            base = downstream.backbone(x)
            candidates, aggregation = retrieve_for_downstream_mode(
                LEARNED_TOPK_ERROR_AWARE,
                pretrained,
                retriever,
                bank,
                data,
                graph,
                batch,
                x,
                observed_x,
                device,
                candidate_protocol=config.target.candidate_protocol,
            )
            if candidates is None or aggregation is None:
                raise RuntimeError("retrieval diagnosis produced no candidates")

            base_valid = observed & torch.isfinite(base) & torch.isfinite(target)
            base_residual = torch.where(base_valid, base - target, 0.0).mean(dim=-1)
            node_base_valid = base_valid.any(dim=-1)

            candidate_valid = (
                aggregation.candidate_masks.bool()
                & observed.unsqueeze(3)
                & torch.isfinite(aggregation.candidate_futures)
                & torch.isfinite(target).unsqueeze(3)
            )
            candidate_count = candidate_valid.sum(dim=-1)
            candidate_error = torch.where(
                candidate_valid,
                (aggregation.candidate_futures - target.unsqueeze(3)).abs(),
                torch.zeros_like(aggregation.candidate_futures),
            ).sum(dim=-1) / candidate_count.clamp_min(1)
            candidate_error = candidate_error.masked_fill(candidate_count == 0, torch.inf)
            best_candidate_error = candidate_error.amin(dim=3)
            base_error = torch.where(
                base_valid, (base - target).abs(), torch.zeros_like(base)
            ).sum(dim=-1) / base_valid.sum(dim=-1).clamp_min(1)
            node_candidate_valid = torch.isfinite(best_candidate_error)
            node_helpful_valid = node_base_valid & node_candidate_valid
            node_helpful = (best_candidate_error < base_error) & node_helpful_valid

            residuals.append(base_residual.cpu().numpy())
            residual_valid.append(node_base_valid.cpu().numpy())
            helpful.append(node_helpful.cpu().numpy())
            helpful_valid.append(node_helpful_valid.cpu().numpy())
            processed_batches += 1

    if not residuals:
        raise ValueError("spatial residual diagnosis processed no samples")
    metrics = spatial_residual_metrics(
        np.concatenate(residuals, axis=0),
        np.concatenate(residual_valid, axis=0),
        np.concatenate(helpful, axis=0),
        np.concatenate(helpful_valid, axis=0),
        graph_cpu.edge_index.cpu().numpy(),
        seed=config.runtime.seed,
    )
    return {
        "schema_version": 1,
        "diagnostic": "spatial_base_residual_and_candidate_helpfulness",
        "split": split,
        "processed_batches": processed_batches,
        "candidate_protocol": config.target.candidate_protocol,
        "node_top_k": config.bank.node_top_k,
        "residual_space": "normalized model space",
        "nonedge_control": "target-degree-matched directed non-edges",
        "future_information_boundary": (
            "query future is used only for offline residual/helpfulness labels; "
            "it is not used by the backbone, encoder, retrieval, or candidate selection"
        ),
        "pretrained_checkpoint": str(Path(pretrained_checkpoint).resolve()),
        "pretrained_fingerprint": pretrained_state.get("model_fingerprint"),
        "base_checkpoint": provenance,
        "bank_path": str(Path(bank_path).resolve()),
        "metrics": metrics,
    }

"""Offline diagnostics for canonical future dynamics profile semantics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.stats import spearmanr

from stanchor.config import ExperimentConfig
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.pretrainer import build_validation_loader
from stanchor.losses.pretraining import build_future_relation_targets
from stanchor.retrieval.semantic_profile import build_cfdp_teacher
from stanchor.utils import resolve_device


def _pairwise_profile_mae(
    profile: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_valid = valid[:, None] & valid[None, :]
    difference = (profile[:, None] - profile[None, :]).abs()
    count = pair_valid.sum(dim=-1)
    distance = torch.where(pair_valid, difference, torch.zeros_like(difference)).sum(dim=-1)
    return distance / count.clamp_min(1), count > 0


def _relation_metrics(
    key_distance: torch.Tensor,
    teacher_distance: torch.Tensor,
    candidate_mask: torch.Tensor,
    top_k: int,
) -> tuple[float, float, int, int]:
    valid = candidate_mask.bool() & torch.isfinite(key_distance) & torch.isfinite(teacher_distance)
    key_values = key_distance.masked_select(valid).detach().double().cpu().numpy()
    teacher_values = teacher_distance.masked_select(valid).detach().double().cpu().numpy()
    if key_values.size < 2 or np.ptp(key_values) == 0.0 or np.ptp(teacher_values) == 0.0:
        spearman = 0.0
    else:
        spearman = float(spearmanr(key_values, teacher_values).statistic)
    recalls = []
    batch, _, nodes = candidate_mask.shape
    for anchor in range(batch):
        for node in range(nodes):
            candidates = torch.where(valid[anchor, :, node])[0]
            if candidates.numel() < top_k:
                continue
            key_local = key_distance[anchor, candidates, node]
            teacher_local = teacher_distance[anchor, candidates, node]
            key_ids = candidates[torch.topk(key_local, top_k, largest=False).indices]
            teacher_ids = candidates[torch.topk(teacher_local, top_k, largest=False).indices]
            intersection = torch.isin(key_ids, teacher_ids).sum()
            recalls.append(float(intersection) / top_k)
    recall = float(np.mean(recalls)) if recalls else 0.0
    return spearman, recall, int(valid.sum().item()), len(recalls)


def cfdp_batch_metrics(
    profile_prediction: torch.Tensor | None,
    profile_teacher: torch.Tensor,
    profile_valid: torch.Tensor,
    profile_keys: torch.Tensor | None,
    total_keys: torch.Tensor,
    od_distance: torch.Tensor,
    candidate_mask: torch.Tensor,
    top_k: int = 5,
) -> dict[str, float | int]:
    """Measure profile prediction and relation alignment for one batch.

    Profile tensors use ``[B,N,Kp]`` and pair tensors use ``[B,B,N]``.
    Query future is used only by this offline diagnostic.
    """
    if profile_prediction is None or profile_keys is None:
        raise ValueError("CFDP diagnostic requires profile prediction and profile keys")
    if profile_prediction.shape != profile_teacher.shape or profile_valid.shape != profile_teacher.shape:
        raise ValueError("profile tensors must share [B,N,Kp]")
    if total_keys.ndim != 3 or profile_keys.ndim != 3:
        raise ValueError("profile and total keys must be [B,N,D]")
    batch, nodes, _ = profile_teacher.shape
    if total_keys.shape[:2] != (batch, nodes) or profile_keys.shape[:2] != (batch, nodes):
        raise ValueError("key batch/node dimensions must align with profile")
    if od_distance.shape != (batch, batch, nodes) or candidate_mask.shape != od_distance.shape:
        raise ValueError("OD relation tensors must be [B,B,N]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    valid = profile_valid.bool()
    point_count = int(valid.sum().item())
    if point_count == 0:
        raise ValueError("CFDP diagnostic has no valid profile points")
    profile_mae = float(
        (profile_prediction - profile_teacher).abs().masked_select(valid).mean()
    )
    masked_prediction = torch.where(valid, profile_prediction, torch.zeros_like(profile_prediction))
    masked_teacher = torch.where(valid, profile_teacher, torch.zeros_like(profile_teacher))
    vector_valid = valid.any(dim=-1) & (
        masked_prediction.norm(dim=-1) > 1.0e-8
    ) & (
        masked_teacher.norm(dim=-1) > 1.0e-8
    )
    cosine = functional.cosine_similarity(masked_prediction, masked_teacher, dim=-1, eps=1.0e-8)
    cosine_count = int(vector_valid.sum().item())
    profile_cosine = float(cosine.masked_select(vector_valid).mean()) if cosine_count else 0.0

    profile_distance, profile_pair_valid = _pairwise_profile_mae(profile_teacher, valid)
    relation_mask = candidate_mask.bool() & profile_pair_valid
    normalized_profile_keys = functional.normalize(profile_keys, dim=-1)
    normalized_total_keys = functional.normalize(total_keys, dim=-1)
    profile_key_distance = 1.0 - torch.einsum(
        "ind,jnd->ijn", normalized_profile_keys, normalized_profile_keys
    )
    total_key_distance = 1.0 - torch.einsum(
        "ind,jnd->ijn", normalized_total_keys, normalized_total_keys
    )
    profile_spearman, profile_recall, relation_pairs, relation_anchors = _relation_metrics(
        profile_key_distance, profile_distance, relation_mask, top_k
    )
    total_spearman, total_recall, od_pairs, od_anchors = _relation_metrics(
        total_key_distance, od_distance, candidate_mask, top_k
    )
    return {
        "profile_mae": profile_mae,
        "profile_cosine": profile_cosine,
        "profile_relation_spearman": profile_spearman,
        "profile_relation_recall_at_k": profile_recall,
        "total_od_relation_spearman": total_spearman,
        "total_od_relation_recall_at_k": total_recall,
        "profile_points": point_count,
        "profile_vectors": cosine_count,
        "profile_relation_pairs": relation_pairs,
        "profile_relation_anchors": relation_anchors,
        "od_relation_pairs": od_pairs,
        "od_relation_anchors": od_anchors,
        "top_k": top_k,
    }


@torch.no_grad()
def diagnose_cfdp_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    split: str = "val",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run future-using CFDP diagnostics on an evaluation split only."""
    if split != "val":
        raise ValueError("CFDP development diagnostic is restricted to val")
    if config.model.profile_dim <= 0:
        raise ValueError("CFDP diagnostic requires a profile-enabled config")
    started = time.perf_counter()
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    model, checkpoint = load_pretrained_model(
        config, checkpoint_path, data.series.slots_per_day, device
    )
    model.eval()
    loader = build_validation_loader(
        data.val,
        config.pretrain.batch_size,
        config.data.num_workers,
        config.runtime.seed,
    )
    records = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        encoding = model.encode_clean(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            batch["retrieval_weekday"].to(device),
            batch["retrieval_slot"].to(device),
            graph,
        )
        teacher, teacher_valid = build_cfdp_teacher(
            batch["y"].to(device),
            batch["y_observed"].to(device),
            batch["x"].to(device),
            batch["x_observed"].to(device),
            profile_size=config.model.profile_dim,
            scale_floor=config.pretrain.profile_scale_floor,
        )
        teacher = teacher.squeeze(-1).permute(0, 2, 1).contiguous()
        teacher_valid = teacher_valid.squeeze(-1).permute(0, 2, 1).contiguous()
        od_targets = build_future_relation_targets(
            future_model=batch["y"].to(device),
            context_statistics=encoding.statistics,
            future_observed=batch["y_observed"].to(device),
            context_start=batch["context_start"].to(device),
            future_end=batch["future_end"].to(device),
            teacher_temperature=config.pretrain.relation_teacher_temperature,
            relation_teacher_mode="offset_decay",
            forecast_context=batch["x"].to(device),
            forecast_context_observed=batch["x_observed"].to(device),
            relation_distance_normalization=config.pretrain.relation_distance_normalization,
        )
        records.append(
            cfdp_batch_metrics(
                encoding.retrieval.profile_prediction,
                teacher,
                teacher_valid,
                encoding.retrieval.profile_keys,
                encoding.retrieval.node_keys,
                od_targets.future_distance,
                od_targets.candidate_mask,
                top_k=min(5, max(1, encoding.retrieval.node_keys.shape[0] - 1)),
            )
        )
    if not records:
        raise ValueError("CFDP diagnostic processed no batches")
    metric_weights = {
        "profile_mae": "profile_points",
        "profile_cosine": "profile_vectors",
        "profile_relation_spearman": "profile_relation_pairs",
        "profile_relation_recall_at_k": "profile_relation_anchors",
        "total_od_relation_spearman": "od_relation_pairs",
        "total_od_relation_recall_at_k": "od_relation_anchors",
    }
    metrics = {}
    for metric, weight_name in metric_weights.items():
        weights = np.asarray([record[weight_name] for record in records], dtype=np.float64)
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        metrics[metric] = float(np.average(values, weights=weights)) if weights.sum() > 0 else None
    return {
        "schema_version": 1,
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "batches": len(records),
        "metrics": metrics,
        "counts": {
            name: int(sum(record[name] for record in records))
            for name in (
                "profile_points",
                "profile_vectors",
                "profile_relation_pairs",
                "profile_relation_anchors",
                "od_relation_pairs",
                "od_relation_anchors",
            )
        },
        "top_k": records[0]["top_k"],
        "future_information_boundary": (
            "query future is used only for offline val diagnostics and never for ranking"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }

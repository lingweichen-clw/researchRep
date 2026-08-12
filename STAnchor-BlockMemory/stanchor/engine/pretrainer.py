"""Source-domain clean/masked pretraining loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.data.graph import GraphData
from stanchor.losses.pretraining import compute_pretraining_loss
from stanchor.models.pretraining import STAnchorPretrainModel
from stanchor.utils import (
    count_parameters,
    create_run_logger,
    require_finite,
    resolve_device,
    set_seed,
)

from .common import build_data_and_graph, build_pretrain_model, save_checkpoint


ProgressCallback = Callable[[int, int, float, float], None]


@dataclass(frozen=True)
class PretrainEpochResult:
    total: float
    reconstruction: float
    retrieval: float
    valid_retrieval_anchors: int
    positive_pairs: int
    hard_negative_pairs: int
    reconstruction_positions: int
    relation_candidate_pairs: int
    teacher_effective_support: float
    student_effective_support: float
    skipped_batches: int
    batches: int
    profile: float = 0.0


def early_stopping_metric(
    retrieval_loss_mode: str,
    result: PretrainEpochResult,
) -> tuple[str, float]:
    """Select the validation objective that controls pretraining duration."""
    if retrieval_loss_mode == "relation":
        return "val_retrieval", float(result.retrieval)
    return "val_total", float(result.total)


def build_validation_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Keep legacy order when viable, otherwise use one reproducible permutation."""
    retrieval_context = int(getattr(dataset, "retrieval_context_length", 0))
    horizon = int(getattr(dataset, "horizon", 0))
    if retrieval_context <= 0 or horizon <= 0:
        raise ValueError("validation dataset must expose retrieval_context_length and horizon")
    if batch_size > retrieval_context + horizon:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    return DataLoader(
        Subset(dataset, order),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


def run_pretrain_epoch(
    model: STAnchorPretrainModel,
    loader: DataLoader,
    graph: GraphData,
    neighbors: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
    progress_interval: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> PretrainEpochResult:
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "reconstruction": 0.0, "retrieval": 0.0, "profile": 0.0}
    anchors = positives = hard_negatives = reconstruction_positions = batches = skipped_batches = 0
    relation_candidates = 0
    teacher_support = student_support = 0.0
    support_anchors = 0
    context = torch.enable_grad() if training else torch.no_grad()
    planned_batches = len(loader)
    if max_batches is not None:
        planned_batches = min(planned_batches, max_batches)
    started = time.perf_counter()

    def emit_progress(completed: int) -> None:
        if progress_callback is None:
            return
        if completed != 1 and completed % progress_interval != 0 and completed != planned_batches:
            return
        elapsed = time.perf_counter() - started
        seconds_per_batch = elapsed / max(completed, 1)
        eta = seconds_per_batch * max(planned_batches - completed, 0)
        progress_callback(completed, planned_batches, elapsed, eta)

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Training follows the configured categorical sampler. Validation
            # alternates deterministically so both proxy tasks remain comparable.
            mask_task = None if training else ("time" if batch_index % 2 == 0 else "space")
            output = model.forward_pretrain(
                x=batch["retrieval_x"].to(device),
                observed=batch["retrieval_observed"].to(device),
                weekday=batch["retrieval_weekday"].to(device),
                slot=batch["retrieval_slot"].to(device),
                graph=graph,
                neighbors=neighbors,
                mask_task=mask_task,
            )
            losses = compute_pretraining_loss(
                output=output,
                future_model=batch["y"].to(device),
                observed_context=batch["retrieval_observed"].to(device),
                observed_future=batch["y_observed"].to(device),
                context_start=batch["context_start"].to(device),
                future_end=batch["future_end"].to(device),
                retrieval_weight=config.pretrain.retrieval_weight,
                retrieval_temperature=config.pretrain.retrieval_temperature,
                positive_quantile=config.pretrain.positive_quantile,
                context_quantile=config.pretrain.context_quantile,
                negative_quantile=config.pretrain.negative_quantile,
                hard_negative_weight=config.pretrain.hard_negative_weight,
                retrieval_loss_mode=config.pretrain.retrieval_loss_mode,
                relation_teacher_temperature=config.pretrain.relation_teacher_temperature,
                relation_student_temperature=config.pretrain.relation_student_temperature,
                forecast_context=batch["x"].to(device),
                forecast_context_observed=batch["x_observed"].to(device),
                relation_teacher_mode=config.pretrain.relation_teacher_mode,
                relation_distance_normalization=(
                    config.pretrain.relation_distance_normalization
                ),
                future_increment_weight=config.pretrain.future_increment_weight,
                profile_loss_weight=config.pretrain.profile_loss_weight,
                profile_scale_floor=config.pretrain.profile_scale_floor,
            )
            if losses.reconstruction_positions == 0 and losses.valid_retrieval_anchors == 0:
                skipped_batches += 1
                emit_progress(batch_index + 1)
                continue
            require_finite(losses.total, "pretraining loss")
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            totals["total"] += float(losses.total.detach())
            totals["reconstruction"] += float(losses.reconstruction.detach())
            totals["retrieval"] += float(losses.retrieval.detach())
            totals["profile"] += float((losses.profile if losses.profile is not None else losses.total * 0.0).detach())
            anchors += losses.valid_retrieval_anchors
            positives += losses.positive_pairs
            hard_negatives += losses.hard_negative_pairs
            reconstruction_positions += losses.reconstruction_positions
            relation_candidates += losses.relation_candidate_pairs
            if losses.valid_retrieval_anchors > 0:
                teacher_support += (
                    losses.teacher_effective_support * losses.valid_retrieval_anchors
                )
                student_support += (
                    losses.student_effective_support * losses.valid_retrieval_anchors
                )
                support_anchors += losses.valid_retrieval_anchors
            batches += 1
            emit_progress(batch_index + 1)
    if batches == 0:
        raise ValueError("pretraining epoch processed no batches")
    return PretrainEpochResult(
        total=totals["total"] / batches,
        reconstruction=totals["reconstruction"] / batches,
        retrieval=totals["retrieval"] / batches,
        valid_retrieval_anchors=anchors,
        positive_pairs=positives,
        hard_negative_pairs=hard_negatives,
        reconstruction_positions=reconstruction_positions,
        relation_candidate_pairs=relation_candidates,
        teacher_effective_support=teacher_support / max(support_anchors, 1),
        student_effective_support=student_support / max(support_anchors, 1),
        profile=totals["profile"] / batches,
        skipped_batches=skipped_batches,
        batches=batches,
    )


def train_pretraining(
    config: ExperimentConfig,
    max_batches: int | None = None,
) -> Path:
    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    neighbors = graph_cpu.dense_neighbors(include_self=False)
    model = build_pretrain_model(config, data.series.slots_per_day).to(device)
    train_loader = DataLoader(
        data.train,
        batch_size=config.pretrain.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=False,
    )
    val_loader = build_validation_loader(
        data.val,
        config.pretrain.batch_size,
        config.data.num_workers,
        config.runtime.seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain.learning_rate,
        weight_decay=config.pretrain.weight_decay,
    )
    run_dir = resolve_project_path(config.runtime.output_dir) / config.runtime.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "pretrain_metrics.jsonl"
    logger = create_run_logger(
        f"stanchor.pretrain.{config.runtime.run_name}",
        run_dir / "pretrain.log",
    )
    best_path = run_dir / "pretrain_best.pt"
    parameter_counts = {
        "embedding": count_parameters(model.embedding),
        "encoder": count_parameters(model.encoder),
        "retrieval_head": count_parameters(model.retrieval_head),
        "reconstruction_head": count_parameters(model.reconstruction_head),
        "total_trainable": count_parameters(model),
        "total": count_parameters(model, trainable_only=False),
    }
    logger.info(
        "Pretraining start | run=%s | device=%s (requested=%s) | seed=%d | output=%s",
        config.runtime.run_name,
        device,
        config.runtime.device,
        config.runtime.seed,
        run_dir,
    )
    logger.info(
        "Data | steps=%d | nodes=%d | channels=%d | train/val/test=%d/%d/%d | "
        "dropped_unobserved=%d/%d/%d",
        data.series.num_steps,
        data.series.num_nodes,
        data.series.num_channels,
        len(data.train),
        len(data.val),
        len(data.test),
        data.train.dropped_unobserved_events,
        data.val.dropped_unobserved_events,
        data.test.dropped_unobserved_events,
    )
    logger.info(
        "Tensor contract | retrieval_x=[B,%d,%d,%d] | forecast_x=[B,%d,%d,%d] | "
        "patches=%d | hidden=%d | retrieval_dim=%d | y=[B,%d,%d,%d] | graph_edges=%d",
        config.data.encoder_context_length,
        data.series.num_nodes,
        config.model.input_channels,
        config.data.context_length,
        data.series.num_nodes,
        config.model.input_channels,
        model.num_patches,
        config.model.hidden_dim,
        config.model.retrieval_dim,
        config.data.horizon,
        data.series.num_nodes,
        config.model.output_channels,
        graph_cpu.edge_index.shape[1],
    )
    logger.info(
        "Parameters | total=%s | trainable=%s | embedding=%s | encoder=%s | "
        "retrieval_head=%s | reconstruction_head=%s",
        f"{parameter_counts['total']:,}",
        f"{parameter_counts['total_trainable']:,}",
        f"{parameter_counts['embedding']:,}",
        f"{parameter_counts['encoder']:,}",
        f"{parameter_counts['retrieval_head']:,}",
        f"{parameter_counts['reconstruction_head']:,}",
    )
    logger.info(
        "Optimization | epochs=%d | batch_size=%d | lr=%.3g | weight_decay=%.3g | "
        "time_mask=%.3f | time_mask_block=%d steps | space_mask=%.3f | "
        "retrieval_loss=%s | teacher_mode=%s | distance_normalization=%s | "
        "future_increment_weight=%.3f | retrieval_weight=%.3f | "
        "student_tau=%.3f | teacher_tau=%.3f | patience=%d",
        config.pretrain.epochs,
        config.pretrain.batch_size,
        config.pretrain.learning_rate,
        config.pretrain.weight_decay,
        config.pretrain.time_mask_ratio,
        config.pretrain.time_mask_block_size,
        config.pretrain.space_mask_ratio,
        config.pretrain.retrieval_loss_mode,
        config.pretrain.relation_teacher_mode,
        config.pretrain.relation_distance_normalization,
        config.pretrain.future_increment_weight,
        config.pretrain.retrieval_weight,
        config.pretrain.relation_student_temperature,
        config.pretrain.relation_teacher_temperature,
        config.pretrain.patience,
    )
    best_value = float("inf")
    best_relation_value = float("inf")
    best_stopping_value = float("inf")
    stale_epochs = 0
    best_relation_path = run_dir / "pretrain_best_relation.pt"

    def checkpoint_payload(record: dict, epoch: int) -> dict:
        return {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "retrieval_state_dict": model.retrieval_state_dict(),
            "retrieval_fingerprint": model.retrieval_fingerprint(),
            "config": config.to_dict(),
            "normalizer": data.scaler.state_dict(),
            "graph_fingerprint": graph_cpu.fingerprint,
            "metrics": record,
            "epoch": epoch,
            "seed": config.runtime.seed,
        }

    for epoch in range(1, config.pretrain.epochs + 1):
        logger.info(
            "Epoch %03d/%03d started | train_batches=%d | val_batches=%d",
            epoch,
            config.pretrain.epochs,
            len(train_loader),
            len(val_loader),
        )
        train_progress = lambda completed, total, elapsed, eta: logger.info(
            "Epoch %03d | train batch=%d/%d | elapsed=%.1f min | eta=%.1f min",
            epoch,
            completed,
            total,
            elapsed / 60.0,
            eta / 60.0,
        )
        train_result = run_pretrain_epoch(
            model,
            train_loader,
            graph,
            neighbors,
            config,
            device,
            optimizer,
            max_batches,
            progress_interval=config.pretrain.progress_interval,
            progress_callback=train_progress,
        )
        val_progress = lambda completed, total, elapsed, eta: logger.info(
            "Epoch %03d | val batch=%d/%d | elapsed=%.1f min | eta=%.1f min",
            epoch,
            completed,
            total,
            elapsed / 60.0,
            eta / 60.0,
        )
        val_result = run_pretrain_epoch(
            model,
            val_loader,
            graph,
            neighbors,
            config,
            device,
            None,
            max_batches,
            progress_interval=config.pretrain.progress_interval,
            progress_callback=val_progress,
        )
        record = {
            "epoch": epoch,
            "train": train_result.__dict__,
            "val": val_result.__dict__,
            "parameters": count_parameters(model),
            "parameter_counts": parameter_counts,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(
            "Epoch %03d | train_total=%.6f | val_total=%.6f | val_mask=%.6f | "
            "val_retrieval=%.6f | val_profile=%.6f | val_anchors=%d | val_positive_pairs=%d | "
            "val_hard_negatives=%d | val_relation_candidates=%d | "
            "val_teacher_keff=%.3f | val_student_keff=%.3f | "
            "val_masked_positions=%d | skipped(train/val)=%d/%d",
            epoch,
            train_result.total,
            val_result.total,
            val_result.reconstruction,
            val_result.retrieval,
            val_result.profile,
            val_result.valid_retrieval_anchors,
            val_result.positive_pairs,
            val_result.hard_negative_pairs,
            val_result.relation_candidate_pairs,
            val_result.teacher_effective_support,
            val_result.student_effective_support,
            val_result.reconstruction_positions,
            train_result.skipped_batches,
            val_result.skipped_batches,
        )
        if val_result.total < best_value:
            best_value = val_result.total
            save_checkpoint(
                best_path,
                checkpoint_payload(record, epoch),
            )
            logger.info(
                "Checkpoint updated | epoch=%d | best_val=%.6f | path=%s",
                epoch,
                best_value,
                best_path,
            )
        if (
            config.pretrain.retrieval_loss_mode == "relation"
            and val_result.retrieval < best_relation_value
        ):
            best_relation_value = val_result.retrieval
            save_checkpoint(
                best_relation_path,
                checkpoint_payload(record, epoch),
            )
            logger.info(
                "Relation checkpoint updated | epoch=%d | best_val_relation=%.6f | path=%s",
                epoch,
                best_relation_value,
                best_relation_path,
            )
        stopping_name, stopping_value = early_stopping_metric(
            config.pretrain.retrieval_loss_mode,
            val_result,
        )
        if stopping_value < best_stopping_value:
            best_stopping_value = stopping_value
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.pretrain.patience:
                logger.info(
                    "Early stopping | epoch=%d | stale_epochs=%d | %s=%.6f",
                    epoch,
                    stale_epochs,
                    stopping_name,
                    best_stopping_value,
                )
                break
    if config.pretrain.retrieval_loss_mode == "relation":
        logger.info(
            "Pretraining finished | best_val=%.6f | best_val_relation=%.6f | "
            "checkpoint=%s | relation_checkpoint=%s",
            best_value,
            best_relation_value,
            best_path,
            best_relation_path,
        )
    else:
        logger.info("Pretraining finished | best_val=%.6f | checkpoint=%s", best_value, best_path)
    return best_path

"""Target-domain retrieval adaptation helpers."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from torch.utils.data import DataLoader

from stanchor.config import (
    AdaptationConfig,
    ExperimentConfig,
    PretrainConfig,
    resolve_project_path,
)
from stanchor.data.graph import GraphData
from stanchor.losses.adaptation import compute_t1_adaptation_loss
from stanchor.losses.pretraining import compute_relation_only_loss
from stanchor.models.pretraining import STAnchorPretrainModel
from stanchor.utils import (
    count_parameters,
    create_run_logger,
    require_finite,
    resolve_device,
    set_seed,
)

from .common import build_data_and_graph, load_pretrained_model, save_checkpoint
from .pretrainer import build_validation_loader, should_validate_epoch


T1_TRAINABLE_PREFIXES = (
    "retrieval_head.pool_projection.",
    "retrieval_head.pool_score.",
    "retrieval_head.key_mlp.",
    "retrieval_head.domain_adapter.",
)
ProgressCallback = Callable[[int, int, float, float], None]


@dataclass(frozen=True)
class RetrievalAdaptationEpochResult:
    total: float
    relation: float
    distillation: float
    valid_retrieval_anchors: int
    relation_candidate_pairs: int
    teacher_effective_support: float
    student_effective_support: float
    skipped_batches: int
    batches: int
    seconds: float


def configure_t1_parameters(
    model: STAnchorPretrainModel,
) -> list[torch.nn.Parameter]:
    """Freeze the model except the retrieval pooling, key MLP, and domain adapter."""
    if model.retrieval_head.key_mlp is None:
        raise ValueError("T1 head adaptation requires retrieval_head.key_mlp")
    if model.retrieval_head.domain_adapter is None:
        raise ValueError("T1 head adaptation requires retrieval_head.domain_adapter")
    trainable: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(T1_TRAINABLE_PREFIXES)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise ValueError("T1 selected no trainable parameters")
    return trainable


def build_t1_checkpoint_payload(
    model: STAnchorPretrainModel,
    config: ExperimentConfig,
    normalizer: dict,
    graph_fingerprint: str,
    source_checkpoint: str,
    source_retrieval_fingerprint: str,
    epoch: int,
    metrics: dict,
) -> dict:
    """Build a Bank-compatible checkpoint with explicit adaptation provenance."""
    return {
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "retrieval_encoder_state_dict": model.retrieval_state_dict(),
        "retrieval_state_dict": model.retrieval_state_dict(),
        "retrieval_fingerprint": model.retrieval_fingerprint(),
        "config": config.to_dict(),
        "normalizer": normalizer,
        "graph_fingerprint": graph_fingerprint,
        "source_checkpoint": source_checkpoint,
        "source_retrieval_fingerprint": source_retrieval_fingerprint,
        "stage": "t1_head_adapter",
        "epoch": epoch,
        "metrics": metrics,
        "seed": config.runtime.seed,
    }


def run_retrieval_adaptation_epoch(
    model: STAnchorPretrainModel,
    source_model: STAnchorPretrainModel,
    loader: Iterable[Mapping[str, torch.Tensor]],
    graph: GraphData,
    config: AdaptationConfig,
    pretrain_config: PretrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RetrievalAdaptationEpochResult:
    """Run one T1 epoch while keeping the frozen representation deterministic."""
    training = optimizer is not None
    model.eval()
    model.retrieval_head.train(training)
    source_model.eval()
    totals = {"total": 0.0, "relation": 0.0, "distillation": 0.0}
    anchors = candidates = skipped = batches = 0
    teacher_support = student_support = 0.0
    started = time.perf_counter()
    planned_batches = len(loader)  # type: ignore[arg-type]
    if max_batches is not None:
        planned_batches = min(planned_batches, max_batches)

    def emit_progress(completed: int) -> None:
        if progress_callback is None:
            return
        if (
            completed != 1
            and completed % config.progress_interval != 0
            and completed != planned_batches
        ):
            return
        elapsed = time.perf_counter() - started
        seconds_per_batch = elapsed / max(completed, 1)
        eta = seconds_per_batch * max(planned_batches - completed, 0)
        progress_callback(completed, planned_batches, elapsed, eta)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            retrieval_x = batch["retrieval_x"].to(device)
            retrieval_observed = batch["retrieval_observed"].to(device).bool()
            clean = model.forward_relation(
                retrieval_x,
                retrieval_observed,
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            # T1 leaves the representation trunk unchanged, so the frozen
            # source head can reuse the detached target hidden state.
            with torch.no_grad():
                teacher_keys = source_model.retrieval_head(
                    clean.hidden.detach()
                ).node_keys
            relation = compute_relation_only_loss(
                clean=clean,
                future_model=batch["y"].to(device),
                observed_future=batch["y_observed"].to(device).bool(),
                context_start=batch["context_start"].to(device),
                future_end=batch["future_end"].to(device),
                retrieval_weight=1.0,
                relation_teacher_temperature=(
                    pretrain_config.relation_teacher_temperature
                ),
                relation_student_temperature=(
                    pretrain_config.relation_student_temperature
                ),
                forecast_context=batch["x"].to(device),
                forecast_context_observed=batch["x_observed"].to(device).bool(),
                relation_teacher_mode=pretrain_config.relation_teacher_mode,
                relation_distance_normalization=(
                    pretrain_config.relation_distance_normalization
                ),
                future_increment_weight=pretrain_config.future_increment_weight,
                rank_loss_weight=pretrain_config.rank_loss_weight,
                rank_positive_count=pretrain_config.rank_positive_count,
                rank_negative_count=pretrain_config.rank_negative_count,
                rank_future_gap=pretrain_config.rank_future_gap,
                rank_margin=pretrain_config.rank_margin,
                rank_temperature=pretrain_config.rank_temperature,
            )
            if relation.valid_retrieval_anchors == 0:
                skipped += 1
                emit_progress(batch_index + 1)
                continue
            loss = compute_t1_adaptation_loss(
                relation_loss=relation.retrieval,
                student_keys=clean.retrieval.node_keys,
                teacher_keys=teacher_keys,
                relation_weight=config.relation_weight,
                distill_weight=config.distill_weight,
            )
            require_finite(loss.total, "T1 adaptation loss")
            if training:
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=config.gradient_clip_norm,
                )
                optimizer.step()
            totals["total"] += float(loss.total.detach())
            totals["relation"] += float(loss.relation.detach())
            totals["distillation"] += float(loss.distillation.detach())
            anchors += relation.valid_retrieval_anchors
            candidates += relation.relation_candidate_pairs
            teacher_support += (
                relation.teacher_effective_support * relation.valid_retrieval_anchors
            )
            student_support += (
                relation.student_effective_support * relation.valid_retrieval_anchors
            )
            batches += 1
            emit_progress(batch_index + 1)
    if batches == 0:
        raise ValueError("T1 adaptation epoch processed no valid retrieval batches")
    return RetrievalAdaptationEpochResult(
        total=totals["total"] / batches,
        relation=totals["relation"] / batches,
        distillation=totals["distillation"] / batches,
        valid_retrieval_anchors=anchors,
        relation_candidate_pairs=candidates,
        teacher_effective_support=teacher_support / max(anchors, 1),
        student_effective_support=student_support / max(anchors, 1),
        skipped_batches=skipped,
        batches=batches,
        seconds=time.perf_counter() - started,
    )


def train_retrieval_adaptation(
    config: ExperimentConfig,
    source_checkpoint: str | Path,
    max_batches: int | None = None,
) -> Path:
    """Run T1 target-domain adaptation and return the best relation checkpoint."""
    config.validate()
    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    source_path = resolve_project_path(source_checkpoint)
    if not source_path.exists():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    source_model, source_payload = load_pretrained_model(
        config,
        source_path,
        data.series.slots_per_day,
        device,
    )
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    source_fingerprint = source_model.retrieval_fingerprint()
    recorded_source_fingerprint = source_payload.get("retrieval_fingerprint")
    if (
        recorded_source_fingerprint is not None
        and recorded_source_fingerprint != source_fingerprint
    ):
        raise ValueError("source checkpoint retrieval fingerprint does not match its weights")

    model = copy.deepcopy(source_model)
    trainable = configure_t1_parameters(model)
    train_loader = DataLoader(
        data.train,
        batch_size=config.adaptation.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=False,
    )
    val_loader = build_validation_loader(
        data.val,
        config.adaptation.batch_size,
        config.data.num_workers,
        config.runtime.seed,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.adaptation.learning_rate,
        weight_decay=config.adaptation.weight_decay,
    )
    run_dir = resolve_project_path(config.runtime.output_dir) / config.runtime.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "retrieval_adaptation_metrics.jsonl"
    best_path = run_dir / "retrieval_t1_best.pt"
    logger = create_run_logger(
        f"stanchor.retrieval_adaptation.{config.runtime.run_name}",
        run_dir / "retrieval_adaptation.log",
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    logger.info(
        "T1 retrieval adaptation start | run=%s | device=%s (requested=%s) | "
        "seed=%d | output=%s",
        config.runtime.run_name,
        device,
        config.runtime.device,
        config.runtime.seed,
        run_dir,
    )
    logger.info(
        "Data | steps=%d | nodes=%d | channels=%d | train/val/test=%d/%d/%d",
        data.series.num_steps,
        data.series.num_nodes,
        data.series.num_channels,
        len(data.train),
        len(data.val),
        len(data.test),
    )
    logger.info(
        "Transfer | stage=t1_head_adapter | source=%s | source_fingerprint=%s | "
        "graph_fingerprint=%s",
        source_path,
        source_fingerprint,
        graph_cpu.fingerprint,
    )
    logger.info(
        "Parameters | total=%s | trainable=%s | trainable_tensors=%d | prefixes=%s",
        f"{count_parameters(model, trainable_only=False):,}",
        f"{count_parameters(model):,}",
        len(trainable_names),
        ",".join(T1_TRAINABLE_PREFIXES),
    )
    logger.info(
        "Optimization | epochs=%d | batch_size=%d | lr=%.3g | weight_decay=%.3g | "
        "relation_weight=%.3f | distill_weight=%.3f | reconstruction_weight=0",
        config.adaptation.epochs,
        config.adaptation.batch_size,
        config.adaptation.learning_rate,
        config.adaptation.weight_decay,
        config.adaptation.relation_weight,
        config.adaptation.distill_weight,
    )
    best_relation = float("inf")
    best_epoch = 0
    for epoch in range(1, config.adaptation.epochs + 1):
        logger.info(
            "Epoch %03d/%03d started | train_batches=%d | val_batches=%d",
            epoch,
            config.adaptation.epochs,
            len(train_loader),
            len(val_loader),
        )
        train_result = run_retrieval_adaptation_epoch(
            model=model,
            source_model=source_model,
            loader=train_loader,
            graph=graph,
            config=config.adaptation,
            pretrain_config=config.pretrain,
            device=device,
            optimizer=optimizer,
            max_batches=max_batches,
            progress_callback=lambda completed, total, elapsed, eta: logger.info(
                "Epoch %03d | train batch=%d/%d | elapsed=%.1f min | eta=%.1f min",
                epoch,
                completed,
                total,
                elapsed / 60.0,
                eta / 60.0,
            ),
        )
        val_evaluated = should_validate_epoch(
            epoch,
            config.adaptation.epochs,
            config.adaptation.validation_interval,
        )
        val_result = None
        if val_evaluated:
            val_result = run_retrieval_adaptation_epoch(
                model=model,
                source_model=source_model,
                loader=val_loader,
                graph=graph,
                config=config.adaptation,
                pretrain_config=config.pretrain,
                device=device,
                optimizer=None,
                max_batches=max_batches,
                progress_callback=lambda completed, total, elapsed, eta: logger.info(
                    "Epoch %03d | val batch=%d/%d | elapsed=%.1f min | eta=%.1f min",
                    epoch,
                    completed,
                    total,
                    elapsed / 60.0,
                    eta / 60.0,
                ),
            )
        record = {
            "epoch": epoch,
            "train": train_result.__dict__,
            "val": val_result.__dict__ if val_result is not None else None,
            "val_evaluated": val_evaluated,
            "trainable_parameters": count_parameters(model),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if val_result is None:
            logger.info(
                "Epoch %03d | train_total=%.6f | train_relation=%.6f | "
                "train_distill=%.6f | seconds=%.2f | val_evaluated=false",
                epoch,
                train_result.total,
                train_result.relation,
                train_result.distillation,
                train_result.seconds,
            )
            continue
        logger.info(
            "Epoch %03d | train_total=%.6f | train_relation=%.6f | "
            "train_distill=%.6f | val_total=%.6f | val_relation=%.6f | "
            "val_distill=%.6f | val_anchors=%d | val_candidates=%d | "
            "val_teacher_keff=%.3f | val_student_keff=%.3f | "
            "seconds(train/val)=%.2f/%.2f",
            epoch,
            train_result.total,
            train_result.relation,
            train_result.distillation,
            val_result.total,
            val_result.relation,
            val_result.distillation,
            val_result.valid_retrieval_anchors,
            val_result.relation_candidate_pairs,
            val_result.teacher_effective_support,
            val_result.student_effective_support,
            train_result.seconds,
            val_result.seconds,
        )
        if val_result.relation < best_relation:
            best_relation = val_result.relation
            best_epoch = epoch
            save_checkpoint(
                best_path,
                build_t1_checkpoint_payload(
                    model=model,
                    config=config,
                    normalizer=data.scaler.state_dict(),
                    graph_fingerprint=graph_cpu.fingerprint,
                    source_checkpoint=str(source_path.resolve()),
                    source_retrieval_fingerprint=source_fingerprint,
                    epoch=epoch,
                    metrics=record,
                ),
            )
            logger.info(
                "Checkpoint updated | epoch=%d | best_val_relation=%.6f | path=%s",
                epoch,
                best_relation,
                best_path,
            )
    if best_epoch == 0 or not best_path.exists():
        raise RuntimeError("T1 adaptation finished without a validation checkpoint")
    logger.info(
        "T1 retrieval adaptation finished | best_epoch=%d | "
        "best_val_relation=%.6f | checkpoint=%s",
        best_epoch,
        best_relation,
        best_path,
    )
    return best_path

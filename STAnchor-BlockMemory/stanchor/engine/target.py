"""Target calibration, safe fusion training, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from stanchor.bank.storage import MemoryBank
from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.data.graph import GraphData
from stanchor.data.normalization import NodeStandardScaler
from stanchor.losses.downstream import compute_downstream_loss
from stanchor.metrics import ForecastMetricAccumulator, select_common_horizon_metrics
from stanchor.modes import (
    BASE_ONLY,
    LEARNED_TOPK_CONFIDENCE,
    LEARNED_TOPK_HORIZON,
    LEARNED_TOPK_OFFSET_DECAY_HORIZON,
    RAW_L1_TOPK_HORIZON,
    WEEKLY_MEAN_HORIZON,
    validate_downstream_mode,
)
from stanchor.models.downstream import (
    ConfidenceHead,
    LightweightForecastBackbone,
    SafeResidualFusion,
    STAnchorDownstreamModel,
)
from stanchor.models.pretraining import STAnchorPretrainModel
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates, TwoStageRetriever
from stanchor.retrieval.strategies import (
    calendar_event_candidates,
    offset_decay_aggregation,
    raw_l1_topk_aggregation,
    weekly_mean_aggregation,
)
from stanchor.utils import count_parameters, create_run_logger, require_finite, resolve_device, set_seed

from .common import build_data_and_graph, load_checkpoint, load_pretrained_model, save_checkpoint


@dataclass(frozen=True)
class TargetEpochResult:
    total_loss: float
    forecast_loss: float
    confidence_loss: float
    metrics: dict
    batches: int


def build_downstream_model(config: ExperimentConfig) -> STAnchorDownstreamModel:
    return STAnchorDownstreamModel(
        backbone=LightweightForecastBackbone(
            context_length=config.data.context_length,
            horizon=config.data.horizon,
            input_channels=config.model.input_channels,
            output_channels=config.model.output_channels,
            hidden_dim=config.target.backbone_hidden_dim,
            dropout=config.model.dropout,
        ),
        confidence_head=ConfidenceHead(config.target.confidence_hidden_dim),
        fusion=SafeResidualFusion(config.data.horizon),
        confidence_level_temperature=config.target.confidence_level_temperature,
        mode=config.target.downstream_mode,
    )


def checkpoint_downstream_mode(checkpoint: dict) -> str:
    mode = checkpoint.get("downstream_mode")
    if mode is None:
        target_config = checkpoint.get("config", {}).get("target", {})
        mode = target_config.get("downstream_mode", LEARNED_TOPK_CONFIDENCE)
    return validate_downstream_mode(str(mode))


def configure_downstream_trainable(
    downstream: STAnchorDownstreamModel,
    mode: str,
) -> None:
    mode = validate_downstream_mode(mode)
    for parameter in downstream.parameters():
        parameter.requires_grad_(True)
    if mode != LEARNED_TOPK_CONFIDENCE:
        for parameter in downstream.confidence_head.parameters():
            parameter.requires_grad_(False)
    if mode == BASE_ONLY:
        for parameter in downstream.fusion.parameters():
            parameter.requires_grad_(False)


@torch.no_grad()
def retrieve_for_downstream_mode(
    mode: str,
    pretrained: STAnchorPretrainModel,
    retriever: TwoStageRetriever,
    bank: MemoryBank,
    data,
    graph: GraphData,
    batch: dict[str, torch.Tensor],
    x: torch.Tensor,
    observed_x: torch.Tensor,
    device: torch.device,
) -> tuple[NodeCandidates | None, AggregationOutput | None]:
    mode = validate_downstream_mode(mode)
    if mode == BASE_ONLY:
        return None, None
    if mode == LEARNED_TOPK_CONFIDENCE:
        encoding = pretrained.encode_clean(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            batch["retrieval_weekday"].to(device),
            batch["retrieval_slot"].to(device),
            graph,
        )
        _, candidates, aggregation = retriever.retrieve(
            query_event_keys=encoding.retrieval.event_keys,
            query_node_keys=encoding.retrieval.node_keys,
            query_levels=encoding.statistics.level_features,
            weekday=batch["query_weekday"].to(device),
            slot=batch["query_slot"].to(device),
            context_start=batch["context_start"].to(device),
        )
        return candidates, aggregation

    events = calendar_event_candidates(
        bank,
        weekday=batch["query_weekday"].to(device),
        slot=batch["query_slot"].to(device),
        context_start=batch["context_start"].to(device),
        max_candidates=retriever.event_top_r,
        device=device,
    )
    if mode == WEEKLY_MEAN_HORIZON:
        return None, weekly_mean_aggregation(bank, events, device)
    if mode == RAW_L1_TOPK_HORIZON:
        return None, raw_l1_topk_aggregation(
            x,
            observed_x,
            bank,
            events,
            data.series,
            data.scaler,
            data.train.context_length,
            retriever.node_top_k,
            device,
        )
    if mode in {LEARNED_TOPK_HORIZON, LEARNED_TOPK_OFFSET_DECAY_HORIZON}:
        encoding = pretrained.encode_clean(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            batch["retrieval_weekday"].to(device),
            batch["retrieval_slot"].to(device),
            graph,
        )
        candidates = retriever.rerank_nodes(
            encoding.retrieval.node_keys,
            encoding.statistics.level_features,
            events,
        )
        if mode == LEARNED_TOPK_OFFSET_DECAY_HORIZON:
            return candidates, offset_decay_aggregation(
                candidates,
                x,
                observed_x,
                bank,
                data.series,
                data.scaler,
                data.train.context_length,
                device,
            )
        return candidates, retriever.aggregate(candidates)
    raise AssertionError(f"unhandled downstream mode: {mode}")


def _validate_bank(
    bank: MemoryBank,
    pretrained: STAnchorPretrainModel,
    graph: GraphData,
    scaler_state: dict,
) -> None:
    if bank.manifest.encoder_fingerprint != pretrained.retrieval_fingerprint():
        raise ValueError("Bank keys were built with a different retrieval encoder")
    if bank.manifest.context_length != pretrained.context_length:
        raise ValueError("Bank retrieval context length does not match the pretrained encoder")
    if bank.manifest.graph_fingerprint != graph.fingerprint:
        raise ValueError("Bank graph fingerprint does not match target graph")
    bank_mean = np.asarray(bank.manifest.scaler["mean"], dtype=np.float32)
    current_mean = np.asarray(scaler_state["mean"], dtype=np.float32)
    bank_std = np.asarray(bank.manifest.scaler["std"], dtype=np.float32)
    current_std = np.asarray(scaler_state["std"], dtype=np.float32)
    if not np.allclose(bank_mean, current_mean) or not np.allclose(bank_std, current_std):
        raise ValueError("Bank and query data use different target scalers")


def run_target_epoch(
    pretrained: STAnchorPretrainModel,
    downstream: STAnchorDownstreamModel,
    retriever: TwoStageRetriever,
    bank: MemoryBank,
    data,
    loader: DataLoader,
    graph: GraphData,
    config: ExperimentConfig,
    scaler: NodeStandardScaler,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> TargetEpochResult:
    training = optimizer is not None
    downstream.train(training)
    pretrained.eval()
    total = forecast = confidence = 0.0
    batches = 0
    metrics = ForecastMetricAccumulator(config.data.horizon)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            observed_x = batch["x_observed"].to(device)
            node_candidates, aggregation = retrieve_for_downstream_mode(
                config.target.downstream_mode,
                pretrained,
                retriever,
                bank,
                data,
                graph,
                batch,
                x,
                observed_x,
                device,
            )
            output = downstream(x, node_candidates, aggregation)
            losses = compute_downstream_loss(
                output,
                target=batch["y"].to(device),
                observed=batch["y_observed"].to(device),
                confidence_weight=config.target.confidence_weight,
                help_margin=config.target.help_margin,
                help_temperature=config.target.help_temperature,
                use_confidence=config.target.downstream_mode == LEARNED_TOPK_CONFIDENCE,
            )
            require_finite(losses.total, "downstream loss")
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(downstream.parameters(), max_norm=5.0)
                optimizer.step()
            total += float(losses.total.detach())
            forecast += float(losses.forecast.detach())
            confidence += float(losses.confidence.detach())
            target_model = batch["y"].to(device)
            metrics.update(
                scaler.inverse_transform_torch(output.final_prediction.detach()),
                scaler.inverse_transform_torch(target_model),
                batch["y_observed"].to(device),
            )
            batches += 1
    if batches == 0:
        raise ValueError("target epoch processed no batches")
    return TargetEpochResult(total / batches, forecast / batches, confidence / batches, metrics.compute(), batches)


def train_downstream(
    config: ExperimentConfig,
    pretrained_checkpoint: str | Path,
    bank_path: str | Path,
    max_batches: int | None = None,
) -> Path:
    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    pretrained, _ = load_pretrained_model(
        config, pretrained_checkpoint, data.series.slots_per_day, device
    )
    for parameter in pretrained.parameters():
        parameter.requires_grad_(False)
    with MemoryBank(bank_path) as bank:
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
        memory_events = int(len(data.train) * config.bank.memory_fraction)
        calibration = Subset(data.train, range(memory_events, len(data.train)))
        train_loader = DataLoader(
            calibration,
            batch_size=config.target.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        val_loader = DataLoader(
            data.val,
            batch_size=config.target.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        downstream = build_downstream_model(config).to(device)
        configure_downstream_trainable(downstream, config.target.downstream_mode)
        trainable_parameters = [
            parameter for parameter in downstream.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("downstream mode has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.target.learning_rate,
            weight_decay=config.target.weight_decay,
        )
        run_dir = resolve_project_path(config.runtime.output_dir) / config.runtime.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "downstream_best.pt"
        metrics_path = run_dir / "target_metrics.jsonl"
        logger = create_run_logger(
            f"stanchor.downstream.{config.runtime.run_name}",
            run_dir / "downstream.log",
        )
        parameter_counts = {
            "frozen_pretrained": count_parameters(pretrained, trainable_only=False),
            "backbone": count_parameters(downstream.backbone),
            "confidence_head": count_parameters(downstream.confidence_head),
            "fusion": count_parameters(downstream.fusion),
            "downstream_trainable": count_parameters(downstream),
            "downstream_total": count_parameters(downstream, trainable_only=False),
        }
        logger.info(
            "Downstream start | run=%s | device=%s (requested=%s) | seed=%d | output=%s",
            config.runtime.run_name,
            device,
            config.runtime.device,
            config.runtime.seed,
            run_dir,
        )
        logger.info("Mode | downstream_mode=%s", config.target.downstream_mode)
        logger.info(
            "Data | steps=%d | nodes=%d | channels=%d | memory/calibration/val/test=%d/%d/%d/%d",
            data.series.num_steps,
            data.series.num_nodes,
            data.series.num_channels,
            memory_events,
            len(calibration),
            len(data.val),
            len(data.test),
        )
        logger.info(
            "Bank/retrieval | dataset=%s | events=%d | nodes=%d | retrieval_dim=%d | "
            "event_top_r=%d | node_top_k=%d | key_dtype=%s",
            bank.manifest.dataset_name,
            bank.manifest.num_events,
            bank.manifest.num_nodes,
            bank.manifest.retrieval_dim,
            config.bank.event_top_r,
            config.bank.node_top_k,
            bank.manifest.key_dtype,
        )
        logger.info(
            "Parameters | downstream_total=%s | downstream_trainable=%s | backbone=%s | "
            "confidence_head=%s | fusion=%s | frozen_pretrained=%s",
            f"{parameter_counts['downstream_total']:,}",
            f"{parameter_counts['downstream_trainable']:,}",
            f"{parameter_counts['backbone']:,}",
            f"{parameter_counts['confidence_head']:,}",
            f"{parameter_counts['fusion']:,}",
            f"{parameter_counts['frozen_pretrained']:,}",
        )
        logger.info(
            "Optimization | epochs=%d | batch_size=%d | lr=%.3g | weight_decay=%.3g | "
            "confidence_weight=%.3f | patience=%d",
            config.target.epochs,
            config.target.batch_size,
            config.target.learning_rate,
            config.target.weight_decay,
            config.target.confidence_weight,
            config.target.patience,
        )
        best_mae = float("inf")
        stale = 0
        for epoch in range(1, config.target.epochs + 1):
            train_result = run_target_epoch(
                pretrained, downstream, retriever, bank, data, train_loader, graph, config, data.scaler, device, optimizer, max_batches
            )
            val_result = run_target_epoch(
                pretrained, downstream, retriever, bank, data, val_loader, graph, config, data.scaler, device, None, max_batches
            )
            record = {
                "epoch": epoch,
                "train": train_result.__dict__,
                "val": val_result.__dict__,
                "parameter_counts": parameter_counts,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                "Epoch %03d | train_total=%.6f | train_forecast=%.6f | "
                "train_confidence=%.6f | val_total=%.6f | val_mae=%.6f | "
                "val_rmse=%.6f | val_mape=%.6f | val_confidence=%.6f",
                epoch,
                train_result.total_loss,
                train_result.forecast_loss,
                train_result.confidence_loss,
                val_result.total_loss,
                val_result.metrics["mae"],
                val_result.metrics["rmse"],
                val_result.metrics["mape"],
                val_result.confidence_loss,
            )
            horizon_metrics = select_common_horizon_metrics(
                val_result.metrics,
                config.data.frequency_minutes,
            )
            if horizon_metrics:
                fields = []
                for label, values in horizon_metrics.items():
                    fields.append(
                        f"{label}_mae={values['mae']:.6f} "
                        f"{label}_rmse={values['rmse']:.6f} "
                        f"{label}_mape={values['mape']:.6f}"
                    )
                logger.info("Val horizons | %s", " | ".join(fields))
            if float(val_result.metrics["mae"]) < best_mae:
                best_mae = float(val_result.metrics["mae"])
                stale = 0
                save_checkpoint(
                    best_path,
                    {
                        "downstream_state_dict": downstream.state_dict(),
                        "downstream_mode": config.target.downstream_mode,
                        "config": config.to_dict(),
                        "normalizer": data.scaler.state_dict(),
                        "bank_manifest": bank.manifest.to_dict(),
                        "pretrained_fingerprint": pretrained.retrieval_fingerprint(),
                        "metrics": record,
                        "epoch": epoch,
                        "seed": config.runtime.seed,
                    },
                )
                logger.info(
                    "Checkpoint updated | epoch=%d | best_val_mae=%.6f | path=%s",
                    epoch,
                    best_mae,
                    best_path,
                )
            else:
                stale += 1
                if stale >= config.target.patience:
                    logger.info(
                        "Early stopping | epoch=%d | stale_epochs=%d | best_val_mae=%.6f",
                        epoch,
                        stale,
                        best_mae,
                    )
                    break
        logger.info(
            "Downstream training finished | best_val_mae=%.6f | checkpoint=%s",
            best_mae,
            best_path,
        )
    return best_path


def evaluate_downstream(
    config: ExperimentConfig,
    pretrained_checkpoint: str | Path,
    downstream_checkpoint: str | Path,
    bank_path: str | Path,
    split: str = "test",
    max_batches: int | None = None,
) -> TargetEpochResult:
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    pretrained, _ = load_pretrained_model(config, pretrained_checkpoint, data.series.slots_per_day, device)
    checkpoint = load_checkpoint(downstream_checkpoint, device)
    mode = checkpoint_downstream_mode(checkpoint)
    config = replace(config, target=replace(config.target, downstream_mode=mode))
    downstream = build_downstream_model(config).to(device)
    downstream.load_state_dict(checkpoint["downstream_state_dict"], strict=True)
    dataset: Dataset = getattr(data, split)
    loader = DataLoader(dataset, batch_size=config.target.batch_size, shuffle=False)
    with MemoryBank(bank_path) as bank:
        _validate_bank(bank, pretrained, graph_cpu, data.scaler.state_dict())
        if checkpoint.get("bank_manifest") != bank.manifest.to_dict():
            raise ValueError("Evaluation bank differs from the bank used for downstream training")
        retriever = TwoStageRetriever(
            bank,
            config.bank.event_top_r,
            config.bank.node_top_k,
            config.bank.level_weight,
            config.bank.level_temperature,
            config.bank.search_temperature,
            device,
        )
        return run_target_epoch(
            pretrained, downstream, retriever, bank, data, loader, graph, config, data.scaler, device, None, max_batches
        )

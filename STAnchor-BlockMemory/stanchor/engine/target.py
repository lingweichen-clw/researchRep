"""Target calibration, safe fusion training, and evaluation."""

from __future__ import annotations

import json
import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from stanchor.bank.storage import MemoryBank
from stanchor.config import (
    FULL_TRAIN,
    POSTHOC_FROZEN_BASE,
    ExperimentConfig,
    resolve_project_path,
)
from stanchor.data.graph import GraphData
from stanchor.data.normalization import NodeStandardScaler
from stanchor.losses.downstream import compute_downstream_loss
from stanchor.metrics import ForecastMetricAccumulator, select_common_horizon_metrics
from stanchor.modes import (
    BASE_ONLY,
    LEARNED_TOPK_CONFIDENCE,
    LEARNED_TOPK_ERROR_AWARE,
    LEARNED_TOPK_HORIZON,
    LEARNED_TOPK_OFFSET_DECAY_HORIZON,
    RAW_L1_TOPK_HORIZON,
    WEEKLY_MEAN_HORIZON,
    validate_downstream_mode,
)
from stanchor.models.downstream import (
    ConfidenceHead,
    HorizonAwareAggregationHead,
    LightweightForecastBackbone,
    SafeResidualFusion,
    STAnchorDownstreamModel,
    StructuredErrorCorrector,
    CandidateSetHorizonCorrector,
    LegacyCandidateSetHorizonCorrector,
)
from stanchor.models.trajectory_calibrator import (
    TrajectoryConditionedCandidateSetHorizonCorrector,
    TransformerCandidateRouter,
)
from stanchor.models.pretraining import STAnchorPretrainModel
from stanchor.models.stgcn import STGCNForecastBackbone
from stanchor.models.graph_wavenet import GraphWaveNetForecastBackbone
from stanchor.models.baseline import ARGCNForecastBackbone, STAEformerForecastBackbone
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates, TwoStageRetriever
from stanchor.retrieval.strategies import (
    calendar_event_candidates,
    offset_decay_aggregation,
    raw_l1_topk_aggregation,
    validate_candidate_protocol,
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
    risk_loss: float = 0.0
    blend_loss: float = 0.0
    candidate_quality_loss: float = 0.0


@dataclass(frozen=True)
class FrozenPathEntry:
    base_prediction: torch.Tensor
    candidates: NodeCandidates | None
    aggregation: AggregationOutput | None


def _node_candidates_device(value: NodeCandidates | None, device: torch.device) -> NodeCandidates | None:
    if value is None:
        return None
    return NodeCandidates(**{name: tensor.to(device) for name, tensor in value.__dict__.items()})


def _aggregation_device(value: AggregationOutput | None, device: torch.device) -> AggregationOutput | None:
    if value is None:
        return None
    return AggregationOutput(**{name: tensor.to(device) for name, tensor in value.__dict__.items()})


def _freeze_path_entry(
    base_prediction: torch.Tensor,
    candidates: NodeCandidates | None,
    aggregation: AggregationOutput | None,
) -> FrozenPathEntry:
    cpu = torch.device("cpu")
    return FrozenPathEntry(
        base_prediction=base_prediction.detach().to(cpu),
        candidates=_node_candidates_device(candidates, cpu),
        aggregation=_aggregation_device(aggregation, cpu),
    )


def _split_frozen_path(
    entry: FrozenPathEntry, sample_ids: torch.Tensor
) -> dict[int, FrozenPathEntry]:
    """Split a batch path into per-sample CPU cache entries."""
    ids = [int(value) for value in sample_ids.detach().cpu().tolist()]
    result: dict[int, FrozenPathEntry] = {}
    for index, sample_id in enumerate(ids):
        candidates = None
        if entry.candidates is not None:
            candidates = NodeCandidates(**{
                name: tensor[index:index + 1].detach().cpu()
                for name, tensor in entry.candidates.__dict__.items()
            })
        aggregation = None
        if entry.aggregation is not None:
            aggregation = AggregationOutput(**{
                name: tensor[index:index + 1].detach().cpu()
                for name, tensor in entry.aggregation.__dict__.items()
            })
        result[sample_id] = FrozenPathEntry(
            base_prediction=entry.base_prediction[index:index + 1].detach().cpu(),
            candidates=candidates,
            aggregation=aggregation,
        )
    return result


def _merge_frozen_paths(
    entries: list[FrozenPathEntry], device: torch.device
) -> FrozenPathEntry:
    """Merge per-sample cache entries in current batch order."""
    if not entries:
        raise ValueError("cannot merge an empty frozen path")
    candidates = None
    if entries[0].candidates is not None:
        candidates = NodeCandidates(**{
            name: torch.cat(
                [getattr(entry.candidates, name) for entry in entries], dim=0
            ).to(device)
            for name in entries[0].candidates.__dict__
        })
    aggregation = None
    if entries[0].aggregation is not None:
        aggregation = AggregationOutput(**{
            name: torch.cat(
                [getattr(entry.aggregation, name) for entry in entries], dim=0
            ).to(device)
            for name in entries[0].aggregation.__dict__
        })
    return FrozenPathEntry(
        base_prediction=torch.cat(
            [entry.base_prediction for entry in entries], dim=0
        ).to(device),
        candidates=candidates,
        aggregation=aggregation,
    )


def build_downstream_model(
    config: ExperimentConfig,
    graph: GraphData | None = None,
) -> STAnchorDownstreamModel:
    error_aware = config.target.downstream_mode == LEARNED_TOPK_ERROR_AWARE
    if config.target.backbone_name == "lightweight":
        backbone = LightweightForecastBackbone(
            context_length=config.data.context_length,
            horizon=config.data.horizon,
            input_channels=config.model.input_channels,
            output_channels=config.model.output_channels,
            hidden_dim=config.target.backbone_hidden_dim,
            dropout=config.model.dropout,
        )
    elif config.target.backbone_name == "stgcn":
        if graph is None:
            raise ValueError("stgcn backbone construction requires graph data")
        backbone = STGCNForecastBackbone(
            context_length=config.data.context_length,
            horizon=config.data.horizon,
            input_channels=config.model.input_channels,
            output_channels=config.model.output_channels,
            graph=graph,
            temporal_kernel=config.target.stgcn_temporal_kernel,
            graph_kernel=config.target.stgcn_graph_kernel,
            block_num=config.target.stgcn_block_num,
            hidden_channels=config.target.stgcn_hidden_channels,
            bottleneck_channels=config.target.stgcn_bottleneck_channels,
            output_hidden_channels=config.target.stgcn_output_hidden_channels,
            dropout=config.target.stgcn_dropout,
        )
    elif config.target.backbone_name == "graph_wavenet":
        if graph is None:
            raise ValueError("graph_wavenet backbone construction requires graph data")
        backbone = GraphWaveNetForecastBackbone(
            context_length=config.data.context_length,
            horizon=config.data.horizon,
            input_channels=config.model.input_channels,
            output_channels=config.model.output_channels,
            graph=graph,
            residual_channels=config.target.graph_wavenet_residual_channels,
            dilation_channels=config.target.graph_wavenet_dilation_channels,
            skip_channels=config.target.graph_wavenet_skip_channels,
            end_channels=config.target.graph_wavenet_end_channels,
            kernel_size=config.target.graph_wavenet_kernel_size,
            blocks=config.target.graph_wavenet_blocks,
            layers=config.target.graph_wavenet_layers,
            dropout=config.target.graph_wavenet_dropout,
            adaptive_adj=config.target.graph_wavenet_adaptive_adj,
            adaptive_dim=config.target.graph_wavenet_adaptive_dim,
        )
    elif config.target.backbone_name in {"argcn", "staeformer"}:
        if graph is None:
            raise ValueError("baseline backbone construction requires graph data")
        adjacency = torch.zeros((graph.num_nodes, graph.num_nodes), device=graph.edge_index.device)
        target, source = graph.edge_index
        adjacency[target, source] = graph.edge_weight.float()
        adjacency.fill_diagonal_(0.0)
        support = (adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1e-8)).unsqueeze(0)
        if config.target.backbone_name == "argcn":
            backbone = ARGCNForecastBackbone(config.data.context_length, config.data.horizon,
                graph.num_nodes, config.model.input_channels, config.model.output_channels,
                support, hidden_dim=128, num_layers=1, cheb_k=3)
        else:
            backbone = STAEformerForecastBackbone(config.data.context_length, config.data.horizon,
                graph.num_nodes, config.model.input_channels, config.model.output_channels,
                steps_per_day=288, input_embedding_dim=24, tod_embedding_dim=24,
                dow_embedding_dim=24, spatial_embedding_dim=0,
                adaptive_embedding_dim=80, feed_forward_dim=256,
                heads=4, layers=3, dropout=0.1)
    else:
        raise ValueError(f"unsupported downstream backbone: {config.target.backbone_name}")
    error_corrector = None
    if error_aware:
        if config.target.validation_correction_variant == "base_as_candidate":
            if config.target.calibrator_arch == "transformer_candidate_router":
                error_corrector = TransformerCandidateRouter(
                    config.data.context_length,
                    config.data.horizon,
                    config.model.input_channels,
                    hidden_dim=config.target.candidate_token_dim,
                    state_dim=config.target.calibrator_state_dim,
                    attention_heads=config.target.candidate_attention_heads,
                    base_logit_init_bias=config.target.base_logit_init_bias,
                    trajectory_hidden_dim=config.target.candidate_trajectory_hidden_dim,
                    routing_hidden_dim=config.target.routing_hidden_dim,
                    mha_dropout=config.target.mha_dropout,
                    use_horizon_embedding=config.target.use_horizon_embedding,
                )
            elif config.target.calibrator_arch == "trajectory_conditioned_base_as_candidate":
                error_corrector = TrajectoryConditionedCandidateSetHorizonCorrector(
                    config.data.context_length,
                    config.data.horizon,
                    config.model.input_channels,
                    hidden_dim=config.target.candidate_token_dim,
                    state_dim=config.target.calibrator_state_dim,
                    attention_heads=config.target.candidate_attention_heads,
                    base_logit_init_bias=config.target.base_logit_init_bias,
                    trajectory_hidden_dim=config.target.candidate_trajectory_hidden_dim,
                    use_horizon_embedding=config.target.use_horizon_embedding,
                )
            else:
                error_corrector = CandidateSetHorizonCorrector(
                    config.data.context_length,
                    config.data.horizon,
                    config.model.input_channels,
                    hidden_dim=config.target.candidate_token_dim,
                    state_dim=config.target.calibrator_state_dim,
                    attention_heads=config.target.candidate_attention_heads,
                    base_logit_init_bias=config.target.base_logit_init_bias,
                )
        elif config.target.validation_correction_variant == "set_attention_horizon":
            error_corrector = LegacyCandidateSetHorizonCorrector(
                config.data.context_length,
                config.data.horizon,
                config.model.input_channels,
                hidden_dim=192,
                state_dim=128,
            )
        else:
            error_corrector = StructuredErrorCorrector(
                config.data.context_length,
                config.data.horizon,
                config.model.input_channels,
                risk_hidden_dim=config.target.risk_hidden_dim,
                evidence_hidden_dim=config.target.fusion_feature_hidden_dim,
                num_features=9,
                initial_weight=0.1,
                correction_variant=config.target.validation_correction_variant,
            )
    return STAnchorDownstreamModel(
        backbone=backbone,
        confidence_head=ConfidenceHead(config.target.confidence_hidden_dim),
        fusion=SafeResidualFusion(config.data.horizon),
        confidence_level_temperature=config.target.confidence_level_temperature,
        mode=config.target.downstream_mode,
        horizon_aggregator=(
            HorizonAwareAggregationHead(hidden_dim=config.target.horizon_aggregation_hidden_dim)
            if error_aware and config.target.validation_correction_variant not in {"set_attention_horizon", "base_as_candidate"}
            else None
        ),
        error_corrector=error_corrector,
    )

def validate_downstream_bank_path(
    mode: str,
    bank_path: str | Path | None,
) -> Path | None:
    """Return the Bank path for retrieval modes, or ``None`` for base-only.

    A base-only backbone does not execute retrieval and therefore must remain
    runnable on machines where the large historical Bank is unavailable.
    Retrieval modes keep the existing explicit ``--bank`` requirement.
    """
    mode = validate_downstream_mode(mode)
    if mode == BASE_ONLY:
        return None
    if bank_path is None:
        raise ValueError(f"downstream mode {mode!r} requires --bank")
    return resolve_project_path(bank_path)


def validate_evaluation_bank_path(
    mode: str,
    bank_path: str | Path | None,
) -> Path | None:
    """Resolve an evaluation Bank only for modes that execute retrieval."""
    return validate_downstream_bank_path(mode, bank_path)


def validate_downstream_pretrained_checkpoint(
    mode: str,
    checkpoint_path: str | Path | None,
) -> Path | None:
    """Require the retrieval encoder checkpoint only when retrieval is used."""
    mode = validate_downstream_mode(mode)
    if mode == BASE_ONLY:
        return None
    if checkpoint_path is None:
        raise ValueError(
            f"downstream mode {mode!r} requires --pretrained-checkpoint"
        )
    return resolve_project_path(checkpoint_path)


def _state_dict_fingerprint(module: torch.nn.Module) -> str:
    """Return a stable short hash for an initialized module state."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def checkpoint_downstream_mode(checkpoint: dict) -> str:
    mode = checkpoint.get("downstream_mode")
    if mode is None:
        target_config = checkpoint.get("config", {}).get("target", {})
        mode = target_config.get("downstream_mode", LEARNED_TOPK_CONFIDENCE)
    return validate_downstream_mode(str(mode))


def checkpoint_candidate_protocol(
    checkpoint: dict,
    expected: str | None = None,
) -> str:
    """Read the candidate protocol saved with a downstream checkpoint."""
    protocol = checkpoint.get("candidate_protocol")
    if protocol is None:
        target_config = checkpoint.get("config", {}).get("target", {})
        protocol = target_config.get("candidate_protocol", "exact_calendar")
    protocol = validate_candidate_protocol(str(protocol))
    if expected is not None and protocol != validate_candidate_protocol(expected):
        raise ValueError(
            f"candidate protocol {expected!r} differs from checkpoint {protocol!r}"
        )
    return protocol


def checkpoint_bank_level_weight(checkpoint: dict, default: float) -> float:
    """Restore the node reranking level weight used during downstream training."""
    bank_config = checkpoint.get("config", {}).get("bank", {})
    value = float(bank_config.get("level_weight", default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("checkpoint bank level_weight must be finite and non-negative")
    return value


def load_frozen_base_backbone(
    downstream: STAnchorDownstreamModel,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, str]:
    """Load only a verified base-only backbone and freeze it."""
    resolved_path = resolve_project_path(checkpoint_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"base checkpoint does not exist: {resolved_path}")
    checkpoint = load_checkpoint(resolved_path, device)
    if checkpoint_downstream_mode(checkpoint) != BASE_ONLY:
        raise ValueError("posthoc base checkpoint must use base_only mode")
    state = checkpoint.get("downstream_state_dict")
    if not isinstance(state, dict):
        raise ValueError("base checkpoint has no downstream_state_dict")
    prefix = "backbone."
    backbone_state = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError("base checkpoint has no backbone parameters")
    try:
        downstream.backbone.load_state_dict(backbone_state, strict=True)
    except RuntimeError as error:
        raise ValueError("base checkpoint has an incompatible backbone") from error
    for parameter in downstream.backbone.parameters():
        parameter.requires_grad_(False)
    return {
        "path": str(resolved_path),
        "fingerprint": _state_dict_fingerprint(downstream.backbone),
    }


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
    if mode != LEARNED_TOPK_ERROR_AWARE:
        if downstream.error_corrector is not None:
            for parameter in downstream.error_corrector.parameters():
                parameter.requires_grad_(False)
    if mode == BASE_ONLY:
        for parameter in downstream.fusion.parameters():
            parameter.requires_grad_(False)
    if mode == LEARNED_TOPK_ERROR_AWARE:
        for parameter in downstream.fusion.parameters():
            parameter.requires_grad_(False)


def select_downstream_training_dataset(
    train_dataset: Dataset,
    config: ExperimentConfig,
) -> tuple[Dataset, int]:
    """Select the full baseline data or the retrieval calibration partition."""
    memory_events = int(len(train_dataset) * config.bank.memory_fraction)
    if config.target.training_data_scope == FULL_TRAIN:
        return train_dataset, memory_events
    return (
        Subset(train_dataset, range(memory_events, len(train_dataset))),
        memory_events,
    )


def build_target_optimizer(
    config: ExperimentConfig,
    parameter_groups: list[dict],
) -> torch.optim.Optimizer:
    optimizer_class = (
        torch.optim.Adam
        if config.target.optimizer_name == "adam"
        else torch.optim.AdamW
    )
    return optimizer_class(
        parameter_groups,
        lr=config.target.learning_rate,
        weight_decay=config.target.weight_decay,
    )


def build_target_scheduler(
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
):
    if config.target.scheduler_name == "none":
        return None
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.target.scheduler_step_size,
        gamma=config.target.scheduler_gamma,
    )


def should_stop_target_stage(
    stale_epochs: int,
    patience: int,
    enabled: bool = True,
) -> bool:
    """Return whether target-stage early stopping should terminate training."""
    if stale_epochs < 0 or patience <= 0:
        raise ValueError("stale_epochs must be non-negative and patience must be positive")
    return bool(enabled and stale_epochs >= patience)


def configure_error_aware_stage(
    downstream: STAnchorDownstreamModel,
    stage: str,
) -> list[dict]:
    """Configure base/calibrator/joint trainability and optimizer groups."""
    if stage not in {"base", "calibrator", "posthoc_calibrator", "joint"}:
        raise ValueError(
            "error-aware stage must be base, calibrator, posthoc_calibrator, or joint"
        )
    for parameter in downstream.parameters():
        parameter.requires_grad_(False)
    if stage in {"base", "joint"}:
        for parameter in downstream.backbone.parameters():
            parameter.requires_grad_(True)
    if stage in {"calibrator", "posthoc_calibrator", "joint"}:
        if downstream.error_corrector is None:
            raise ValueError("error-aware stages require StructuredErrorCorrector")
        for parameter in downstream.error_corrector.parameters():
            parameter.requires_grad_(True)
        if downstream.horizon_aggregator is not None:
            for parameter in downstream.horizon_aggregator.parameters():
                parameter.requires_grad_(True)
    groups = []
    backbone_parameters = [
        parameter for parameter in downstream.backbone.parameters() if parameter.requires_grad
    ]
    calibrator_parameters = [
        parameter for parameter in downstream.error_corrector.parameters()
        if downstream.error_corrector is not None and parameter.requires_grad
    ]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "role": "backbone"})
    if calibrator_parameters:
        groups.append({"params": calibrator_parameters, "role": "calibrator"})
    aggregation_parameters = (
        [parameter for parameter in downstream.horizon_aggregator.parameters() if parameter.requires_grad]
        if downstream.horizon_aggregator is not None
        else []
    )
    if aggregation_parameters:
        groups.append({"params": aggregation_parameters, "role": "aggregation"})
    return groups


def build_downstream_training_stages(
    config: ExperimentConfig,
    base_checkpoint_path: str | Path | None,
) -> list[tuple[str, int]]:
    """Return an explicit stage schedule for the selected target protocol."""
    if config.target.training_protocol == POSTHOC_FROZEN_BASE:
        if base_checkpoint_path is None:
            raise ValueError("posthoc_frozen_base requires a base checkpoint")
        return [("posthoc_calibrator", config.target.epochs)]
    if base_checkpoint_path is not None:
        raise ValueError(
            "a base checkpoint can only be used with posthoc_frozen_base"
        )
    if config.target.downstream_mode == LEARNED_TOPK_ERROR_AWARE:
        stages: list[tuple[str, int]] = []
        if config.target.base_warmup_epochs > 0:
            stages.append(("base", config.target.base_warmup_epochs))
        if config.target.calibrator_warmup_epochs > 0:
            stages.append(("calibrator", config.target.calibrator_warmup_epochs))
        stages.append(("joint", config.target.epochs))
        return stages
    return [("standard", config.target.epochs)]


@torch.no_grad()
def retrieve_for_downstream_mode(
    mode: str,
    pretrained: STAnchorPretrainModel | None,
    retriever: TwoStageRetriever | None,
    bank: MemoryBank | None,
    data,
    graph: GraphData,
    batch: dict[str, torch.Tensor],
    x: torch.Tensor,
    observed_x: torch.Tensor,
    device: torch.device,
    candidate_protocol: str = "exact_calendar",
) -> tuple[NodeCandidates | None, AggregationOutput | None]:
    mode = validate_downstream_mode(mode)
    candidate_protocol = validate_candidate_protocol(candidate_protocol)
    if mode == BASE_ONLY:
        return None, None
    if pretrained is None or retriever is None or bank is None:
        raise ValueError(f"downstream mode {mode!r} requires retrieval assets")
    if mode in {LEARNED_TOPK_CONFIDENCE, LEARNED_TOPK_ERROR_AWARE}:
        encoding = pretrained.encode_clean(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            batch["retrieval_weekday"].to(device),
            batch["retrieval_slot"].to(device),
            graph,
        )
        if candidate_protocol == "exact_calendar":
            # The legal exact-calendar pool is already small (about eight
            # events/query on METR-LA).  Event keys are mean-pooled across
            # nodes and are not independently supervised, so they are
            # intentionally bypassed here; node keys perform the actual Top-K
            # selection over the complete legal pool.
            events = calendar_event_candidates(
                bank,
                weekday=batch["query_weekday"].to(device),
                slot=batch["query_slot"].to(device),
                context_start=batch["context_start"].to(device),
                max_candidates=retriever.event_top_r,
                device=device,
                candidate_protocol=candidate_protocol,
            )
            candidates = retriever.rerank_nodes(
                encoding.retrieval.node_keys,
                encoding.statistics.level_features,
                events,
            )
            aggregation = retriever.aggregate(candidates)
        else:
            events = calendar_event_candidates(
                bank,
                weekday=batch["query_weekday"].to(device),
                slot=batch["query_slot"].to(device),
                context_start=batch["context_start"].to(device),
                max_candidates=retriever.event_top_r,
                device=device,
                candidate_protocol=candidate_protocol,
            )
            candidates = retriever.rerank_nodes(
                encoding.retrieval.node_keys,
                encoding.statistics.level_features,
                events,
            )
            aggregation = retriever.aggregate(candidates)
        if mode == LEARNED_TOPK_ERROR_AWARE:
            aggregation = offset_decay_aggregation(
                candidates,
                x,
                observed_x,
                bank,
                data.series,
                data.scaler,
                data.train.context_length,
                device,
            )
        return candidates, aggregation

    events = calendar_event_candidates(
        bank,
        weekday=batch["query_weekday"].to(device),
        slot=batch["query_slot"].to(device),
        context_start=batch["context_start"].to(device),
        max_candidates=retriever.event_top_r,
        device=device,
        candidate_protocol=candidate_protocol,
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
    expected_schema = 2 if pretrained.model_config.profile_dim > 0 else 1
    if bank.manifest.schema_version != expected_schema:
        raise ValueError(
            f"Bank schema version {bank.manifest.schema_version} does not match "
            f"retrieval model expected schema {expected_schema}"
        )
    if expected_schema == 2:
        if bank.manifest.key_layout != "canonical_profile_latent":
            raise ValueError("profile-enabled retrieval requires canonical_profile_latent Bank")
        if bank.manifest.profile_dim != pretrained.model_config.profile_dim:
            raise ValueError("Bank profile_dim does not match retrieval model")
        if bank.manifest.latent_dim != pretrained.model_config.latent_dim:
            raise ValueError("Bank latent_dim does not match retrieval model")
        if not np.isclose(
            bank.manifest.profile_weight, pretrained.model_config.profile_weight
        ):
            raise ValueError("Bank profile_weight does not match retrieval model")
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
    pretrained: STAnchorPretrainModel | None,
    downstream: STAnchorDownstreamModel,
    retriever: TwoStageRetriever | None,
    bank: MemoryBank | None,
    data,
    loader: DataLoader,
    graph: GraphData,
    config: ExperimentConfig,
    scaler: NodeStandardScaler,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
    frozen_cache: dict[int, FrozenPathEntry] | None = None,
) -> TargetEpochResult:
    training = optimizer is not None
    downstream.train(training)
    if pretrained is not None:
        pretrained.eval()
    total = forecast = confidence = risk = blend = candidate_quality = 0.0
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
            sample_ids = batch["sample_id"]
            cached_entries = (
                [frozen_cache[int(value)] for value in sample_ids.detach().cpu().tolist()]
                if frozen_cache is not None
                and all(int(value) in frozen_cache for value in sample_ids.detach().cpu().tolist())
                else None
            )
            if cached_entries is not None:
                entry = _merge_frozen_paths(cached_entries, device)
                node_candidates = entry.candidates
                aggregation = entry.aggregation
                base_prediction = entry.base_prediction
            else:
                node_candidates, aggregation = retrieve_for_downstream_mode(
                    config.target.downstream_mode, pretrained, retriever, bank, data, graph,
                    batch, x, observed_x, device,
                    candidate_protocol=config.target.candidate_protocol,
                )
                def _backbone_forward(inp):
                    if config.target.backbone_name == "staeformer":
                        if config.target.staeformer_time_feature_mode == "fallback":
                            return downstream.backbone(inp)
                        return downstream.backbone(
                            inp,
                            tod=batch["slot"].to(device),
                            dow=batch["weekday"].to(device),
                        )
                    return downstream.backbone(inp)

                if config.target.training_protocol == POSTHOC_FROZEN_BASE:
                    with torch.no_grad():
                        base_prediction = _backbone_forward(x)
                else:
                    base_prediction = _backbone_forward(x)
                if frozen_cache is not None:
                    frozen_cache.update(
                        _split_frozen_path(
                            _freeze_path_entry(base_prediction, node_candidates, aggregation),
                            sample_ids,
                        )
                    )
            output = downstream(
                x, node_candidates, aggregation, base_override=base_prediction
            )
            losses = compute_downstream_loss(
                output,
                target=batch["y"].to(device),
                observed=batch["y_observed"].to(device),
                confidence_weight=config.target.confidence_weight,
                help_margin=config.target.help_margin,
                help_temperature=config.target.help_temperature,
                use_confidence=config.target.downstream_mode == LEARNED_TOPK_CONFIDENCE,
                use_error_aware=config.target.downstream_mode == LEARNED_TOPK_ERROR_AWARE,
                risk_weight=config.target.risk_weight,
                blend_weight=config.target.blend_weight,
                blend_minimum_direction_norm=(
                    config.target.blend_minimum_direction_norm
                ),
                loss_variant=config.target.validation_loss_variant,
                candidate_quality_weight=(config.target.candidate_quality_weight if training else 0.0),
                candidate_quality_temperature=config.target.candidate_quality_temperature,
            )
            require_finite(losses.total, "downstream loss")
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(downstream.parameters(), max_norm=5.0)
                optimizer.step()
            total += float(losses.total.detach())
            forecast += float(losses.forecast.detach())
            confidence += float(losses.confidence.detach())
            risk += float((losses.risk if losses.risk is not None else losses.total * 0.0).detach())
            blend += float((losses.blend if losses.blend is not None else losses.total * 0.0).detach())
            candidate_quality += float((losses.candidate_quality if losses.candidate_quality is not None else losses.total * 0.0).detach())
            target_model = batch["y"].to(device)
            metrics.update(
                scaler.inverse_transform_torch(output.final_prediction.detach()),
                scaler.inverse_transform_torch(target_model),
                batch["y_observed"].to(device),
            )
            batches += 1
    if batches == 0:
        raise ValueError("target epoch processed no batches")
    return TargetEpochResult(
        total / batches,
        forecast / batches,
        confidence / batches,
        metrics.compute(),
        batches,
        risk / batches,
        blend / batches,
        candidate_quality / batches,
    )


def train_downstream(
    config: ExperimentConfig,
    pretrained_checkpoint: str | Path | None = None,
    bank_path: str | Path | None = None,
    base_checkpoint_path: str | Path | None = None,
    max_batches: int | None = None,
) -> Path:
    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    resolved_pretrained_checkpoint = validate_downstream_pretrained_checkpoint(
        config.target.downstream_mode,
        pretrained_checkpoint,
    )
    if resolved_pretrained_checkpoint is None:
        pretrained = None
    else:
        pretrained, _ = load_pretrained_model(
            config,
            resolved_pretrained_checkpoint,
            data.series.slots_per_day,
            device,
        )
        for parameter in pretrained.parameters():
            parameter.requires_grad_(False)
    resolved_bank_path = validate_downstream_bank_path(
        config.target.downstream_mode,
        bank_path,
    )
    bank_context = (
        MemoryBank(
            resolved_bank_path,
            expected_schema_version=(
                2
                if pretrained is not None
                and pretrained.model_config.profile_dim > 0
                else 1
            ),
        )
        if resolved_bank_path is not None
        else nullcontext(None)
    )
    with bank_context as bank:
        if bank is not None:
            if pretrained is None:
                raise ValueError("Bank retrieval requires a pretrained encoder")
            _validate_bank(bank, pretrained, graph_cpu, data.scaler.state_dict())
            retriever: TwoStageRetriever | None = TwoStageRetriever(
                bank,
                config.bank.event_top_r,
                config.bank.node_top_k,
                config.bank.level_weight,
                config.bank.level_temperature,
                config.bank.search_temperature,
                device,
            )
        else:
            retriever = None
        training_dataset, memory_events = select_downstream_training_dataset(
            data.train, config
        )
        train_loader = DataLoader(
            training_dataset,
            batch_size=config.target.batch_size,
            shuffle=not config.target.fixed_batch_order,
            num_workers=config.data.num_workers,
        )
        val_loader = DataLoader(
            data.val,
            batch_size=config.target.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        train_cache = {} if config.target.frozen_path_cache else None
        val_cache = {} if config.target.frozen_path_cache else None
        # Loading a frozen encoder consumes RNG state. Reset before constructing
        # the downstream model so encoder variants share identical initialization.
        set_seed(config.runtime.seed)
        downstream = build_downstream_model(config, graph).to(device)
        configure_downstream_trainable(downstream, config.target.downstream_mode)
        base_provenance = None
        if config.target.training_protocol == POSTHOC_FROZEN_BASE:
            if base_checkpoint_path is None:
                raise ValueError("posthoc_frozen_base requires a base checkpoint")
            base_provenance = load_frozen_base_backbone(
                downstream, base_checkpoint_path, device
            )
        downstream_init_hash = _state_dict_fingerprint(downstream)
        run_dir = resolve_project_path(config.runtime.output_dir) / config.runtime.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_path = run_dir / "downstream_best.pt"
        metrics_path = run_dir / "target_metrics.jsonl"
        logger = create_run_logger(
            f"stanchor.downstream.{config.runtime.run_name}",
            run_dir / "downstream.log",
        )
        parameter_counts = {
            "frozen_pretrained": (
                count_parameters(pretrained, trainable_only=False)
                if pretrained is not None
                else 0
            ),
            "backbone": count_parameters(downstream.backbone),
            "confidence_head": count_parameters(downstream.confidence_head),
            "fusion": count_parameters(downstream.fusion),
            "structured_error_corrector": (
                count_parameters(downstream.error_corrector)
                if downstream.error_corrector is not None
                else 0
            ),
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
        logger.info(
            "Mode | downstream_mode=%s | training_protocol=%s | candidate_protocol=%s",
            config.target.downstream_mode,
            config.target.training_protocol,
            config.target.candidate_protocol,
        )
        if config.target.backbone_name == "staeformer":
            logger.info(
                "STAEformer temporal covariates | mode=%s | source=%s",
                config.target.staeformer_time_feature_mode,
                "dataset slot/weekday" if config.target.staeformer_time_feature_mode == "calendar" else "backbone fallback",
            )
        logger.info(
            "Downstream initialization | seed=%d | state_hash=%s",
            config.runtime.seed,
            downstream_init_hash,
        )
        if base_provenance is not None:
            logger.info(
                "Frozen base | checkpoint=%s | fingerprint=%s",
                base_provenance["path"],
                base_provenance["fingerprint"],
            )
        logger.info(
            "Data | steps=%d | nodes=%d | channels=%d | training_scope=%s | "
            "memory/post_memory/train_selected/val/test=%d/%d/%d/%d/%d",
            data.series.num_steps,
            data.series.num_nodes,
            data.series.num_channels,
            config.target.training_data_scope,
            memory_events,
            len(data.train) - memory_events,
            len(training_dataset),
            len(data.val),
            len(data.test),
        )
        if bank is None:
            logger.info("Bank/retrieval | disabled (base_only)")
        else:
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
            "Optimization | epochs=%d | batch_size=%d | optimizer=%s | lr=%.3g | "
            "weight_decay=%.3g | scheduler=%s | step_size=%d | gamma=%.3f | "
            "confidence_weight=%.3f | patience=%d | early_stopping=%s",
            config.target.epochs,
            config.target.batch_size,
            config.target.optimizer_name,
            config.target.learning_rate,
            config.target.weight_decay,
            config.target.scheduler_name,
            config.target.scheduler_step_size,
            config.target.scheduler_gamma,
            config.target.confidence_weight,
            config.target.patience,
            "enabled" if config.target.early_stopping_enabled else "disabled",
        )
        stages = build_downstream_training_stages(config, base_checkpoint_path)

        best_mae = float("inf")
        global_epoch = 0
        for stage, stage_epochs in stages:
            if stage == "standard":
                downstream.mode = config.target.downstream_mode
                configure_downstream_trainable(downstream, config.target.downstream_mode)
                parameter_groups = [
                    {
                        "params": [
                            parameter
                            for parameter in downstream.parameters()
                            if parameter.requires_grad
                        ],
                        "lr": config.target.learning_rate,
                    }
                ]
                stage_config = config
            else:
                groups = configure_error_aware_stage(downstream, stage)
                parameter_groups = []
                for group in groups:
                    learning_rate = config.target.learning_rate
                    if stage == "joint" and group["role"] == "backbone":
                        learning_rate *= config.target.backbone_learning_rate_scale
                    parameter_groups.append(
                        {"params": group["params"], "lr": learning_rate}
                    )
                if stage == "base":
                    downstream.mode = BASE_ONLY
                    stage_config = replace(
                        config,
                        target=replace(config.target, downstream_mode=BASE_ONLY),
                    )
                else:
                    downstream.mode = LEARNED_TOPK_ERROR_AWARE
                    stage_config = config
            if not parameter_groups or not any(group["params"] for group in parameter_groups):
                raise ValueError(f"downstream stage {stage} has no trainable parameters")
            optimizer = build_target_optimizer(config, parameter_groups)
            scheduler = build_target_scheduler(config, optimizer)
            logger.info(
                "Stage start | stage=%s | epochs=%d | optimizer_lrs=%s",
                stage,
                stage_epochs,
                [group["lr"] for group in optimizer.param_groups],
            )
            stale = 0
            for stage_epoch in range(1, stage_epochs + 1):
                global_epoch += 1
                epoch = global_epoch
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                epoch_started = time.perf_counter()
                epoch_learning_rate = [
                    group["lr"] for group in optimizer.param_groups
                ]
                train_result = run_target_epoch(
                    pretrained, downstream, retriever, bank, data, train_loader, graph,
                    stage_config, data.scaler, device, optimizer, max_batches,
                    frozen_cache=train_cache,
                )
                if base_provenance is not None:
                    current_base_fingerprint = _state_dict_fingerprint(
                        downstream.backbone
                    )
                    if current_base_fingerprint != base_provenance["fingerprint"]:
                        raise RuntimeError(
                            "posthoc frozen base backbone changed during training"
                        )
                val_result = run_target_epoch(
                    pretrained, downstream, retriever, bank, data, val_loader, graph,
                    stage_config, data.scaler, device, None, max_batches,
                    frozen_cache=val_cache,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    cuda_peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                else:
                    cuda_peak_mb = 0.0
                epoch_seconds = time.perf_counter() - epoch_started
                if scheduler is not None:
                    scheduler.step()
                record = {
                    "epoch": epoch,
                    "stage": stage,
                    "stage_epoch": stage_epoch,
                    "train": train_result.__dict__,
                    "val": val_result.__dict__,
                    "parameter_counts": parameter_counts,
                    "epoch_seconds": epoch_seconds,
                    "learning_rates": epoch_learning_rate,
                    "cuda_peak_allocated_mb": cuda_peak_mb,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(
                    "Epoch %03d | stage=%s | train_total=%.6f | train_forecast=%.6f | "
                    "train_confidence=%.6f | train_risk=%.6f | train_blend=%.6f | "
                    "val_total=%.6f | val_mae=%.6f | val_rmse=%.6f | val_mape=%.6f | "
                    "val_confidence=%.6f | val_risk=%.6f | val_blend=%.6f | "
                    "batches=%d/%d | seconds=%.2f | lr=%s | cuda_peak_mb=%.1f",
                    epoch,
                    stage,
                    train_result.total_loss,
                    train_result.forecast_loss,
                    train_result.confidence_loss,
                    train_result.risk_loss,
                    train_result.blend_loss,
                    val_result.total_loss,
                    val_result.metrics["mae"],
                    val_result.metrics["rmse"],
                    val_result.metrics["mape"],
                    val_result.confidence_loss,
                    val_result.risk_loss,
                    val_result.blend_loss,
                    train_result.batches,
                    val_result.batches,
                    epoch_seconds,
                    epoch_learning_rate,
                    cuda_peak_mb,
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
                # Base warm-up is an initialization stage, not a selectable
                # error-aware checkpoint because its calibrator is untrained.
                if stage != "base" and float(val_result.metrics["mae"]) < best_mae:
                    best_mae = float(val_result.metrics["mae"])
                    stale = 0
                    save_checkpoint(
                        best_path,
                        {
                            "downstream_state_dict": downstream.state_dict(),
                            "downstream_mode": config.target.downstream_mode,
                            "training_protocol": config.target.training_protocol,
                            "candidate_protocol": config.target.candidate_protocol,
                            "training_stage": stage,
                            "base_checkpoint_provenance": base_provenance,
                            "config": config.to_dict(),
                            "normalizer": data.scaler.state_dict(),
                            "bank_manifest": (
                                bank.manifest.to_dict() if bank is not None else None
                            ),
                            "pretrained_fingerprint": (
                                pretrained.retrieval_fingerprint()
                                if pretrained is not None
                                else None
                            ),
                            "metrics": record,
                            "epoch": epoch,
                            "seed": config.runtime.seed,
                        },
                    )
                    logger.info(
                        "Checkpoint updated | epoch=%d | stage=%s | best_val_mae=%.6f | path=%s",
                        epoch,
                        stage,
                        best_mae,
                        best_path,
                    )
                elif stage != "base":
                    stale += 1
                    if should_stop_target_stage(
                        stale,
                        config.target.patience,
                        config.target.early_stopping_enabled,
                    ):
                        logger.info(
                            "Stage early stopping | epoch=%d | stage=%s | stale_epochs=%d | best_val_mae=%.6f",
                            epoch,
                            stage,
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
    candidate_protocol: str | None = None,
) -> TargetEpochResult:
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    checkpoint = load_checkpoint(downstream_checkpoint, device)
    mode = checkpoint_downstream_mode(checkpoint)
    if mode == BASE_ONLY:
        pretrained = None
    else:
        if pretrained_checkpoint is None:
            raise ValueError(f"downstream mode {mode!r} requires --pretrained-checkpoint")
        pretrained, _ = load_pretrained_model(
            config, pretrained_checkpoint, data.series.slots_per_day, device
        )
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
    dataset: Dataset = getattr(data, split)
    loader = DataLoader(dataset, batch_size=config.target.batch_size, shuffle=False)
    resolved_bank_path = validate_evaluation_bank_path(mode, bank_path)
    if resolved_bank_path is None:
        return run_target_epoch(
            pretrained, downstream, None, None, data, loader, graph, config,
            data.scaler, device, None, max_batches
        )
    with MemoryBank(
        resolved_bank_path,
        expected_schema_version=(2 if pretrained.model_config.profile_dim > 0 else 1),
    ) as bank:
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
            pretrained, downstream, retriever, bank, data, loader, graph, config,
            data.scaler, device, None, max_batches
        )














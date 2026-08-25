"""Typed configuration for all STAnchor experiment stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from stanchor.modes import (
    LEARNED_TOPK_CONFIDENCE,
    LEARNED_TOPK_ERROR_AWARE,
    validate_downstream_mode,
)
from stanchor.retrieval.strategies import validate_candidate_protocol


STAGED_JOINT = "staged_joint"
POSTHOC_FROZEN_BASE = "posthoc_frozen_base"
TARGET_TRAINING_PROTOCOLS = (STAGED_JOINT, POSTHOC_FROZEN_BASE)
POST_MEMORY_CALIBRATION = "post_memory_calibration"
FULL_TRAIN = "full_train"
TARGET_TRAINING_DATA_SCOPES = (POST_MEMORY_CALIBRATION, FULL_TRAIN)
TARGET_OPTIMIZERS = ("adamw", "adam")
TARGET_SCHEDULERS = ("none", "step_lr")


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    adjacency_path: str
    context_length: int = 12
    retrieval_context_length: int | None = None
    horizon: int = 12
    frequency_minutes: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    zero_is_missing: bool = True
    num_workers: int = 0

    @property
    def encoder_context_length(self) -> int:
        return (
            self.context_length
            if self.retrieval_context_length is None
            else self.retrieval_context_length
        )


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 1
    output_channels: int = 1
    patch_size: int = 3
    hidden_dim: int = 64
    retrieval_dim: int = 32
    num_heads: int = 4
    encoder_layers: int = 2
    ffn_multiplier: int = 2
    dropout: float = 0.1
    graph_bias: float = 1.0
    route_enabled: bool = False
    route_dim: int = 16
    route_top_k: int = 10
    route_local_quota: int = 4
    route_prior_weight: float = 0.25
    route_temperature: float = 0.1
    route_gate_bias: float = -2.0
    profile_dim: int = 0
    latent_dim: int = 0
    profile_weight: float = 0.25
    dynamics_adapter_mode: str = "none"
    dynamics_bottleneck_dim: int = 16
    dynamics_gate_bias: float = -2.0
    dynamics_gate_groups: int = 8


@dataclass(frozen=True)
class PretrainConfig:
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    objective: str = "joint"
    reconstruction_weight: float = 1.0
    validation_interval: int = 1
    time_mask_ratio: float = 0.25
    time_mask_block_size: int = 3
    space_mask_ratio: float = 0.25
    time_task_probability: float = 0.5
    retrieval_weight: float = 0.1
    retrieval_loss_mode: str = "hard_negative"
    retrieval_temperature: float = 0.1
    relation_teacher_temperature: float = 0.1
    relation_student_temperature: float = 0.1
    relation_teacher_mode: str = "context_normalized"
    relation_distance_normalization: str = "none"
    future_increment_weight: float = 0.0
    rank_loss_weight: float = 0.0
    rank_positive_count: int = 2
    rank_negative_count: int = 2
    rank_future_gap: float = 0.05
    rank_margin: float = 0.05
    rank_temperature: float = 0.1
    profile_loss_weight: float = 0.0
    profile_scale_floor: float = 0.1
    hard_negative_weight: float = 2.0
    positive_quantile: float = 0.1
    context_quantile: float = 0.2
    negative_quantile: float = 0.8
    patience: int = 10
    # Pretraining uses a fixed epoch budget by default; legacy convergence
    # experiments may explicitly opt into early stopping.
    early_stopping_enabled: bool = False
    progress_interval: int = 10


@dataclass(frozen=True)
class BankConfig:
    output_dir: str = "artifacts/bank"
    memory_fraction: float = 0.7
    event_top_r: int = 32
    node_top_k: int = 5
    level_weight: float = 0.25
    level_temperature: float = 1.0
    search_temperature: float = 0.1
    key_dtype: str = "float16"


@dataclass(frozen=True)
class TargetConfig:
    downstream_mode: str = LEARNED_TOPK_CONFIDENCE
    training_protocol: str = STAGED_JOINT
    training_data_scope: str = POST_MEMORY_CALIBRATION
    candidate_protocol: str = "exact_calendar"
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    optimizer_name: str = "adamw"
    scheduler_name: str = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.95
    confidence_hidden_dim: int = 32
    confidence_weight: float = 1.0
    confidence_level_temperature: float = 1.0
    help_margin: float = 0.0
    help_temperature: float = 0.1
    backbone_name: str = "lightweight"
    backbone_hidden_dim: int = 64
    stgcn_temporal_kernel: int = 3
    stgcn_graph_kernel: int = 3
    stgcn_block_num: int = 2
    stgcn_hidden_channels: int = 64
    stgcn_bottleneck_channels: int = 16
    stgcn_output_hidden_channels: int = 128
    stgcn_dropout: float = 0.5
    graph_wavenet_residual_channels: int = 32
    graph_wavenet_dilation_channels: int = 32
    graph_wavenet_skip_channels: int = 256
    graph_wavenet_end_channels: int = 512
    graph_wavenet_kernel_size: int = 2
    graph_wavenet_blocks: int = 4
    graph_wavenet_layers: int = 2
    graph_wavenet_dropout: float = 0.3
    graph_wavenet_adaptive_dim: int = 10
    graph_wavenet_adaptive_adj: bool = True
    patience: int = 10
    early_stopping_enabled: bool = True
    # Current mainline Structured Error Corrector widths (~158k parameters).
    # Compact post-hoc budget shared by the horizon selector and legacy corrector.
    risk_hidden_dim: int = 256
    fusion_feature_hidden_dim: int = 64
    horizon_aggregation_hidden_dim: int = 256
    risk_weight: float = 0.1
    blend_weight: float = 0.1
    blend_minimum_direction_norm: float = 1.0e-4
    validation_loss_variant: str = "forecast_risk_blend"
    validation_correction_variant: str = "scalar_gate"
    base_warmup_epochs: int = 0
    calibrator_warmup_epochs: int = 5
    backbone_learning_rate_scale: float = 0.1


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda:0"
    output_dir: str = "artifacts"
    run_name: str = "stanchor_v1"


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    bank: BankConfig = field(default_factory=BankConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        validate_downstream_mode(self.target.downstream_mode)
        validate_candidate_protocol(self.target.candidate_protocol)
        if self.target.training_protocol not in TARGET_TRAINING_PROTOCOLS:
            choices = ", ".join(TARGET_TRAINING_PROTOCOLS)
            raise ValueError(f"training_protocol must be one of: {choices}")
        if self.target.training_data_scope not in TARGET_TRAINING_DATA_SCOPES:
            choices = ", ".join(TARGET_TRAINING_DATA_SCOPES)
            raise ValueError(f"training_data_scope must be one of: {choices}")
        if self.target.optimizer_name not in TARGET_OPTIMIZERS:
            choices = ", ".join(TARGET_OPTIMIZERS)
            raise ValueError(f"optimizer_name must be one of: {choices}")
        if self.target.scheduler_name not in TARGET_SCHEDULERS:
            choices = ", ".join(TARGET_SCHEDULERS)
            raise ValueError(f"scheduler_name must be one of: {choices}")
        if self.target.scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be positive")
        if not 0.0 < self.target.scheduler_gamma <= 1.0:
            raise ValueError("scheduler_gamma must be in (0,1]")
        if (
            self.target.training_protocol == POSTHOC_FROZEN_BASE
            and self.target.downstream_mode != LEARNED_TOPK_ERROR_AWARE
        ):
            raise ValueError(
                "posthoc_frozen_base requires learned_topk_error_aware mode"
            )
        if self.target.backbone_name not in {"lightweight", "stgcn", "graph_wavenet"}:
            raise ValueError("backbone_name must be lightweight, stgcn, or graph_wavenet")
        if self.data.context_length <= 0 or self.data.horizon <= 0:
            raise ValueError("context_length and horizon must be positive")
        if self.data.encoder_context_length < self.data.context_length:
            raise ValueError("retrieval_context_length must be at least context_length")
        if self.data.encoder_context_length % self.model.patch_size != 0:
            raise ValueError("retrieval context length must be divisible by patch_size")
        if not 0 < self.pretrain.time_mask_block_size <= self.data.encoder_context_length:
            raise ValueError("time_mask_block_size must be in [1, retrieval context length]")
        if self.pretrain.time_mask_block_size % self.model.patch_size != 0:
            raise ValueError("time_mask_block_size must be divisible by patch_size")
        if self.model.hidden_dim % self.model.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.pretrain.objective not in {
            "joint",
            "relation_only",
            "masked_relation_single_view",
        }:
            raise ValueError(
                "pretrain objective must be joint, relation_only, or "
                "masked_relation_single_view"
            )
        if self.pretrain.reconstruction_weight < 0.0:
            raise ValueError("reconstruction_weight must be non-negative")
        if self.pretrain.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if (
            self.pretrain.objective == "relation_only"
            and self.pretrain.reconstruction_weight != 0.0
        ):
            raise ValueError(
                "relation_only objective requires reconstruction_weight=0"
            )
        if (
            self.pretrain.objective == "masked_relation_single_view"
            and self.pretrain.reconstruction_weight <= 0.0
        ):
            raise ValueError(
                "masked_relation_single_view requires reconstruction_weight>0"
            )
        if self.model.route_dim <= 0 or self.model.route_dim > 2 * self.model.hidden_dim:
            raise ValueError("route_dim must be positive and no larger than 2 * hidden_dim")
        if self.model.route_top_k <= 0:
            raise ValueError("route_top_k must be positive")
        if not 0 <= self.model.route_local_quota <= self.model.route_top_k:
            raise ValueError("route_local_quota must be in [0, route_top_k]")
        if self.model.route_prior_weight < 0.0:
            raise ValueError("route_prior_weight must be non-negative")
        if self.model.route_temperature <= 0.0:
            raise ValueError("route_temperature must be positive")
        if self.model.route_gate_bias >= 0.0:
            raise ValueError("route_gate_bias must be negative")
        if self.model.dynamics_adapter_mode not in {
            "none",
            "local",
            "local_graph",
            "context_conditioned",
        }:
            raise ValueError(
                "dynamics_adapter_mode must be none, local, local_graph, "
                "or context_conditioned"
            )
        if not 0 < self.model.dynamics_bottleneck_dim <= self.model.hidden_dim:
            raise ValueError(
                "dynamics_bottleneck_dim must be in [1, hidden_dim]"
            )
        if self.model.dynamics_gate_bias >= 0.0:
            raise ValueError("dynamics_gate_bias must be negative")
        if self.model.dynamics_adapter_mode == "context_conditioned" and (
            self.model.dynamics_gate_groups <= 0
            or self.model.hidden_dim % self.model.dynamics_gate_groups != 0
        ):
            raise ValueError(
                "dynamics_gate_groups must be positive and divide hidden_dim"
            )
        if self.model.input_channels != self.model.output_channels:
            raise ValueError("v1 requires input_channels == output_channels")
        if self.model.profile_dim < 0 or self.model.latent_dim < 0:
            raise ValueError("profile_dim and latent_dim must be non-negative")
        if (self.model.profile_dim == 0) != (self.model.latent_dim == 0):
            raise ValueError(
                "profile_dim and latent_dim must either both be zero or both be positive"
            )
        if self.model.profile_dim > 0 and self.model.profile_dim + self.model.latent_dim != self.model.retrieval_dim:
            raise ValueError("profile_dim + latent_dim must equal retrieval_dim")
        if self.model.profile_dim > 0:
            if self.model.profile_dim != 12:
                raise ValueError("E5-Final requires a fixed 12-D canonical profile")
            if self.model.input_channels != 1:
                raise ValueError("E5-Final CFDP currently requires one input channel")
            if self.pretrain.relation_distance_normalization != "symmetric_geometric_mean":
                raise ValueError(
                    "profile-enabled E5-Final requires symmetric_geometric_mean relation normalization"
                )
        if not 0.0 <= self.model.profile_weight <= 1.0:
            raise ValueError("profile_weight must be in [0, 1]")
        if self.pretrain.profile_loss_weight < 0:
            raise ValueError("profile_loss_weight must be non-negative")
        if self.pretrain.profile_scale_floor <= 0:
            raise ValueError("profile_scale_floor must be positive")
        if self.model.profile_dim == 0 and self.pretrain.profile_loss_weight != 0.0:
            raise ValueError("profile_loss_weight requires a profile-enabled retrieval head")
        if self.model.profile_dim > 0 and self.pretrain.profile_loss_weight <= 0.0:
            raise ValueError("profile-enabled retrieval requires positive profile_loss_weight")
        if self.pretrain.retrieval_loss_mode not in {"hard_negative", "relation", "hard_negative_offset_decay"}:
            raise ValueError(
                "retrieval_loss_mode must be hard_negative, relation, or hard_negative_offset_decay"
            )
        teacher_mode = self.pretrain.relation_teacher_mode
        if teacher_mode not in {
            "context_normalized",
            "offset_decay",
            "offset_decay_increment",
        }:
            raise ValueError(
                "relation_teacher_mode must be context_normalized, offset_decay, "
                "or offset_decay_increment"
            )
        distance_normalization = self.pretrain.relation_distance_normalization
        if distance_normalization not in {
            "none",
            "anchor_mean",
            "symmetric_geometric_mean",
        }:
            raise ValueError(
                "relation_distance_normalization must be none, anchor_mean, "
                "or symmetric_geometric_mean"
            )
        increment_weight = self.pretrain.future_increment_weight
        if not 0.0 <= increment_weight <= 1.0:
            raise ValueError("future_increment_weight must be in [0, 1]")
        if self.pretrain.rank_loss_weight < 0.0:
            raise ValueError("rank_loss_weight must be non-negative")
        if self.pretrain.rank_positive_count <= 0:
            raise ValueError("rank_positive_count must be positive")
        if self.pretrain.rank_negative_count <= 0:
            raise ValueError("rank_negative_count must be positive")
        if self.pretrain.rank_future_gap < 0.0:
            raise ValueError("rank_future_gap must be non-negative")
        if self.pretrain.rank_margin < 0.0:
            raise ValueError("rank_margin must be non-negative")
        if self.pretrain.rank_temperature <= 0.0:
            raise ValueError("rank_temperature must be positive")
        if (
            self.pretrain.rank_loss_weight > 0.0
            and self.pretrain.retrieval_loss_mode != "relation"
        ):
            raise ValueError(
                "rank_loss_weight requires retrieval_loss_mode=relation"
            )
        if teacher_mode == "context_normalized":
            if distance_normalization != "none" or increment_weight != 0.0:
                raise ValueError(
                    "context_normalized teacher requires normalization=none and "
                    "future_increment_weight=0"
                )
        else:
            if distance_normalization not in {
                "anchor_mean",
                "symmetric_geometric_mean",
            }:
                raise ValueError(
                    "OffsetDecay relation teachers require anchor_mean or "
                    "symmetric_geometric_mean distance normalization"
                )
            if teacher_mode == "offset_decay" and increment_weight != 0.0:
                raise ValueError(
                    "offset_decay teacher requires future_increment_weight=0"
                )
            if teacher_mode == "offset_decay_increment" and increment_weight != 0.5:
                raise ValueError(
                    "offset_decay_increment teacher requires future_increment_weight=0.5"
                )
        if not 0.0 < self.data.train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")
        if not 0.0 <= self.data.val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.data.train_ratio + self.data.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be smaller than 1")
        for name, value in (
            ("time_mask_ratio", self.pretrain.time_mask_ratio),
            ("space_mask_ratio", self.pretrain.space_mask_ratio),
            ("time_task_probability", self.pretrain.time_task_probability),
            ("memory_fraction", self.bank.memory_fraction),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.bank.event_top_r < self.bank.node_top_k:
            raise ValueError("event_top_r must be greater than or equal to node_top_k")
        if self.bank.key_dtype not in {"float16", "float32"}:
            raise ValueError("key_dtype must be float16 or float32")
        for name, value in (
            ("retrieval_temperature", self.pretrain.retrieval_temperature),
            (
                "relation_teacher_temperature",
                self.pretrain.relation_teacher_temperature,
            ),
            (
                "relation_student_temperature",
                self.pretrain.relation_student_temperature,
            ),
            ("hard_negative_weight", self.pretrain.hard_negative_weight),
            ("level_temperature", self.bank.level_temperature),
            ("search_temperature", self.bank.search_temperature),
            ("confidence_level_temperature", self.target.confidence_level_temperature),
            ("help_temperature", self.target.help_temperature),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        quantiles = (
            self.pretrain.positive_quantile,
            self.pretrain.context_quantile,
            self.pretrain.negative_quantile,
        )
        if not all(0.0 < value < 1.0 for value in quantiles):
            raise ValueError("retrieval quantiles must be in (0, 1)")
        if self.pretrain.positive_quantile >= self.pretrain.negative_quantile:
            raise ValueError("positive_quantile must be smaller than negative_quantile")
        if self.pretrain.progress_interval <= 0:
            raise ValueError("progress_interval must be positive")
        for name, value in (
            ("risk_hidden_dim", self.target.risk_hidden_dim),
            ("fusion_feature_hidden_dim", self.target.fusion_feature_hidden_dim),
            ("horizon_aggregation_hidden_dim", self.target.horizon_aggregation_hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("risk_weight", self.target.risk_weight),
            ("blend_weight", self.target.blend_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.target.blend_minimum_direction_norm <= 0:
            raise ValueError("blend_minimum_direction_norm must be positive")
        if self.target.validation_loss_variant not in {"forecast_risk_blend", "forecast_risk", "forecast_only"}:
            raise ValueError("unsupported validation_loss_variant")
        if self.target.validation_correction_variant not in {"scalar_gate", "vector_residual", "residual_additive"}:
            raise ValueError("unsupported validation_correction_variant")
        if self.target.backbone_name == "stgcn":
            if self.target.stgcn_temporal_kernel < 2:
                raise ValueError("stgcn_temporal_kernel must be at least 2")
            if self.target.stgcn_graph_kernel <= 0 or self.target.stgcn_block_num <= 0:
                raise ValueError("stgcn graph kernel and block_num must be positive")
            for name, value in (
                ("stgcn_hidden_channels", self.target.stgcn_hidden_channels),
                ("stgcn_bottleneck_channels", self.target.stgcn_bottleneck_channels),
                ("stgcn_output_hidden_channels", self.target.stgcn_output_hidden_channels),
            ):
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
            if not 0.0 <= self.target.stgcn_dropout < 1.0:
                raise ValueError("stgcn_dropout must be in [0,1)")
            stgcn_output_length = self.data.context_length - 2 * (
                self.target.stgcn_temporal_kernel - 1
            ) * self.target.stgcn_block_num
            if stgcn_output_length < 1:
                raise ValueError(
                    "context_length is too short for the requested STGCN blocks"
                )
        if self.target.backbone_name == "graph_wavenet":
            for name, value in (
                ("graph_wavenet_residual_channels", self.target.graph_wavenet_residual_channels),
                ("graph_wavenet_dilation_channels", self.target.graph_wavenet_dilation_channels),
                ("graph_wavenet_skip_channels", self.target.graph_wavenet_skip_channels),
                ("graph_wavenet_end_channels", self.target.graph_wavenet_end_channels),
                ("graph_wavenet_blocks", self.target.graph_wavenet_blocks),
                ("graph_wavenet_layers", self.target.graph_wavenet_layers),
                ("graph_wavenet_adaptive_dim", self.target.graph_wavenet_adaptive_dim),
            ):
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
            if self.target.graph_wavenet_kernel_size <= 1:
                raise ValueError("graph_wavenet_kernel_size must be at least 2")
            if not 0.0 <= self.target.graph_wavenet_dropout < 1.0:
                raise ValueError("graph_wavenet_dropout must be in [0,1)")
        if self.target.base_warmup_epochs < 0 or self.target.calibrator_warmup_epochs < 0:
            raise ValueError("downstream warmup epochs must be non-negative")
        if self.target.training_protocol == POSTHOC_FROZEN_BASE and (
            self.target.base_warmup_epochs != 0
            or self.target.calibrator_warmup_epochs != 0
        ):
            raise ValueError("posthoc_frozen_base requires zero warmup epochs")
        if not 0.0 < self.target.backbone_learning_rate_scale <= 1.0:
            raise ValueError("backbone_learning_rate_scale must be in (0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct_dataclass(cls: type[T], values: Mapping[str, Any] | None) -> T:
    values = dict(values or {})
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config and reject silent misspellings."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Config root must be a mapping")
    allowed_sections = {item.name for item in fields(ExperimentConfig)}
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {unknown_sections}")
    if "data" not in raw:
        raise ValueError("Config must contain a data section")
    config = ExperimentConfig(
        data=_construct_dataclass(DataConfig, raw.get("data")),
        model=_construct_dataclass(ModelConfig, raw.get("model")),
        pretrain=_construct_dataclass(PretrainConfig, raw.get("pretrain")),
        bank=_construct_dataclass(BankConfig, raw.get("bank")),
        target=_construct_dataclass(TargetConfig, raw.get("target")),
        runtime=_construct_dataclass(RuntimeConfig, raw.get("runtime")),
    )
    if not is_dataclass(config):
        raise TypeError("Failed to construct ExperimentConfig")
    config.validate()
    return config


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (project_root() / candidate).resolve()


"""Shared clean/masked encoder flow for source-domain pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from stanchor.config import ModelConfig, PretrainConfig
from stanchor.data.graph import GraphData
from stanchor.data.masking import MaskBatch, StructuredMaskSampler
from stanchor.data.normalization import WindowStatistics, normalize_window
from stanchor.utils import tensor_mapping_sha256

from .dynamics_adapter import DynamicsAdapterOutput, HistoryDynamicsAdapter
from .encoder import FactorizedSTEncoder
from .patch_embedding import TemporalPatchEmbedding
from .retrieval_head import RetrievalHead, RetrievalOutput


@dataclass(frozen=True)
class CleanEncoding:
    hidden: torch.Tensor  # [B, P, N, D]
    retrieval: RetrievalOutput
    statistics: WindowStatistics
    dynamics: DynamicsAdapterOutput | None = None


@dataclass(frozen=True)
class PretrainForwardOutput:
    clean: CleanEncoding
    masked_hidden: torch.Tensor  # [B, P, N, D]
    reconstruction: torch.Tensor  # [B, T, N, C]
    reconstruction_target: torch.Tensor  # [B, T, N, C]
    mask: MaskBatch
    masked_dynamics: DynamicsAdapterOutput | None = None


class STAnchorPretrainModel(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        pretrain_config: PretrainConfig,
        context_length: int,
        slots_per_day: int,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.context_length = context_length
        self.num_patches = context_length // model_config.patch_size
        self.embedding = TemporalPatchEmbedding(
            model_config.input_channels,
            model_config.patch_size,
            model_config.hidden_dim,
            slots_per_day,
        )
        self.encoder = FactorizedSTEncoder(
            hidden_dim=model_config.hidden_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.encoder_layers,
            ffn_multiplier=model_config.ffn_multiplier,
            dropout=model_config.dropout,
            graph_bias=model_config.graph_bias,
            route_enabled=model_config.route_enabled,
            route_dim=model_config.route_dim,
            route_top_k=model_config.route_top_k,
            route_local_quota=model_config.route_local_quota,
            route_prior_weight=model_config.route_prior_weight,
            route_temperature=model_config.route_temperature,
            route_gate_bias=model_config.route_gate_bias,
        )
        self.retrieval_head = RetrievalHead(
            model_config.hidden_dim,
            model_config.retrieval_dim,
            profile_dim=model_config.profile_dim,
            latent_dim=model_config.latent_dim,
            profile_weight=model_config.profile_weight,
            adapter_bottleneck_dim=model_config.adapter_bottleneck_dim,
        )
        self.reconstruction_head = nn.Linear(
            model_config.hidden_dim,
            model_config.patch_size * model_config.input_channels,
        )
        self.dynamics_adapter = (
            None
            if model_config.dynamics_adapter_mode == "none"
            else HistoryDynamicsAdapter(
                input_channels=model_config.input_channels,
                patch_size=model_config.patch_size,
                hidden_dim=model_config.hidden_dim,
                bottleneck_dim=model_config.dynamics_bottleneck_dim,
                mode=model_config.dynamics_adapter_mode,
                gate_bias=model_config.dynamics_gate_bias,
                gate_groups=model_config.dynamics_gate_groups,
            )
        )
        self.mask_sampler = StructuredMaskSampler(
            context_length=context_length,
            patch_size=model_config.patch_size,
            time_ratio=pretrain_config.time_mask_ratio,
            space_ratio=pretrain_config.space_mask_ratio,
            time_probability=pretrain_config.time_task_probability,
            time_block_size=pretrain_config.time_mask_block_size,
        )

    def retrieval_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("embedding", self.embedding),
            ("encoder", self.encoder),
            ("retrieval_head", self.retrieval_head),
        ):
            state.update({f"{prefix}.{name}": value for name, value in module.state_dict().items()})
        if self.dynamics_adapter is not None:
            state.update(
                {
                    f"dynamics_adapter.{name}": value
                    for name, value in self.dynamics_adapter.state_dict().items()
                }
            )
        return state

    def retrieval_fingerprint(self) -> str:
        return tensor_mapping_sha256(self.retrieval_state_dict())

    def _embed_clean(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
    ) -> tuple[torch.Tensor, WindowStatistics]:
        if x.ndim != 4 or x.shape[1] != self.context_length:
            raise ValueError(
                f"x must match configured context length {self.context_length}: [B, T, N, C]"
            )
        if observed.shape != x.shape:
            raise ValueError("observed must have the same shape as x")
        if weekday.shape != x.shape[:2] or slot.shape != x.shape[:2]:
            raise ValueError("weekday and slot must be [B, T] and align with x")
        statistics = normalize_window(x, observed)
        tokens = self.embedding(
            statistics.normalized,
            statistics.level_features,
            statistics.level_valid,
            weekday,
            slot,
        )
        return tokens, statistics

    def encode_clean(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
        graph: GraphData,
    ) -> CleanEncoding:
        tokens, statistics = self._embed_clean(x, observed, weekday, slot)
        hidden = self.encoder(tokens, graph)
        dynamics = None
        if self.dynamics_adapter is not None:
            dynamics = self.dynamics_adapter(
                hidden,
                statistics.normalized,
                observed.bool(),
                graph,
            )
            hidden = dynamics.hidden
        return CleanEncoding(
            hidden,
            self.retrieval_head(hidden),
            statistics,
            dynamics,
        )

    def forward_relation(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
        graph: GraphData,
    ) -> CleanEncoding:
        """Encode one clean history for relation-only pretraining.

        No mask sampler or reconstruction head is touched in this path; the
        returned key is trained only by the future-relation objective.
        """
        return self.encode_clean(x, observed, weekday, slot, graph)

    def forward_pretrain(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
        graph: GraphData,
        neighbors: torch.Tensor,
        mask_task: str | None = None,
        generator: torch.Generator | None = None,
    ) -> PretrainForwardOutput:
        if x.ndim != 4 or observed.shape != x.shape:
            raise ValueError("x and observed must be [B, T, N, C]")
        batch, time, nodes, channels = x.shape
        if time != self.context_length or channels != self.model_config.input_channels:
            raise ValueError("input does not match model context/channel configuration")
        clean_tokens, clean_statistics = self._embed_clean(x, observed, weekday, slot)
        mask = self.mask_sampler.sample(
            batch_size=batch,
            num_nodes=nodes,
            num_channels=channels,
            neighbors=neighbors,
            device=x.device,
            observed=observed,
            task=mask_task,
            generator=generator,
        )
        visible = observed.bool() & ~mask.value_mask
        masked_statistics = normalize_window(x, visible)
        masked_tokens = self.embedding(
            masked_statistics.normalized,
            masked_statistics.level_features,
            masked_statistics.level_valid,
            weekday,
            slot,
            patch_mask=mask.patch_mask,
            mask_task=mask.task,
        )
        combined_hidden = self.encoder(torch.cat((clean_tokens, masked_tokens), dim=0), graph)
        clean_hidden, masked_hidden = combined_hidden[:batch], combined_hidden[batch:]
        clean_dynamics = masked_dynamics = None
        if self.dynamics_adapter is not None:
            clean_dynamics = self.dynamics_adapter(
                clean_hidden,
                clean_statistics.normalized,
                observed.bool(),
                graph,
            )
            masked_dynamics = self.dynamics_adapter(
                masked_hidden,
                masked_statistics.normalized,
                visible,
                graph,
            )
            clean_hidden = clean_dynamics.hidden
            masked_hidden = masked_dynamics.hidden
        retrieval = self.retrieval_head(clean_hidden)
        patch_values = self.reconstruction_head(masked_hidden)
        reconstruction = self._unpatchify(patch_values)
        reconstruction_target = (
            x - masked_statistics.mean.unsqueeze(1)
        ) / (masked_statistics.std.unsqueeze(1) + 1.0e-6)
        return PretrainForwardOutput(
            clean=CleanEncoding(
                clean_hidden,
                retrieval,
                clean_statistics,
                clean_dynamics,
            ),
            masked_hidden=masked_hidden,
            reconstruction=reconstruction,
            reconstruction_target=reconstruction_target,
            mask=mask,
            masked_dynamics=masked_dynamics,
        )

    def forward_pretrain_single_view(
        self,
        x: torch.Tensor,
        observed: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
        graph: GraphData,
        neighbors: torch.Tensor,
        mask_task: str | None = None,
        generator: torch.Generator | None = None,
    ) -> PretrainForwardOutput:
        """Run masked reconstruction and retrieval relation on one masked view.

        Unlike :meth:`forward_pretrain`, this path does not build a clean view
        and a masked view in a concatenated batch.  The masked representation
        is used for both the retrieval key and the reconstruction head, so the
        encoder is evaluated exactly once per input batch.
        """
        if x.ndim != 4 or observed.shape != x.shape:
            raise ValueError("x and observed must be [B, T, N, C]")
        batch, time, nodes, channels = x.shape
        if time != self.context_length or channels != self.model_config.input_channels:
            raise ValueError("input does not match model context/channel configuration")

        mask = self.mask_sampler.sample(
            batch_size=batch,
            num_nodes=nodes,
            num_channels=channels,
            neighbors=neighbors,
            device=x.device,
            observed=observed,
            task=mask_task,
            generator=generator,
        )
        visible = observed.bool() & ~mask.value_mask
        masked_statistics = normalize_window(x, visible)
        masked_tokens = self.embedding(
            masked_statistics.normalized,
            masked_statistics.level_features,
            masked_statistics.level_valid,
            weekday,
            slot,
            patch_mask=mask.patch_mask,
            mask_task=mask.task,
        )
        hidden = self.encoder(masked_tokens, graph)
        dynamics = None
        if self.dynamics_adapter is not None:
            dynamics = self.dynamics_adapter(
                hidden,
                masked_statistics.normalized,
                visible,
                graph,
            )
            hidden = dynamics.hidden
        retrieval = self.retrieval_head(hidden)
        patch_values = self.reconstruction_head(hidden)
        reconstruction = self._unpatchify(patch_values)
        reconstruction_target = (
            x - masked_statistics.mean.unsqueeze(1)
        ) / (masked_statistics.std.unsqueeze(1) + 1.0e-6)
        # ``clean`` is retained in the output contract so the existing loss
        # code can consume this single-view encoding without a second API.
        return PretrainForwardOutput(
            clean=CleanEncoding(hidden, retrieval, masked_statistics, dynamics),
            masked_hidden=hidden,
            reconstruction=reconstruction,
            reconstruction_target=reconstruction_target,
            mask=mask,
            masked_dynamics=dynamics,
        )

    def _unpatchify(self, patch_values: torch.Tensor) -> torch.Tensor:
        batch, patches, nodes, patch_channels = patch_values.shape
        channels = self.model_config.input_channels
        expected = self.model_config.patch_size * channels
        if patches != self.num_patches or patch_channels != expected:
            raise ValueError("reconstruction patch tensor has invalid shape")
        values = patch_values.view(batch, patches, nodes, self.model_config.patch_size, channels)
        values = values.permute(0, 1, 3, 2, 4).contiguous()
        return values.view(batch, self.context_length, nodes, channels)




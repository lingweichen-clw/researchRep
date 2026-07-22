"""Temporal patch embedding with explicit time, level, and mask semantics."""

from __future__ import annotations

import torch
from torch import nn


class TemporalPatchEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        hidden_dim: int,
        slots_per_day: int,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.slots_per_day = slots_per_day
        self.value_projection = nn.Linear(patch_size * input_channels, hidden_dim)
        self.level_projection = nn.Linear(4 * input_channels, hidden_dim, bias=False)
        self.weekday_embedding = nn.Embedding(7, hidden_dim)
        self.slot_embedding = nn.Embedding(slots_per_day, hidden_dim)
        self.time_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.space_mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.unknown_level_token = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.time_mask_token, std=0.02)
        nn.init.normal_(self.space_mask_token, std=0.02)
        nn.init.normal_(self.unknown_level_token, std=0.02)

    def forward(
        self,
        normalized: torch.Tensor,
        level_features: torch.Tensor,
        level_valid: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        mask_task: str | None = None,
    ) -> torch.Tensor:
        """Create tokens with shape [B, P, N, D]."""
        if normalized.ndim != 4:
            raise ValueError("normalized must be [B, T, N, C]")
        batch, time, nodes, channels = normalized.shape
        if channels != self.input_channels:
            raise ValueError("input channel mismatch")
        if time % self.patch_size != 0:
            raise ValueError("time dimension must be divisible by patch_size")
        if level_features.shape != (batch, nodes, 4 * channels):
            raise ValueError("level_features must be [B, N, 4C]")
        if level_valid.shape != (batch, nodes, 1):
            raise ValueError("level_valid must be [B, N, 1]")
        if weekday.shape != (batch, time) or slot.shape != (batch, time):
            raise ValueError("weekday and slot must be [B, T]")
        if bool((weekday < 0).any()) or bool((weekday >= 7).any()):
            raise ValueError("weekday ids must be in [0, 6]")
        if bool((slot < 0).any()) or bool((slot >= self.slots_per_day).any()):
            raise ValueError("slot ids are outside configured slots_per_day")

        patches = time // self.patch_size
        # [B, T, N, C] -> [B, P, N, pC]
        patch_values = normalized.reshape(batch, patches, self.patch_size, nodes, channels)
        patch_values = patch_values.permute(0, 1, 3, 2, 4).contiguous()
        patch_values = patch_values.reshape(batch, patches, nodes, self.patch_size * channels)
        value_tokens = self.value_projection(patch_values)

        if patch_mask is not None:
            if patch_mask.shape != (batch, patches, nodes):
                raise ValueError("patch_mask must be [B, P, N]")
            if mask_task == "time":
                mask_token = self.time_mask_token
            elif mask_task == "space":
                mask_token = self.space_mask_token
            else:
                raise ValueError("mask_task must be time or space when patch_mask is provided")
            value_tokens = torch.where(
                patch_mask.unsqueeze(-1),
                mask_token.view(1, 1, 1, -1),
                value_tokens,
            )

        level_tokens = self.level_projection(level_features)
        level_tokens = torch.where(
            level_valid.bool(),
            level_tokens,
            self.unknown_level_token.view(1, 1, -1),
        )
        patch_end = torch.arange(
            self.patch_size - 1,
            time,
            self.patch_size,
            device=normalized.device,
        )
        patch_weekday = weekday.index_select(1, patch_end)
        patch_slot = slot.index_select(1, patch_end)
        time_tokens = self.weekday_embedding(patch_weekday) + self.slot_embedding(patch_slot)
        tokens = value_tokens + level_tokens[:, None, :, :] + time_tokens[:, :, None, :]
        return self.norm(tokens)


"""Train-only dataset scaling and mask-aware window normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class NodeStandardScaler:
    """Per-node, per-channel scaler fitted only on a training segment."""

    mean: np.ndarray  # [N, C]
    std: np.ndarray  # [N, C]
    eps: float = 1.0e-6

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        observed: np.ndarray,
        eps: float = 1.0e-6,
    ) -> "NodeStandardScaler":
        if values.ndim != 3 or observed.shape != values.shape:
            raise ValueError("values and observed must both be [L, N, C]")
        valid = observed.astype(bool) & np.isfinite(values)
        counts = valid.sum(axis=0)
        safe_counts = np.maximum(counts, 1)
        sums = np.where(valid, values, 0.0).sum(axis=0)
        mean = sums / safe_counts
        global_valid = values[valid]
        global_mean = float(global_valid.mean()) if global_valid.size else 0.0
        mean = np.where(counts > 0, mean, global_mean)
        centered = np.where(valid, values - mean[None, ...], 0.0)
        variance = np.square(centered).sum(axis=0) / safe_counts
        global_std = float(global_valid.std()) if global_valid.size else 1.0
        global_std = max(global_std, eps)
        std = np.sqrt(variance + eps)
        std = np.where(counts > 1, std, global_std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), eps=eps)

    def transform(self, values: np.ndarray, observed: np.ndarray | None = None) -> np.ndarray:
        transformed = (values - self.mean[None, ...]) / (self.std[None, ...] + self.eps)
        if observed is not None:
            transformed = np.where(observed, transformed, 0.0)
        return transformed.astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * (self.std[None, ...] + self.eps) + self.mean[None, ...]

    def inverse_transform_torch(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[2:] != self.mean.shape:
            raise ValueError("values must be [B, T/H, N, C] and match scaler nodes/channels")
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * (std[None, None, ...] + self.eps) + mean[None, None, ...]

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "NodeStandardScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            eps=float(state.get("eps", 1.0e-6)),
        )


@dataclass
class WindowStatistics:
    normalized: torch.Tensor  # [B, T, N, C]
    level_features: torch.Tensor  # [B, N, 4C]
    level_valid: torch.Tensor  # [B, N, 1]
    mean: torch.Tensor  # [B, N, C]
    std: torch.Tensor  # [B, N, C]


def _first_last_visible(
    values: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # values/observed: [B, T, N, C]
    batch, time, nodes, channels = values.shape
    time_index = torch.arange(time, device=values.device).view(1, time, 1, 1)
    first_index = torch.where(observed, time_index, time).amin(dim=1)
    last_index = torch.where(observed, time_index, -1).amax(dim=1)
    gather_shape = (batch, 1, nodes, channels)
    first = values.gather(1, first_index.clamp(min=0, max=time - 1).view(gather_shape)).squeeze(1)
    last = values.gather(1, last_index.clamp(min=0, max=time - 1).view(gather_shape)).squeeze(1)
    has_value = observed.any(dim=1)
    return torch.where(has_value, first, torch.zeros_like(first)), torch.where(
        has_value, last, torch.zeros_like(last)
    )


def normalize_window(
    values: torch.Tensor,
    observed: torch.Tensor,
    eps: float = 1.0e-6,
) -> WindowStatistics:
    """Normalize each sample/node/channel using only visible observations."""
    if values.ndim != 4 or observed.shape != values.shape:
        raise ValueError("values and observed must both be [B, T, N, C]")
    if observed.dtype is not torch.bool:
        observed = observed.bool()
    counts = observed.sum(dim=1)  # [B, N, C]
    safe_counts = counts.clamp_min(1)
    sums = torch.where(observed, values, torch.zeros_like(values)).sum(dim=1)
    mean = sums / safe_counts
    centered = torch.where(observed, values - mean.unsqueeze(1), torch.zeros_like(values))
    variance = centered.square().sum(dim=1) / safe_counts
    std = torch.sqrt(variance + eps)
    valid_channel = counts > 0
    mean = torch.where(valid_channel, mean, torch.zeros_like(mean))
    std = torch.where(valid_channel, std, torch.ones_like(std))
    normalized = (values - mean.unsqueeze(1)) / (std.unsqueeze(1) + eps)
    normalized = torch.where(observed, normalized, torch.zeros_like(normalized))
    first, last = _first_last_visible(values, observed)
    slope = last - first
    level_features = torch.cat((mean, std, last, slope), dim=-1)
    level_valid = valid_channel.all(dim=-1, keepdim=True)
    return WindowStatistics(
        normalized=normalized,
        level_features=level_features,
        level_valid=level_valid,
        mean=mean,
        std=std,
    )


def normalize_future_with_context(
    future: torch.Tensor,
    context_stats: WindowStatistics,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if future.ndim != 4:
        raise ValueError("future must be [B, H, N, C]")
    return (future - context_stats.mean.unsqueeze(1)) / (context_stats.std.unsqueeze(1) + eps)

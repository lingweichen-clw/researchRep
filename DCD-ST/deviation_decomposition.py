"""Deviation decomposition blocks for DCD-ST.

The modules here operate on traffic windows shaped as ``(B, T, N, C)`` and
return node-level deviation descriptors for the forecasting model.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class MovingAverageDecomposition(nn.Module):
    """Split current-anchor residuals into slow trend and fast residual parts."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep temporal length.")
        self.kernel_size = kernel_size

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return trend and residual components.

        Args:
            residual: Tensor with shape ``(B, T, N, C)``.
        """
        if residual.dim() != 4:
            raise ValueError(f"Expected residual with shape (B,T,N,C), got {tuple(residual.shape)}")
        batch_size, steps, num_nodes, channels = residual.shape
        temporal = residual.permute(0, 2, 3, 1).reshape(batch_size * num_nodes * channels, 1, steps)
        pad = (self.kernel_size - 1) // 2
        if pad > 0:
            temporal = F.pad(temporal, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(temporal, kernel_size=self.kernel_size, stride=1)
        trend = trend.reshape(batch_size, num_nodes, channels, steps).permute(0, 3, 1, 2)
        short_residual = residual - trend
        return trend, short_residual


class TemporalDeviationNorm(nn.Module):
    """Window-wise temporal normalization of current-anchor residuals."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        mean = residual.mean(dim=1, keepdim=True)
        var = residual.var(dim=1, keepdim=True, unbiased=False)
        return (residual - mean) / torch.sqrt(var + self.eps)


class SpatialDeviationNorm(nn.Module):
    """Spatial normalization of current-anchor residuals.

    The first version uses global node-wise normalization. A graph-neighborhood
    variant can be added later without changing the caller contract.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        mean = residual.mean(dim=2, keepdim=True)
        var = residual.var(dim=2, keepdim=True, unbiased=False)
        return (residual - mean) / torch.sqrt(var + self.eps)


class DeviationFeatureExtractor(nn.Module):
    """Build continuous deviation descriptors from current and anchor windows."""

    feature_dim = 8

    def __init__(
        self,
        kernel_size: int = 3,
        use_temporal_norm: bool = True,
        use_spatial_norm: bool = True,
    ):
        super().__init__()
        self.use_temporal_norm = use_temporal_norm
        self.use_spatial_norm = use_spatial_norm
        self.decomposition = MovingAverageDecomposition(kernel_size)
        self.temporal_norm = TemporalDeviationNorm()
        self.spatial_norm = SpatialDeviationNorm()

    @staticmethod
    def _pool_signed_and_abs(sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        signed = sequence.mean(dim=(1, 3))
        magnitude = sequence.abs().mean(dim=(1, 3))
        return signed, magnitude

    def forward(self, x: torch.Tensor, x_his: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract residual components and node-level descriptors.

        Args:
            x: Current input window, shape ``(B, T, N, 1)``.
            x_his: Timestamp-aligned historical anchor, shape ``(B, T, N, 1)``.
        """
        if x.shape != x_his.shape:
            raise ValueError(f"x and x_his must share shape, got {tuple(x.shape)} and {tuple(x_his.shape)}")
        residual = x - x_his
        trend, short_residual = self.decomposition(residual)
        if self.use_temporal_norm:
            temporal_deviation = self.temporal_norm(residual)
        else:
            temporal_deviation = torch.zeros_like(residual)
        if self.use_spatial_norm:
            spatial_deviation = self.spatial_norm(residual)
        else:
            spatial_deviation = torch.zeros_like(residual)

        feature_parts = []
        for component in (trend, short_residual, temporal_deviation, spatial_deviation):
            feature_parts.extend(self._pool_signed_and_abs(component))
        z_dev = torch.stack(feature_parts, dim=-1)
        return {
            "z_dev": z_dev,
            "r_raw": residual,
            "r_trend": trend,
            "r_residual": short_residual,
            "d_t": temporal_deviation,
            "d_s": spatial_deviation,
        }


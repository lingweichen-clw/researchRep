"""Project-native adapter for the canonical ST-Norm WaveNet baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _SpatialNorm(nn.Module):
    """Normalize nodes independently at each context time step."""

    def __init__(self, channels: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2, keepdim=True)
        variance = x.var(dim=2, keepdim=True, unbiased=True)
        normalized = (x - mean) / (variance + self.eps).sqrt()
        return normalized * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


class _TemporalNorm(nn.Module):
    """Normalize each node/channel using context-time running statistics."""

    def __init__(
        self,
        num_nodes: int,
        channels: int,
        momentum: float = 0.1,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer("running_mean", torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer("running_var", torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(dim=(0, 3), keepdim=True)
            variance = x.var(dim=(0, 3), keepdim=True, unbiased=False)
            sample_count = x.shape[0] * x.shape[3]
            corrected_variance = variance
            if sample_count > 1:
                corrected_variance = variance * sample_count / (sample_count - 1)
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * corrected_variance)
        else:
            mean = self.running_mean
            variance = self.running_var
        normalized = (x - mean) / (variance + self.eps).sqrt()
        return normalized * self.gamma + self.beta


class STNormForecastBackbone(nn.Module):
    """Canonical ST-Norm WaveNet adapted to the project forecast contract.

    The reference implementation uses one block with four dilated layers. The
    input is padded on the left when the 12-step context is shorter than the
    receptive field, so no future values are introduced.
    """

    def __init__(
        self,
        context_length: int,
        horizon: int,
        input_channels: int,
        output_channels: int,
        num_nodes: int,
        channels: int = 16,
        kernel_size: int = 2,
        blocks: int = 1,
        layers: int = 4,
        use_snorm: bool = True,
        use_tnorm: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_channels != 1 or output_channels != 1:
            raise ValueError("ST-Norm adapter currently supports one input/output channel")
        if min(context_length, horizon, num_nodes, channels, kernel_size, blocks, layers) <= 0:
            raise ValueError("ST-Norm dimensions must be positive")
        if kernel_size < 2:
            raise ValueError("ST-Norm kernel_size must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("ST-Norm dropout must be in [0,1)")

        self.context_length = context_length
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.use_snorm = use_snorm
        self.use_tnorm = use_tnorm
        normalization_inputs = 1 + int(use_snorm) + int(use_tnorm)

        self.start_conv = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.snorms = nn.ModuleList()
        self.tnorms = nn.ModuleList()

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for _ in range(layers):
                if use_snorm:
                    self.snorms.append(_SpatialNorm(channels))
                if use_tnorm:
                    self.tnorms.append(_TemporalNorm(num_nodes, channels))
                self.filter_convs.append(
                    nn.Conv2d(
                        normalization_inputs * channels,
                        channels,
                        kernel_size=(1, kernel_size),
                        dilation=dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        normalization_inputs * channels,
                        channels,
                        kernel_size=(1, kernel_size),
                        dilation=dilation,
                    )
                )
                self.residual_convs.append(nn.Conv2d(channels, channels, kernel_size=1))
                self.skip_convs.append(nn.Conv2d(channels, channels, kernel_size=1))
                receptive_field += additional_scope
                additional_scope *= 2
                dilation *= 2

        self.receptive_field = receptive_field
        self.end_conv_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.end_conv_2 = nn.Conv2d(channels, horizon, kernel_size=1)
        # Kept for checkpoint/config compatibility with the reference code.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.context_length or x.shape[2] != self.num_nodes:
            raise ValueError("ST-Norm expects [B, context_length, num_nodes, 1]")
        if x.shape[-1] != 1:
            raise ValueError("ST-Norm expects one input channel")
        x = x.permute(0, 3, 2, 1).contiguous()
        if x.shape[-1] < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.shape[-1], 0, 0, 0))
        x = self.start_conv(x)
        skip: torch.Tensor | None = None
        norm_index = 0
        for index in range(len(self.filter_convs)):
            residual = x
            normalized = [x]
            if self.use_tnorm:
                normalized.append(self.tnorms[norm_index](x))
            if self.use_snorm:
                normalized.append(self.snorms[norm_index](x))
            norm_index += 1
            mixed = torch.cat(normalized, dim=1)
            filtered = torch.tanh(self.filter_convs[index](mixed))
            gated = torch.sigmoid(self.gate_convs[index](mixed))
            x = filtered * gated
            skip_value = self.skip_convs[index](x)
            if skip is None:
                skip = skip_value
            else:
                skip = skip[..., -skip_value.shape[-1] :] + skip_value
            x = self.residual_convs[index](x)
            x = x + residual[..., -x.shape[-1] :]

        if skip is None:
            raise RuntimeError("ST-Norm has no temporal layers")
        output = self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))
        return output[..., -1:].contiguous()

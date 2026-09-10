"""Project-native DLinear forecasting baseline."""

from __future__ import annotations

import torch
from torch import nn


class _MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("DLinear moving_avg_kernel must be a positive odd integer")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        front = x[..., :1].expand(*x.shape[:-1], padding)
        end = x[..., -1:].expand(*x.shape[:-1], padding)
        padded = torch.cat((front, x, end), dim=-1)
        return self.pool(padded)


class DLinearForecastBackbone(nn.Module):
    """Trend/seasonal linear model with shared weights across nodes."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        input_channels: int,
        output_channels: int,
        moving_avg_kernel: int = 3,
        individual: bool = False,
        num_nodes: int | None = None,
    ) -> None:
        super().__init__()
        if input_channels != 1 or output_channels != 1:
            raise ValueError("DLinear adapter currently supports one input/output channel")
        if context_length <= 0 or horizon <= 0:
            raise ValueError("DLinear context_length and horizon must be positive")
        if moving_avg_kernel <= 0 or moving_avg_kernel % 2 == 0:
            raise ValueError("DLinear moving_avg_kernel must be a positive odd integer")
        if individual and (num_nodes is None or num_nodes <= 0):
            raise ValueError("DLinear individual mode requires a positive num_nodes")

        self.context_length = context_length
        self.horizon = horizon
        self.individual = individual
        self.decomposition = _MovingAverage(moving_avg_kernel)
        # Official LTSF-Linear leaves Linear at the PyTorch default init.
        # The 1/T constant init is commented out in the official source and
        # should not be enabled in this adapter.
        if individual:
            self.seasonal = nn.ModuleList(
                [nn.Linear(context_length, horizon) for _ in range(int(num_nodes))]
            )
            self.trend = nn.ModuleList(
                [nn.Linear(context_length, horizon) for _ in range(int(num_nodes))]
            )
        else:
            self.seasonal = nn.Linear(context_length, horizon)
            self.trend = nn.Linear(context_length, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.context_length or x.shape[-1] != 1:
            raise ValueError("DLinear expects [B, context_length, num_nodes, 1]")
        values = x[..., 0].permute(0, 2, 1).contiguous()
        trend = self.decomposition(values)
        seasonal = values - trend
        if self.individual:
            seasonal_output = torch.stack(
                [layer(seasonal[:, node, :]) for node, layer in enumerate(self.seasonal)],
                dim=1,
            )
            trend_output = torch.stack(
                [layer(trend[:, node, :]) for node, layer in enumerate(self.trend)],
                dim=1,
            )
        else:
            seasonal_output = self.seasonal(seasonal)
            trend_output = self.trend(trend)
        output = (seasonal_output + trend_output).permute(0, 2, 1).contiguous()
        return output.unsqueeze(-1)

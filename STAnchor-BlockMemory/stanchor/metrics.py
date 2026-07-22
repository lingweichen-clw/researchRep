"""Streaming observed-value forecasting metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def select_common_horizon_metrics(
    metrics: dict[str, float | list[float]],
    frequency_minutes: int,
    requested_minutes: tuple[int, ...] = (15, 30, 60),
) -> dict[str, dict[str, float]]:
    """Select standard traffic horizons from full horizon metric vectors."""
    if frequency_minutes <= 0:
        raise ValueError("frequency_minutes must be positive")
    vectors = {}
    for name in ("mae", "rmse", "mape"):
        value = metrics.get(f"horizon_{name}")
        if not isinstance(value, list):
            raise ValueError(f"metrics must contain horizon_{name} as a list")
        vectors[name] = value
    lengths = {len(value) for value in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("horizon metric vectors must have identical lengths")
    horizon = next(iter(lengths))
    selected: dict[str, dict[str, float]] = {}
    for minutes in requested_minutes:
        if minutes <= 0 or minutes % frequency_minutes != 0:
            continue
        index = minutes // frequency_minutes - 1
        if index >= horizon:
            continue
        selected[f"{minutes}min"] = {
            name: float(values[index]) for name, values in vectors.items()
        }
    return selected


@dataclass
class ForecastMetricAccumulator:
    horizon: int
    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    percentage_sum: float = 0.0
    percentage_count: int = 0
    count: int = 0
    horizon_absolute_sum: torch.Tensor = field(init=False)
    horizon_squared_sum: torch.Tensor = field(init=False)
    horizon_percentage_sum: torch.Tensor = field(init=False)
    horizon_percentage_count: torch.Tensor = field(init=False)
    horizon_count: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.horizon_absolute_sum = torch.zeros(self.horizon, dtype=torch.float64)
        self.horizon_squared_sum = torch.zeros(self.horizon, dtype=torch.float64)
        self.horizon_percentage_sum = torch.zeros(self.horizon, dtype=torch.float64)
        self.horizon_percentage_count = torch.zeros(self.horizon, dtype=torch.float64)
        self.horizon_count = torch.zeros(self.horizon, dtype=torch.float64)

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor) -> None:
        if prediction.shape != target.shape or observed.shape != target.shape:
            raise ValueError("metric tensors must have identical shapes")
        valid = observed.bool()
        error = prediction - target
        absolute = error.abs()
        self.absolute_sum += float(absolute.masked_select(valid).sum().cpu())
        self.squared_sum += float(error.square().masked_select(valid).sum().cpu())
        percentage_valid = valid & (target.abs() > 1.0e-5)
        self.percentage_sum += float(
            (absolute / target.abs().clamp_min(1.0e-5)).masked_select(percentage_valid).sum().cpu()
        )
        self.percentage_count += int(percentage_valid.sum().item())
        self.count += int(valid.sum().item())
        reduce_dims = (0, 2, 3)
        self.horizon_absolute_sum += torch.where(valid, absolute, torch.zeros_like(absolute)).sum(
            dim=reduce_dims
        ).double().cpu()
        self.horizon_squared_sum += torch.where(
            valid, error.square(), torch.zeros_like(error)
        ).sum(dim=reduce_dims).double().cpu()
        percentage_error = absolute / target.abs().clamp_min(1.0e-5)
        self.horizon_percentage_sum += torch.where(
            percentage_valid,
            percentage_error,
            torch.zeros_like(percentage_error),
        ).sum(dim=reduce_dims).double().cpu()
        self.horizon_percentage_count += percentage_valid.sum(dim=reduce_dims).double().cpu()
        self.horizon_count += valid.sum(dim=reduce_dims).double().cpu()

    def compute(self) -> dict[str, float | list[float]]:
        if self.count == 0:
            raise ValueError("no observations were accumulated")
        horizon_mae = self.horizon_absolute_sum / self.horizon_count.clamp_min(1)
        horizon_rmse = torch.sqrt(
            self.horizon_squared_sum / self.horizon_count.clamp_min(1)
        )
        horizon_mape = 100.0 * self.horizon_percentage_sum / self.horizon_percentage_count.clamp_min(1)
        return {
            "mae": self.absolute_sum / self.count,
            "rmse": (self.squared_sum / self.count) ** 0.5,
            "mape": 100.0 * self.percentage_sum / max(self.percentage_count, 1),
            "horizon_mae": horizon_mae.tolist(),
            "horizon_rmse": horizon_rmse.tolist(),
            "horizon_mape": horizon_mape.tolist(),
        }

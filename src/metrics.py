"""Masked metrics for traffic forecasting."""

from __future__ import annotations

from typing import Dict

import torch


def _mask(labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = (labels != null_val).float()
    mean = mask.mean()
    if mean > 0:
        mask = mask / mean
    return torch.nan_to_num(mask)


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = _mask(labels, null_val)
    loss = torch.abs(preds - labels) * mask
    return torch.nan_to_num(loss).mean()


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = _mask(labels, null_val)
    loss = torch.pow(preds - labels, 2) * mask
    return torch.nan_to_num(loss).mean()


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = _mask(labels, null_val)
    loss = torch.abs((preds - labels) / torch.clamp(labels, min=1e-5)) * mask
    return torch.nan_to_num(loss).mean()


def horizon_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Return overall and common-horizon metrics."""
    result = {
        "mae": float(masked_mae(preds, labels).detach().cpu()),
        "rmse": float(masked_rmse(preds, labels).detach().cpu()),
        "mape": float(masked_mape(preds, labels).detach().cpu()),
    }
    for horizon_idx, name in ((2, "15min"), (5, "30min"), (11, "60min")):
        if preds.shape[1] > horizon_idx:
            p = preds[:, horizon_idx]
            y = labels[:, horizon_idx]
            result[f"mae_{name}"] = float(masked_mae(p, y).detach().cpu())
            result[f"rmse_{name}"] = float(masked_rmse(p, y).detach().cpu())
            result[f"mape_{name}"] = float(masked_mape(p, y).detach().cpu())
    return result

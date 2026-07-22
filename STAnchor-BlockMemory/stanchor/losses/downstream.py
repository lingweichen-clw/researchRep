"""Observed-value forecasting and confidence calibration losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from stanchor.models.downstream import DownstreamOutput, confidence_soft_target


@dataclass(frozen=True)
class DownstreamLoss:
    total: torch.Tensor
    forecast: torch.Tensor
    confidence: torch.Tensor
    confidence_target: torch.Tensor


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("prediction, target, and observed must share a shape")
    valid = observed.bool()
    if not bool(valid.any()):
        raise ValueError("masked MAE has no observed targets")
    return (prediction - target).abs().masked_select(valid).mean()


def compute_downstream_loss(
    output: DownstreamOutput,
    target: torch.Tensor,
    observed: torch.Tensor,
    confidence_weight: float,
    help_margin: float,
    help_temperature: float,
    use_confidence: bool = True,
) -> DownstreamLoss:
    forecast = masked_mae(output.final_prediction, target, observed)
    if not use_confidence:
        connected_zero = output.confidence.sum() * 0.0
        return DownstreamLoss(
            total=forecast,
            forecast=forecast,
            confidence=connected_zero,
            confidence_target=torch.zeros_like(output.confidence),
        )
    confidence_target = confidence_soft_target(
        output.base_prediction.detach(),
        output.memory_prediction.detach(),
        target,
        output.memory_valid,
        help_margin,
        help_temperature,
    )
    confidence_mask = output.memory_valid & observed.any(dim=-1, keepdim=True)
    if bool(confidence_mask.any()):
        confidence = (output.confidence - confidence_target).square().masked_select(confidence_mask).mean()
    else:
        confidence = output.confidence.sum() * 0.0
    return DownstreamLoss(
        total=forecast + confidence_weight * confidence,
        forecast=forecast,
        confidence=confidence,
        confidence_target=confidence_target,
    )

"""Observed-value forecasting and confidence calibration losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from stanchor.models.downstream import DownstreamOutput, confidence_soft_target


@dataclass(frozen=True)
class DownstreamLoss:
    total: torch.Tensor
    forecast: torch.Tensor
    confidence: torch.Tensor
    confidence_target: torch.Tensor
    risk: torch.Tensor | None = None
    blend: torch.Tensor | None = None
    risk_target: torch.Tensor | None = None
    blend_target: torch.Tensor | None = None


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError("prediction, target, and observed must share a shape")
    # Missing values may still be present in tensors even when an upstream
    # observed mask is supplied.  Exclude non-finite prediction/target pairs
    # before taking the reduction so one invalid sensor value cannot turn the
    # whole downstream loss into NaN/Inf.
    valid = observed.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        raise ValueError("masked MAE has no observed targets")
    return (prediction - target).abs().masked_select(valid).mean()


def build_huber_risk_target(
    base: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel-mean Huber base error and validity as [B,H,N,1]."""
    if base.shape != target.shape or observed.shape != target.shape:
        raise ValueError("base, target, and observed must share a shape")
    valid = observed.bool() & torch.isfinite(base) & torch.isfinite(target)
    element = functional.smooth_l1_loss(base, target, reduction="none")
    count = valid.sum(dim=-1, keepdim=True)
    risk = torch.where(valid, element, torch.zeros_like(element)).sum(dim=-1, keepdim=True)
    risk = risk / count.clamp_min(1).to(risk.dtype)
    risk_valid = count > 0
    return torch.where(risk_valid, risk, torch.zeros_like(risk)), risk_valid


def build_blend_target(
    base: torch.Tensor,
    memory: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    memory_valid: torch.Tensor,
    minimum_direction_norm: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the oracle convex step along the memory-minus-base direction."""
    if base.shape != memory.shape or target.shape != base.shape or observed.shape != base.shape:
        raise ValueError("base, memory, target, and observed must share a shape")
    if memory_valid.shape != base.shape[:-1] + (1,):
        raise ValueError("memory_valid must be [B,H,N,1]")
    if minimum_direction_norm <= 0:
        raise ValueError("minimum_direction_norm must be positive")
    valid = observed.bool() & memory_valid.expand_as(observed)
    direction = torch.where(valid, memory - base, torch.zeros_like(base))
    target_offset = torch.where(valid, target - base, torch.zeros_like(base))
    numerator = (target_offset * direction).sum(dim=-1, keepdim=True)
    denominator = direction.square().sum(dim=-1, keepdim=True)
    blend = (numerator / denominator.clamp_min(1.0e-8)).clamp(0.0, 1.0)
    blend_valid = memory_valid & valid.any(dim=-1, keepdim=True) & (
        denominator >= minimum_direction_norm**2
    )
    return torch.where(blend_valid, blend, torch.zeros_like(blend)), blend_valid


def compute_downstream_loss(
    output: DownstreamOutput,
    target: torch.Tensor,
    observed: torch.Tensor,
    confidence_weight: float,
    help_margin: float,
    help_temperature: float,
    use_confidence: bool = True,
    use_error_aware: bool = False,
    risk_weight: float = 0.1,
    blend_weight: float = 0.1,
    blend_minimum_direction_norm: float = 1.0e-4,
    loss_variant: str = "forecast_risk_blend",
) -> DownstreamLoss:
    forecast = masked_mae(output.final_prediction, target, observed)
    if use_error_aware:
        if loss_variant == "forecast_only":
            connected_zero = output.confidence.sum() * 0.0
            return DownstreamLoss(
                total=forecast, forecast=forecast, confidence=connected_zero,
                confidence_target=torch.zeros_like(output.confidence),
            )
        if output.predicted_base_risk is None:
            raise ValueError("error-aware loss requires predicted_base_risk")
        risk_target, risk_valid = build_huber_risk_target(
            output.base_prediction.detach(), target, observed
        )
        risk_mask = risk_valid
        if bool(risk_mask.any()):
            risk = functional.smooth_l1_loss(
                output.predicted_base_risk,
                risk_target.detach(),
                reduction="none",
            ).masked_select(risk_mask).mean()
        else:
            risk = output.predicted_base_risk.sum() * 0.0
        blend_target, blend_valid = build_blend_target(
            output.base_prediction.detach(),
            output.memory_prediction.detach(),
            target,
            observed,
            output.memory_valid,
            minimum_direction_norm=blend_minimum_direction_norm,
        )
        if bool(blend_valid.any()):
            blend = functional.smooth_l1_loss(
                output.fusion_weight,
                blend_target.detach(),
                reduction="none",
            ).masked_select(blend_valid).mean()
        else:
            blend = output.fusion_weight.sum() * 0.0
        return DownstreamLoss(
            total=(forecast + risk_weight * risk + blend_weight * blend
                   if loss_variant == "forecast_risk_blend"
                   else forecast + risk_weight * risk
                   if loss_variant == "forecast_risk"
                   else forecast),
            forecast=forecast,
            confidence=output.confidence.sum() * 0.0,
            confidence_target=torch.zeros_like(output.confidence),
            risk=risk,
            blend=blend,
            risk_target=risk_target,
            blend_target=blend_target,
        )
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

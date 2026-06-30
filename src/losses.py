"""Loss composition for ST-SSDL baseline and DCD-ST."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .metrics import masked_mae


@dataclass
class LossWeights:
    contrastive: float = 0.01
    deviation: float = 1.0
    gate_sparse: float = 0.0
    gate_smooth: float = 0.0
    use_contrastive: bool = True
    use_deviation: bool = True


def compute_training_loss(
    model_output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    scaler,
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    """Compute prediction, prototype, deviation, and DCD gate losses."""
    prediction = model_output["prediction"]
    y_pred = scaler.inverse_transform(prediction)
    mae_loss = masked_mae(y_pred, target)

    if weights.use_contrastive and weights.contrastive > 0 and model_output["query"].shape[-1] > 0:
        contrastive_loss = F.triplet_margin_loss(
            model_output["query"][0].detach(),
            model_output["pos"][0],
            model_output["neg"][0],
            margin=0.5,
        )
    else:
        contrastive_loss = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    if weights.use_deviation and weights.deviation > 0 and model_output["prototype_dis"].numel() > 0:
        deviation_loss = F.l1_loss(
            model_output["latent_dis"].detach(),
            model_output["prototype_dis"],
        )
    else:
        deviation_loss = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    zero_loss = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    gate_sparse_loss = model_output.get("gate_sparse_loss", zero_loss)
    gate_smooth_loss = model_output.get("gate_smooth_loss", zero_loss)

    total = (
        mae_loss
        + weights.contrastive * contrastive_loss
        + weights.deviation * deviation_loss
        + weights.gate_sparse * gate_sparse_loss
        + weights.gate_smooth * gate_smooth_loss
    )
    return {
        "total": total,
        "mae": mae_loss.detach(),
        "contrastive": contrastive_loss.detach(),
        "deviation": deviation_loss.detach(),
        "gate_sparse": gate_sparse_loss.detach(),
        "gate_smooth": gate_smooth_loss.detach(),
    }

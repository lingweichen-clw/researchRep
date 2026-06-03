"""Loss composition for the region-aware ST-SSDL model."""

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
    region: float = 0.05
    graph_reg: float = 0.001
    use_contrastive: bool = True
    use_deviation: bool = True


def compute_training_loss(
    model_output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    scaler,
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    """Compute prediction, prototype, deviation, and graph-region losses."""
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
    region_loss = model_output["region_loss"]
    graph_reg_loss = model_output["graph_reg_loss"]

    total = (
        mae_loss
        + weights.contrastive * contrastive_loss
        + weights.deviation * deviation_loss
        + weights.region * region_loss
        + weights.graph_reg * graph_reg_loss
    )
    return {
        "total": total,
        "mae": mae_loss.detach(),
        "contrastive": contrastive_loss.detach(),
        "deviation": deviation_loss.detach(),
        "region": region_loss.detach(),
        "graph_reg": graph_reg_loss.detach(),
    }

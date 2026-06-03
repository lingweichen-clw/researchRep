"""Region-prototype graph denoising inspired by DarkFarseer SDGS."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RegionPrototypeBuilder(nn.Module):
    """Aggregate node hidden states into BCC-region prototypes."""

    def __init__(self, positive_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("positive_mask", positive_mask.float())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjd->bid", self.positive_mask, hidden)

    def contrastive_loss(self, hidden: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        hidden_norm = F.normalize(hidden, dim=-1)
        prototype_norm = F.normalize(prototypes, dim=-1)
        logits = torch.einsum("bid,bjd->bij", hidden_norm, prototype_norm) / temperature
        labels = torch.arange(hidden.shape[1], device=hidden.device)
        labels = labels.unsqueeze(0).expand(hidden.shape[0], -1).reshape(-1)
        return F.cross_entropy(logits.reshape(-1, hidden.shape[1]), labels)


class GraphDenoisingLayer(nn.Module):
    """Softly downweight unreliable dynamic edges with region prototypes."""

    def __init__(
        self,
        static_adj: torch.Tensor,
        sp_degree: int = 3,
        static_weight: float = 0.15,
        eps: float = 1e-6,
    ):
        super().__init__()
        static_adj = static_adj.float()
        eye = torch.eye(static_adj.shape[0], dtype=static_adj.dtype, device=static_adj.device)
        static_adj = torch.clamp(static_adj + eye, min=0.0)
        static_adj = static_adj / torch.clamp(static_adj.sum(dim=-1, keepdim=True), min=eps)
        self.register_buffer("static_adj", static_adj)
        self.sp_degree = max(int(sp_degree), 1)
        self.static_weight = static_weight
        self.eps = eps

    def forward(
        self,
        base_support: torch.Tensor,
        hidden: torch.Tensor,
        region_prototypes: torch.Tensor,
    ):
        hidden_norm = F.normalize(hidden, dim=-1)
        region_norm = F.normalize(region_prototypes, dim=-1)
        sim_h = torch.einsum("bid,bjd->bij", hidden_norm, hidden_norm)
        sim_s = torch.einsum("bid,bjd->bij", region_norm, region_norm)
        edge_weight = (1.0 - 1.0 / self.sp_degree) * sim_h + (1.0 / self.sp_degree) * sim_s
        reliability = torch.sigmoid(edge_weight).detach()

        cleaned = base_support * reliability
        static_prior = self.static_adj.unsqueeze(0).expand_as(cleaned)
        cleaned = (1.0 - self.static_weight) * cleaned + self.static_weight * static_prior
        cleaned = cleaned / torch.clamp(cleaned.sum(dim=-1, keepdim=True), min=self.eps)

        graph_reg_loss = torch.mean(torch.abs(cleaned - base_support.detach()))
        return cleaned, graph_reg_loss, reliability

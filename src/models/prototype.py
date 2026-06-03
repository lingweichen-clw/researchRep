"""Learnable prototype module adapted from ST-SSDL."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class PrototypeMemory(nn.Module):
    """Map node hidden states into a learnable prototype space."""

    def __init__(self, hidden_dim: int, prototype_num: int, prototype_dim: int):
        super().__init__()
        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim
        self.prototypes = nn.Parameter(torch.empty(prototype_num, prototype_dim))
        self.query_projection = nn.Parameter(torch.empty(hidden_dim, prototype_dim))
        nn.init.xavier_normal_(self.prototypes)
        nn.init.xavier_normal_(self.query_projection)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        query = torch.matmul(hidden, self.query_projection)
        att_score = torch.softmax(torch.matmul(query, self.prototypes.t()), dim=-1)
        value = torch.matmul(att_score, self.prototypes)
        _, indices = torch.topk(att_score, k=2, dim=-1)
        pos = self.prototypes[indices[:, :, 0]]
        neg = self.prototypes[indices[:, :, 1]]
        mask = torch.stack([indices[:, :, 0], indices[:, :, 1]], dim=-1)
        return {
            "value": value,
            "query": query,
            "pos": pos,
            "neg": neg,
            "mask": mask,
            "attention": att_score,
        }


def l1_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Node-wise L1 distance over the feature dimension."""
    return torch.sum(torch.abs(left - right), dim=-1)

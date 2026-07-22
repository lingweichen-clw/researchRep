"""Node-level and event-level normalized retrieval keys."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class RetrievalOutput:
    node_keys: torch.Tensor  # [B, N, Dr]
    event_keys: torch.Tensor  # [B, Dr]
    pooling_weights: torch.Tensor  # [B, P, N]


class RetrievalHead(nn.Module):
    def __init__(self, hidden_dim: int, retrieval_dim: int) -> None:
        super().__init__()
        self.pool_projection = nn.Linear(hidden_dim, hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1, bias=False)
        self.key_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, retrieval_dim),
        )

    def forward(self, hidden: torch.Tensor) -> RetrievalOutput:
        if hidden.ndim != 4:
            raise ValueError("hidden must be [B, P, N, D]")
        logits = self.pool_score(torch.tanh(self.pool_projection(hidden))).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        node_hidden = (weights.unsqueeze(-1) * hidden).sum(dim=1)
        node_keys = functional.normalize(self.key_mlp(node_hidden), p=2, dim=-1, eps=1.0e-8)
        event_keys = functional.normalize(node_keys.mean(dim=1), p=2, dim=-1, eps=1.0e-8)
        return RetrievalOutput(node_keys=node_keys, event_keys=event_keys, pooling_weights=weights)


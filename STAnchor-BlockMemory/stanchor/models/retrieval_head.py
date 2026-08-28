"""Node-level and event-level normalized retrieval keys."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from stanchor.retrieval.semantic_profile import compose_profile_latent_key


@dataclass(frozen=True)
class RetrievalOutput:
    node_keys: torch.Tensor  # [B, N, Dr]
    event_keys: torch.Tensor  # [B, Dr]
    pooling_weights: torch.Tensor  # [B, P, N]
    profile_prediction: torch.Tensor | None = None  # [B, N, Dp]
    profile_keys: torch.Tensor | None = None  # [B, N, Dp]
    latent_keys: torch.Tensor | None = None  # [B, N, Dl]
    event_profile_keys: torch.Tensor | None = None  # [B, Dp]
    event_latent_keys: torch.Tensor | None = None  # [B, Dl]


class RetrievalHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        retrieval_dim: int,
        profile_dim: int = 0,
        latent_dim: int = 0,
        profile_weight: float = 0.25,
        adapter_bottleneck_dim: int = 0,
    ) -> None:
        super().__init__()
        if profile_dim < 0 or latent_dim < 0:
            raise ValueError("profile_dim and latent_dim must be non-negative")
        if (profile_dim == 0) != (latent_dim == 0):
            raise ValueError("profile_dim and latent_dim must either both be zero or both be positive")
        if profile_dim > 0 and profile_dim + latent_dim != retrieval_dim:
            raise ValueError("profile_dim + latent_dim must equal retrieval_dim")
        if not 0.0 <= profile_weight <= 1.0:
            raise ValueError("profile_weight must be in [0, 1]")
        self.pool_projection = nn.Linear(hidden_dim, hidden_dim)
        self.pool_score = nn.Linear(hidden_dim, 1, bias=False)
        self.profile_dim = profile_dim
        self.latent_dim = latent_dim
        self.profile_weight = profile_weight
        self.domain_adapter = (nn.Sequential(nn.Linear(hidden_dim, adapter_bottleneck_dim), nn.GELU(), nn.Linear(adapter_bottleneck_dim, hidden_dim)) if adapter_bottleneck_dim else None)
        if profile_dim > 0:
            self.profile_head = nn.Linear(hidden_dim, profile_dim)
            self.latent_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.key_mlp = None
        else:
            self.profile_head = None
            self.latent_mlp = None
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
        if self.domain_adapter is not None:
            node_hidden = node_hidden + self.domain_adapter(node_hidden)
        if self.profile_head is None or self.latent_mlp is None:
            if self.key_mlp is None:
                raise RuntimeError("legacy retrieval head is not initialized")
            node_keys = functional.normalize(
                self.key_mlp(node_hidden), p=2, dim=-1, eps=1.0e-8
            )
            profile_prediction = profile_keys = latent_keys = None
            event_profile_keys = event_latent_keys = None
            event_keys = functional.normalize(
                node_keys.mean(dim=1), p=2, dim=-1, eps=1.0e-8
            )
        else:
            profile_prediction = self.profile_head(node_hidden)
            latent_prediction = self.latent_mlp(node_hidden)
            node_keys, profile_keys, latent_keys = compose_profile_latent_key(
                profile_prediction,
                latent_prediction,
                self.profile_weight,
            )
            event_profile_keys = functional.normalize(
                profile_keys.mean(dim=1), p=2, dim=-1, eps=1.0e-8
            )
            event_latent_keys = functional.normalize(
                latent_keys.mean(dim=1), p=2, dim=-1, eps=1.0e-8
            )
            event_keys = torch.cat(
                (
                    self.profile_weight**0.5 * event_profile_keys,
                    (1.0 - self.profile_weight) ** 0.5 * event_latent_keys,
                ),
                dim=-1,
            )
        return RetrievalOutput(
            node_keys=node_keys,
            event_keys=event_keys,
            pooling_weights=weights,
            profile_prediction=profile_prediction,
            profile_keys=profile_keys,
            latent_keys=latent_keys,
            event_profile_keys=event_profile_keys,
            event_latent_keys=event_latent_keys,
        )




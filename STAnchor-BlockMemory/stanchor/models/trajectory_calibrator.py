"""Trajectory-conditioned Base-as-candidate calibrator.

The module keeps the established residual-mixture contract while adding a
deployment-available representation of each candidate's complete horizon.
"""

from __future__ import annotations

import torch
from torch import nn

from stanchor.models.downstream import CandidateSetHorizonCorrector


class TrajectoryConditionedCandidateSetHorizonCorrector(CandidateSetHorizonCorrector):
    """Base-as-candidate mixer with a full residual-trajectory token."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channels: int,
        hidden_dim: int = 384,
        state_dim: int = 288,
        attention_heads: int = 4,
        base_logit_init_bias: float = 1.0,
        trajectory_hidden_dim: int = 96,
        use_horizon_embedding: bool = True,
    ) -> None:
        # Keep the parent implementation's stable state/query/token blocks.
        # The new modules are owned here so the legacy constructor remains an
        # exact, explicit fallback for matched comparisons.
        super().__init__(
            context_length,
            horizon,
            channels,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            attention_heads=attention_heads,
            base_logit_init_bias=base_logit_init_bias,
            trajectory_hidden_dim=0,
            use_horizon_embedding=False,
        )
        if trajectory_hidden_dim <= 0:
            raise ValueError("trajectory_hidden_dim must be positive")
        self.trajectory_hidden_dim = trajectory_hidden_dim
        self.use_horizon_embedding = use_horizon_embedding
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(horizon * channels, trajectory_hidden_dim),
            nn.GELU(),
            nn.Linear(trajectory_hidden_dim, hidden_dim),
        )
        self.horizon_embedding = (
            nn.Parameter(torch.empty(horizon, hidden_dim))
            if use_horizon_embedding
            else None
        )
        if self.horizon_embedding is not None:
            nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        self.last_trajectory_token_norm = None

    def forward(
        self,
        history,
        base,
        memory,
        features,
        memory_valid,
        risk_state=None,
        candidates=None,
        aggregation=None,
    ):
        if candidates is None or aggregation is None:
            raise ValueError(
                "TrajectoryConditionedCandidateSetHorizonCorrector requires "
                "candidates and aggregation"
            )
        cand = torch.nan_to_num(
            aggregation.candidate_futures, nan=0.0, posinf=0.0, neginf=0.0
        )
        masks = aggregation.candidate_masks.bool()
        valid = masks.any(-1)
        b, h, n, k, c = cand.shape
        if h != self.horizon or c != self.channels:
            raise ValueError("candidate future shape does not match calibrator")
        if risk_state is None:
            risk_state = self._state(history, base)

        delta = cand - base.unsqueeze(3)
        sim = torch.nan_to_num(
            candidates.shape_scores[:, None, :, :, None].expand(b, h, n, k, 1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        level = torch.nan_to_num(
            (-candidates.level_distances)[:, None, :, :, None].expand(b, h, n, k, 1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        pos = torch.linspace(
            0, 1, h, device=base.device, dtype=base.dtype
        ).view(1, h, 1, 1, 1).expand(b, h, n, k, 1)
        cand_tok = self.candidate_encoder(
            torch.cat((delta, delta.abs(), sim, level, pos), -1)
        )
        trajectory_input = delta.permute(0, 2, 3, 1, 4).reshape(b, n, k, h * c)
        trajectory_tok = self.trajectory_encoder(trajectory_input)
        self.last_trajectory_token_norm = trajectory_tok.detach().norm(dim=-1).mean()
        cand_tok = cand_tok + trajectory_tok.unsqueeze(1)
        if self.horizon_embedding is not None:
            horizon_tok = self.horizon_embedding.view(1, h, 1, 1, self.hidden_dim)
            cand_tok = cand_tok + horizon_tok
        cand_tok = torch.where(valid.unsqueeze(-1), cand_tok, torch.zeros_like(cand_tok))

        history_finite = torch.isfinite(history)
        history_safe = torch.where(history_finite, history, torch.zeros_like(history))
        history_count = history_finite.sum(1).clamp_min(1).to(history.dtype)
        history_mean = history_safe.sum(1) / history_count
        history_centered = (
            history_safe - history_mean.unsqueeze(1)
        ) * history_finite.to(history.dtype)
        ctx_std = (
            history_centered.square().sum(1) / history_count
        ).clamp_min(0.0).sqrt().mean(-1, keepdim=True)
        base_risk = torch.nn.functional.softplus(
            self.risk_probe(risk_state)
        ).permute(0, 2, 1).unsqueeze(-1)
        ctx_std = ctx_std[:, None, :, :].expand(-1, h, -1, -1)
        pos_scalar = torch.linspace(
            0, 1, h, device=base.device, dtype=base.dtype
        ).view(1, h, 1, 1).expand(b, h, n, 1)
        base_type_scalar = torch.ones_like(pos_scalar)
        base_feat = torch.cat(
            (base_risk, ctx_std, pos_scalar, base_type_scalar), dim=-1
        )
        base_tok = self.base_encoder(base_feat) + self.base_type
        query = self.query_proj(risk_state)[:, None, :, :].expand(b, h, n, -1)
        if self.horizon_embedding is not None:
            horizon_tok = self.horizon_embedding.view(1, h, 1, self.hidden_dim)
            query = query + horizon_tok
            base_tok = base_tok + horizon_tok

        all_tok = torch.cat((cand_tok, base_tok.unsqueeze(3)), 3)
        all_tok = all_tok + self.token_refiner(all_tok)
        logits = (all_tok * query.unsqueeze(3)).sum(-1) / (self.hidden_dim**0.5)
        logits[..., -1] = logits[..., -1] + self.base_bias
        all_valid = torch.cat(
            (valid, torch.ones(b, h, n, 1, dtype=torch.bool, device=base.device)),
            -1,
        )
        attn = torch.softmax(logits.masked_fill(~all_valid, -1e9), -1)
        self.current_attention, self.last_attention = attn, attn.detach()
        hist_attn = attn[..., :k]
        residual = (hist_attn.unsqueeze(-1) * delta).sum(3)
        final = base + residual
        final = torch.where(valid.any(-1, keepdim=True), final, base)
        historical_mass = attn[..., :k].sum(-1, keepdim=True)
        dispersion = (
            hist_attn.unsqueeze(-1)
            * (delta - residual.unsqueeze(3)).square()
        ).sum(3).sqrt().mean(-1, keepdim=True)
        contributions = torch.cat(
            (residual.abs().mean(-1, keepdim=True), dispersion), -1
        )
        learned_memory = base + residual
        return final, historical_mass, contributions, learned_memory

"""Trajectory-conditioned Base-as-candidate calibrator.

The module keeps the established residual-mixture contract while adding a
deployment-available representation of each candidate's complete horizon.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

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

class TransformerCandidateRouter(nn.Module):
    """Unified Base/history residual router with standard Q/K/V/O attention.

    Multi-head attention produces a candidate-conditioned query. A shared
    per-candidate routing head then makes the final K+1 decision, so the
    attention value/output projections stay on the prediction gradient path.
    """
    uses_candidate_routing = True

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channels: int,
        hidden_dim: int = 256,
        state_dim: int = 256,
        attention_heads: int = 4,
        base_logit_init_bias: float = 1.0,
        trajectory_hidden_dim: int = 64,
        routing_hidden_dim: int = 128,
        mha_dropout: float = 0.05,
        use_horizon_embedding: bool = True,
    ) -> None:
        super().__init__()
        if min(context_length, horizon, channels, hidden_dim, state_dim) <= 0:
            raise ValueError("router dimensions must be positive")
        if attention_heads <= 0 or hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if trajectory_hidden_dim <= 0 or routing_hidden_dim <= 0:
            raise ValueError("trajectory and routing hidden dimensions must be positive")
        if not 0.0 <= mha_dropout < 1.0:
            raise ValueError("mha_dropout must be in [0, 1)")
        self.context_length = context_length
        self.horizon = horizon
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.attention_heads = attention_heads
        self.trajectory_hidden_dim = trajectory_hidden_dim
        self.routing_hidden_dim = routing_hidden_dim
        self.use_horizon_embedding = use_horizon_embedding

        self.state_encoder = nn.Sequential(
            nn.Linear((context_length + horizon) * channels, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(channels * 2 + 3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.base_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(horizon * channels, trajectory_hidden_dim),
            nn.GELU(),
            nn.Linear(trajectory_hidden_dim, hidden_dim),
        )
        self.query_proj = nn.Linear(state_dim, hidden_dim)
        self.risk_probe = nn.Linear(state_dim, horizon)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=mha_dropout,
            batch_first=True,
        )
        self.routing_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, routing_hidden_dim),
            nn.GELU(),
            nn.Linear(routing_hidden_dim, 1),
        )

        self.base_type = nn.Parameter(torch.zeros(hidden_dim))
        self.base_bias = nn.Parameter(torch.tensor(float(base_logit_init_bias)))
        self.horizon_embedding = (
            nn.Parameter(torch.empty(horizon, hidden_dim))
            if use_horizon_embedding
            else None
        )
        if self.horizon_embedding is not None:
            nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)

        self.current_attention = None
        self.last_attention = None
        self.last_routing_weights = None
        self.last_mha_attention = None
        self.last_base_usage = None
        self.last_routing_entropy = None
        self.last_trajectory_token_norm = None

    def _state(self, history: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or base.ndim != 4:
            raise ValueError("history and base must be [B,T/H,N,C]")
        batch, time, nodes, channels = history.shape
        if time != self.context_length or channels != self.channels:
            raise ValueError("history does not match router configuration")
        if base.shape != (batch, self.horizon, nodes, channels):
            raise ValueError("base does not match router configuration")
        finite = torch.isfinite(history)
        safe = torch.where(finite, history, torch.zeros_like(history))
        count = finite.sum(dim=1, keepdim=True).clamp_min(1).to(history.dtype)
        mean = safe.sum(dim=1, keepdim=True) / count
        centered = torch.where(finite, history - mean, torch.zeros_like(history))
        variance = centered.square().sum(dim=1, keepdim=True) / count
        normalized = centered / (variance + 1.0e-6).sqrt()
        safe_base = torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        state_input = torch.cat((normalized, safe_base), dim=1)
        state_input = state_input.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
        return self.state_encoder(state_input)

    def predict_risk(self, history: torch.Tensor, base: torch.Tensor):
        state = self._state(history, base)
        risk = F.softplus(self.risk_probe(state))
        return risk.permute(0, 2, 1).unsqueeze(-1).contiguous(), state

    def forward(
        self,
        history: torch.Tensor,
        base: torch.Tensor,
        memory: torch.Tensor | None,
        features: torch.Tensor | None,
        memory_valid: torch.Tensor | None,
        risk_state: torch.Tensor | None = None,
        candidates=None,
        aggregation=None,
    ):
        del memory, features, memory_valid
        if candidates is None or aggregation is None:
            raise ValueError("TransformerCandidateRouter requires candidates and aggregation")
        candidate_futures = aggregation.candidate_futures
        candidate_masks = aggregation.candidate_masks.bool()
        if candidate_futures.ndim != 5 or candidate_masks.shape != candidate_futures.shape:
            raise ValueError("candidate futures and masks must be [B,H,N,K,C]")
        batch, horizon, nodes, top_k, channels = candidate_futures.shape
        if horizon != self.horizon or channels != self.channels:
            raise ValueError("candidate future shape does not match router")
        if base.shape != (batch, horizon, nodes, channels):
            raise ValueError("base and candidate futures do not align")
        if candidates.valid.shape != (batch, nodes, top_k):
            raise ValueError("node candidate validity does not align")
        if risk_state is None:
            risk_state = self._state(history, base)
        if risk_state.shape != (batch, nodes, self.state_dim):
            raise ValueError("risk_state has an invalid shape")

        safe_candidates = torch.where(
            candidate_masks,
            torch.nan_to_num(candidate_futures, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(candidate_futures),
        )
        delta = safe_candidates - base.unsqueeze(3)
        candidate_valid = candidate_masks.any(dim=-1)
        candidate_valid = candidate_valid & candidates.valid.bool()[:, None, :, :]
        position = torch.linspace(
            0.0, 1.0, horizon, dtype=base.dtype, device=base.device
        ).view(1, horizon, 1, 1, 1)
        shape_score = torch.nan_to_num(
            candidates.shape_scores[:, None, :, :, None], nan=0.0, posinf=0.0, neginf=0.0
        ).expand(batch, horizon, nodes, top_k, 1)
        level_score = torch.nan_to_num(
            -candidates.level_distances[:, None, :, :, None], nan=0.0, posinf=0.0, neginf=0.0
        ).expand(batch, horizon, nodes, top_k, 1)
        local_features = torch.cat(
            (delta, delta.abs(), shape_score, level_score,
             position.expand(batch, horizon, nodes, top_k, 1)), dim=-1
        )
        candidate_tokens = self.candidate_encoder(local_features)
        trajectory_input = delta.permute(0, 2, 3, 1, 4).reshape(
            batch, nodes, top_k, horizon * channels
        )
        trajectory_tokens = self.trajectory_encoder(trajectory_input)
        self.last_trajectory_token_norm = trajectory_tokens.detach().norm(dim=-1).mean()
        candidate_tokens = candidate_tokens + trajectory_tokens.unsqueeze(1)
        horizon_embedding = None
        if self.horizon_embedding is not None:
            horizon_embedding = self.horizon_embedding.view(1, horizon, 1, 1, self.hidden_dim)
            candidate_tokens = candidate_tokens + horizon_embedding
        candidate_tokens = self.token_norm(candidate_tokens)
        candidate_tokens = torch.where(
            candidate_valid.unsqueeze(-1), candidate_tokens, torch.zeros_like(candidate_tokens)
        )

        context_finite = torch.isfinite(history)
        context_safe = torch.where(context_finite, history, torch.zeros_like(history))
        context_count = context_finite.sum(dim=1).clamp_min(1).to(history.dtype)
        context_mean = context_safe.sum(dim=1) / context_count
        context_centered = (
            context_safe - context_mean.unsqueeze(1)
        ) * context_finite.to(history.dtype)
        context_std = (
            context_centered.square().sum(dim=1) / context_count
        ).clamp_min(0.0).sqrt().mean(dim=-1, keepdim=True)
        base_risk = F.softplus(self.risk_probe(risk_state)).permute(0, 2, 1).unsqueeze(-1)
        base_features = torch.cat(
            (
                base_risk,
                context_std[:, None, :, :].expand(-1, horizon, -1, -1),
                position.view(1, horizon, 1, 1).expand(batch, horizon, nodes, 1),
                torch.ones(batch, horizon, nodes, 1, dtype=base.dtype, device=base.device),
            ), dim=-1
        )
        base_token = self.base_encoder(base_features) + self.base_type
        if horizon_embedding is not None:
            base_token = base_token + horizon_embedding.squeeze(3)
        base_token = self.token_norm(base_token)

        all_tokens = torch.cat((candidate_tokens, base_token.unsqueeze(3)), dim=3)
        all_valid = torch.cat(
            (
                candidate_valid,
                torch.ones(batch, horizon, nodes, 1, dtype=torch.bool, device=base.device),
            ), dim=-1
        )
        query = self.query_proj(risk_state)[:, None, :, :].expand(
            batch, horizon, nodes, self.hidden_dim
        )
        if horizon_embedding is not None:
            query = query + horizon_embedding.squeeze(3)

        flat = batch * horizon * nodes
        query_sequence = query.reshape(flat, 1, self.hidden_dim)
        token_sequence = all_tokens.reshape(flat, top_k + 1, self.hidden_dim)
        key_padding_mask = ~all_valid.reshape(flat, top_k + 1)
        mha_output, mha_weights = self.mha(
            query_sequence, token_sequence, token_sequence,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        query_conditioned = self.query_norm(
            query_sequence.squeeze(1) + mha_output.squeeze(1)
        ).reshape(batch, horizon, nodes, self.hidden_dim)
        mha_weights = mha_weights.squeeze(2).reshape(
            batch, horizon, nodes, self.attention_heads, top_k + 1
        )
        self.last_mha_attention = mha_weights.detach()

        route_query = query_conditioned.unsqueeze(3).expand(
            batch, horizon, nodes, top_k + 1, self.hidden_dim
        )
        route_features = torch.cat(
            (route_query, all_tokens, route_query * all_tokens), dim=-1
        )
        logits = self.routing_head(route_features).squeeze(-1)
        base_bias = torch.zeros_like(logits)
        base_bias[..., -1] = self.base_bias
        logits = logits + base_bias
        logits = logits.masked_fill(~all_valid, torch.finfo(logits.dtype).min)
        routing_weights = torch.softmax(logits, dim=-1)
        self.current_attention = routing_weights
        self.last_attention = routing_weights.detach()
        self.last_routing_weights = routing_weights.detach()
        self.last_base_usage = routing_weights[..., -1:].detach()
        safe_weights = routing_weights.clamp_min(1.0e-8)
        self.last_routing_entropy = (
            -(safe_weights * safe_weights.log()).sum(dim=-1, keepdim=True).detach()
        )

        historical_weights = routing_weights[..., :top_k]
        residual = (historical_weights.unsqueeze(-1) * delta).sum(dim=3)
        final = base + residual
        historical_valid = candidate_valid.any(dim=-1, keepdim=True)
        final = torch.where(historical_valid, final, base)
        dispersion = (
            historical_weights.unsqueeze(-1)
            * (delta - residual.unsqueeze(3)).square()
        ).sum(dim=3).sqrt().mean(dim=-1, keepdim=True)
        contributions = torch.cat(
            (residual.abs().mean(dim=-1, keepdim=True), dispersion), dim=-1
        )
        learned_memory = base + residual
        return final, historical_weights.sum(dim=-1, keepdim=True), contributions, learned_memory

"""Retrieval-aware multi-head residual routing for downstream calibration."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RetrievalAwareMHAResidualRouter(nn.Module):
    """Use standard MHA output to route K residual experts plus Base fallback."""

    uses_candidate_routing = True
    uses_retrieval_node_keys = True

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channels: int,
        retrieval_dim: int = 64,
        hidden_dim: int = 256,
        retrieval_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        candidate_hidden_dim: int = 128,
        routing_dim: int = 128,
        attention_heads: int = 4,
        mha_dropout: float = 0.05,
        base_logit_init_bias: float = 1.0,
    ) -> None:
        super().__init__()
        dims = (context_length, horizon, channels, retrieval_dim, hidden_dim,
                retrieval_hidden_dim, fusion_hidden_dim, candidate_hidden_dim,
                routing_dim)
        if any(v <= 0 for v in dims):
            raise ValueError("router dimensions must be positive")
        if attention_heads <= 0 or hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0.0 <= mha_dropout < 1.0:
            raise ValueError("mha_dropout must be in [0, 1)")
        self.context_length = context_length
        self.horizon = horizon
        self.channels = channels
        self.retrieval_dim = retrieval_dim
        self.hidden_dim = hidden_dim
        self.state_dim = fusion_hidden_dim
        self.attention_heads = attention_heads
        self.routing_dim = routing_dim

        self.state_encoder = nn.Sequential(
            nn.Linear((context_length + horizon) * channels, fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
        )
        self.retrieval_encoder = nn.Sequential(
            nn.Linear(retrieval_dim, retrieval_hidden_dim),
            nn.GELU(),
            nn.Linear(retrieval_hidden_dim, hidden_dim),
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(fusion_hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_summary_encoder = nn.Sequential(
            nn.Linear(channels * 5 + 4, candidate_hidden_dim),
            nn.GELU(),
            nn.Linear(candidate_hidden_dim, hidden_dim),
        )
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(horizon * channels * 2, candidate_hidden_dim),
            nn.GELU(),
            nn.Linear(candidate_hidden_dim, hidden_dim),
        )
        self.base_encoder = nn.Sequential(
            nn.Linear(fusion_hidden_dim + hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mha = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=mha_dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_routing = nn.Linear(hidden_dim, routing_dim)
        self.candidate_routing = nn.Linear(hidden_dim, routing_dim)
        self.horizon_embedding = nn.Parameter(torch.empty(horizon, hidden_dim))
        self.base_type = nn.Parameter(torch.zeros(hidden_dim))
        self.base_bias = nn.Parameter(torch.tensor(float(base_logit_init_bias)))
        self.risk_probe = nn.Linear(fusion_hidden_dim, horizon)
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)

        self.current_attention = None
        self.last_attention = None
        self.last_routing_weights = None
        self.last_mha_attention = None
        self.last_base_usage = None
        self.last_routing_entropy = None

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
        memory=None,
        features=None,
        memory_valid=None,
        risk_state: torch.Tensor | None = None,
        candidates=None,
        aggregation=None,
        retrieval_node_keys: torch.Tensor | None = None,
    ):
        del memory, features, memory_valid
        if candidates is None or aggregation is None:
            raise ValueError("router requires candidates and aggregation")
        future = aggregation.candidate_futures
        mask = aggregation.candidate_masks.bool()
        if future.ndim != 5 or mask.shape != future.shape:
            raise ValueError("candidate future/mask must be [B,H,N,K,C]")
        batch, horizon, nodes, top_k, channels = future.shape
        if (horizon, channels) != (self.horizon, self.channels):
            raise ValueError("candidate future shape does not match router")
        if base.shape != (batch, horizon, nodes, channels):
            raise ValueError("base and candidate futures do not align")
        if candidates.valid.shape != (batch, nodes, top_k):
            raise ValueError("node candidate validity does not align")
        if risk_state is None:
            risk_state = self._state(history, base)
        if risk_state.shape != (batch, nodes, self.state_dim):
            raise ValueError("risk_state has an invalid shape")

        safe_future = torch.where(
            mask, torch.nan_to_num(future, nan=0.0, posinf=0.0, neginf=0.0),
            torch.zeros_like(future)
        )
        delta = safe_future - base.unsqueeze(3)
        valid = mask.any(dim=-1).any(dim=1) & candidates.valid.bool()
        mask_f = mask.to(delta.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean = (delta * mask_f).sum(dim=1) / denom
        centered = delta - mean.unsqueeze(1)
        std = ((centered.square() * mask_f).sum(dim=1) / denom).clamp_min(0).sqrt()
        last = delta[:, -1]
        abs_mean = (delta.abs() * mask_f).sum(dim=1) / denom
        direction = (((delta > 0).to(delta.dtype) * mask_f).sum(dim=1) / denom)
        shape = torch.nan_to_num(candidates.shape_scores, nan=0.0, posinf=0.0, neginf=0.0)
        level = torch.nan_to_num(-candidates.level_distances, nan=0.0, posinf=0.0, neginf=0.0)
        rank = torch.linspace(1.0, 0.0, top_k, device=base.device, dtype=base.dtype)
        rank = rank.view(1, 1, top_k).expand(batch, nodes, top_k)
        valid_ratio = mask_f.mean(dim=(1, 4))
        summary = torch.cat((mean, std, last, abs_mean, direction,
                             shape.unsqueeze(-1), level.unsqueeze(-1),
                             rank.unsqueeze(-1), valid_ratio.unsqueeze(-1)), dim=-1)
        tokens = self.candidate_summary_encoder(summary)
        trajectory = torch.cat((delta, delta.abs()), dim=-1)
        trajectory = trajectory.permute(0, 2, 3, 1, 4).reshape(
            batch, nodes, top_k, horizon * channels * 2
        )
        tokens = tokens + self.trajectory_encoder(trajectory)
        tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))

        if retrieval_node_keys is None:
            retrieval_node_keys = torch.zeros(
                batch, nodes, self.retrieval_dim, dtype=base.dtype, device=base.device
            )
        if retrieval_node_keys.shape != (batch, nodes, self.retrieval_dim):
            raise ValueError("retrieval_node_keys must be [B,N,retrieval_dim]")
        key_token = self.retrieval_encoder(torch.nan_to_num(
            retrieval_node_keys, nan=0.0, posinf=0.0, neginf=0.0
        ))
        query_base = self.query_fusion(torch.cat((risk_state, key_token), dim=-1))
        query = query_base.unsqueeze(2) + self.horizon_embedding.view(
            1, 1, horizon, self.hidden_dim
        )
        context_std = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0).std(
            dim=1, unbiased=False
        ).mean(-1, keepdim=True)
        base_risk = F.softplus(self.risk_probe(risk_state)).mean(-1, keepdim=True)
        base_token = self.base_encoder(torch.cat((risk_state, key_token,
                                                    context_std, base_risk), dim=-1))
        base_token = base_token + self.base_type
        all_tokens = torch.cat((tokens, base_token.unsqueeze(2)), dim=2)
        all_valid = torch.cat((valid, torch.ones(batch, nodes, 1,
                                                  dtype=torch.bool, device=base.device)), dim=-1)

        q_seq = query.reshape(batch * nodes, horizon, self.hidden_dim)
        c_seq = all_tokens.reshape(batch * nodes, top_k + 1, self.hidden_dim)
        mha_out, mha_weights = self.mha(
            q_seq, c_seq, c_seq,
            key_padding_mask=~all_valid.reshape(batch * nodes, top_k + 1),
            need_weights=True,
            average_attn_weights=False,
        )
        conditioned = self.query_norm(q_seq + mha_out).reshape(
            batch, nodes, horizon, self.hidden_dim
        )
        self.last_mha_attention = mha_weights.detach().reshape(
            batch, nodes, self.attention_heads, horizon, top_k + 1
        )
        query_r = self.query_routing(conditioned)
        candidate_r = self.candidate_routing(all_tokens)
        logits = torch.matmul(query_r, candidate_r.transpose(-1, -2)) / (self.routing_dim ** 0.5)
        logits[..., -1] = logits[..., -1] + self.base_bias
        logits = logits.masked_fill(~all_valid.unsqueeze(2), torch.finfo(logits.dtype).min)
        routing = torch.softmax(logits, dim=-1)
        self.current_attention = routing.permute(0, 2, 1, 3).contiguous()
        self.last_attention = routing.detach()
        self.last_routing_weights = routing.detach()
        self.last_base_usage = routing[..., -1:].detach()
        safe_routing = routing.clamp_min(1.0e-8)
        self.last_routing_entropy = (-(safe_routing * safe_routing.log()).sum(-1, keepdim=True)).detach()

        weights = routing[..., :top_k].permute(0, 2, 1, 3)
        residual = (weights.unsqueeze(-1) * delta).sum(3)
        final = base + residual
        final = torch.where(valid.any(-1)[:, None, :, None], final, base)
        dispersion = (weights.unsqueeze(-1) * (delta - residual.unsqueeze(3)).square()).sum(3).clamp_min(0).sqrt().mean(-1, keepdim=True)
        contributions = torch.cat((residual.abs().mean(-1, keepdim=True), dispersion), dim=-1)
        learned_memory = base + residual
        historical_mass = routing[..., :top_k].sum(-1).permute(0, 2, 1).unsqueeze(-1)
        return final, historical_mass, contributions, learned_memory

"""Calendar-filtered event search, node reranking, and value retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from stanchor.bank.storage import MemoryBank


@dataclass(frozen=True)
class EventCandidates:
    event_ids: torch.Tensor  # [B, R], -1 means padding
    scores: torch.Tensor  # [B, R]
    valid: torch.Tensor  # [B, R]


@dataclass(frozen=True)
class NodeCandidates:
    event_ids: torch.Tensor  # [B, N, K]
    total_scores: torch.Tensor  # [B, N, K]
    shape_scores: torch.Tensor  # [B, N, K]
    level_distances: torch.Tensor  # [B, N, K]
    weights: torch.Tensor  # [B, N, K]
    valid: torch.Tensor  # [B, N, K]
    profile_scores: torch.Tensor | None = None  # [B, N, K]
    latent_scores: torch.Tensor | None = None  # [B, N, K]


@dataclass(frozen=True)
class AggregationOutput:
    prediction: torch.Tensor  # [B, H, N, C]
    variance: torch.Tensor  # [B, H, N, C]
    valid: torch.Tensor  # [B, H, N, C]
    candidate_futures: torch.Tensor  # [B, H, N, K, C]
    candidate_masks: torch.Tensor  # [B, H, N, K, C]


class TwoStageRetriever:
    def __init__(
        self,
        bank: MemoryBank,
        event_top_r: int,
        node_top_k: int,
        level_weight: float,
        level_temperature: float,
        search_temperature: float,
        device: torch.device,
        profile_weight_override: float | None = None,
    ) -> None:
        if event_top_r < node_top_k:
            raise ValueError("event_top_r must be >= node_top_k")
        if level_temperature <= 0 or search_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if profile_weight_override is not None:
            if bank.manifest.schema_version != 2:
                raise ValueError("profile_weight_override requires a v2 profile/latent Bank")
            if not 0.0 <= profile_weight_override <= 1.0:
                raise ValueError("profile_weight_override must be in [0, 1]")
        self.bank = bank
        self.event_top_r = event_top_r
        self.node_top_k = node_top_k
        self.level_weight = level_weight
        self.level_temperature = level_temperature
        self.search_temperature = search_temperature
        self.profile_weight_override = profile_weight_override
        self.device = device
        self.event_keys = torch.from_numpy(bank.event_keys_memory).to(device)
        calendar_event_ids = getattr(bank, "calendar_event_ids_padded", None)
        if calendar_event_ids is None:
            calendar_event_ids = bank.calendar.padded_event_ids()
        calendar_future_end = getattr(bank, "calendar_future_end_padded", None)
        if calendar_future_end is None:
            calendar_future_end = np.full_like(calendar_event_ids, -1, dtype=np.int64)
            valid_calendar = calendar_event_ids >= 0
            if bool(valid_calendar.any()):
                calendar_future_end[valid_calendar] = np.asarray(bank.future_end)[
                    calendar_event_ids[valid_calendar]
                ]
        self._calendar_event_ids = torch.from_numpy(calendar_event_ids).to(device)
        self._calendar_future_end = torch.from_numpy(calendar_future_end).to(device)

    @torch.no_grad()
    def search_events(
        self,
        query_event_keys: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
        context_start: torch.Tensor,
    ) -> EventCandidates:
        if query_event_keys.ndim != 2:
            raise ValueError("query_event_keys must be [B, Dr]")
        batch = query_event_keys.shape[0]
        ids = torch.full((batch, self.event_top_r), -1, dtype=torch.long, device=self.device)
        scores = torch.full((batch, self.event_top_r), -torch.inf, device=self.device)
        valid = torch.zeros((batch, self.event_top_r), dtype=torch.bool, device=self.device)
        bucket = weekday.to(device=self.device, dtype=torch.long) * int(
            self.bank.manifest.slots_per_day
        ) + slot.to(device=self.device, dtype=torch.long)
        bucket_ids = self._calendar_event_ids.index_select(0, bucket)
        bucket_valid = bucket_ids >= 0
        causal = self._calendar_future_end.index_select(0, bucket) < context_start.to(
            device=self.device, dtype=torch.long
        ).unsqueeze(1)
        legal_mask = bucket_valid & causal
        safe_ids = bucket_ids.clamp_min(0)
        candidate_keys = self.event_keys.index_select(0, safe_ids.reshape(-1)).reshape(
            safe_ids.shape[0], safe_ids.shape[1], -1
        )
        candidate_scores = torch.einsum(
            "bd,brd->br", query_event_keys, candidate_keys
        ).masked_fill(~legal_mask, -torch.inf)
        count = min(self.event_top_r, candidate_scores.shape[1])
        ids = torch.full(
            (batch, self.event_top_r), -1, dtype=torch.long, device=self.device
        )
        scores = torch.full(
            (batch, self.event_top_r), -torch.inf, dtype=torch.float32, device=self.device
        )
        valid = torch.zeros(
            (batch, self.event_top_r), dtype=torch.bool, device=self.device
        )
        if count:
            top_scores, local_ids = torch.topk(candidate_scores, count, dim=1)
            top_ids = safe_ids.gather(1, local_ids)
            top_valid = legal_mask.gather(1, local_ids) & torch.isfinite(top_scores)
            ids[:, :count] = torch.where(
                top_valid, top_ids, torch.full_like(top_ids, -1)
            )
            scores[:, :count] = torch.where(
                top_valid, top_scores, torch.full_like(top_scores, -torch.inf)
            )
            valid[:, :count] = top_valid
        return EventCandidates(ids, scores, valid)

    @torch.no_grad()
    def rerank_nodes(
        self,
        query_node_keys: torch.Tensor,
        query_levels: torch.Tensor,
        events: EventCandidates,
    ) -> NodeCandidates:
        if query_node_keys.ndim != 3:
            raise ValueError("query_node_keys must be [B, N, Dr]")
        batch, nodes, retrieval_dim = query_node_keys.shape
        if nodes != self.bank.manifest.num_nodes or retrieval_dim != self.bank.manifest.retrieval_dim:
            raise ValueError("query keys do not match bank schema")
        safe_ids = events.event_ids.clamp_min(0).cpu().numpy()
        candidate_keys = torch.from_numpy(
            np.asarray(self.bank.node_keys[safe_ids], dtype=np.float32)
        ).to(self.device)  # [B, R, N, Dr]
        candidate_levels = torch.from_numpy(
            np.asarray(self.bank.level_features[safe_ids], dtype=np.float32)
        ).to(self.device)  # [B, R, N, 4C]
        shape = torch.einsum("bnd,brnd->bnr", query_node_keys, candidate_keys)
        profile = latent = None
        if self.bank.manifest.schema_version == 2:
            profile_dim = self.bank.manifest.profile_dim
            if profile_dim <= 0 or profile_dim >= retrieval_dim:
                raise ValueError("invalid profile/latent dimensions in v2 Bank")
            query_profile = functional.normalize(
                query_node_keys[..., :profile_dim], dim=-1
            )
            candidate_profile = functional.normalize(
                candidate_keys[..., :profile_dim], dim=-1
            )
            query_latent = functional.normalize(
                query_node_keys[..., profile_dim:], dim=-1
            )
            candidate_latent = functional.normalize(
                candidate_keys[..., profile_dim:], dim=-1
            )
            profile = torch.einsum(
                "bnd,brnd->bnr", query_profile, candidate_profile
            )
            latent = torch.einsum(
                "bnd,brnd->bnr", query_latent, candidate_latent
            )
            if self.profile_weight_override is not None:
                gamma = self.profile_weight_override
                shape = gamma * profile + (1.0 - gamma) * latent
        level_distance = (query_levels[:, None, :, :] - candidate_levels).abs().mean(dim=-1).permute(0, 2, 1)
        total = shape + self.level_weight * torch.exp(-level_distance / self.level_temperature)
        event_valid = events.valid[:, None, :].expand(batch, nodes, -1)
        total = total.masked_fill(~event_valid, -torch.inf)
        top_scores, local_top = torch.topk(total, self.node_top_k, dim=-1)
        expanded_event_ids = events.event_ids[:, None, :].expand(batch, nodes, -1)
        global_ids = expanded_event_ids.gather(-1, local_top)
        shape_top = shape.gather(-1, local_top)
        level_top = level_distance.gather(-1, local_top)
        profile_top = None if profile is None else profile.gather(-1, local_top)
        latent_top = None if latent is None else latent.gather(-1, local_top)
        valid_top = event_valid.gather(-1, local_top) & torch.isfinite(top_scores)
        logits = top_scores / self.search_temperature
        stable = logits.masked_fill(~valid_top, -torch.inf)
        max_value = stable.amax(dim=-1, keepdim=True)
        max_value = torch.where(torch.isfinite(max_value), max_value, torch.zeros_like(max_value))
        exponent = torch.where(valid_top, torch.exp(logits - max_value), torch.zeros_like(logits))
        weights = exponent / exponent.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return NodeCandidates(
            global_ids,
            top_scores,
            shape_top,
            level_top,
            weights,
            valid_top,
            profile_top,
            latent_top,
        )

    @torch.no_grad()
    def aggregate(self, candidates: NodeCandidates) -> AggregationOutput:
        event_ids = candidates.event_ids.clamp_min(0).cpu().numpy()
        valid_candidate = candidates.valid.cpu().numpy()
        batch, nodes, top_k = event_ids.shape
        node_ids = np.broadcast_to(np.arange(nodes)[None, :, None], (batch, nodes, top_k))
        flat_events = event_ids.reshape(-1)
        flat_nodes = node_ids.reshape(-1)
        selected_future = np.asarray(
            self.bank.future_values[flat_events, :, flat_nodes, :], dtype=np.float32
        )  # [B*N*K, H, C]
        selected_mask = np.asarray(
            self.bank.future_masks[flat_events, :, flat_nodes, :], dtype=np.uint8
        ).astype(bool)
        horizon, channels = selected_future.shape[1], selected_future.shape[2]
        future = torch.from_numpy(
            selected_future.reshape(batch, nodes, top_k, horizon, channels).transpose(0, 3, 1, 2, 4)
        ).to(self.device)
        mask = torch.from_numpy(
            selected_mask.reshape(batch, nodes, top_k, horizon, channels).transpose(0, 3, 1, 2, 4)
        ).to(self.device)
        mask = mask & torch.from_numpy(valid_candidate[:, None, :, :, None]).to(self.device)
        base_weights = candidates.weights[:, None, :, :, None]
        effective_weights = base_weights * mask.to(base_weights.dtype)
        denominator = effective_weights.sum(dim=3)
        prediction = (effective_weights * future).sum(dim=3) / denominator.clamp_min(1.0e-8)
        difference = future - prediction.unsqueeze(3)
        variance = (effective_weights * difference.square()).sum(dim=3) / denominator.clamp_min(1.0e-8)
        valid = denominator > 0
        prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
        variance = torch.where(valid, variance, torch.zeros_like(variance))
        return AggregationOutput(prediction, variance, valid, future, mask)

    @torch.no_grad()
    def retrieve(
        self,
        query_event_keys: torch.Tensor,
        query_node_keys: torch.Tensor,
        query_levels: torch.Tensor,
        weekday: torch.Tensor,
        slot: torch.Tensor,
        context_start: torch.Tensor,
    ) -> tuple[EventCandidates, NodeCandidates, AggregationOutput]:
        events = self.search_events(query_event_keys, weekday, slot, context_start)
        nodes = self.rerank_nodes(query_node_keys, query_levels, events)
        return events, nodes, self.aggregate(nodes)

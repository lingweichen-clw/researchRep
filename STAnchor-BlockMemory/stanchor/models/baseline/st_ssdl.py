"""Complete ST-SSDL forecasting backbone adapted from the official METR-LA code."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional


class AGCN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, num_support: int) -> None:
        super().__init__()
        if cheb_k < 1 or num_support < 1:
            raise ValueError("cheb_k and num_support must be positive")
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.empty(num_support * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("AGCN expects [B, N, Din]")
        features = []
        for support in supports:
            identity = torch.eye(support.shape[-1], device=x.device, dtype=x.dtype)
            if support.ndim == 2:
                basis = [identity, support]
                for _ in range(2, self.cheb_k):
                    basis.append(2.0 * support @ basis[-1] - basis[-2])
                features.extend(torch.einsum("nm,bmc->bnc", graph, x) for graph in basis[: self.cheb_k])
            else:
                identity = identity.unsqueeze(0).expand(support.shape[0], -1, -1)
                basis = [identity, support]
                for _ in range(2, self.cheb_k):
                    basis.append(torch.matmul(2.0 * support, basis[-1]) - basis[-2])
                features.extend(torch.einsum("bnm,bmc->bnc", graph, x) for graph in basis[: self.cheb_k])
        stacked = torch.cat(features, dim=-1)
        return torch.einsum("bni,io->bno", stacked, self.weights.to(x.dtype)) + self.bias.to(x.dtype)


class AGCRNCell(nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, num_support: int) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + dim_out, 2 * dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in + dim_out, dim_out, cheb_k, num_support)

    def forward(self, x: torch.Tensor, state: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        state = state.to(device=x.device, dtype=x.dtype)
        input_and_state = torch.cat((x, state), dim=-1)
        reset_update = torch.sigmoid(self.gate(input_and_state, supports))
        update_gate, reset_gate = torch.split(reset_update, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, update_gate * state), dim=-1)
        hidden_candidate = torch.tanh(self.update(candidate, supports))
        return reset_gate * state + (1.0 - reset_gate) * hidden_candidate


class ADCRNNEncoder(nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, rnn_layers: int, num_support: int) -> None:
        super().__init__()
        if rnn_layers < 1:
            raise ValueError("rnn_layers must be >= 1")
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in if index == 0 else dim_out, dim_out, cheb_k, num_support) for index in range(rnn_layers)]
        )

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 4 or x.shape[2] != self.node_num or x.shape[3] != self.input_dim:
            raise ValueError("encoder expects [B, T, N, Din]")
        current = x
        final_states = []
        for cell in self.cells:
            state = current.new_zeros(current.shape[0], self.node_num, cell.hidden_dim)
            outputs = []
            for time_index in range(current.shape[1]):
                state = cell(current[:, time_index], state, supports)
                outputs.append(state)
            current = torch.stack(outputs, dim=1)
            final_states.append(state)
        return current, final_states


class ADCRNNDecoder(nn.Module):
    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, rnn_layers: int, num_support: int) -> None:
        super().__init__()
        if rnn_layers < 1:
            raise ValueError("rnn_layers must be >= 1")
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.cells = nn.ModuleList(
            [AGCRNCell(node_num, dim_in if index == 0 else dim_out, dim_out, cheb_k, num_support) for index in range(rnn_layers)]
        )

    def forward(self, xt: torch.Tensor, states: list[torch.Tensor], supports: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if xt.ndim != 3 or xt.shape[1] != self.node_num or xt.shape[2] != self.input_dim:
            raise ValueError("decoder expects [B, N, Din]")
        current = xt
        next_states = []
        for cell, state in zip(self.cells, states):
            state = cell(current, state, supports)
            next_states.append(state)
            current = state
        return current, next_states


class STSSDLForecastBackbone(nn.Module):
    """Official ST-SSDL model with a project-native [B,H,N,C] interface."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        num_nodes: int,
        input_channels: int,
        output_channels: int,
        adj_supports: torch.Tensor,
        rnn_units: int = 128,
        rnn_layers: int = 1,
        cheb_k: int = 3,
        prototype_num: int = 20,
        prototype_dim: int = 64,
        tod_embed_dim: int = 20,
        node_embedding_dim: int = 25,
        input_embedding_dim: int = 3,
        adaptive_embedding_dim: int = 0,
        use_ste: bool = True,
        use_curriculum_learning: bool = True,
        cl_decay_steps: int = 2000,
        triplet_margin: float = 0.5,
        slots_per_day: int = 288,
    ) -> None:
        super().__init__()
        if input_channels != 1 or output_channels != 1:
            raise ValueError("ST-SSDL adapter currently supports one input/output channel")
        if adj_supports.ndim != 3 or adj_supports.shape[-1] != num_nodes or adj_supports.shape[-2] != num_nodes:
            raise ValueError("adj_supports must be [num_support, N, N]")
        self.context_length = context_length
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.input_dim = input_channels
        self.output_dim = output_channels
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.tod_embed_dim = tod_embed_dim
        self.node_embedding_dim = node_embedding_dim
        self.input_embedding_dim = input_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.use_ste = use_ste
        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps
        self.tday = slots_per_day
        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim
        self.triplet_margin = triplet_margin
        self.total_embedding_dim = tod_embed_dim + adaptive_embedding_dim + node_embedding_dim
        self.batches_seen = 0
        self._auxiliary: dict[str, torch.Tensor] | None = None
        self.register_buffer("adj_supports", adj_supports.float())
        self.register_buffer("history_table", torch.zeros(num_nodes, 7 * slots_per_day), persistent=False)
        self._history_ready = False

        self.prototypes = nn.Parameter(torch.empty(prototype_num, prototype_dim))
        self.prototype_query = nn.Parameter(torch.empty(rnn_units, prototype_dim))
        nn.init.xavier_normal_(self.prototypes)
        nn.init.xavier_normal_(self.prototype_query)

        if use_ste:
            if adaptive_embedding_dim > 0:
                self.adaptive_embedding = nn.Parameter(torch.empty(context_length, num_nodes, adaptive_embedding_dim))
                nn.init.xavier_uniform_(self.adaptive_embedding)
            if input_embedding_dim > 0:
                self.input_proj = nn.Linear(input_channels, input_embedding_dim)
            self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
            self.time_embedding = nn.Parameter(torch.empty(slots_per_day, tod_embed_dim))
            nn.init.xavier_uniform_(self.node_embedding)
            nn.init.xavier_uniform_(self.time_embedding)
            encoder_dim = input_embedding_dim + self.total_embedding_dim
            decoder_in = input_embedding_dim + self.total_embedding_dim - adaptive_embedding_dim
        else:
            encoder_dim = input_channels
            decoder_in = output_channels + 1
        self.encoder = ADCRNNEncoder(num_nodes, encoder_dim, rnn_units, cheb_k, rnn_layers, adj_supports.shape[0])
        self.decoder_dim = rnn_units + prototype_dim
        self.decoder = ADCRNNDecoder(num_nodes, decoder_in, self.decoder_dim, cheb_k, rnn_layers, 1)
        self.proj = nn.Linear(self.decoder_dim, output_channels)
        self.hypernet = nn.Linear(self.decoder_dim * 2, tod_embed_dim if tod_embed_dim > 0 else 16)
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin)

    def set_history_table(self, table: torch.Tensor) -> None:
        if table.ndim != 2 or table.shape[0] != self.num_nodes:
            raise ValueError("history table must be [N, 7 * slots_per_day]")
        self.history_table = table.to(device=self.history_table.device, dtype=self.history_table.dtype)
        self._history_ready = True

    def lookup_history(self, weekday: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        if not self._history_ready:
            raise ValueError("ST-SSDL history table has not been attached")
        if weekday.shape != slot.shape or weekday.ndim != 2:
            raise ValueError("weekday and slot must be [B, T]")
        keys = weekday.long() * self.tday + slot.long()
        keys = keys.clamp(0, self.history_table.shape[1] - 1)
        values = self.history_table.transpose(0, 1)[keys]
        return values.unsqueeze(-1)

    def future_slots(self, context_slot: torch.Tensor) -> torch.Tensor:
        if context_slot.ndim != 2:
            raise ValueError("context_slot must be [B, T]")
        last = context_slot[:, -1]
        steps = torch.arange(1, self.horizon + 1, device=context_slot.device)
        return (last.unsqueeze(1) + steps) % self.tday

    def pop_auxiliary_losses(self) -> dict[str, torch.Tensor] | None:
        auxiliary = self._auxiliary
        self._auxiliary = None
        return auxiliary

    def _supports(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [support.to(device=x.device, dtype=x.dtype) for support in self.adj_supports]

    def _time_embedding(self, slot: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if slot.ndim == 2:
            slot = slot.unsqueeze(-1).expand(-1, -1, num_nodes)
        index = slot.long().clamp(0, self.tday - 1)
        return self.time_embedding.to(slot.device)[index]

    def _ste_features(self, values: torch.Tensor, slot: torch.Tensor, include_adaptive: bool) -> torch.Tensor:
        features = [self.input_proj(values) if self.input_embedding_dim > 0 else values]
        if self.tod_embed_dim > 0:
            features.append(self._time_embedding(slot, values.shape[2]))
        if include_adaptive and self.adaptive_embedding_dim > 0:
            features.append(self.adaptive_embedding.expand(values.shape[0], *self.adaptive_embedding.shape))
        if self.node_embedding_dim > 0:
            node = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(values.shape[0], values.shape[1], -1, -1)
            features.append(node)
        return torch.cat(features, dim=-1)

    def _query_prototypes(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = hidden @ self.prototype_query
        score = torch.softmax(query @ self.prototypes.transpose(0, 1), dim=-1)
        value = score @ self.prototypes
        _, index = torch.topk(score, k=2, dim=-1)
        positive = self.prototypes[index[:, :, 0]]
        negative = self.prototypes[index[:, :, 1]]
        mask = torch.stack((index[:, :, 0], index[:, :, 1]), dim=-1)
        return value, query, positive, negative, mask

    def _curriculum_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))

    @staticmethod
    def _dynamic_support(node_embeddings: torch.Tensor) -> torch.Tensor:
        similarity = torch.einsum("bnc,bmc->bnm", node_embeddings, node_embeddings)
        return functional.softmax(torch.sigmoid(similarity), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        tod: torch.Tensor | None = None,
        x_his: torch.Tensor | None = None,
        y_tod: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.context_length or x.shape[2] != self.num_nodes or x.shape[-1] != 1:
            raise ValueError("ST-SSDL expects x as [B, T, N, 1]")
        if tod is None or x_his is None or y_tod is None:
            raise ValueError("ST-SSDL requires tod, x_his, and y_tod")
        if x_his.shape != x.shape:
            raise ValueError("x_his must match x")
        if y_tod.ndim != 2 or y_tod.shape[0] != x.shape[0] or y_tod.shape[1] != self.horizon:
            raise ValueError("y_tod must be [B, H]")
        supports = self._supports(x)
        if self.use_ste:
            current = self._ste_features(x, tod, include_adaptive=True)
            history = self._ste_features(x_his, tod, include_adaptive=True)
        else:
            current = x
            history = x_his
        encoded, _ = self.encoder(current, supports)
        history_encoded, _ = self.encoder(history, supports)
        hidden = encoded[:, -1]
        history_hidden = history_encoded[:, -1]
        value, query, positive, negative, mask = self._query_prototypes(hidden)
        history_value, history_query, history_positive, history_negative, history_mask = self._query_prototypes(history_hidden)
        latent_distance = (query - history_query).abs().sum(dim=-1)
        prototype_distance = (positive - history_positive).abs().sum(dim=-1)
        decoder_state = torch.cat((hidden, value), dim=-1)
        augmented = torch.cat((hidden, value, history_hidden, history_value), dim=-1)
        node_embeddings = self.hypernet(augmented)
        dynamic = self._dynamic_support(node_embeddings)
        states = [decoder_state] * self.rnn_layers
        go = x.new_zeros(x.shape[0], self.num_nodes, self.output_dim)
        outputs = []
        for step in range(self.horizon):
            if self.use_ste:
                decoder_input = self._ste_features(go.unsqueeze(1), y_tod[:, step : step + 1], include_adaptive=False).squeeze(1)
            else:
                decoder_input = torch.cat((go, y_tod[:, step].float().view(x.shape[0], 1, 1).expand(-1, self.num_nodes, 1)), dim=-1)
            decoder_state, states = self.decoder(decoder_input, states, [dynamic])
            go = self.proj(decoder_state)
            outputs.append(go)
            if self.training and self.use_curriculum_learning and labels is not None:
                if float(np.random.uniform(0.0, 1.0)) < self._curriculum_threshold(self.batches_seen):
                    go = labels[:, step]
        if self.training and labels is not None:
            self.batches_seen += 1
            contrastive = self.triplet_loss(query.detach(), positive, negative)
            deviation = functional.l1_loss(latent_distance.detach(), prototype_distance)
            self._auxiliary = {"contrastive": contrastive, "deviation": deviation}
        else:
            self._auxiliary = None
        return torch.stack(outputs, dim=1)

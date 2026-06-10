"""Original ST-SSDL baseline architecture used for fair ablations."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .agcrn import ADCRNNDecoder, ADCRNNEncoder


class STSSDLBaseline(nn.Module):
    """Architecture-matched ST-SSDL baseline.

    This class mirrors the original `ST-SSDL/model_STSSDL/STSSDL.py` model path
    used by the METR-LA configuration: STE embeddings, dual current/history
    encoders, learnable prototypes, dynamic graph generation, and AGCRN decoder.
    """

    def __init__(
        self,
        num_nodes: int,
        supports: Sequence[torch.Tensor],
        input_dim: int = 1,
        output_dim: int = 1,
        horizon: int = 12,
        rnn_units: int = 128,
        rnn_layers: int = 1,
        cheb_k: int = 3,
        prototype_num: int = 20,
        prototype_dim: int = 64,
        input_embedding_dim: int = 3,
        tod_embed_dim: int = 20,
        node_embedding_dim: int = 25,
        adaptive_embedding_dim: int = 0,
        tday: int = 288,
        cl_decay_steps: int = 2000,
        use_curriculum_learning: bool = True,
        use_ssdl: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.prototype_num = prototype_num
        self.prototype_dim = prototype_dim if use_ssdl else 0
        self.input_embedding_dim = input_embedding_dim
        self.tod_embed_dim = tod_embed_dim
        self.node_embedding_dim = node_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.tday = tday
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.use_ssdl = use_ssdl
        self.supports = list(supports)

        encoder_dim = input_embedding_dim + tod_embed_dim + adaptive_embedding_dim + node_embedding_dim
        decoder_input_dim = input_embedding_dim + tod_embed_dim + node_embedding_dim
        decoder_dim = rnn_units + self.prototype_dim

        if use_ssdl:
            self.prototypes = self.construct_prototypes(prototype_dim)
        else:
            self.prototypes = None

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
        self.time_embedding = nn.Parameter(torch.empty(tday, tod_embed_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        nn.init.xavier_uniform_(self.time_embedding)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(torch.empty(horizon, num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None

        self.encoder = ADCRNNEncoder(
            node_num=num_nodes,
            dim_in=encoder_dim,
            dim_out=rnn_units,
            cheb_k=cheb_k,
            rnn_layers=rnn_layers,
            num_support=len(self.supports),
        )
        self.decoder = ADCRNNDecoder(
            node_num=num_nodes,
            dim_in=decoder_input_dim,
            dim_out=decoder_dim,
            cheb_k=cheb_k,
            rnn_layers=rnn_layers,
            num_support=1,
        )
        self.proj = nn.Sequential(nn.Linear(decoder_dim, output_dim, bias=True))
        self.hypernet = nn.Sequential(nn.Linear(decoder_dim * 2, tod_embed_dim, bias=True))

    def construct_prototypes(self, prototype_dim: int) -> nn.ParameterDict:
        prototypes_dict = nn.ParameterDict()
        prototypes_dict["prototypes"] = nn.Parameter(torch.randn(self.prototype_num, prototype_dim), requires_grad=True)
        prototypes_dict["Wq"] = nn.Parameter(torch.randn(self.rnn_units, prototype_dim), requires_grad=True)
        for param in prototypes_dict.values():
            nn.init.xavier_normal_(param)
        return prototypes_dict

    def compute_sampling_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def _tod_indices(self, tod: torch.Tensor) -> torch.Tensor:
        idx = (tod.squeeze(-1) * self.tday).long()
        return torch.clamp(idx, min=0, max=self.tday - 1)

    def _embed_sequence(self, values: torch.Tensor, tod: torch.Tensor) -> torch.Tensor:
        features = [self.input_proj(values)]
        if self.tod_embed_dim > 0:
            features.append(self.time_embedding[self._tod_indices(tod)])
        if self.adaptive_embedding is not None:
            adaptive = self.adaptive_embedding.unsqueeze(0).expand(values.shape[0], -1, -1, -1)
            features.append(adaptive[:, : values.shape[1]])
        if self.node_embedding_dim > 0:
            node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(values.shape[0], values.shape[1], -1, -1)
            features.append(node_emb)
        return torch.cat(features, dim=-1)

    def _embed_decoder_input(self, values: torch.Tensor, tod: torch.Tensor) -> torch.Tensor:
        features = [self.input_proj(values)]
        if self.tod_embed_dim > 0:
            features.append(self.time_embedding[self._tod_indices(tod)])
        if self.node_embedding_dim > 0:
            node_emb = self.node_embedding.unsqueeze(0).expand(values.shape[0], -1, -1)
            features.append(node_emb)
        return torch.cat(features, dim=-1)

    def query_prototypes(self, hidden: torch.Tensor):
        query = torch.matmul(hidden, self.prototypes["Wq"])
        att_score = torch.softmax(torch.matmul(query, self.prototypes["prototypes"].t()), dim=-1)
        value = torch.matmul(att_score, self.prototypes["prototypes"])
        _, indices = torch.topk(att_score, k=2, dim=-1)
        pos = self.prototypes["prototypes"][indices[:, :, 0]]
        neg = self.prototypes["prototypes"][indices[:, :, 1]]
        mask = torch.stack([indices[:, :, 0], indices[:, :, 1]], dim=-1)
        return value, query, pos, neg, mask, att_score

    @staticmethod
    def calculate_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(left - right), dim=-1)

    def _encode_pair(self, x: torch.Tensor, x_cov: torch.Tensor, x_his: torch.Tensor):
        init_state = self.encoder.init_hidden(x.shape[0], device=x.device)
        h_cur_seq, _ = self.encoder(self._embed_sequence(x, x_cov), init_state, self.supports)
        h_his_seq, _ = self.encoder(self._embed_sequence(x_his, x_cov), init_state, self.supports)
        return h_cur_seq[:, -1, :, :], h_his_seq[:, -1, :, :]

    def forward(
        self,
        x: torch.Tensor,
        x_cov: torch.Tensor,
        x_his: torch.Tensor,
        y_cov: torch.Tensor,
        labels: torch.Tensor | None = None,
        batches_seen: int | None = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        h_t, h_a = self._encode_pair(x, x_cov, x_his)
        if self.use_ssdl:
            v_t, q_t, p_t, n_t, mask_t, att_c = self.query_prototypes(h_t)
            v_a, q_a, p_a, n_a, mask_a, att_a = self.query_prototypes(h_a)
            latent_dis = self.calculate_distance(q_t, q_a)
            prototype_dis = self.calculate_distance(p_t, p_a)
            query = torch.stack([q_t, q_a], dim=0)
            pos = torch.stack([p_t, p_a], dim=0)
            neg = torch.stack([n_t, n_a], dim=0)
            mask = torch.stack([mask_t, mask_a], dim=0)
        else:
            v_t = h_t.new_zeros((*h_t.shape[:-1], 0))
            v_a = h_a.new_zeros((*h_a.shape[:-1], 0))
            latent_dis = h_t.new_zeros(h_t.shape[:-1])
            prototype_dis = h_t.new_zeros(h_t.shape[:-1])
            query = h_t.new_zeros((2, *h_t.shape[:-1], 0))
            pos = h_t.new_zeros((2, *h_t.shape[:-1], 0))
            neg = h_t.new_zeros((2, *h_t.shape[:-1], 0))
            mask = torch.zeros((2, *h_t.shape[:-1], 2), device=x.device, dtype=torch.long)
            att_c = h_t.new_zeros((*h_t.shape[:-1], 0))
            att_a = h_t.new_zeros((*h_t.shape[:-1], 0))

        h_de = torch.cat([h_t, v_t], dim=-1)
        h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1)
        node_embeddings = self.hypernet(h_aug)
        support = F.softmax(
            F.relu(torch.einsum("bnc,bmc->bnm", node_embeddings, node_embeddings)),
            dim=-1,
        )

        ht_list = [h_de] * self.rnn_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(self.horizon):
            decoder_input = self._embed_decoder_input(go, y_cov[:, step, :, :])
            h_de, ht_list = self.decoder(decoder_input, ht_list, [support])
            go = self.proj(h_de)
            outputs.append(go)
            if self.training and self.use_curriculum_learning and labels is not None and batches_seen is not None:
                if np.random.uniform(0, 1) < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, step, :, :]

        zero_loss = x.new_zeros(())
        prediction = torch.stack(outputs, dim=1)
        output = {
            "prediction": prediction,
            "query": query,
            "pos": pos,
            "neg": neg,
            "mask": mask,
            "latent_dis": latent_dis,
            "prototype_dis": prototype_dis,
            "region_loss": zero_loss,
            "graph_reg_loss": zero_loss,
            "clean_support": support,
            "edge_reliability": torch.ones_like(support),
        }
        if return_intermediates:
            output.update(
                {
                    "attention_c": att_c,
                    "attention_a": att_a,
                    "h_c": h_t,
                    "h_a": h_a,
                    "v_c": v_t,
                    "v_a": v_a,
                }
            )
        return output

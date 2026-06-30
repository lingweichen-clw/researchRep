"""DCD-ST model implementation.

DCD-ST keeps the ST-SSDL current/anchor encoder-decoder backbone and replaces
learnable prototypes with continuous deviation decomposition and gated fusion.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _load_deviation_extractor_class():
    module_path = Path(__file__).with_name("deviation_decomposition.py")
    module_name = "_dcd_st_deviation_decomposition"
    if module_name in sys.modules:
        return sys.modules[module_name].DeviationFeatureExtractor
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load deviation decomposition module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.DeviationFeatureExtractor


try:
    from src.models.agcrn import ADCRNNDecoder, ADCRNNEncoder
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.models.agcrn import ADCRNNDecoder, ADCRNNEncoder


DeviationFeatureExtractor = _load_deviation_extractor_class()


class FeedForwardBlock(nn.Module):
    """Small residual MLP for node-level features."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class DeviationGate(nn.Module):
    """Identity-conditioned gate that injects continuous deviation corrections."""

    def __init__(
        self,
        rnn_units: int,
        dev_feature_dim: int,
        dev_embed_dim: int,
        node_embedding_dim: int,
        tod_embed_dim: int,
        gate_hidden_dim: int,
    ):
        super().__init__()
        self.dev_encoder = nn.Sequential(
            nn.Linear(dev_feature_dim, dev_embed_dim),
            nn.ReLU(),
            nn.Linear(dev_embed_dim, dev_embed_dim),
            nn.ReLU(),
        )
        id_dim = node_embedding_dim + tod_embed_dim
        self.gate = FeedForwardBlock(rnn_units + dev_embed_dim + id_dim, gate_hidden_dim, rnn_units)
        self.delta = FeedForwardBlock(rnn_units + dev_embed_dim, gate_hidden_dim, rnn_units)

    def forward(
        self,
        h_cur: torch.Tensor,
        h_anchor: torch.Tensor,
        z_dev: torch.Tensor,
        node_identity: torch.Tensor,
        time_identity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_delta = h_cur - h_anchor
        z_dev_emb = self.dev_encoder(z_dev)
        z_id = torch.cat([node_identity, time_identity], dim=-1)
        gate_input = torch.cat([z_delta, z_dev_emb, z_id], dim=-1)
        delta_input = torch.cat([z_delta, z_dev_emb], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        delta_h = self.delta(delta_input)
        h_de = h_cur + gate * delta_h
        return h_de, gate, delta_h, z_dev_emb


class DCDST(nn.Module):
    """Deviation-Calibrated Decomposition model built on the ST-SSDL backbone."""

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
        input_embedding_dim: int = 3,
        tod_embed_dim: int = 20,
        node_embedding_dim: int = 25,
        adaptive_embedding_dim: int = 0,
        tday: int = 288,
        cl_decay_steps: int = 2000,
        use_curriculum_learning: bool = True,
        decomp_kernel_size: int = 3,
        dev_embed_dim: int = 32,
        gate_hidden_dim: int = 128,
        use_temporal_deviation_norm: bool = True,
        use_spatial_deviation_norm: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.input_embedding_dim = input_embedding_dim
        self.tod_embed_dim = tod_embed_dim
        self.node_embedding_dim = node_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.tday = tday
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.supports = list(supports)

        encoder_dim = input_embedding_dim + tod_embed_dim + adaptive_embedding_dim + node_embedding_dim
        decoder_input_dim = input_embedding_dim + tod_embed_dim + node_embedding_dim
        decoder_dim = rnn_units

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
        self.deviation_extractor = DeviationFeatureExtractor(
            kernel_size=decomp_kernel_size,
            use_temporal_norm=use_temporal_deviation_norm,
            use_spatial_norm=use_spatial_deviation_norm,
        )
        self.deviation_gate = DeviationGate(
            rnn_units=rnn_units,
            dev_feature_dim=self.deviation_extractor.feature_dim,
            dev_embed_dim=dev_embed_dim,
            node_embedding_dim=node_embedding_dim,
            tod_embed_dim=tod_embed_dim,
            gate_hidden_dim=gate_hidden_dim,
        )
        self.hypernet = nn.Sequential(nn.Linear(rnn_units * 4, tod_embed_dim, bias=True))
        self.proj = nn.Sequential(nn.Linear(decoder_dim, output_dim, bias=True))

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

    def _encode_pair(self, x: torch.Tensor, x_cov: torch.Tensor, x_his: torch.Tensor):
        init_state = self.encoder.init_hidden(x.shape[0], device=x.device)
        h_cur_seq, _ = self.encoder(self._embed_sequence(x, x_cov), init_state, self.supports)
        h_his_seq, _ = self.encoder(self._embed_sequence(x_his, x_cov), init_state, self.supports)
        return h_cur_seq[:, -1, :, :], h_his_seq[:, -1, :, :]

    def _identity_context(self, x_cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_cov.shape[0]
        node_identity = self.node_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        if self.tod_embed_dim > 0:
            time_indices = self._tod_indices(x_cov[:, -1, :, :])
            time_identity = self.time_embedding[time_indices]
        else:
            time_identity = x_cov.new_zeros((batch_size, self.num_nodes, 0))
        return node_identity, time_identity

    def _build_dynamic_support(self, h_de: torch.Tensor, h_anchor: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h_aug = torch.cat([h_de, h_anchor, h_de - h_anchor, gate], dim=-1)
        node_embeddings = self.hypernet(h_aug)
        support = torch.einsum("bnc,bmc->bnm", node_embeddings, node_embeddings)
        return F.softmax(F.relu(support), dim=-1)

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
        h_cur, h_anchor = self._encode_pair(x, x_cov, x_his)
        deviation = self.deviation_extractor(x, x_his)
        node_identity, time_identity = self._identity_context(x_cov)
        h_de, gate, delta_h, z_dev_emb = self.deviation_gate(
            h_cur,
            h_anchor,
            deviation["z_dev"],
            node_identity,
            time_identity,
        )
        support = self._build_dynamic_support(h_de, h_anchor, gate)

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

        prediction = torch.stack(outputs, dim=1)
        zero_loss = x.new_zeros(())
        zero_node = h_cur.new_zeros(h_cur.shape[:-1])
        zero_query = h_cur.new_zeros((2, *h_cur.shape[:-1], 0))
        zero_mask = torch.zeros((2, *h_cur.shape[:-1], 2), device=x.device, dtype=torch.long)
        output = {
            "prediction": prediction,
            "query": zero_query,
            "pos": zero_query,
            "neg": zero_query,
            "mask": zero_mask,
            "latent_dis": zero_node,
            "prototype_dis": zero_node,
            "region_loss": zero_loss,
            "graph_reg_loss": zero_loss,
            "gate_sparse_loss": gate.mean(),
            "gate_smooth_loss": zero_loss,
            "clean_support": support,
            "edge_reliability": torch.ones_like(support),
        }
        if return_intermediates:
            output.update(
                {
                    "h_c": h_cur,
                    "h_a": h_anchor,
                    "h_de": h_de,
                    "delta_h": delta_h,
                    "g_dev": gate,
                    "z_dev": deviation["z_dev"],
                    "z_dev_emb": z_dev_emb,
                    "r_raw": deviation["r_raw"],
                    "r_trend": deviation["r_trend"],
                    "r_residual": deviation["r_residual"],
                    "d_t": deviation["d_t"],
                    "d_s": deviation["d_s"],
                }
            )
        return output


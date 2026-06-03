"""ST-SSDL-style backbone with optional deviation-learning ablations."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .agcrn import ADCRNNDecoder, ADCRNNEncoder
from .graph_denoise import GraphDenoisingLayer, RegionPrototypeBuilder
from .graph_regions import BCCRegionSelector
from .prototype import PrototypeMemory, l1_distance


class RegionAwareSTSSDL(nn.Module):
    """Minimal ST-SSDL-style model with explicit ablation switches."""

    def __init__(
        self,
        num_nodes: int,
        supports: Sequence[torch.Tensor],
        raw_adj: np.ndarray | torch.Tensor,
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
        dataset_name: str = "METR-LA",
        bcc_edge_threshold: float | None = None,
        region_temperature: float = 0.5,
        graph_static_weight: float = 0.15,
        use_ssdl: bool = True,
        use_region_loss: bool = True,
        use_graph_denoise: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.tday = tday
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.input_embedding_dim = input_embedding_dim
        self.tod_embed_dim = tod_embed_dim
        self.node_embedding_dim = node_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.region_temperature = region_temperature
        self.use_ssdl = use_ssdl
        self.use_region_loss = use_region_loss
        self.use_graph_denoise = use_graph_denoise

        self.supports = list(supports)
        encoder_dim = input_embedding_dim + tod_embed_dim + node_embedding_dim + adaptive_embedding_dim
        decoder_input_dim = input_embedding_dim + tod_embed_dim + node_embedding_dim
        self.prototype_dim = prototype_dim if use_ssdl else 0
        decoder_dim = rnn_units + self.prototype_dim

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.time_embedding = nn.Parameter(torch.empty(tday, tod_embed_dim))
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
        nn.init.xavier_uniform_(self.time_embedding)
        nn.init.xavier_uniform_(self.node_embedding)
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
        self.prototypes = PrototypeMemory(rnn_units, prototype_num, prototype_dim) if use_ssdl else None
        self.output_proj = nn.Linear(decoder_dim, output_dim)
        self.hypernet = nn.Linear(decoder_dim * 2, tod_embed_dim)

        raw_adj_np = raw_adj.detach().cpu().numpy() if isinstance(raw_adj, torch.Tensor) else raw_adj
        selector = BCCRegionSelector(raw_adj_np, dataset=dataset_name, edge_threshold=bcc_edge_threshold)
        positive_mask = selector.positive_mask_tensor(device=self.supports[0].device)
        self.region_builder = RegionPrototypeBuilder(positive_mask)
        raw_adj_tensor = torch.as_tensor(raw_adj_np, dtype=torch.float32, device=self.supports[0].device)
        self.graph_denoiser = GraphDenoisingLayer(
            static_adj=raw_adj_tensor,
            sp_degree=3,
            static_weight=graph_static_weight,
        )

    def compute_sampling_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def _tod_indices(self, tod: torch.Tensor) -> torch.Tensor:
        idx = torch.floor(torch.clamp(tod.squeeze(-1), min=0.0, max=0.999999) * self.tday).long()
        return torch.clamp(idx, min=0, max=self.tday - 1)

    def _embed_sequence(self, values: torch.Tensor, tod: torch.Tensor) -> torch.Tensor:
        features = [self.input_proj(values)]
        features.append(self.time_embedding[self._tod_indices(tod)])
        node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(values.shape[0], values.shape[1], -1, -1)
        features.append(node_emb)
        if self.adaptive_embedding is not None:
            adaptive = self.adaptive_embedding.unsqueeze(0).expand(values.shape[0], -1, -1, -1)
            features.append(adaptive[:, : values.shape[1]])
        return torch.cat(features, dim=-1)

    def _embed_decoder_input(self, values: torch.Tensor, tod: torch.Tensor) -> torch.Tensor:
        features = [self.input_proj(values)]
        features.append(self.time_embedding[self._tod_indices(tod)])
        node_emb = self.node_embedding.unsqueeze(0).expand(values.shape[0], -1, -1)
        features.append(node_emb)
        return torch.cat(features, dim=-1)

    def _encode_pair(self, x: torch.Tensor, x_cov: torch.Tensor, x_his: torch.Tensor):
        init_state = self.encoder.init_hidden(x.shape[0], device=x.device)
        h_cur_seq, _ = self.encoder(self._embed_sequence(x, x_cov), init_state, self.supports)
        h_cur = h_cur_seq[:, -1, :, :]
        h_his_seq, _ = self.encoder(self._embed_sequence(x_his, x_cov), init_state, self.supports)
        h_his = h_his_seq[:, -1, :, :]
        return h_cur, h_his

    def forward(
        self,
        x: torch.Tensor,
        x_cov: torch.Tensor,
        x_his: torch.Tensor,
        y_cov: torch.Tensor,
        labels: torch.Tensor | None = None,
        batches_seen: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        h_cur, h_his = self._encode_pair(x, x_cov, x_his)
        if self.use_ssdl:
            cur_proto = self.prototypes(h_cur)
            his_proto = self.prototypes(h_his)
            cur_value = cur_proto["value"]
            his_value = his_proto["value"]
            latent_dis = l1_distance(cur_proto["query"], his_proto["query"])
            prototype_dis = l1_distance(cur_proto["pos"], his_proto["pos"])
            query = torch.stack([cur_proto["query"], his_proto["query"]], dim=0)
            pos = torch.stack([cur_proto["pos"], his_proto["pos"]], dim=0)
            neg = torch.stack([cur_proto["neg"], his_proto["neg"]], dim=0)
            mask = torch.stack([cur_proto["mask"], his_proto["mask"]], dim=0)
        else:
            cur_value = h_cur.new_zeros((*h_cur.shape[:-1], 0))
            his_value = h_his.new_zeros((*h_his.shape[:-1], 0))
            latent_dis = h_cur.new_zeros(h_cur.shape[:-1])
            prototype_dis = h_cur.new_zeros(h_cur.shape[:-1])
            query = h_cur.new_zeros((2, *h_cur.shape[:-1], 0))
            pos = h_cur.new_zeros((2, *h_cur.shape[:-1], 0))
            neg = h_cur.new_zeros((2, *h_cur.shape[:-1], 0))
            mask = torch.zeros((2, *h_cur.shape[:-1], 2), device=x.device, dtype=torch.long)

        region_prototypes = self.region_builder(h_cur)
        if self.use_region_loss:
            region_loss = self.region_builder.contrastive_loss(
                h_cur,
                region_prototypes,
                temperature=self.region_temperature,
            )
        else:
            region_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        h_de = torch.cat([h_cur, cur_value], dim=-1)
        h_aug = torch.cat([h_cur, cur_value, h_his, his_value], dim=-1)
        node_embeddings = self.hypernet(h_aug)
        base_support = F.softmax(
            F.relu(torch.einsum("bnc,bmc->bnm", node_embeddings, node_embeddings)),
            dim=-1,
        )
        if self.use_graph_denoise:
            clean_support, graph_reg_loss, edge_reliability = self.graph_denoiser(
                base_support,
                h_cur,
                region_prototypes,
            )
        else:
            clean_support = base_support
            graph_reg_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            edge_reliability = torch.ones_like(base_support)

        ht_list = [h_de] * self.rnn_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(self.horizon):
            dec_input = self._embed_decoder_input(go, y_cov[:, step, :, :])
            h_de, ht_list = self.decoder(dec_input, ht_list, [clean_support])
            go = self.output_proj(h_de)
            outputs.append(go)
            if self.training and self.use_curriculum_learning and labels is not None and batches_seen is not None:
                if np.random.uniform(0, 1) < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, step, :, :]

        prediction = torch.stack(outputs, dim=1)
        return {
            "prediction": prediction,
            "query": query,
            "pos": pos,
            "neg": neg,
            "mask": mask,
            "latent_dis": latent_dis,
            "prototype_dis": prototype_dis,
            "region_loss": region_loss,
            "graph_reg_loss": graph_reg_loss,
            "clean_support": clean_support,
            "edge_reliability": edge_reliability,
        }

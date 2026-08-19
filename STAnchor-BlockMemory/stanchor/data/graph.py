"""Graph loading and sparse edge representation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stanchor.utils import array_sha256


@dataclass(frozen=True)
class GraphData:
    edge_index: torch.Tensor  # [2, E], rows are target and source
    edge_weight: torch.Tensor  # [E]
    num_nodes: int
    fingerprint: str

    def validate(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2, E]")
        if self.edge_weight.ndim != 1 or self.edge_weight.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_weight must align with edge_index")
        if self.edge_index.numel() == 0:
            raise ValueError("graph must contain at least one edge")
        if int(self.edge_index.min()) < 0 or int(self.edge_index.max()) >= self.num_nodes:
            raise ValueError("edge_index contains out-of-range node ids")
        target, source = self.edge_index
        self_loop_nodes = torch.unique(target[target == source])
        if self_loop_nodes.numel() != self.num_nodes:
            raise ValueError("graph must contain one or more self-loops for every node")
        if not torch.isfinite(self.edge_weight).all() or bool((self.edge_weight <= 0).any()):
            raise ValueError("edge weights must be finite and positive")

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            num_nodes=self.num_nodes,
            fingerprint=self.fingerprint,
        )

    def dense_neighbors(self, include_self: bool = False) -> torch.Tensor:
        neighbors = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.bool)
        target, source = self.edge_index.cpu()
        neighbors[target, source] = True
        if not include_self:
            neighbors.fill_diagonal_(False)
        return neighbors

    def random_walk_diffusion_prior(self) -> torch.Tensor:
        """Return a fixed two/three-hop prior without self-loop paths.

        The returned matrix is ``A_rw @ A_rw + A_rw @ A_rw @ A_rw`` where
        ``A_rw`` is the row-normalized adjacency after removing self-loops.
        It is a graph-only prior and contains no trainable parameters.
        """
        adjacency = torch.zeros(
            (self.num_nodes, self.num_nodes),
            dtype=self.edge_weight.dtype,
            device=self.edge_weight.device,
        )
        target, source = self.edge_index
        non_self = target != source
        adjacency[target[non_self], source[non_self]] = self.edge_weight[non_self]
        degree = adjacency.sum(dim=1, keepdim=True)
        random_walk = adjacency / degree.clamp_min(1.0e-8)
        two_hop = random_walk @ random_walk
        three_hop = two_hop @ random_walk
        prior = two_hop + three_hop
        prior.fill_diagonal_(0.0)
        return prior

    def mixed_range_candidate_indices(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded first-order and remote source indices once per graph.

        Each index tensor has shape ``[N, N-1]``.  Valid entries are sorted
        source ids and padding is marked by the corresponding boolean mask.
        """
        device = self.edge_index.device
        direct = torch.zeros(
            (self.num_nodes, self.num_nodes), dtype=torch.bool, device=device
        )
        target, source = self.edge_index
        direct[target, source] = True
        direct.fill_diagonal_(False)
        remote = ~direct
        remote.fill_diagonal_(False)
        source_ids = torch.arange(self.num_nodes, device=device).expand(
            self.num_nodes, self.num_nodes
        )

        def pack(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            padded = source_ids.masked_fill(~mask, self.num_nodes)
            padded = padded.sort(dim=1).values[:, : self.num_nodes - 1]
            valid = padded != self.num_nodes
            return padded.clamp_max(self.num_nodes - 1), valid

        local_ids, local_valid = pack(direct)
        remote_ids, remote_valid = pack(remote)
        return local_ids, local_valid, remote_ids, remote_valid

    def higher_order_candidate_indices(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return direct and non-direct candidates restricted to 2/3-hop support.

        The local tensors are retained for the shared route-selection contract;
        a route with ``route_local_quota=0`` ignores them.  If a node has too
        few positive 2/3-hop paths, the remaining non-direct nodes are used as
        a deterministic fallback so fixed-size top-k routing remains valid.
        """
        device = self.edge_index.device
        direct = self.dense_neighbors(include_self=False).to(device)
        diffusion = self.random_walk_diffusion_prior()
        higher = (diffusion > 0.0) & ~direct
        higher.fill_diagonal_(False)
        remote = (~direct).clone()
        remote.fill_diagonal_(False)
        higher_count = higher.sum(dim=1)
        remote_count = remote.sum(dim=1)
        fallback = higher_count == 0
        remote = torch.where(fallback[:, None], remote, higher)
        source_ids = torch.arange(self.num_nodes, device=device).expand(
            self.num_nodes, self.num_nodes
        )

        def pack(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            padded = source_ids.masked_fill(~mask, self.num_nodes)
            padded = padded.sort(dim=1).values[:, : self.num_nodes - 1]
            valid = padded != self.num_nodes
            return padded.clamp_max(self.num_nodes - 1), valid

        local_ids, local_valid = pack(direct)
        remote_ids, remote_valid = pack(remote)
        return local_ids, local_valid, remote_ids, remote_valid


def _loads_pickle(payload: bytes):
    try:
        return pickle.loads(payload)
    except UnicodeDecodeError:
        return pickle.loads(payload, encoding="latin1")


def _load_pickle(path: Path):
    payload = path.read_bytes()
    try:
        return _loads_pickle(payload)
    except (pickle.UnpicklingError, ModuleNotFoundError) as original_error:
        # Protocol-0 pickle is line-oriented and can be corrupted when a
        # transfer tool rewrites LF as CRLF. Retry only that reversible case.
        normalized = payload.replace(b"\r\n", b"\n")
        if normalized == payload:
            raise
        try:
            return _loads_pickle(normalized)
        except (pickle.UnpicklingError, ModuleNotFoundError, UnicodeDecodeError) as normalized_error:
            raise original_error from normalized_error


def load_dense_adjacency(path: str | Path) -> np.ndarray:
    value = _load_pickle(Path(path))
    if isinstance(value, (tuple, list)):
        value = value[-1]
    adjacency = np.asarray(value, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("adjacency must be finite and non-negative")
    return adjacency


def graph_from_dense(adjacency: np.ndarray, add_self_loops: bool = True) -> GraphData:
    adjacency = np.asarray(adjacency, dtype=np.float32).copy()
    if add_self_loops:
        diagonal = np.diag_indices_from(adjacency)
        adjacency[diagonal] = np.maximum(adjacency[diagonal], 1.0)
    target, source = np.nonzero(adjacency > 0)
    weights = adjacency[target, source]
    graph = GraphData(
        edge_index=torch.from_numpy(np.stack((target, source))).long(),
        edge_weight=torch.from_numpy(weights).float(),
        num_nodes=adjacency.shape[0],
        fingerprint=array_sha256(adjacency),
    )
    graph.validate()
    return graph


def load_graph(path: str | Path) -> GraphData:
    return graph_from_dense(load_dense_adjacency(path), add_self_loops=True)

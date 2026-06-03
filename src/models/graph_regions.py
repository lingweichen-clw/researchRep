"""BCC-based region selection adapted from DarkFarseer."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch


class Graph:
    """Undirected graph with biconnected component extraction."""

    def __init__(self, vertices: int):
        self.vertices = vertices
        self.graph: Dict[int, List[int]] = defaultdict(list)
        self.components: List[List[Tuple[int, int]]] = []
        self.time = 0

    def add_edge(self, u: int, v: int) -> None:
        self.graph[u].append(v)
        self.graph[v].append(u)

    def _bcc_util(self, u: int, parent: List[int], low: List[int], disc: List[int], stack: List[Tuple[int, int]]):
        children = 0
        disc[u] = self.time
        low[u] = self.time
        self.time += 1

        for v in self.graph[u]:
            if disc[v] == -1:
                parent[v] = u
                children += 1
                stack.append((u, v))
                self._bcc_util(v, parent, low, disc, stack)
                low[u] = min(low[u], low[v])
                is_root_articulation = parent[u] == -1 and children > 1
                is_articulation = parent[u] != -1 and low[v] >= disc[u]
                if is_root_articulation or is_articulation:
                    comp = []
                    edge = (-1, -1)
                    while edge != (u, v):
                        edge = stack.pop()
                        comp.append(edge)
                    self.components.append(comp)
            elif v != parent[u] and low[u] > disc[v]:
                low[u] = min(low[u], disc[v])
                stack.append((u, v))

    def biconnected_components(self) -> List[List[Tuple[int, int]]]:
        disc = [-1] * self.vertices
        low = [-1] * self.vertices
        parent = [-1] * self.vertices
        stack: List[Tuple[int, int]] = []
        for node in range(self.vertices):
            if disc[node] == -1:
                self._bcc_util(node, parent, low, disc, stack)
            if stack:
                comp = []
                while stack:
                    comp.append(stack.pop())
                self.components.append(comp)
        return self.components


def get_1hop_neighbors(adj: np.ndarray) -> Dict[int, List[int]]:
    """Return undirected one-hop neighbors from a possibly directed matrix."""
    neighbors = {}
    for node in range(adj.shape[0]):
        neighbors[node] = list(
            set(np.where(adj[node] > 0)[0].tolist() + np.where(adj[:, node] > 0)[0].tolist())
        )
    return neighbors


def default_edge_threshold(dataset: str) -> float:
    if dataset in {"PEMS04", "PEMS03"}:
        return 0.0
    if dataset == "AIR36":
        return 0.5
    if dataset in {"METR-LA", "METRLA", "PEMS-BAY", "PEMSBAY"}:
        return 0.7
    if dataset in {"NREL-PA", "USHCN"}:
        return 0.9
    return 0.0


def get_bcc_regions(
    adj: np.ndarray,
    dataset: str = "METR-LA",
    edge_threshold: float | None = None,
):
    """DarkFarseer-style BCC positive/negative region selection."""
    threshold = default_edge_threshold(dataset) if edge_threshold is None else edge_threshold
    graph = Graph(adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if i != j and adj[i, j] > threshold:
                graph.add_edge(i, j)

    components = []
    for edge_list in graph.biconnected_components():
        nodes = set()
        for edge in edge_list:
            nodes.update(edge)
        if nodes:
            components.append(sorted(nodes))

    assigned = set(node for comp in components for node in comp)
    not_map = sorted(set(range(adj.shape[0])) - assigned)
    positive_select: Dict[int, List] = {node: None for node in not_map}
    negative_select: Dict[int, List] = {}
    neighbors = get_1hop_neighbors(adj)

    for node in range(adj.shape[0]):
        if node in not_map:
            positive_select[node] = deepcopy(neighbors[node]) + [node]
            negative_select[node] = [idx for idx in not_map if idx not in neighbors[node] and idx != node]
            continue

        memory_set = set()
        for component in components:
            if node in component:
                positive_select.setdefault(node, [])
                positive_select[node].append(component)
                memory_set.update(component)

        memory_set.discard(node)
        used = {component_id: 0 for component_id in range(len(components))}
        for member in memory_set:
            for component_id, component in enumerate(components):
                if member in component:
                    used[component_id] += 1
        negative_select[node] = [components[component_id] for component_id, count in used.items() if count == 0]

    bcc_node_index = np.zeros_like(adj, dtype=np.float32)
    for node, positives in positive_select.items():
        flat_nodes = _flatten_nodes(positives)
        for other in flat_nodes:
            if node != other:
                bcc_node_index[node, other] = 1.0
    return bcc_node_index, not_map, positive_select, negative_select


def _flatten_nodes(nodes) -> List[int]:
    if nodes is None:
        return []
    flat: List[int] = []
    for item in nodes:
        if isinstance(item, (list, tuple, set)):
            flat.extend(int(x) for x in item)
        else:
            flat.append(int(item))
    return sorted(set(flat))


class BCCRegionSelector:
    """Precompute normalized positive masks for region prototypes."""

    def __init__(
        self,
        adj: np.ndarray,
        dataset: str = "METR-LA",
        edge_threshold: float | None = None,
    ):
        self.adj = np.asarray(adj, dtype=np.float32)
        self.bcc_node_index, self.not_map, self.positive_select, self.negative_select = get_bcc_regions(
            self.adj,
            dataset=dataset,
            edge_threshold=edge_threshold,
        )
        self.positive_mask = self._build_positive_mask()

    def _build_positive_mask(self) -> np.ndarray:
        num_nodes = self.adj.shape[0]
        mask = np.eye(num_nodes, dtype=np.float32)
        for node in range(num_nodes):
            positives = _flatten_nodes(self.positive_select.get(node))
            if not positives:
                positives = [node]
            mask[node, positives] = 1.0
        denom = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        return mask / denom

    def positive_mask_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.positive_mask, dtype=torch.float32, device=device)

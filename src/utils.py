"""Utilities for traffic data, graph loading, and reproducibility."""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch


def set_seed(seed: int) -> None:
    """Set common random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path: str | Path):
    """Load pickle files produced by both Python 2 and Python 3."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def extract_adj_matrix(pickle_obj) -> np.ndarray:
    """Extract adjacency matrix from common traffic pickle formats."""
    if isinstance(pickle_obj, tuple):
        return np.asarray(pickle_obj[-1], dtype=np.float32)
    if isinstance(pickle_obj, list):
        return np.asarray(pickle_obj[-1], dtype=np.float32)
    return np.asarray(pickle_obj, dtype=np.float32)


def sym_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetrically normalize an adjacency matrix."""
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (
        adj_sp.dot(d_mat_inv_sqrt)
        .transpose()
        .dot(d_mat_inv_sqrt)
        .astype(np.float32)
        .todense()
        .A
    )


def asym_adj(adj: np.ndarray) -> np.ndarray:
    """Row-normalize an adjacency matrix."""
    adj_sp = sp.coo_matrix(adj)
    rowsum = np.array(adj_sp.sum(1)).flatten()
    d_inv = np.power(rowsum, -1.0)
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj_sp).astype(np.float32).todense().A


def load_adj(
    path: str | Path,
    adj_type: str = "symadj",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load raw adjacency and construct graph supports.

    Returns:
        supports: list of normalized supports, each `(N, N)`.
        raw_adj: original adjacency matrix `(N, N)`.
    """
    raw_adj = extract_adj_matrix(load_pickle(path))
    if adj_type == "symadj":
        supports = [sym_adj(raw_adj)]
    elif adj_type == "transition":
        supports = [asym_adj(raw_adj)]
    elif adj_type == "doubletransition":
        supports = [asym_adj(raw_adj), asym_adj(raw_adj.T)]
    elif adj_type == "identity":
        supports = [np.eye(raw_adj.shape[0], dtype=np.float32)]
    else:
        raise ValueError(f"Unsupported adj_type: {adj_type}")
    return supports, raw_adj.astype(np.float32)


def to_torch_supports(
    supports: Sequence[np.ndarray],
    device: torch.device,
) -> List[torch.Tensor]:
    """Move numpy graph supports to a torch device."""
    return [torch.as_tensor(support, dtype=torch.float32, device=device) for support in supports]


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def project_root() -> Path:
    """Return the repository root for this package."""
    return Path(__file__).resolve().parents[1]

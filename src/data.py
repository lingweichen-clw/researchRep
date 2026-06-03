"""Data loaders and tensor preparation for short-horizon traffic forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class StandardScaler:
    """Standard scaler compatible with numpy arrays and torch tensors."""

    mean: float
    std: float

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def prepare_x_y(x: torch.Tensor, y: torch.Tensor):
    """Split ST-SSDL style multi-channel windows.

    Args:
        x: `(B, T, N, 3)` with value, time-in-day, history anchor.
        y: `(B, H, N, 3)` with target value and future covariates.
    """
    x0 = x[..., 0:1]
    x_cov = x[..., 1:2]
    x_his = x[..., 2:3]
    y0 = y[..., 0:1]
    y_cov = y[..., 1:2]
    return x0, x_cov, x_his, y0, y_cov


def load_npz_splits(data_dir: str | Path) -> Dict[str, np.ndarray]:
    """Load `trainhis/valhis/testhis.npz` arrays."""
    data_dir = Path(data_dir)
    data: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}his.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with np.load(path) as npz:
            data[f"x_{split}"] = np.nan_to_num(npz["x"]).astype(np.float32)
            data[f"y_{split}"] = np.nan_to_num(npz["y"]).astype(np.float32)
    return data


def normalize_splits(data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """Normalize value and history-anchor channels using train statistics."""
    train_values = data["x_train"][..., 0]
    scaler = StandardScaler(mean=float(train_values.mean()), std=float(train_values.std() + 1e-6))
    normalized = {key: value.copy() for key, value in data.items()}
    for split in ("train", "val", "test"):
        normalized[f"x_{split}"][..., 0] = scaler.transform(normalized[f"x_{split}"][..., 0])
        normalized[f"x_{split}"][..., 2] = scaler.transform(normalized[f"x_{split}"][..., 2])
    return normalized, scaler


def build_loaders(
    data: Dict[str, np.ndarray],
    batch_size: int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Build PyTorch DataLoaders from normalized numpy splits."""
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = TensorDataset(
            torch.from_numpy(data[f"x_{split}"]).float(),
            torch.from_numpy(data[f"y_{split}"]).float(),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders

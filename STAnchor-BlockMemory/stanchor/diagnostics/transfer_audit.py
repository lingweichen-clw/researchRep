"""Dataset and graph audits used before cross-dataset retrieval transfer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _channel_stats(values: np.ndarray) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for channel in range(values.shape[-1]):
        current = np.asarray(values[..., channel], dtype=np.float64)
        finite = np.isfinite(current)
        valid = current[finite]
        result.append(
            {
                "channel": channel,
                "finite_fraction": float(finite.mean()),
                "zero_fraction": float((current == 0).mean()),
                "min": float(valid.min()) if valid.size else None,
                "max": float(valid.max()) if valid.size else None,
                "mean": float(valid.mean()) if valid.size else None,
                "std": float(valid.std()) if valid.size else None,
            }
        )
    return result


def _audit_values(
    values: np.ndarray,
    *,
    source_path: Path,
    timestamp_source: str,
    timestamps_ns: np.ndarray | None = None,
) -> dict[str, Any]:
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(f"traffic array must be [T,N,C], got {array.shape}")
    finite = np.isfinite(array)
    if timestamps_ns is not None:
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        if timestamps.shape != (array.shape[0],):
            raise ValueError("timestamps must have one entry per time step")
        gaps_minutes = np.diff(timestamps).astype(np.float64) / 60_000_000_000.0
        time_summary = {
            "start": int(timestamps[0]),
            "end": int(timestamps[-1]),
            "duplicate_count": int(np.sum(gaps_minutes == 0)),
            "non_increasing_count": int(np.sum(gaps_minutes <= 0)),
            "gap_gt_5min_count": int(np.sum(gaps_minutes > 5.0)),
            "gap_min_minutes": float(gaps_minutes.min()) if gaps_minutes.size else 0.0,
            "gap_max_minutes": float(gaps_minutes.max()) if gaps_minutes.size else 0.0,
        }
    else:
        time_summary = {
            "start": 0,
            "end": int(array.shape[0] - 1),
            "duplicate_count": 0,
            "non_increasing_count": 0,
            "gap_gt_5min_count": 0,
            "gap_min_minutes": 5.0,
            "gap_max_minutes": 5.0,
        }
    return {
        "source_path": str(source_path.resolve()),
        "sha256": _file_sha256(source_path),
        "shape": [int(v) for v in array.shape],
        "dtype": str(array.dtype),
        "steps": int(array.shape[0]),
        "nodes": int(array.shape[1]),
        "channels": int(array.shape[2]),
        "finite_fraction": float(finite.mean()),
        "nan_or_inf_count": int((~finite).sum()),
        "zero_count": int((array == 0).sum()),
        "timestamp_source": timestamp_source,
        "time": time_summary,
        "channel_stats": _channel_stats(array),
    }


def audit_npz_array(path: str | Path, key: str = "data") -> dict[str, Any]:
    """Audit a traffic NPZ whose selected array has shape ``[T,N,C]``."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if key not in archive.files:
            raise ValueError(f"NPZ does not contain key {key!r}")
        values = np.asarray(archive[key])
    return _audit_values(
        values,
        source_path=source,
        timestamp_source="inferred_from_row_index",
    )


def audit_hdf(path: str | Path) -> dict[str, Any]:
    """Audit an HDF traffic frame with a DatetimeIndex."""
    source = Path(path)
    frame = pd.read_hdf(source)
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("HDF must contain a DataFrame with a DatetimeIndex")
    values = frame.to_numpy(dtype=np.float32)[..., None]
    return _audit_values(
        values,
        source_path=source,
        timestamp_source="hdf_datetime_index",
        timestamps_ns=frame.index.view("int64").astype(np.int64),
    )


def audit_edge_csv(path: str | Path, num_nodes: int) -> dict[str, Any]:
    """Audit a ``from,to,cost`` edge CSV and report isolated nodes."""
    source = Path(path)
    frame = pd.read_csv(source)
    required = {"from", "to"}
    if not required.issubset(frame.columns):
        raise ValueError("edge CSV must contain from and to columns")
    source_nodes = frame["from"].to_numpy(dtype=np.int64)
    target_nodes = frame["to"].to_numpy(dtype=np.int64)
    in_range = (
        (source_nodes >= 0)
        & (source_nodes < int(num_nodes))
        & (target_nodes >= 0)
        & (target_nodes < int(num_nodes))
    )
    degree = np.zeros(int(num_nodes), dtype=np.int64)
    for node in np.concatenate((source_nodes[in_range], target_nodes[in_range])):
        degree[int(node)] += 1
    return {
        "source_path": str(source.resolve()),
        "sha256": _file_sha256(source),
        "num_nodes": int(num_nodes),
        "edge_count": int(len(frame)),
        "valid_edge_count": int(in_range.sum()),
        "out_of_range_edges": int((~in_range).sum()),
        "self_loop_count": int(np.sum(source_nodes[in_range] == target_nodes[in_range])),
        "isolated_nodes": int(np.sum(degree == 0)),
        "degree_min": int(degree.min()) if degree.size else 0,
        "degree_max": int(degree.max()) if degree.size else 0,
        "degree_mean": float(degree.mean()) if degree.size else 0.0,
    }


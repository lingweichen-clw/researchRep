"""Distributional historical-anchor preprocessing for DHA-DCD-ST.

This script keeps the existing ST-SSDL data contract:

    x/y: (B, T/H, N, 3) = value, time_in_day, history_anchor

The difference is that the third channel can be generated from different
train-only same-weekday-slot statistics instead of only the historical mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ANCHOR_MODES = ("mean", "median", "q25", "q75", "recent")


def generate_windows(
    data: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate seq2seq windows from a traffic tensor.

    Args:
        data: `(L, N, C)` traffic array.
        x_offsets: offsets ending at the current time, e.g. `[-11, ..., 0]`.
        y_offsets: future offsets, e.g. `[1, ..., 12]`.

    Returns:
        x: `(B, T, N, C)`.
        y: `(B, H, N, C)`.
    """
    num_samples = data.shape[0]
    min_t = abs(int(min(x_offsets)))
    max_t = num_samples - abs(int(max(y_offsets)))
    x, y = [], []
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    return np.stack(x, axis=0), np.stack(y, axis=0)


def _time_in_day(index: pd.DatetimeIndex, num_nodes: int) -> np.ndarray:
    time_ind = (index.values - index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    return np.tile(time_ind, [num_nodes, 1]).T.astype(np.float32)


def _weekday_slot(index: pd.DatetimeIndex, slots_per_day: int) -> np.ndarray:
    minutes_per_day = 24 * 60
    if minutes_per_day % slots_per_day != 0:
        raise ValueError("slots_per_day must evenly divide 1440 minutes.")
    minutes_per_slot = minutes_per_day // slots_per_day
    slot_in_day = (index.hour.to_numpy() * 60 + index.minute.to_numpy()) // minutes_per_slot
    slot_in_day = np.clip(slot_in_day, 0, slots_per_day - 1)
    return index.weekday.to_numpy() * slots_per_day + slot_in_day


def _sensor_fallback(train_values: np.ndarray) -> np.ndarray:
    valid = train_values != 0
    summed = np.where(valid, train_values, 0.0).sum(axis=0)
    counts = valid.sum(axis=0)
    return np.divide(
        summed,
        np.maximum(counts, 1),
        out=np.zeros_like(summed, dtype=np.float32),
        where=counts > 0,
    ).astype(np.float32)


def _aggregate_slot_values(
    slot_values: np.ndarray,
    anchor_mode: str,
    fallback: np.ndarray,
) -> np.ndarray:
    """Aggregate one weekday-slot history matrix into one anchor vector.

    `slot_values` has shape `(S, N)` where `S` is the number of training
    timestamps that share the same weekday-slot.
    """
    if anchor_mode not in ANCHOR_MODES:
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")

    num_nodes = slot_values.shape[1]
    anchor = fallback.copy()
    valid = slot_values != 0

    if anchor_mode == "mean":
        counts = valid.sum(axis=0)
        summed = np.where(valid, slot_values, 0.0).sum(axis=0)
        return np.divide(
            summed,
            np.maximum(counts, 1),
            out=anchor,
            where=counts > 0,
        ).astype(np.float32)

    if anchor_mode == "recent":
        for node_idx in range(num_nodes):
            node_values = slot_values[:, node_idx]
            node_values = node_values[node_values != 0]
            if node_values.size > 0:
                anchor[node_idx] = node_values[-1]
        return anchor.astype(np.float32)

    percentile = {"q25": 25.0, "median": 50.0, "q75": 75.0}[anchor_mode]
    for node_idx in range(num_nodes):
        node_values = slot_values[:, node_idx]
        node_values = node_values[node_values != 0]
        if node_values.size > 0:
            anchor[node_idx] = np.percentile(node_values, percentile)
    return anchor.astype(np.float32)


def build_distributional_history_anchor(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    slots_per_day: int = 288,
    anchor_mode: str = "mean",
) -> np.ndarray:
    """Build train-only same-weekday-slot historical anchors.

    The anchor table is computed only from the raw training segment. Validation
    and test timestamps reuse this train-derived table, so future observations
    never leak into the anchor channel.

    Args:
        df: traffic DataFrame `(L, N)` with a `DatetimeIndex`.
        train_ratio: raw-time split ratio used for the anchor table.
        slots_per_day: usually 288 for 5-minute traffic data.
        anchor_mode: one of `mean`, `median`, `q25`, `q75`, `recent`.

    Returns:
        Anchor tensor `(L, N)` aligned with `df.index`.
    """
    if anchor_mode not in ANCHOR_MODES:
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")

    values = df.values.astype(np.float32)
    num_steps, num_nodes = values.shape
    train_end = int(num_steps * train_ratio)
    slot_ids = _weekday_slot(df.index, slots_per_day)
    train_slots = slot_ids[:train_end]
    train_values = values[:train_end]
    num_slots = 7 * slots_per_day
    fallback = _sensor_fallback(train_values)

    anchor_table = np.tile(fallback.reshape(1, num_nodes), (num_slots, 1)).astype(np.float32)
    for slot in range(num_slots):
        mask = train_slots == slot
        if not np.any(mask):
            continue
        anchor_table[slot] = _aggregate_slot_values(
            train_values[mask],
            anchor_mode=anchor_mode,
            fallback=fallback,
        )

    return anchor_table[slot_ids].astype(np.float32)


def generate_metrla_dha_splits(
    traffic_h5: str | Path,
    output_dir: str | Path,
    anchor_mode: str = "mean",
    seq_len: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    slots_per_day: int = 288,
    max_windows: int | None = None,
) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Generate `trainhis/valhis/testhis.npz` files for DHA-DCD-ST."""
    traffic_h5 = Path(traffic_h5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_hdf(traffic_h5)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Traffic h5 must contain a DataFrame with DatetimeIndex.")

    values = df.values.astype(np.float32)
    tod = _time_in_day(df.index, values.shape[1])
    history_anchor = build_distributional_history_anchor(
        df,
        train_ratio=train_ratio,
        slots_per_day=slots_per_day,
        anchor_mode=anchor_mode,
    )
    data = np.stack([values, tod, history_anchor], axis=-1).astype(np.float32)

    x_offsets = np.arange(-(seq_len - 1), 1, dtype=np.int32)
    y_offsets = np.arange(1, horizon + 1, dtype=np.int32)
    x, y = generate_windows(data, x_offsets=x_offsets, y_offsets=y_offsets)
    if max_windows is not None:
        x = x[:max_windows]
        y = y[:max_windows]

    num_samples = x.shape[0]
    num_train = round(num_samples * train_ratio)
    num_val = round(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    splits = {
        "train": (x[:num_train], y[:num_train]),
        "val": (x[num_train : num_train + num_val], y[num_train : num_train + num_val]),
        "test": (x[-num_test:], y[-num_test:]),
    }

    summary: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for name, (x_part, y_part) in splits.items():
        np.savez_compressed(
            output_dir / f"{name}his.npz",
            x=x_part,
            y=y_part,
            x_offsets=x_offsets.reshape(-1, 1),
            y_offsets=y_offsets.reshape(-1, 1),
        )
        summary[name] = (x_part.shape, y_part.shape)

    metadata = {
        "anchor_mode": anchor_mode,
        "traffic_h5": str(traffic_h5),
        "seq_len": seq_len,
        "horizon": horizon,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "slots_per_day": slots_per_day,
        "max_windows": max_windows,
        "splits": {
            split: {"x": list(x_shape), "y": list(y_shape)}
            for split, (x_shape, y_shape) in summary.items()
        },
        "channel_order": ["value", "time_in_day", f"history_anchor_{anchor_mode}"],
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DHA-DCD-ST single-anchor splits.")
    parser.add_argument("--traffic-h5", default="data/METRLA_data/METR-LA.h5")
    parser.add_argument("--output-dir", default="data/METRLA_anchor_mean")
    parser.add_argument("--anchor-mode", default="mean", choices=ANCHOR_MODES)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--slots-per-day", type=int, default=288)
    parser.add_argument("--max-windows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_metrla_dha_splits(
        args.traffic_h5,
        args.output_dir,
        anchor_mode=args.anchor_mode,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        slots_per_day=args.slots_per_day,
        max_windows=args.max_windows,
    )
    for split_name, (x_shape, y_shape) in summary.items():
        print(f"{split_name}: x={x_shape}, y={y_shape}")


if __name__ == "__main__":
    main()

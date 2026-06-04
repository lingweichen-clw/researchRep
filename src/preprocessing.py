"""Split-aware METR-LA preprocessing with ST-SSDL style history anchors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def generate_windows(
    data: np.ndarray,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate seq2seq windows.

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
    return (
        index.weekday.to_numpy() * slots_per_day
        + (index.hour.to_numpy() * 60 + index.minute.to_numpy()) // 5
    )


def build_history_anchor(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    slots_per_day: int = 288,
) -> np.ndarray:
    """Build train-only weekday-slot historical means.

    The anchor is computed from the raw train segment only to avoid temporal
    leakage into validation and test windows.
    """
    values = df.values.astype(np.float32)
    num_steps, num_nodes = values.shape
    train_end = int(num_steps * train_ratio)
    slot_ids = _weekday_slot(df.index, slots_per_day)
    num_slots = 7 * slots_per_day

    history = np.zeros((num_slots, num_nodes), dtype=np.float32)
    counts = np.zeros((num_slots, num_nodes), dtype=np.float32)
    train_slots = slot_ids[:train_end]
    train_values = values[:train_end]
    for slot in range(num_slots):
        mask = train_slots == slot
        if not np.any(mask):
            continue
        slot_values = train_values[mask]
        nonzero = slot_values != 0
        counts[slot] = nonzero.sum(axis=0)
        summed = np.where(nonzero, slot_values, 0.0).sum(axis=0)
        history[slot] = np.divide(
            summed,
            counts[slot],
            out=np.zeros_like(summed, dtype=np.float32),
            where=counts[slot] > 0,
        )

    sensor_mean = np.divide(
        train_values.sum(axis=0),
        np.maximum((train_values != 0).sum(axis=0), 1),
    ).astype(np.float32)
    empty = counts == 0
    history[empty] = np.take(sensor_mean, np.where(empty)[1])
    return history[slot_ids]


def generate_metrla_splits(
    traffic_h5: str | Path,
    output_dir: str | Path,
    seq_len: int = 12,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    slots_per_day: int = 288,
    max_windows: int | None = None,
) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Generate `trainhis/valhis/testhis.npz` files for METR-LA."""
    traffic_h5 = Path(traffic_h5)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_hdf(traffic_h5)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("METR-LA h5 must contain a DataFrame with DatetimeIndex.")

    values = df.values.astype(np.float32)
    tod = _time_in_day(df.index, values.shape[1])
    history_anchor = build_history_anchor(df, train_ratio=train_ratio, slots_per_day=slots_per_day)
    data = np.stack([values, tod, history_anchor], axis=-1)

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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate METR-LA ST-SSDL style splits.")
    parser.add_argument("--traffic-h5", default="data/METRLA_data/METR-LA.h5")
    parser.add_argument("--output-dir", default="data/METRLA")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    summary = generate_metrla_splits(
        args.traffic_h5,
        args.output_dir,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_windows=args.max_windows,
    )
    for split_name, (x_shape, y_shape) in summary.items():
        print(f"{split_name}: x={x_shape}, y={y_shape}")


if __name__ == "__main__":
    main()

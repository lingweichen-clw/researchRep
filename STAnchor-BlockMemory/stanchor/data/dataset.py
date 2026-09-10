"""Leakage-safe traffic series loading and window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import NodeStandardScaler


@dataclass(frozen=True)
class TrafficSeries:
    values: np.ndarray  # [L, N, C], raw physical values
    observed: np.ndarray  # [L, N, C]
    timestamps_ns: np.ndarray  # [L]
    weekday: np.ndarray  # [L]
    slot: np.ndarray  # [L]
    slots_per_day: int

    @property
    def num_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.values.shape[1])

    @property
    def num_channels(self) -> int:
        return int(self.values.shape[2])


class TrafficWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Windows fully contained in one chronological split."""

    def __init__(
        self,
        series: TrafficSeries,
        scaler: NodeStandardScaler,
        split_start: int,
        split_end: int,
        context_length: int,
        horizon: int,
        retrieval_context_length: int | None = None,
    ) -> None:
        if not 0 <= split_start < split_end <= series.num_steps:
            raise ValueError("invalid split bounds")
        if split_end - split_start < context_length + horizon:
            raise ValueError("split is too short for one context/future event")
        self.series = series
        self.scaler = scaler
        self.context_length = int(context_length)
        self.retrieval_context_length = int(
            context_length if retrieval_context_length is None else retrieval_context_length
        )
        self.horizon = int(horizon)
        if self.retrieval_context_length < self.context_length:
            raise ValueError("retrieval_context_length must be at least context_length")
        if split_end - split_start < self.retrieval_context_length + horizon:
            raise ValueError("split is too short for one retrieval context/future event")
        first_end = split_start + self.retrieval_context_length - 1
        last_end = split_end - horizon - 1
        candidates = np.arange(first_end, last_end + 1, dtype=np.int64)
        # Events with no context observation or no future supervision cannot
        # contribute to either pretraining objective or forecasting metrics.
        observed_per_step = series.observed.reshape(series.num_steps, -1).sum(axis=1)
        prefix = np.concatenate(([0], np.cumsum(observed_per_step, dtype=np.int64)))
        context_start = candidates - self.retrieval_context_length + 1
        context_count = prefix[candidates + 1] - prefix[context_start]
        future_count = prefix[candidates + horizon + 1] - prefix[candidates + 1]
        supervised = (context_count > 0) & (future_count > 0)
        self.context_end_indices = candidates[supervised]
        self.dropped_unobserved_events = int((~supervised).sum())
        if self.context_end_indices.size == 0:
            raise ValueError("split contains no event with observed context and future")

    def __len__(self) -> int:
        return int(self.context_end_indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        context_end = int(self.context_end_indices[index])
        context_start = context_end - self.retrieval_context_length + 1
        forecast_context_start = context_end - self.context_length + 1
        future_start = context_end + 1
        future_end = context_end + self.horizon
        retrieval_raw = self.series.values[context_start : context_end + 1]
        y_raw = self.series.values[future_start : future_end + 1]
        retrieval_observed = self.series.observed[context_start : context_end + 1]
        y_observed = self.series.observed[future_start : future_end + 1]
        retrieval_model = self.scaler.transform(retrieval_raw, retrieval_observed)
        x_model = retrieval_model[-self.context_length :]
        x_observed = retrieval_observed[-self.context_length :]
        y_model = self.scaler.transform(y_raw, y_observed)
        return {
            "x": torch.from_numpy(x_model),
            "y": torch.from_numpy(y_model),
            "x_observed": torch.from_numpy(x_observed),
            "y_observed": torch.from_numpy(y_observed),
            "weekday": torch.from_numpy(
                self.series.weekday[forecast_context_start : context_end + 1]
            ).long(),
            "slot": torch.from_numpy(
                self.series.slot[forecast_context_start : context_end + 1]
            ).long(),
            "retrieval_x": torch.from_numpy(retrieval_model),
            "retrieval_observed": torch.from_numpy(retrieval_observed),
            "retrieval_weekday": torch.from_numpy(
                self.series.weekday[context_start : context_end + 1]
            ).long(),
            "retrieval_slot": torch.from_numpy(
                self.series.slot[context_start : context_end + 1]
            ).long(),
            "query_weekday": torch.tensor(self.series.weekday[context_end], dtype=torch.long),
            "query_slot": torch.tensor(self.series.slot[context_end], dtype=torch.long),
            "context_start": torch.tensor(context_start, dtype=torch.long),
            "forecast_context_start": torch.tensor(
                forecast_context_start, dtype=torch.long
            ),
            "context_end": torch.tensor(context_end, dtype=torch.long),
            "future_end": torch.tensor(future_end, dtype=torch.long),
            "timestamp_ns": torch.tensor(self.series.timestamps_ns[context_end], dtype=torch.long),
            "sample_id": torch.tensor(context_end, dtype=torch.long),
        }


@dataclass(frozen=True)
class TrafficDataBundle:
    series: TrafficSeries
    scaler: NodeStandardScaler
    train: TrafficWindowDataset
    val: TrafficWindowDataset
    test: TrafficWindowDataset
    train_end: int
    val_end: int


def load_hdf_series(
    path: str | Path,
    frequency_minutes: int = 5,
    zero_is_missing: bool = True,
) -> TrafficSeries:
    frame = pd.read_hdf(Path(path))
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("HDF must contain a DataFrame with a DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and sorted")
    values = frame.to_numpy(dtype=np.float32)[..., None]
    observed = np.isfinite(values)
    if zero_is_missing:
        observed &= values != 0
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    slots_per_day = (24 * 60) // frequency_minutes
    minute_of_day = frame.index.hour.to_numpy() * 60 + frame.index.minute.to_numpy()
    slot = (minute_of_day // frequency_minutes).astype(np.int64)
    return TrafficSeries(
        values=values,
        observed=observed.astype(bool),
        timestamps_ns=frame.index.view("int64").astype(np.int64),
        weekday=frame.index.weekday.to_numpy(dtype=np.int64),
        slot=slot,
        slots_per_day=slots_per_day,
    )


def load_npz_series(
    path: str | Path,
    frequency_minutes: int = 5,
    zero_is_missing: bool = True,
    npz_key: str = "data",
    channel_index: int = 0,
    start_weekday: int = 0,
    start_slot: int = 0,
) -> TrafficSeries:
    """Load a traffic NPZ and select one physical variable as ``[T,N,1]``.

    NPZ files in the transfer datasets do not contain timestamps.  Their rows
    are therefore interpreted as consecutive ``frequency_minutes`` samples,
    with an explicit inferred calendar origin.
    """
    if frequency_minutes <= 0 or (24 * 60) % frequency_minutes != 0:
        raise ValueError("frequency_minutes must divide one day")
    if not 0 <= start_weekday <= 6:
        raise ValueError("start_weekday must be in [0, 6]")
    slots_per_day = (24 * 60) // frequency_minutes
    if not 0 <= start_slot < slots_per_day:
        raise ValueError("start_slot must be within one day")
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if npz_key not in archive.files:
            raise ValueError(f"NPZ does not contain key {npz_key!r}")
        raw = np.asarray(archive[npz_key])
    if raw.ndim != 3:
        raise ValueError(f"NPZ traffic array must be [T,N,C], got {raw.shape}")
    if not 0 <= channel_index < raw.shape[-1]:
        raise ValueError(
            f"channel_index {channel_index} is outside NPZ channel range {raw.shape[-1]}"
        )
    values = np.asarray(raw[..., channel_index : channel_index + 1], dtype=np.float32)
    observed = np.isfinite(values)
    if zero_is_missing:
        observed &= values != 0
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
    steps = np.arange(values.shape[0], dtype=np.int64)
    absolute_slots = start_slot + steps
    weekday = ((start_weekday + absolute_slots // slots_per_day) % 7).astype(np.int64)
    slot = (absolute_slots % slots_per_day).astype(np.int64)
    timestamps_ns = (
        absolute_slots.astype(np.int64)
        * int(frequency_minutes)
        * 60
        * 1_000_000_000
    )
    return TrafficSeries(
        values=values,
        observed=observed.astype(bool),
        timestamps_ns=timestamps_ns,
        weekday=weekday,
        slot=slot,
        slots_per_day=slots_per_day,
    )


def build_hdf_datasets(
    path: str | Path,
    context_length: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    frequency_minutes: int = 5,
    zero_is_missing: bool = True,
    retrieval_context_length: int | None = None,
) -> TrafficDataBundle:
    series = load_hdf_series(path, frequency_minutes, zero_is_missing)
    train_end = int(series.num_steps * train_ratio)
    val_end = int(series.num_steps * (train_ratio + val_ratio))
    scaler = NodeStandardScaler.fit(
        series.values[:train_end],
        series.observed[:train_end],
    )
    return TrafficDataBundle(
        series=series,
        scaler=scaler,
        train=TrafficWindowDataset(
            series,
            scaler,
            0,
            train_end,
            context_length,
            horizon,
            retrieval_context_length,
        ),
        val=TrafficWindowDataset(
            series,
            scaler,
            train_end,
            val_end,
            context_length,
            horizon,
            retrieval_context_length,
        ),
        test=TrafficWindowDataset(
            series,
            scaler,
            val_end,
            series.num_steps,
            context_length,
            horizon,
            retrieval_context_length,
        ),
        train_end=train_end,
        val_end=val_end,
    )


def build_npz_datasets(
    path: str | Path,
    context_length: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    frequency_minutes: int = 5,
    zero_is_missing: bool = True,
    retrieval_context_length: int | None = None,
    npz_key: str = "data",
    channel_index: int = 0,
    start_weekday: int = 0,
    start_slot: int = 0,
) -> TrafficDataBundle:
    series = load_npz_series(
        path,
        frequency_minutes,
        zero_is_missing,
        npz_key,
        channel_index,
        start_weekday,
        start_slot,
    )
    train_end = int(series.num_steps * train_ratio)
    val_end = int(series.num_steps * (train_ratio + val_ratio))
    scaler = NodeStandardScaler.fit(series.values[:train_end], series.observed[:train_end])
    return TrafficDataBundle(
        series=series,
        scaler=scaler,
        train=TrafficWindowDataset(
            series, scaler, 0, train_end, context_length, horizon, retrieval_context_length
        ),
        val=TrafficWindowDataset(
            series, scaler, train_end, val_end, context_length, horizon, retrieval_context_length
        ),
        test=TrafficWindowDataset(
            series, scaler, val_end, series.num_steps, context_length, horizon, retrieval_context_length
        ),
        train_end=train_end,
        val_end=val_end,
    )

def build_normalized_weekday_slot_mean_table(
    series: TrafficSeries,
    train_end: int,
    scaler: NodeStandardScaler,
) -> np.ndarray:
    """Build a train-only weekday-slot mean table in normalized units.

    The table has shape [N, 7 * slots_per_day]. Empty bins fall back to the
    node-level training mean, so the normalized value is zero.
    """
    if not 0 < train_end <= series.num_steps:
        raise ValueError("train_end must lie within the series")
    values = series.values[:train_end, :, 0]
    observed = series.observed[:train_end, :, 0]
    weekday = series.weekday[:train_end]
    slot = series.slot[:train_end]
    slot_count = 7 * series.slots_per_day
    keys = weekday * series.slots_per_day + slot
    node_count = series.num_nodes
    totals = np.zeros((node_count, slot_count), dtype=np.float64)
    counts = np.zeros((node_count, slot_count), dtype=np.float64)
    for node in range(node_count):
        valid = observed[:, node]
        if not np.any(valid):
            continue
        np.add.at(totals[node], keys[valid], values[valid, node])
        np.add.at(counts[node], keys[valid], 1.0)
    means = np.zeros((node_count, slot_count), dtype=np.float32)
    filled = counts > 0
    means[filled] = (totals[filled] / counts[filled]).astype(np.float32)
    node_mean = np.asarray(scaler.mean[:, 0], dtype=np.float32)
    for node in range(node_count):
        means[node, ~filled[node]] = node_mean[node]
    physical = means.T[:, :, None]
    return scaler.transform(physical)[:, :, 0].T.astype(np.float32)

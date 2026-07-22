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
    ) -> None:
        if not 0 <= split_start < split_end <= series.num_steps:
            raise ValueError("invalid split bounds")
        if split_end - split_start < context_length + horizon:
            raise ValueError("split is too short for one context/future event")
        self.series = series
        self.scaler = scaler
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        first_end = split_start + context_length - 1
        last_end = split_end - horizon - 1
        candidates = np.arange(first_end, last_end + 1, dtype=np.int64)
        # Events with no context observation or no future supervision cannot
        # contribute to either pretraining objective or forecasting metrics.
        observed_per_step = series.observed.reshape(series.num_steps, -1).sum(axis=1)
        prefix = np.concatenate(([0], np.cumsum(observed_per_step, dtype=np.int64)))
        context_start = candidates - context_length + 1
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
        context_start = context_end - self.context_length + 1
        future_start = context_end + 1
        future_end = context_end + self.horizon
        x_raw = self.series.values[context_start : context_end + 1]
        y_raw = self.series.values[future_start : future_end + 1]
        x_observed = self.series.observed[context_start : context_end + 1]
        y_observed = self.series.observed[future_start : future_end + 1]
        x_model = self.scaler.transform(x_raw, x_observed)
        y_model = self.scaler.transform(y_raw, y_observed)
        return {
            "x": torch.from_numpy(x_model),
            "y": torch.from_numpy(y_model),
            "x_observed": torch.from_numpy(x_observed),
            "y_observed": torch.from_numpy(y_observed),
            "weekday": torch.from_numpy(self.series.weekday[context_start : context_end + 1]).long(),
            "slot": torch.from_numpy(self.series.slot[context_start : context_end + 1]).long(),
            "query_weekday": torch.tensor(self.series.weekday[context_end], dtype=torch.long),
            "query_slot": torch.tensor(self.series.slot[context_end], dtype=torch.long),
            "context_start": torch.tensor(context_start, dtype=torch.long),
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


def build_hdf_datasets(
    path: str | Path,
    context_length: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    frequency_minutes: int = 5,
    zero_is_missing: bool = True,
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
        train=TrafficWindowDataset(series, scaler, 0, train_end, context_length, horizon),
        val=TrafficWindowDataset(series, scaler, train_end, val_end, context_length, horizon),
        test=TrafficWindowDataset(series, scaler, val_end, series.num_steps, context_length, horizon),
        train_end=train_end,
        val_end=val_end,
    )

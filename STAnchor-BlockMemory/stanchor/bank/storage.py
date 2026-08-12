"""NumPy mmap storage for immutable event-aligned memory banks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stanchor.utils import load_json, save_json

from .schema import BankManifest


ARRAY_FILES = {
    "event_keys": "event_keys.npy",
    "node_keys": "node_keys.npy",
    "future_values": "future_values.npy",
    "future_masks": "future_masks.npy",
    "level_features": "level_features.npy",
    "weekday": "weekday.npy",
    "slot": "slot.npy",
    "context_start": "context_start.npy",
    "context_end": "context_end.npy",
    "future_end": "future_end.npy",
    "sample_id": "sample_id.npy",
}


@dataclass(frozen=True)
class CalendarIndex:
    offsets: np.ndarray  # [7 * slots_per_day + 1]
    event_ids: np.ndarray  # [M]
    slots_per_day: int

    @classmethod
    def build(cls, weekday: np.ndarray, slot: np.ndarray, slots_per_day: int) -> "CalendarIndex":
        weekday = np.asarray(weekday, dtype=np.int64)
        slot = np.asarray(slot, dtype=np.int64)
        if weekday.ndim != 1 or slot.shape != weekday.shape:
            raise ValueError("weekday and slot must be aligned vectors")
        if (weekday < 0).any() or (weekday >= 7).any():
            raise ValueError("weekday ids must be in [0, 6]")
        if (slot < 0).any() or (slot >= slots_per_day).any():
            raise ValueError("slot ids outside slots_per_day")
        bucket = weekday * slots_per_day + slot
        order = np.argsort(bucket, kind="stable")
        counts = np.bincount(bucket, minlength=7 * slots_per_day)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return cls(offsets=offsets, event_ids=order.astype(np.int64), slots_per_day=slots_per_day)

    def lookup(self, weekday: int, slot: int) -> np.ndarray:
        if not 0 <= weekday < 7 or not 0 <= slot < self.slots_per_day:
            raise ValueError("invalid calendar query")
        bucket = weekday * self.slots_per_day + slot
        return self.event_ids[self.offsets[bucket] : self.offsets[bucket + 1]]


class BankWriter:
    """Write every array on the shared event axis; manifest is committed last."""

    def __init__(self, output_dir: str | Path, manifest: BankManifest) -> None:
        manifest.validate()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / "manifest.json").exists():
            raise FileExistsError(f"Refusing to overwrite existing bank: {self.output_dir}")
        unexpected = [path.name for path in self.output_dir.iterdir() if path.is_file()]
        if unexpected:
            raise FileExistsError(f"Bank output directory is not empty: {unexpected[:5]}")
        self.manifest = manifest
        m, n, h, c, d = (
            manifest.num_events,
            manifest.num_nodes,
            manifest.horizon,
            manifest.channels,
            manifest.retrieval_dim,
        )
        key_dtype = np.dtype(manifest.key_dtype)
        self.arrays: dict[str, np.memmap] = {
            "event_keys": np.lib.format.open_memmap(
                self.output_dir / ARRAY_FILES["event_keys"], mode="w+", dtype=key_dtype, shape=(m, d)
            ),
            "node_keys": np.lib.format.open_memmap(
                self.output_dir / ARRAY_FILES["node_keys"], mode="w+", dtype=key_dtype, shape=(m, n, d)
            ),
            "future_values": np.lib.format.open_memmap(
                self.output_dir / ARRAY_FILES["future_values"], mode="w+", dtype=np.float32, shape=(m, h, n, c)
            ),
            "future_masks": np.lib.format.open_memmap(
                self.output_dir / ARRAY_FILES["future_masks"], mode="w+", dtype=np.uint8, shape=(m, h, n, c)
            ),
            "level_features": np.lib.format.open_memmap(
                self.output_dir / ARRAY_FILES["level_features"], mode="w+", dtype=np.float32, shape=(m, n, 4 * c)
            ),
        }
        for name in ("weekday", "slot"):
            self.arrays[name] = np.lib.format.open_memmap(
                self.output_dir / ARRAY_FILES[name], mode="w+", dtype=np.int16, shape=(m,)
            )
        for name in ("context_start", "context_end", "future_end", "sample_id"):
            self.arrays[name] = np.lib.format.open_memmap(
                self.output_dir / ARRAY_FILES[name], mode="w+", dtype=np.int64, shape=(m,)
            )
        self.offset = 0

    def write(self, batch: dict[str, np.ndarray]) -> None:
        required = set(self.arrays)
        if set(batch) != required:
            missing = sorted(required - set(batch))
            extra = sorted(set(batch) - required)
            raise ValueError(f"Bank batch fields mismatch; missing={missing}, extra={extra}")
        batch_size = int(np.asarray(batch["event_keys"]).shape[0])
        end = self.offset + batch_size
        if end > self.manifest.num_events:
            raise ValueError("bank writer received more events than declared")
        for name, target in self.arrays.items():
            value = np.asarray(batch[name])
            if value.shape[0] != batch_size or value.shape[1:] != target.shape[1:]:
                raise ValueError(f"{name} has shape {value.shape}, expected batch axis + {target.shape[1:]}")
            target[self.offset : end] = value.astype(target.dtype, copy=False)
        self.offset = end

    def finalize(self) -> None:
        if self.offset != self.manifest.num_events:
            raise ValueError(f"bank incomplete: wrote {self.offset} of {self.manifest.num_events} events")
        for array in self.arrays.values():
            array.flush()
        calendar = CalendarIndex.build(
            np.asarray(self.arrays["weekday"]),
            np.asarray(self.arrays["slot"]),
            self.manifest.slots_per_day,
        )
        np.save(self.output_dir / "calendar_offsets.npy", calendar.offsets)
        np.save(self.output_dir / "calendar_event_ids.npy", calendar.event_ids)
        save_json(self.output_dir / "manifest.json", self.manifest.to_dict())
        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays = {}


class MemoryBank:
    def __init__(self, path: str | Path, expected_schema_version: int | None = None) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Bank is incomplete or missing manifest: {manifest_path}")
        self.manifest = BankManifest.from_dict(load_json(manifest_path))
        if expected_schema_version is not None and self.manifest.schema_version != expected_schema_version:
            raise ValueError(
                f"Bank schema version {self.manifest.schema_version} does not match "
                f"expected schema version {expected_schema_version}"
            )
        self.arrays: dict[str, np.ndarray] = {
            name: np.load(self.path / filename, mmap_mode="r") for name, filename in ARRAY_FILES.items()
        }
        self._validate_shapes()
        self.calendar = CalendarIndex(
            offsets=np.load(self.path / "calendar_offsets.npy", mmap_mode="r"),
            event_ids=np.load(self.path / "calendar_event_ids.npy", mmap_mode="r"),
            slots_per_day=self.manifest.slots_per_day,
        )
        # Event keys are small enough to keep resident for exact coarse search.
        self.event_keys_memory = np.array(self.arrays["event_keys"], dtype=np.float32, copy=True)

    def _validate_shapes(self) -> None:
        m, n, h, c, d = (
            self.manifest.num_events,
            self.manifest.num_nodes,
            self.manifest.horizon,
            self.manifest.channels,
            self.manifest.retrieval_dim,
        )
        expected: dict[str, tuple[int, ...]] = {
            "event_keys": (m, d),
            "node_keys": (m, n, d),
            "future_values": (m, h, n, c),
            "future_masks": (m, h, n, c),
            "level_features": (m, n, 4 * c),
            "weekday": (m,),
            "slot": (m,),
            "context_start": (m,),
            "context_end": (m,),
            "future_end": (m,),
            "sample_id": (m,),
        }
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"Bank array {name} has shape {self.arrays[name].shape}, expected {shape}")

    def __getattr__(self, name: str) -> Any:
        if name != "arrays" and "arrays" in self.__dict__ and name in self.arrays:
            return self.arrays[name]
        raise AttributeError(name)

    def close(self) -> None:
        mapped_arrays = list(self.arrays.values())
        mapped_arrays.extend((self.calendar.offsets, self.calendar.event_ids))
        for array in mapped_arrays:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays = {}

    def __enter__(self) -> "MemoryBank":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

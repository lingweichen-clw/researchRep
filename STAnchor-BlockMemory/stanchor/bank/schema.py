"""Versioned schema for on-disk memory banks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BankManifest:
    schema_version: int
    dataset_name: str
    num_events: int
    num_nodes: int
    context_length: int
    horizon: int
    channels: int
    retrieval_dim: int
    slots_per_day: int
    key_dtype: str
    future_dtype: str
    encoder_fingerprint: str
    graph_fingerprint: str
    scaler: dict[str, Any]

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported bank schema version: {self.schema_version}")
        for name in (
            "num_events",
            "num_nodes",
            "context_length",
            "horizon",
            "channels",
            "retrieval_dim",
            "slots_per_day",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.key_dtype not in {"float16", "float32"}:
            raise ValueError("key_dtype must be float16 or float32")
        if self.future_dtype != "float32":
            raise ValueError("v1 bank future_dtype must be float32")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BankManifest":
        manifest = cls(**value)
        manifest.validate()
        return manifest


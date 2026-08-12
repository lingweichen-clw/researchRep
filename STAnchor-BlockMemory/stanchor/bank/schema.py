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
    key_layout: str = "legacy"
    profile_dim: int = 0
    latent_dim: int = 0
    profile_weight: float = 0.0
    profile_grid_size: int = 0
    profile_scale_floor: float = 0.1
    relation_distance_normalization: str = "none"

    def validate(self) -> None:
        if self.schema_version not in {1, 2}:
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
            raise ValueError("bank future_dtype must be float32")
        if self.schema_version == 1:
            if self.key_layout != "legacy":
                raise ValueError("v1 bank must use legacy key layout")
            return
        if self.key_layout != "canonical_profile_latent":
            raise ValueError("v2 bank must use canonical_profile_latent key layout")
        if self.profile_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("v2 profile_dim and latent_dim must be positive")
        if self.profile_dim + self.latent_dim != self.retrieval_dim:
            raise ValueError("v2 key dimensions must sum to retrieval_dim")
        if not 0.0 <= self.profile_weight <= 1.0:
            raise ValueError("v2 profile_weight must be in [0, 1]")
        if self.profile_grid_size != self.profile_dim:
            raise ValueError("v2 profile_grid_size must equal profile_dim")
        if self.profile_scale_floor <= 0:
            raise ValueError("v2 profile_scale_floor must be positive")
        if self.relation_distance_normalization != "symmetric_geometric_mean":
            raise ValueError("v2 bank must record symmetric_geometric_mean normalization")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BankManifest":
        manifest = cls(**value)
        manifest.validate()
        return manifest

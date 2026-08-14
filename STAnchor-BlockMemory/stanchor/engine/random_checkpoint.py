"""Deterministic untrained checkpoint construction for pretraining controls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from stanchor.config import ExperimentConfig
from stanchor.utils import count_parameters, set_seed

from .common import build_pretrain_model


def _cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def build_random_checkpoint_payload(
    config: ExperimentConfig,
    slots_per_day: int,
    normalizer: dict[str, Any],
    graph_fingerprint: str,
    seed: int,
) -> dict[str, Any]:
    """Build a reproducible checkpoint whose parameters have received zero updates."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if slots_per_day <= 0:
        raise ValueError("slots_per_day must be positive")
    if not graph_fingerprint:
        raise ValueError("graph_fingerprint must not be empty")

    set_seed(seed)
    model = build_pretrain_model(config, slots_per_day=slots_per_day).cpu()
    retrieval_fingerprint = model.retrieval_fingerprint()
    return {
        "model_state_dict": _cpu_state_dict(dict(model.state_dict())),
        "encoder_state_dict": _cpu_state_dict(dict(model.encoder.state_dict())),
        "retrieval_encoder_state_dict": _cpu_state_dict(model.retrieval_state_dict()),
        "retrieval_state_dict": _cpu_state_dict(model.retrieval_state_dict()),
        "retrieval_fingerprint": retrieval_fingerprint,
        "config": config.to_dict(),
        "normalizer": deepcopy(normalizer),
        "graph_fingerprint": graph_fingerprint,
        "metrics": {
            "status": "untrained_random_control",
            "trained_steps": 0,
        },
        "parameter_counts": {
            "total": count_parameters(model, trainable_only=False),
            "dynamics_adapter": (
                count_parameters(model.dynamics_adapter, trainable_only=False)
                if model.dynamics_adapter is not None
                else 0
            ),
            "retrieval_path": sum(
                tensor.numel() for tensor in model.retrieval_state_dict().values()
            ),
        },
        "checkpoint_kind": "target_random_untrained",
        "epoch": 0,
        "seed": seed,
        "trained_steps": 0,
    }

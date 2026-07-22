"""Shared experiment construction and checkpoint contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from stanchor.config import ExperimentConfig, resolve_project_path
from stanchor.data.dataset import TrafficDataBundle, build_hdf_datasets
from stanchor.data.graph import GraphData, load_graph
from stanchor.models.pretraining import STAnchorPretrainModel


def build_data_and_graph(config: ExperimentConfig) -> tuple[TrafficDataBundle, GraphData]:
    data = build_hdf_datasets(
        path=resolve_project_path(config.data.raw_path),
        context_length=config.data.context_length,
        horizon=config.data.horizon,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        frequency_minutes=config.data.frequency_minutes,
        zero_is_missing=config.data.zero_is_missing,
    )
    graph = load_graph(resolve_project_path(config.data.adjacency_path))
    if graph.num_nodes != data.series.num_nodes:
        raise ValueError(
            f"Graph has {graph.num_nodes} nodes but data has {data.series.num_nodes}"
        )
    return data, graph


def build_pretrain_model(config: ExperimentConfig, slots_per_day: int) -> STAnchorPretrainModel:
    return STAnchorPretrainModel(
        model_config=config.model,
        pretrain_config=config.pretrain,
        context_length=config.data.context_length,
        slots_per_day=slots_per_day,
    )


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_pretrained_model(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    slots_per_day: int,
    device: torch.device,
) -> tuple[STAnchorPretrainModel, dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_pretrain_model(config, slots_per_day).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


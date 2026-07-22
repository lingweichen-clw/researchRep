"""Encode chronological target history into an immutable memory bank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stanchor.data.graph import GraphData
from stanchor.data.normalization import NodeStandardScaler
from stanchor.models.pretraining import STAnchorPretrainModel

from .schema import BankManifest
from .storage import BankWriter


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@torch.no_grad()
def build_memory_bank(
    model: STAnchorPretrainModel,
    dataset: Dataset,
    graph: GraphData,
    scaler: NodeStandardScaler,
    output_dir: str | Path,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    key_dtype: str = "float16",
) -> BankManifest:
    if len(dataset) <= 0:
        raise ValueError("cannot build an empty bank")
    graph_device = graph.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sample = dataset[0]
    nodes = int(sample["x"].shape[1])
    channels = int(sample["x"].shape[2])
    manifest = BankManifest(
        schema_version=1,
        dataset_name=dataset_name,
        num_events=len(dataset),
        num_nodes=nodes,
        context_length=model.context_length,
        horizon=int(sample["y"].shape[0]),
        channels=channels,
        retrieval_dim=model.model_config.retrieval_dim,
        slots_per_day=model.embedding.slots_per_day,
        key_dtype=key_dtype,
        future_dtype="float32",
        encoder_fingerprint=model.retrieval_fingerprint(),
        graph_fingerprint=graph.fingerprint,
        scaler=scaler.state_dict(),
    )
    writer = BankWriter(output_dir, manifest)
    model.eval()
    for batch in loader:
        x = batch["retrieval_x"].to(device)
        observed = batch["retrieval_observed"].to(device)
        encoding = model.encode_clean(
            x=x,
            observed=observed,
            weekday=batch["retrieval_weekday"].to(device),
            slot=batch["retrieval_slot"].to(device),
            graph=graph_device,
        )
        writer.write(
            {
                "event_keys": _numpy(encoding.retrieval.event_keys),
                "node_keys": _numpy(encoding.retrieval.node_keys),
                "future_values": _numpy(batch["y"]),
                "future_masks": _numpy(batch["y_observed"]).astype(np.uint8),
                "level_features": _numpy(encoding.statistics.level_features),
                "weekday": _numpy(batch["query_weekday"]),
                "slot": _numpy(batch["query_slot"]),
                "context_start": _numpy(batch["context_start"]),
                "context_end": _numpy(batch["context_end"]),
                "future_end": _numpy(batch["future_end"]),
                "sample_id": _numpy(batch["sample_id"]),
            }
        )
    writer.finalize()
    return manifest

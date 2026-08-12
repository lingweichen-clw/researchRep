from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.bank.builder import build_memory_bank
from stanchor.config import load_config, resolve_project_path
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable target history bank.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--max-events", type=int, default=None, help="Smoke-only event cap.")
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config.runtime.device)
    data, graph = build_data_and_graph(config)
    model, checkpoint = load_pretrained_model(
        config, args.checkpoint, data.series.slots_per_day, device
    )
    if checkpoint.get("retrieval_fingerprint") != model.retrieval_fingerprint():
        raise ValueError("Loaded checkpoint retrieval fingerprint is inconsistent")
    memory_events = int(len(data.train) * config.bank.memory_fraction)
    if args.max_events is not None:
        if args.max_events <= 0:
            raise ValueError("max-events must be positive")
        memory_events = min(memory_events, args.max_events)
    dataset = Subset(data.train, range(memory_events))
    output_dir = resolve_project_path(args.output_dir or config.bank.output_dir)
    manifest = build_memory_bank(
        model=model,
        dataset=dataset,
        graph=graph,
        scaler=data.scaler,
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        batch_size=config.target.batch_size,
        num_workers=config.data.num_workers,
        device=device,
        key_dtype=config.bank.key_dtype,
        profile_scale_floor=config.pretrain.profile_scale_floor,
        relation_distance_normalization=config.pretrain.relation_distance_normalization,
    )
    print(f"bank: {output_dir}")
    print(f"events={manifest.num_events} nodes={manifest.num_nodes} key_dim={manifest.retrieval_dim}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.engine.common import (
    build_data_and_graph,
    load_pretrained_model,
    save_checkpoint,
)
from stanchor.engine.random_checkpoint import build_random_checkpoint_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic untrained encoder-selector checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.runtime.seed if args.seed is None else args.seed
    output = resolve_project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output}")

    data, graph = build_data_and_graph(config)
    payload = build_random_checkpoint_payload(
        config=config,
        slots_per_day=data.series.slots_per_day,
        normalizer=data.scaler.state_dict(),
        graph_fingerprint=graph.fingerprint,
        seed=seed,
    )
    save_checkpoint(output, payload)

    restored, checkpoint = load_pretrained_model(
        config,
        output,
        data.series.slots_per_day,
        torch.device("cpu"),
    )
    if checkpoint.get("retrieval_fingerprint") != restored.retrieval_fingerprint():
        raise RuntimeError("Saved random checkpoint failed fingerprint verification")
    if checkpoint.get("trained_steps") != 0:
        raise RuntimeError("Random control checkpoint must have zero trained steps")

    print(f"random checkpoint: {output}")
    print(f"seed={seed} trained_steps=0")
    print(f"retrieval_fingerprint={restored.retrieval_fingerprint()}")
    print(
        "parameters="
        f"{checkpoint['parameter_counts']['total']} "
        f"retrieval_path={checkpoint['parameter_counts']['retrieval_path']}"
    )


if __name__ == "__main__":
    main()

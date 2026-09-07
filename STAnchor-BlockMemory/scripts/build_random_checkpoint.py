from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.engine.common import build_data_and_graph, save_checkpoint
from stanchor.engine.random_checkpoint import build_random_checkpoint_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic untrained retrieval checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    config.validate()
    data, graph = build_data_and_graph(config)
    payload = build_random_checkpoint_payload(
        config,
        slots_per_day=data.series.slots_per_day,
        normalizer=data.scaler.state_dict(),
        graph_fingerprint=graph.fingerprint,
        seed=args.seed,
    )
    output = resolve_project_path(args.output)
    save_checkpoint(output, payload)
    print(f"random checkpoint: {output.resolve()}")
    print(f"retrieval_fingerprint: {payload['retrieval_fingerprint']}")


if __name__ == "__main__":
    main()

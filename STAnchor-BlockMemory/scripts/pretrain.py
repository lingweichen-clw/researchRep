from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config
from stanchor.engine.pretrainer import train_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the STAnchor retrieval encoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Temporary debug override.")
    parser.add_argument("--run-name", default=None, help="Override the artifact directory name.")
    parser.add_argument("--seed", type=int, default=None, help="Override only the pretraining seed.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("epochs must be positive")
        config = replace(config, pretrain=replace(config.pretrain, epochs=args.epochs))
    if args.run_name is not None:
        config = replace(config, runtime=replace(config.runtime, run_name=args.run_name))
    if args.seed is not None:
        config = replace(config, runtime=replace(config.runtime, seed=args.seed))
    config.validate()
    checkpoint = train_pretraining(config, max_batches=args.max_batches)
    print(f"pretraining checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

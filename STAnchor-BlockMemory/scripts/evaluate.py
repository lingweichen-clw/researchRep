from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config
from stanchor.engine.target import evaluate_downstream


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained STAnchor downstream model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--downstream-checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    result = evaluate_downstream(
        load_config(args.config),
        pretrained_checkpoint=args.pretrained_checkpoint,
        downstream_checkpoint=args.downstream_checkpoint,
        bank_path=args.bank,
        split=args.split,
        max_batches=args.max_batches,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


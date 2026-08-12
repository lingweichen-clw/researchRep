from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config
from stanchor.diagnostics.retrieval_temperature import (
    DEFAULT_TEMPERATURES,
    run_retrieval_temperature_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep deployed E5A Top-5 aggregation temperatures without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--random-checkpoint", required=True)
    parser.add_argument("--random-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument(
        "--candidate-protocol",
        choices=("relaxed_calendar",),
        default="relaxed_calendar",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEMPERATURES),
        help="Positive softmax temperatures; uniform and hard-Top1 are added automatically.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Engineering smoke only. Omit for a formal complete-validation run.",
    )
    args = parser.parse_args()
    result = run_retrieval_temperature_sweep(
        config=load_config(args.config),
        checkpoint_path=args.checkpoint,
        bank_path=args.bank,
        random_checkpoint_path=args.random_checkpoint,
        random_bank_path=args.random_bank,
        output_dir=args.output_dir,
        split=args.split,
        candidate_protocol=args.candidate_protocol,
        temperatures=args.temperatures,
        max_batches=args.max_batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"temperature diagnostic output: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

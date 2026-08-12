from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config
from stanchor.diagnostics.teacher_metric_diagnostic import run_teacher_metric_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-bank, zero-training E5A teacher metric diagnostic."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-protocol",
        choices=("relaxed_calendar",),
        default="relaxed_calendar",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Engineering smoke only; omit for complete validation.",
    )
    parser.add_argument(
        "--clip-delta",
        type=float,
        default=None,
        help="Optional fixed standardized OD residual clip. Default: source-Bank p95.",
    )
    parser.add_argument(
        "--perturb-rate",
        type=float,
        default=0.1,
        help="Fraction of event-node series receiving one mid-horizon perturbation.",
    )
    args = parser.parse_args()
    result = run_teacher_metric_diagnostic(
        config=load_config(args.config),
        checkpoint_path=args.checkpoint,
        bank_path=args.bank,
        split=args.split,
        output_dir=args.output_dir,
        candidate_protocol=args.candidate_protocol,
        max_batches=args.max_batches,
        clip_delta=args.clip_delta,
        perturb_rate=args.perturb_rate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"diagnostic output: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

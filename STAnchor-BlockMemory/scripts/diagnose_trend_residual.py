from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.trend_residual import diagnose_trend_residual_value
from stanchor.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the E5 T0 trend-residual retrieval diagnostic."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--trend-length", type=int, default=12)
    parser.add_argument(
        "--output",
        default="artifacts/e5_t0_trend_residual.json",
        help="JSON output path relative to the project root.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    result = diagnose_trend_residual_value(
        load_config(args.config),
        checkpoint_path=args.checkpoint,
        bank_path=args.bank,
        split=args.split,
        trend_length=args.trend_length,
        max_batches=args.max_batches,
    )
    output = resolve_project_path(args.output)
    save_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"diagnostic output: {output}")


if __name__ == "__main__":
    main()

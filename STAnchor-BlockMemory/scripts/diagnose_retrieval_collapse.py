from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.retrieval_collapse import run_retrieval_collapse_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose retrieval collapse and context reliance with frozen inputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--version",
        choices=("hn_offset_decay_v2", "hn_offset_only_v1"),
        default="hn_offset_decay_v2",
    )
    parser.add_argument("--candidate-protocol", default="weekday_radius1_overlap")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1024,
        help="Frozen validation query cap; omit with --all-queries for full validation.",
    )
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--geometry-samples", type=int, default=40000)
    parser.add_argument("--random-bank", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    max_queries = None if args.all_queries else args.max_queries
    result = run_retrieval_collapse_diagnostic(
        config=load_config(args.config),
        checkpoint_path=args.checkpoint,
        bank_path=args.bank,
        output_dir=args.output_dir,
        version=args.version,
        candidate_protocol=args.candidate_protocol,
        max_queries=max_queries,
        batch_size=args.batch_size,
        geometry_samples=args.geometry_samples,
        random_bank_path=args.random_bank,
        device_override=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"diagnostic output: {resolve_project_path(args.output_dir)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.key_payload_factorial import run_key_payload_factorial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen 2-Key x 2-payload retrieval-memory factorial."
    )
    parser.add_argument("--offset-only-config", required=True)
    parser.add_argument("--offset-only-checkpoint", required=True)
    parser.add_argument("--offset-only-bank", required=True)
    parser.add_argument("--offset-decay-config", required=True)
    parser.add_argument("--offset-decay-checkpoint", required=True)
    parser.add_argument("--offset-decay-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-protocol", default="weekday_radius1_overlap")
    parser.add_argument("--max-queries", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = run_key_payload_factorial(
        offset_only_config=load_config(args.offset_only_config),
        offset_only_checkpoint=args.offset_only_checkpoint,
        offset_only_bank=args.offset_only_bank,
        offset_decay_config=load_config(args.offset_decay_config),
        offset_decay_checkpoint=args.offset_decay_checkpoint,
        offset_decay_bank=args.offset_decay_bank,
        output_dir=args.output_dir,
        candidate_protocol=args.candidate_protocol,
        max_queries=args.max_queries,
        batch_size=args.batch_size,
        bootstrap_resamples=args.bootstrap_resamples,
        device_override=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"diagnostic output: {resolve_project_path(args.output_dir)}")


if __name__ == "__main__":
    main()

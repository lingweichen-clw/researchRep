from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.retrieval import diagnose_retrieval_value
from stanchor.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose learned historical retrieval value.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--output",
        default="artifacts/retrieval_diagnostics.json",
        help="JSON output path relative to the project root.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    result = diagnose_retrieval_value(
        load_config(args.config),
        checkpoint_path=args.checkpoint,
        bank_path=args.bank,
        split=args.split,
        max_batches=args.max_batches,
    )
    output = resolve_project_path(args.output)
    save_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"diagnostic output: {output}")


if __name__ == "__main__":
    main()

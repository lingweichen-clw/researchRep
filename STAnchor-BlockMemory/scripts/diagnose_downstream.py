from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.downstream import diagnose_downstream_checkpoint
from stanchor.utils import save_json
from stanchor.retrieval.strategies import CANDIDATE_PROTOCOLS


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose downstream branch and confidence value.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--downstream-checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--candidate-protocol", choices=CANDIDATE_PROTOCOLS, default=None)
    args = parser.parse_args()
    result = diagnose_downstream_checkpoint(
        load_config(args.config),
        pretrained_checkpoint=args.pretrained_checkpoint,
        downstream_checkpoint=args.downstream_checkpoint,
        bank_path=args.bank,
        split=args.split,
        max_batches=args.max_batches,
        candidate_protocol=args.candidate_protocol,
    )
    output = resolve_project_path(args.output)
    save_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"diagnostic output: {output}")


if __name__ == "__main__":
    main()

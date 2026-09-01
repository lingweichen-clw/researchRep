from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.spatial_residual import diagnose_spatial_residuals
from stanchor.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose spatial structure in base residuals and candidate utility"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    result = diagnose_spatial_residuals(
        load_config(args.config),
        args.pretrained_checkpoint,
        args.base_checkpoint,
        args.bank,
        split=args.split,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    output = resolve_project_path(args.output)
    save_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"diagnostic output: {output}")


if __name__ == "__main__":
    main()

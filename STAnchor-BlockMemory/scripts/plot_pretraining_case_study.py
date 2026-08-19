from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.diagnostics.pretraining_curves import render_pretraining_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a pretraining history figure.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    output = render_pretraining_history(args.metrics, args.output, args.title)
    print(f"pretraining figure: {output.resolve()}")


if __name__ == "__main__":
    main()

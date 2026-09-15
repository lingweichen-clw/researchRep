from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.diagnostics.case_study_report_figures import (
    render_case_study_report_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render publication-facing figures from completed case-study metrics."
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--broad-metrics", required=True)
    parser.add_argument("--exact-metrics", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--frequency-minutes", type=int, default=5)
    parser.add_argument(
        "--secondary-protocol-label",
        default="Weekday ±1, same slot (deployment-like)",
    )
    parser.add_argument("--context-surface")
    parser.add_argument("--context-quadrants")
    parser.add_argument("--context-partials")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = render_case_study_report_figures(
        history_path=args.history,
        broad_metrics_path=args.broad_metrics,
        exact_metrics_path=args.exact_metrics,
        model_label=args.model_label,
        objective=args.objective,
        frequency_minutes=args.frequency_minutes,
        output_dir=args.output_dir,
        secondary_protocol_label=args.secondary_protocol_label,
        context_surface_path=args.context_surface,
        context_quadrant_path=args.context_quadrants,
        context_partial_path=args.context_partials,
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()

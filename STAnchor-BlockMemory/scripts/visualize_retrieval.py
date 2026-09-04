from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config
from stanchor.diagnostics.retrieval_visualization import (
    CURRENT_VISUALIZATION_VERSION,
    run_retrieval_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize teacher-aligned historical retrieval for HN-OffsetDecay v2."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--random-checkpoint", required=True)
    parser.add_argument("--random-bank", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-protocol",
        choices=(
            "exact_calendar",
            "relaxed_calendar",
            "relaxed_calendar_diverse",
            "weekday_radius1_overlap",
            "broad_causal",
            "pretrain_broad_causal",
        ),
        default="weekday_radius1_overlap",
        help=(
            "Validation-only candidate attribution protocol. broad_causal uses "
            "model-independent chronological quantile sampling up to event_top_r."
        ),
    )
    parser.add_argument(
        "--level-weight",
        type=float,
        default=None,
        help=(
            "Optional diagnostic override for endpoint-level reranking. "
            "Use 0 to attribute retrieval entirely to learned key similarity."
        ),
    )
    parser.add_argument(
        "--event-top-r",
        type=int,
        default=None,
        help=(
            "Validation-only candidate-pool width override. Use 96 for "
            "weekday_radius1_overlap and broad_causal/pretrain_broad_causal "
            "to avoid truncating the current analysis pool."
        ),
    )
    parser.add_argument(
        "--node-top-k",
        type=int,
        default=None,
        help="Validation-only node-level Top-K override.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Engineering smoke only. Omit for a formal complete-validation run.",
    )
    parser.add_argument(
        "--profile-weight-override",
        type=float,
        default=None,
        help=(
            "Validation-only override for profile/latent cosine composition. "
            "Reuses the same v2 checkpoint and Bank; valid values are in [0, 1]."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.level_weight is not None:
        if args.level_weight < 0:
            raise ValueError("level-weight must be non-negative")
        config = replace(
            config,
            bank=replace(config.bank, level_weight=args.level_weight),
        )
        config.validate()
    if args.event_top_r is not None:
        if args.event_top_r <= 0:
            raise ValueError("event-top-r must be positive")
        config = replace(
            config,
            bank=replace(config.bank, event_top_r=args.event_top_r),
        )
        config.validate()
    result = run_retrieval_visualization(
        version=CURRENT_VISUALIZATION_VERSION,
        config=config,
        checkpoint_path=args.checkpoint,
        bank_path=args.bank,
        random_checkpoint_path=args.random_checkpoint,
        random_bank_path=args.random_bank,
        split=args.split,
        output_dir=args.output_dir,
        max_batches=args.max_batches,
        candidate_protocol=args.candidate_protocol,
        profile_weight_override=args.profile_weight_override,
        node_top_k_override=args.node_top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"visualization output: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

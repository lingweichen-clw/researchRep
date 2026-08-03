"""Canonical downstream ablation mode names."""

from __future__ import annotations

BASE_ONLY = "base_only"
WEEKLY_MEAN_HORIZON = "weekly_mean_horizon"
RAW_L1_TOPK_HORIZON = "raw_l1_topk_horizon"
LEARNED_TOPK_HORIZON = "learned_topk_horizon"
LEARNED_TOPK_OFFSET_DECAY_HORIZON = "learned_topk_offset_decay_horizon"
LEARNED_TOPK_CONFIDENCE = "learned_topk_confidence"

DOWNSTREAM_MODES = (
    BASE_ONLY,
    WEEKLY_MEAN_HORIZON,
    RAW_L1_TOPK_HORIZON,
    LEARNED_TOPK_HORIZON,
    LEARNED_TOPK_OFFSET_DECAY_HORIZON,
    LEARNED_TOPK_CONFIDENCE,
)
HORIZON_ONLY_MODES = frozenset(
    {
        WEEKLY_MEAN_HORIZON,
        RAW_L1_TOPK_HORIZON,
        LEARNED_TOPK_HORIZON,
        LEARNED_TOPK_OFFSET_DECAY_HORIZON,
    }
)
MEMORY_MODES = frozenset(set(HORIZON_ONLY_MODES) | {LEARNED_TOPK_CONFIDENCE})


def validate_downstream_mode(mode: str) -> str:
    if mode not in DOWNSTREAM_MODES:
        choices = ", ".join(DOWNSTREAM_MODES)
        raise ValueError(f"downstream mode must be one of: {choices}")
    return mode

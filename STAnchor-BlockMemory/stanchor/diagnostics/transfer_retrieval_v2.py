"""Compatibility wrapper for Stage 1 retrieval diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .transfer_retrieval import diagnose_transfer_retrieval as _diagnose


def diagnose_transfer_retrieval(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _diagnose(*args, **kwargs)
    offsets = result.get("candidate_pool", {}).get("weekday_offset_counts")
    if isinstance(offsets, dict) and "other" in offsets:
        # The protocol emits only -1/0/+1. The original diagnostic encoded
        # +1 as the string "1" and accidentally counted it as "other".
        offsets["+1"] = int(offsets.get("+1", 0)) + int(offsets["other"])
        offsets["other"] = 0
    result["diagnostic_version"] = 2
    return result


"""DHA-DCD-ST model entry for the first non-mean-anchor implementation.

DHA v1 changes the historical-anchor construction while preserving the DCD-ST
backbone tensor contract:

    x:     (B, T, N, 1) current traffic
    x_cov: (B, T, N, 1) time-in-day
    x_his: (B, T, N, 1) distributional historical anchor

Keeping the downstream model identical makes the first experiment a clean
anchor-quality ablation: mean/median/q25/q75/recent differ only in the anchor
channel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_base_dcdst_class():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "DCD-ST" / "dcd_st.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Base DCD-ST model file not found: {model_path}")
    module_name = "_dha_dcd_st_base_dcdst"
    if module_name in sys.modules:
        return sys.modules[module_name].DCDST
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base DCD-ST model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.DCDST


_BaseDCDST = _load_base_dcdst_class()


class DHADCDST(_BaseDCDST):
    """Distributional-anchor variant of DCD-ST.

    The architecture is intentionally inherited unchanged in v1. The innovation
    under test is whether a richer train-only historical anchor gives the same
    deviation-calibrated predictor a better reference signal.
    """

    pass

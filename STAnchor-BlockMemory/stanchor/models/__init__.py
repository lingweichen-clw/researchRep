"""Neural modules for STAnchor pretraining and downstream fusion."""

from .dynamics_adapter import (
    DynamicsAdapterOutput,
    HistoryDynamicsAdapter,
    summarize_adapter_output,
)
from .encoder import FactorizedSTEncoder
from .downstream import (
    ConfidenceHead,
    DownstreamOutput,
    LightweightForecastBackbone,
    SafeResidualFusion,
    STAnchorDownstreamModel,
    build_confidence_features,
    confidence_soft_target,
)
from .patch_embedding import TemporalPatchEmbedding
from .pretraining import PretrainForwardOutput, STAnchorPretrainModel
from .retrieval_head import RetrievalHead, RetrievalOutput
from .stgcn import STGCNForecastBackbone, build_stgcn_gso

__all__ = [
    "FactorizedSTEncoder",
    "DynamicsAdapterOutput",
    "HistoryDynamicsAdapter",
    "summarize_adapter_output",
    "ConfidenceHead",
    "DownstreamOutput",
    "LightweightForecastBackbone",
    "PretrainForwardOutput",
    "RetrievalHead",
    "RetrievalOutput",
    "STAnchorPretrainModel",
    "STAnchorDownstreamModel",
    "SafeResidualFusion",
    "TemporalPatchEmbedding",
    "build_confidence_features",
    "confidence_soft_target",
    "STGCNForecastBackbone",
    "build_stgcn_gso",
]

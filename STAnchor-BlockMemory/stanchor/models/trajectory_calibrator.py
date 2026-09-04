"""Compatibility import for the finalized retrieval-aware calibrator.

Older experiments placed several candidate-router implementations in this
module.  The supported runtime now has one calibrator, implemented in
``stanchor.models.retrieval_router``.  Keeping this import path avoids breaking
small external utilities while ensuring that no obsolete architecture can be
constructed accidentally.
"""

from stanchor.models.retrieval_router import RetrievalAwareMHAResidualRouter

__all__ = ["RetrievalAwareMHAResidualRouter"]

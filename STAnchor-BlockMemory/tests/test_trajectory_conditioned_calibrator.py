import sys

sys.path.insert(0, "d:/projects/researchProjects/TrafficRobustST/STAnchor-BlockMemory")

import torch

from stanchor.models.trajectory_calibrator import (
    TrajectoryConditionedCandidateSetHorizonCorrector,
)
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


def _inputs(batch=2, horizon=12, nodes=4, candidates=5, channels=1, context=12):
    history = torch.randn(batch, context, nodes, channels)
    base = torch.randn(batch, horizon, nodes, channels)
    candidate_futures = torch.randn(batch, horizon, nodes, candidates, channels)
    candidate_masks = torch.ones_like(candidate_futures, dtype=torch.bool)
    node_candidates = NodeCandidates(
        event_ids=torch.zeros(batch, nodes, candidates, dtype=torch.long),
        weights=torch.full((batch, nodes, candidates), 1.0 / candidates),
        shape_scores=torch.rand(batch, nodes, candidates),
        level_distances=torch.rand(batch, nodes, candidates),
        total_scores=torch.rand(batch, nodes, candidates),
        valid=torch.ones(batch, nodes, candidates, dtype=torch.bool),
    )
    aggregation = AggregationOutput(
        prediction=torch.zeros_like(base),
        variance=torch.zeros_like(base),
        valid=torch.ones(batch, horizon, nodes, channels, dtype=torch.bool),
        candidate_futures=candidate_futures,
        candidate_masks=candidate_masks,
    )
    return history, base, node_candidates, aggregation


def test_trajectory_conditioned_structure_and_gradients():
    model = TrajectoryConditionedCandidateSetHorizonCorrector(
        context_length=12,
        horizon=12,
        channels=1,
        hidden_dim=384,
        state_dim=288,
        attention_heads=4,
        trajectory_hidden_dim=96,
        use_horizon_embedding=True,
    )
    assert hasattr(model, "trajectory_encoder")
    assert hasattr(model, "horizon_embedding")
    assert model.horizon_embedding.shape == (12, 384)
    assert not hasattr(model, "value_proj")
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 690_000 <= count <= 705_000, count

    history, base, candidates, aggregation = _inputs()
    final, _, _, _ = model(
        history,
        base,
        None,
        None,
        None,
        candidates=candidates,
        aggregation=aggregation,
    )
    assert model.current_attention.shape == (2, 12, 4, 6)
    loss = final.square().mean()
    loss.backward()
    for name in (
        "trajectory_encoder.0.weight",
        "trajectory_encoder.2.weight",
        "horizon_embedding",
    ):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_trajectory_conditioned_fallback_stays_base():
    model = TrajectoryConditionedCandidateSetHorizonCorrector(
        context_length=12,
        horizon=12,
        channels=1,
        hidden_dim=384,
        state_dim=288,
        attention_heads=4,
        trajectory_hidden_dim=96,
        use_horizon_embedding=True,
    )
    history, base, candidates, aggregation = _inputs()
    aggregation = AggregationOutput(
        prediction=base.clone(),
        variance=torch.zeros_like(base),
        valid=torch.ones_like(base, dtype=torch.bool),
        candidate_futures=aggregation.candidate_futures,
        candidate_masks=torch.zeros_like(aggregation.candidate_masks),
    )
    final, history_mass, _, _ = model(
        history,
        base,
        None,
        None,
        None,
        candidates=candidates,
        aggregation=aggregation,
    )
    assert torch.allclose(final, base)
    assert torch.all(history_mass < 1e-6)
    assert torch.allclose(model.current_attention[..., -1], torch.ones_like(history_mass[..., 0]))

import torch

from stanchor.models.trajectory_calibrator import RetrievalAwareMHAResidualRouter
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


def _inputs(batch=2, context=12, horizon=12, nodes=3, top_k=12, channels=1):
    torch.manual_seed(7)
    history = torch.randn(batch, context, nodes, channels)
    base = torch.randn(batch, horizon, nodes, channels)
    futures = torch.randn(batch, horizon, nodes, top_k, channels)
    masks = torch.ones_like(futures, dtype=torch.bool)
    candidates = NodeCandidates(
        event_ids=torch.arange(batch * nodes * top_k).reshape(batch, nodes, top_k),
        total_scores=torch.rand(batch, nodes, top_k),
        shape_scores=torch.rand(batch, nodes, top_k),
        level_distances=torch.rand(batch, nodes, top_k),
        weights=torch.full((batch, nodes, top_k), 1.0 / top_k),
        valid=torch.ones(batch, nodes, top_k, dtype=torch.bool),
    )
    aggregation = AggregationOutput(
        prediction=futures.mean(dim=3),
        variance=futures.var(dim=3, unbiased=False),
        valid=masks,
        candidate_futures=futures,
        candidate_masks=masks,
    )
    return history, base, candidates, aggregation


def _router():
    return RetrievalAwareMHAResidualRouter(
        context_length=12,
        horizon=12,
        channels=1,
        retrieval_dim=64,
        hidden_dim=256,
        retrieval_hidden_dim=128,
        fusion_hidden_dim=256,
        candidate_hidden_dim=128,
        routing_dim=128,
        attention_heads=4,
        mha_dropout=0.0,
    )


def test_retrieval_aware_router_has_horizon_routing_and_mha_gradients():
    model = _router()
    history, base, candidates, aggregation = _inputs()
    final, history_mass, _, _ = model(
        history,
        base,
        None,
        None,
        None,
        retrieval_node_keys=torch.randn(2, 3, 64),
        candidates=candidates,
        aggregation=aggregation,
    )
    assert final.shape == base.shape
    assert history_mass.shape == (2, 12, 3, 1)
    assert model.last_routing_weights.shape == (2, 3, 12, 13)
    assert model.last_mha_attention.shape == (2, 3, 4, 12, 13)
    assert torch.allclose(
        model.last_routing_weights.sum(dim=-1), torch.ones(2, 3, 12), atol=1e-5
    )
    final.square().mean().backward()
    for name in ("mha.in_proj_weight", "mha.out_proj.weight"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_retrieval_aware_router_base_fallback_is_exact():
    model = _router()
    history, base, candidates, aggregation = _inputs()
    invalid = AggregationOutput(
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
        retrieval_node_keys=torch.randn(2, 3, 64),
        candidates=candidates,
        aggregation=invalid,
    )
    assert torch.equal(final, base)
    assert torch.allclose(history_mass, torch.zeros_like(history_mass), atol=1e-7)
    assert torch.allclose(
        model.last_routing_weights[..., -1],
        torch.ones_like(model.last_routing_weights[..., -1]), atol=1e-7
    )

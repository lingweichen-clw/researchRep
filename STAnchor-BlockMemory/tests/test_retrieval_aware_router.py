from dataclasses import replace

import pytest
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
        node_keys=torch.nn.functional.normalize(torch.randn(batch, nodes, top_k, 64), dim=-1),
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
    for name, parameter in model.named_parameters():
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


def _candidate_key_router():
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
        use_candidate_key_context=True,
        candidate_key_bottleneck_dim=16,
    )


def test_candidate_key_router_consumes_selected_candidate_keys():
    model = _candidate_key_router()
    history, base, candidates, aggregation = _inputs()
    final, history_mass, _, _ = model(
        history,
        base,
        candidates=candidates,
        aggregation=aggregation,
        retrieval_node_keys=torch.randn(2, 3, 64),
    )
    assert final.shape == base.shape
    assert history_mass.shape == (2, 12, 3, 1)
    assert model.candidate_key_encoder is not None
    assert model.last_routing_weights.shape == (2, 3, 12, 13)


def test_candidate_key_router_starts_as_exact_retained_router():
    retained = _router()
    enhanced = _candidate_key_router()
    incompatible = enhanced.load_state_dict(retained.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        "candidate_key_encoder.0.weight",
        "candidate_key_encoder.0.bias",
        "candidate_key_encoder.2.weight",
        "candidate_key_encoder.2.bias",
    }
    history, base, candidates, aggregation = _inputs()
    query_keys = torch.randn(2, 3, 64)
    retained_output = retained(
        history,
        base,
        candidates=candidates,
        aggregation=aggregation,
        retrieval_node_keys=query_keys,
    )
    enhanced_output = enhanced(
        history,
        base,
        candidates=candidates,
        aggregation=aggregation,
        retrieval_node_keys=query_keys,
    )
    for expected, actual in zip(retained_output, enhanced_output):
        assert torch.equal(actual, expected)


def test_candidate_key_router_adds_only_small_bottleneck_branch():
    retained = _router()
    enhanced = _candidate_key_router()
    retained_parameters = sum(parameter.numel() for parameter in retained.parameters())
    enhanced_parameters = sum(parameter.numel() for parameter in enhanced.parameters())
    assert enhanced_parameters - retained_parameters == 5_392


def test_candidate_key_router_requires_selected_candidate_keys():
    model = _candidate_key_router()
    history, base, candidates, aggregation = _inputs()
    candidates = replace(candidates, node_keys=None)
    with pytest.raises(ValueError, match="candidate node_keys"):
        model(
            history,
            base,
            candidates=candidates,
            aggregation=aggregation,
            retrieval_node_keys=torch.randn(2, 3, 64),
        )


def test_candidate_key_router_rejects_wrong_candidate_key_shape():
    model = _candidate_key_router()
    history, base, candidates, aggregation = _inputs()
    candidates = replace(candidates, node_keys=torch.randn(2, 3, 12, 32))
    with pytest.raises(ValueError, match="candidate node_keys"):
        model(
            history,
            base,
            candidates=candidates,
            aggregation=aggregation,
            retrieval_node_keys=torch.randn(2, 3, 64),
        )


def test_retained_router_still_ignores_candidate_keys():
    model = RetrievalAwareMHAResidualRouter(
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
    history, base, candidates, aggregation = _inputs()
    candidates_without_keys = replace(candidates, node_keys=None)
    final, history_mass, _, _ = model(
        history,
        base,
        candidates=candidates_without_keys,
        aggregation=aggregation,
        retrieval_node_keys=torch.randn(2, 3, 64),
    )
    assert final.shape == base.shape
    assert history_mass.shape == (2, 12, 3, 1)

import torch

from stanchor.models.trajectory_calibrator import TransformerCandidateRouter
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
    return TransformerCandidateRouter(
        context_length=12,
        horizon=12,
        channels=1,
        hidden_dim=256,
        state_dim=256,
        attention_heads=4,
        trajectory_hidden_dim=64,
        routing_hidden_dim=128,
        mha_dropout=0.0,
    )


def test_router_outputs_k_plus_one_weights_and_uses_mha_output():
    model = _router()
    history, base, candidates, aggregation = _inputs()

    final, history_mass, _, _ = model(
        history, base, None, None, None, candidates=candidates, aggregation=aggregation
    )

    assert final.shape == base.shape
    assert history_mass.shape == (2, 12, 3, 1)
    assert model.last_routing_weights.shape == (2, 12, 3, 13)
    assert model.last_mha_attention.shape == (2, 12, 3, 4, 13)
    assert torch.allclose(
        model.last_routing_weights.sum(dim=-1),
        torch.ones(2, 12, 3),
        atol=1e-5,
    )

    loss = final.square().mean()
    loss.backward()
    for name in (
        "mha.in_proj_weight",
        "mha.out_proj.weight",
        "routing_head.0.weight",
        "trajectory_encoder.0.weight",
    ):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_router_base_fallback_is_exact_when_history_candidates_are_invalid():
    model = _router()
    history, base, candidates, aggregation = _inputs()
    aggregation = AggregationOutput(
        prediction=base.clone(),
        variance=torch.zeros_like(base),
        valid=torch.ones_like(base, dtype=torch.bool),
        candidate_futures=aggregation.candidate_futures,
        candidate_masks=torch.zeros_like(aggregation.candidate_masks),
    )

    final, history_mass, _, _ = model(
        history, base, None, None, None, candidates=candidates, aggregation=aggregation
    )

    assert torch.equal(final, base)
    assert torch.allclose(history_mass, torch.zeros_like(history_mass), atol=1e-7)
    assert torch.allclose(
        model.last_routing_weights[..., -1],
        torch.ones_like(model.last_routing_weights[..., -1]),
        atol=1e-7,
    )


def test_router_is_permutation_equivariant_over_history_candidates():
    model = _router().eval()
    history, base, candidates, aggregation = _inputs()
    permutation = torch.tensor([3, 0, 7, 2, 11, 1, 9, 5, 4, 8, 6, 10])
    perm_candidates = NodeCandidates(
        **{
            name: value[..., permutation]
            for name, value in candidates.__dict__.items()
        }
    )
    perm_aggregation = AggregationOutput(
        prediction=aggregation.prediction,
        variance=aggregation.variance,
        valid=aggregation.valid[..., permutation, :],
        candidate_futures=aggregation.candidate_futures[..., permutation, :],
        candidate_masks=aggregation.candidate_masks[..., permutation, :],
    )

    first, _, _, _ = model(
        history, base, None, None, None, candidates=candidates, aggregation=aggregation
    )
    first_weights = model.last_routing_weights.detach().clone()
    second, _, _, _ = model(
        history,
        base,
        None,
        None,
        None,
        candidates=perm_candidates,
        aggregation=perm_aggregation,
    )
    second_weights = model.last_routing_weights.detach()

    assert torch.allclose(first, second, atol=1e-5)
    assert torch.allclose(first_weights[..., :12], second_weights[..., :12][..., permutation.argsort()], atol=1e-5)
    assert torch.allclose(first_weights[..., -1], second_weights[..., -1], atol=1e-5)

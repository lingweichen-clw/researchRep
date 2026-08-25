import torch

from stanchor.retrieval.retriever import horizon_aware_candidate_weights


def test_horizon_aware_weights_shape_and_normalization():
    base = torch.tensor([[[0.8, 0.2]]])
    future = torch.tensor([[[[[1.0], [1.2]]], [[[2.0], [5.0]]]]])
    mask = torch.ones_like(future, dtype=torch.bool)
    weights = horizon_aware_candidate_weights(base, future, mask)
    assert weights.shape == (1, 2, 1, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 2, 1))
    assert weights[0, 0, 0, 0] > weights[0, 0, 0, 1]


def test_horizon_aware_respects_invalid_candidates():
    base = torch.tensor([[[0.5, 0.5]]])
    future = torch.tensor([[[[[1.0], [9.0]]]]])
    mask = torch.tensor([[[[[True], [False]]]]])
    weights = horizon_aware_candidate_weights(base, future, mask)
    assert weights[0, 0, 0, 1] == 0
    assert weights[0, 0, 0, 0] == 1


def test_zero_horizon_scores_preserve_node_weights():
    base = torch.tensor([[[0.7, 0.3]]])
    future = torch.tensor([[[[[1.0], [8.0]]], [[[2.0], [7.0]]]]])
    mask = torch.ones_like(future, dtype=torch.bool)
    scores = torch.zeros(1, 2, 1, 2)
    weights = horizon_aware_candidate_weights(base, future, mask, horizon_scores=scores)
    expected = base[:, None].expand_as(weights)
    assert torch.allclose(weights, expected, atol=1e-6)

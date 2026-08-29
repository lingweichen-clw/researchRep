import torch
from stanchor.losses.downstream import candidate_quality_kl_loss, compute_downstream_loss
from stanchor.models.downstream import DownstreamOutput

def test_candidate_quality_loss_is_low_when_attention_matches_teacher():
    errors=torch.tensor([[[[0.0,1.0]]]])
    attention=torch.softmax(torch.tensor([[[[0.0,-10.0]]]]),dim=-1)
    valid=torch.ones_like(errors,dtype=torch.bool)
    loss=candidate_quality_kl_loss(attention,errors,valid,temperature=0.1)
    assert float(loss)<1e-3


def test_candidate_quality_accepts_base_as_k_plus_one_token():
    """The Base token is appended for teacher supervision, while payload has K futures."""
    base = torch.zeros(1, 2, 1, 1)
    candidates = torch.tensor([1.0, 0.5]).view(1, 1, 1, 2, 1).expand(1, 2, 1, 2, 1).clone()
    target = torch.ones_like(base)
    output = DownstreamOutput(
        base_prediction=base,
        memory_prediction=base,
        confidence_features=torch.zeros(1, 2, 1, 1),
        confidence=torch.zeros(1, 2, 1, 1),
        fusion_weight=torch.zeros(1, 2, 1, 1),
        final_prediction=base,
        memory_valid=torch.ones(1, 2, 1, 1, dtype=torch.bool),
        candidate_attention=torch.softmax(torch.zeros(1, 2, 1, 3), dim=-1),
        candidate_futures=candidates,
        candidate_masks=torch.ones_like(candidates, dtype=torch.bool),
    )
    losses = compute_downstream_loss(
        output,
        target=target,
        observed=torch.ones_like(target, dtype=torch.bool),
        confidence_weight=0.0,
        help_margin=0.0,
        help_temperature=0.1,
        use_confidence=False,
        use_error_aware=True,
        loss_variant="forecast_only",
        candidate_quality_weight=0.05,
        candidate_quality_temperature=0.2,
    )
    assert losses.candidate_quality is not None
    assert torch.isfinite(losses.candidate_quality)
    assert torch.isfinite(losses.total)

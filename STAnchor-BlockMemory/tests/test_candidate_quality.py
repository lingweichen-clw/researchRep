import torch
from stanchor.losses.downstream import candidate_quality_kl_loss

def test_candidate_quality_loss_is_low_when_attention_matches_teacher():
    errors=torch.tensor([[[[0.0,1.0]]]])
    attention=torch.softmax(torch.tensor([[[[0.0,-10.0]]]]),dim=-1)
    valid=torch.ones_like(errors,dtype=torch.bool)
    loss=candidate_quality_kl_loss(attention,errors,valid,temperature=0.1)
    assert float(loss)<1e-3
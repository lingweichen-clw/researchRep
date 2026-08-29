from stanchor.models.downstream import CandidateSetHorizonCorrector


def test_base_as_candidate_has_no_dead_value_projection():
    model = CandidateSetHorizonCorrector(
        context_length=12,
        horizon=12,
        channels=1,
        hidden_dim=384,
        state_dim=320,
        attention_heads=4,
        base_logit_init_bias=1.0,
    )

    assert not hasattr(model, "value_proj")
    assert not any(name.startswith("value_proj.") for name, _ in model.named_parameters())

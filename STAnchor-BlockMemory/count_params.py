import torch
from stanchor.models.trajectory_calibrator import TransformerCandidateRouter

# Count parameters
model = TransformerCandidateRouter(
    context_length=12,
    horizon=12,
    channels=1,
    hidden_dim=256,
    state_dim=256,
    attention_heads=4,
    trajectory_hidden_dim=64,
    routing_hidden_dim=128
)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'TransformerCandidateRouter total params: {total:,}')
print(f'Trainable: {trainable:,}')

# Breakdown
for name, param in model.named_parameters():
    print(f'{name}: {param.numel():,}')

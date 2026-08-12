"""Dataset-agnostic future dynamics profiles and retrieval-key composition.

The profile functions are used by the source-train teacher only.  Query
inference receives history-derived keys and never constructs a future profile.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _validate_future(values: torch.Tensor, observed: torch.Tensor) -> None:
    if values.ndim != 4 or observed.shape != values.shape:
        raise ValueError("values and observed must be [B, H, N, C]")
    if values.shape[1] <= 0:
        raise ValueError("future horizon must be positive")


def resample_future_profile(
    values: torch.Tensor,
    observed: torch.Tensor,
    profile_size: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample a future to a fixed relative-time grid without crossing gaps."""
    _validate_future(values, observed)
    if profile_size <= 0:
        raise ValueError("profile_size must be positive")
    finite = observed.bool() & torch.isfinite(values)
    masked_values = torch.where(finite, values, torch.zeros_like(values))
    horizon = values.shape[1]
    if profile_size == horizon:
        return masked_values, finite

    positions = torch.linspace(
        0.0,
        float(horizon - 1),
        profile_size,
        dtype=values.dtype,
        device=values.device,
    )
    left = positions.floor().long()
    right = positions.ceil().long()
    alpha = (positions - left.to(values.dtype)).view(1, profile_size, 1, 1)
    left_values = masked_values.index_select(1, left)
    right_values = masked_values.index_select(1, right)
    left_valid = finite.index_select(1, left)
    right_valid = finite.index_select(1, right)
    valid = left_valid & right_valid
    interpolated = (1.0 - alpha) * left_values + alpha * right_values
    return torch.where(valid, interpolated, torch.zeros_like(interpolated)), valid


def _context_statistics(
    context: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if context.ndim != 4 or observed.shape != context.shape:
        raise ValueError("context and observed must be [B, T, N, C]")
    finite = observed.bool() & torch.isfinite(context)
    count = finite.sum(dim=1)
    safe_count = count.clamp_min(1).to(context.dtype)
    clean = torch.where(finite, context, torch.zeros_like(context))
    mean = clean.sum(dim=1) / safe_count
    centered = torch.where(finite, context - mean.unsqueeze(1), torch.zeros_like(context))
    variance = centered.square().sum(dim=1) / safe_count
    std = variance.sqrt()
    endpoint_visible = finite[:, -1]
    endpoint = torch.where(endpoint_visible, context[:, -1], mean)
    return mean, std, endpoint, count > 0


def build_cfdp_teacher(
    future: torch.Tensor,
    future_observed: torch.Tensor,
    context: torch.Tensor,
    context_observed: torch.Tensor,
    profile_size: int = 12,
    scale_floor: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed-length, event-self-normalized future dynamics profile.

    The output is dimensionless ``[B, K, N, C]`` and is constructed only from
    source-train future/context inside a no-grad teacher branch.
    """
    _validate_future(future, future_observed)
    if context.ndim != 4 or context_observed.shape != context.shape:
        raise ValueError("context and observed must be [B, T, N, C]")
    if future.shape[0] != context.shape[0] or future.shape[2:] != context.shape[2:]:
        raise ValueError("future and context batch/node/channel dimensions must align")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")

    with torch.no_grad():
        sampled, sampled_valid = resample_future_profile(
            future, future_observed, profile_size
        )
        mean, std, endpoint, context_valid = _context_statistics(
            context, context_observed
        )
        scale = std.clamp_min(scale_floor)
        positions = torch.linspace(
            0.0,
            1.0,
            profile_size,
            dtype=future.dtype,
            device=future.device,
        )
        decay = (1.0 - positions).view(1, profile_size, 1, 1)
        profile = (sampled - decay * endpoint.unsqueeze(1) - (1.0 - decay) * mean.unsqueeze(1)) / scale.unsqueeze(1)
        valid = sampled_valid & context_valid.unsqueeze(1)
        return torch.where(valid, profile, torch.zeros_like(profile)), valid


def symmetric_geometric_mean_normalize(
    distances: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Normalize pair distances by the two anchor mean scales.

    When ``valid`` is a symmetric pair mask, the returned distance is exactly
    symmetric.  Invalid entries are returned as zero for finite downstream
    masking.
    """
    if distances.ndim != 3 or valid.shape != distances.shape:
        raise ValueError("distances and valid must be [B, B, N]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    finite_valid = valid.bool() & torch.isfinite(distances)
    count = finite_valid.sum(dim=1)
    total = torch.where(finite_valid, distances, torch.zeros_like(distances)).sum(dim=1)
    mean = total / count.clamp_min(1).to(distances.dtype)
    pair_scale = torch.sqrt(
        (mean.unsqueeze(1) + eps) * (mean.unsqueeze(0) + eps)
    )
    normalized = distances / pair_scale.clamp_min(eps)
    return torch.where(finite_valid, normalized, torch.zeros_like(normalized))


def compose_profile_latent_key(
    profile: torch.Tensor,
    latent: torch.Tensor,
    gamma: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose normalized profile/latent node keys with an interpretable weight."""
    if profile.ndim != 3 or latent.ndim != 3 or profile.shape[:2] != latent.shape[:2]:
        raise ValueError("profile and latent must be [B, N, D] with aligned batch/nodes")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    profile_key = functional.normalize(profile, p=2, dim=-1, eps=1.0e-8)
    latent_key = functional.normalize(latent, p=2, dim=-1, eps=1.0e-8)
    total = torch.cat(
        (gamma**0.5 * profile_key, (1.0 - gamma) ** 0.5 * latent_key), dim=-1
    )
    return functional.normalize(total, p=2, dim=-1, eps=1.0e-8), profile_key, latent_key

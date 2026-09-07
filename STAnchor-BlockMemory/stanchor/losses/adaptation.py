"""Losses for parameter-efficient target-domain retrieval adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class RetrievalAdaptationLoss:
    total: torch.Tensor
    relation: torch.Tensor
    distillation: torch.Tensor


def cosine_key_distillation_loss(
    student_keys: torch.Tensor,
    teacher_keys: torch.Tensor,
) -> torch.Tensor:
    """Keep adapted node keys near frozen source keys without teacher gradients."""
    if student_keys.ndim != 3 or teacher_keys.shape != student_keys.shape:
        raise ValueError("student_keys and teacher_keys must align as [B, N, D]")
    student = functional.normalize(student_keys, dim=-1, eps=1.0e-8)
    teacher = functional.normalize(teacher_keys.detach(), dim=-1, eps=1.0e-8)
    return (1.0 - (student * teacher).sum(dim=-1)).mean()


def compute_t1_adaptation_loss(
    relation_loss: torch.Tensor,
    student_keys: torch.Tensor,
    teacher_keys: torch.Tensor,
    relation_weight: float,
    distill_weight: float,
) -> RetrievalAdaptationLoss:
    if relation_loss.ndim != 0:
        raise ValueError("relation_loss must be a scalar")
    if relation_weight <= 0.0:
        raise ValueError("relation_weight must be positive")
    if distill_weight < 0.0:
        raise ValueError("distill_weight must be non-negative")
    distillation = cosine_key_distillation_loss(student_keys, teacher_keys)
    total = relation_weight * relation_loss + distill_weight * distillation
    return RetrievalAdaptationLoss(
        total=total,
        relation=relation_loss,
        distillation=distillation,
    )

"""Decoupled temporal-patch and whole-node mask sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MaskBatch:
    task: str
    patch_mask: torch.Tensor  # [B, P, N]
    value_mask: torch.Tensor  # [B, T, N, C]


class StructuredMaskSampler:
    def __init__(
        self,
        context_length: int,
        patch_size: int,
        time_ratio: float = 0.25,
        space_ratio: float = 0.25,
        time_probability: float = 0.5,
        time_block_size: int | None = None,
    ) -> None:
        if context_length % patch_size != 0:
            raise ValueError("context_length must be divisible by patch_size")
        for name, value in (
            ("time_ratio", time_ratio),
            ("space_ratio", space_ratio),
            ("time_probability", time_probability),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_patches = context_length // patch_size
        self.time_block_size = patch_size if time_block_size is None else int(time_block_size)
        if not 0 < self.time_block_size <= context_length:
            raise ValueError("time_block_size must be in [1, context_length]")
        if self.time_block_size % patch_size != 0:
            raise ValueError("time_block_size must be divisible by patch_size")
        self.time_block_patches = self.time_block_size // patch_size
        self.time_ratio = time_ratio
        self.space_ratio = space_ratio
        self.time_probability = time_probability

    def sample(
        self,
        batch_size: int,
        num_nodes: int,
        num_channels: int,
        neighbors: torch.Tensor,
        device: torch.device | str,
        observed: torch.Tensor | None = None,
        task: str | None = None,
        generator: torch.Generator | None = None,
    ) -> MaskBatch:
        if neighbors.shape != (num_nodes, num_nodes):
            raise ValueError("neighbors must be [N, N]")
        if task is None:
            draw = torch.rand((), generator=generator).item()
            task = "time" if draw < self.time_probability else "space"
        if task not in {"time", "space"}:
            raise ValueError("task must be time or space")
        if observed is None:
            observed = torch.ones(
                (batch_size, self.context_length, num_nodes, num_channels),
                dtype=torch.bool,
            )
        if observed.shape != (batch_size, self.context_length, num_nodes, num_channels):
            raise ValueError("observed must be [B, T, N, C]")
        observed_cpu = observed.bool().cpu()
        if task == "time":
            patch_available = observed_cpu.reshape(
                batch_size,
                self.num_patches,
                self.patch_size,
                num_nodes,
                num_channels,
            ).any(dim=(2, 3, 4))
            patch_mask = self._sample_time(
                batch_size, num_nodes, patch_available, device, generator
            )
        else:
            node_available = observed_cpu.any(dim=(1, 3))
            patch_mask = self._sample_space(
                batch_size,
                num_nodes,
                neighbors.bool().cpu(),
                node_available,
                generator,
            ).to(device)
        value_mask = patch_mask.repeat_interleave(self.patch_size, dim=1).unsqueeze(-1)
        value_mask = value_mask.expand(-1, -1, -1, num_channels)
        return MaskBatch(task=task, patch_mask=patch_mask, value_mask=value_mask)

    def _sample_time(
        self,
        batch_size: int,
        num_nodes: int,
        patch_available: torch.Tensor,
        device: torch.device | str,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        target_patch_count = max(1, int(self.time_ratio * self.num_patches))
        target_blocks = max(1, target_patch_count // self.time_block_patches)
        target_blocks = min(target_blocks, self.num_patches // self.time_block_patches)
        mask = torch.zeros((batch_size, self.num_patches), dtype=torch.bool)
        for batch_index in range(batch_size):
            block_available = patch_available[batch_index].unfold(
                0,
                self.time_block_patches,
                1,
            ).any(dim=-1)
            eligible_starts = torch.where(block_available)[0]
            if eligible_starts.numel() == 0:
                continue
            best = torch.zeros(self.num_patches, dtype=torch.bool)
            for _attempt in range(32):
                current = torch.zeros(self.num_patches, dtype=torch.bool)
                selected_blocks = 0
                order = eligible_starts[
                    torch.randperm(eligible_starts.numel(), generator=generator)
                ]
                for start_tensor in order:
                    start = int(start_tensor)
                    end = start + self.time_block_patches
                    if bool(current[start:end].any()):
                        continue
                    current[start:end] = True
                    selected_blocks += 1
                    if selected_blocks == target_blocks:
                        break
                if int(current.sum()) > int(best.sum()):
                    best = current
                if selected_blocks == target_blocks:
                    break
            mask[batch_index] = best
        return mask.to(device).unsqueeze(-1).expand(-1, -1, num_nodes)

    def _sample_space(
        self,
        batch_size: int,
        num_nodes: int,
        neighbors: torch.Tensor,
        node_available: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        count = max(1, int(self.space_ratio * num_nodes))
        topology_eligible = neighbors.any(dim=1)
        all_masks: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            eligible = torch.where(topology_eligible & node_available[batch_index])[0]
            actual_count = min(count, int(eligible.numel()))
            if actual_count == 0:
                all_masks.append(torch.zeros(num_nodes, dtype=torch.bool))
                continue
            best = torch.zeros(num_nodes, dtype=torch.bool)
            for _attempt in range(32):
                current = torch.zeros(num_nodes, dtype=torch.bool)
                order = eligible[torch.randperm(eligible.numel(), generator=generator)]
                for candidate in order.tolist():
                    proposal = current.clone()
                    proposal[candidate] = True
                    visible = ~proposal
                    has_visible_neighbor = (neighbors & visible.unsqueeze(0)).any(dim=1)
                    if bool(has_visible_neighbor[proposal].all()):
                        current = proposal
                    if int(current.sum()) == actual_count:
                        break
                if int(current.sum()) > int(best.sum()):
                    best = current
                if int(current.sum()) == actual_count:
                    break
            if int(best.sum()) != actual_count:
                raise ValueError("cannot satisfy exact spatial mask count and visible-neighbor constraint")
            all_masks.append(best)
        node_mask = torch.stack(all_masks, dim=0)  # [B, N]
        return node_mask[:, None, :].expand(-1, self.num_patches, -1)

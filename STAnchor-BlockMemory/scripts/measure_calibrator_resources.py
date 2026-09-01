"""Measure standalone router parameter count, latency, and CUDA peak memory."""

from __future__ import annotations

import argparse
import time

import torch

from stanchor.models.retrieval_router import RetrievalAwareMHAResidualRouter
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


def build_inputs(batch: int, nodes: int, top_k: int, horizon: int, channels: int, device):
    history = torch.randn(batch, 12, nodes, channels, device=device)
    base = torch.randn(batch, horizon, nodes, channels, device=device)
    future = torch.randn(batch, horizon, nodes, top_k, channels, device=device)
    mask = torch.ones_like(future, dtype=torch.bool)
    candidates = NodeCandidates(
        event_ids=torch.zeros(batch, nodes, top_k, dtype=torch.long, device=device),
        total_scores=torch.rand(batch, nodes, top_k, device=device),
        shape_scores=torch.rand(batch, nodes, top_k, device=device),
        level_distances=torch.rand(batch, nodes, top_k, device=device),
        weights=torch.full((batch, nodes, top_k), 1.0 / top_k, device=device),
        valid=torch.ones(batch, nodes, top_k, dtype=torch.bool, device=device),
    )
    aggregation = AggregationOutput(
        prediction=future.mean(dim=3),
        variance=future.var(dim=3, unbiased=False),
        valid=torch.ones(batch, horizon, nodes, channels, dtype=torch.bool, device=device),
        candidate_futures=future,
        candidate_masks=mask,
    )
    keys = torch.randn(batch, nodes, 64, device=device)
    return history, base, candidates, aggregation, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = RetrievalAwareMHAResidualRouter(12, 12, 1).to(device)
    params = sum(p.numel() for p in model.parameters())
    inputs = build_inputs(args.batch_size, args.nodes, args.top_k, 12, 1, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = model(inputs[0], inputs[1], None, None, None,
                   candidates=inputs[2], aggregation=inputs[3],
                   retrieval_node_keys=inputs[4])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    output[0].square().mean().backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - start
    print(f"parameters={params}")
    print(f"forward_seconds={forward_seconds:.6f}")
    print(f"backward_seconds={backward_seconds:.6f}")
    print(f"output_shape={tuple(output[0].shape)}")
    print(f"routing_shape={tuple(model.last_routing_weights.shape)}")
    print(f"mha_shape={tuple(model.last_mha_attention.shape)}")
    print(f"finite={all(torch.isfinite(x).all().item() for x in output if torch.is_tensor(x))}")
    if device.type == "cuda":
        print(f"cuda_peak_mb={torch.cuda.max_memory_allocated(device) / 1024**2:.2f}")


if __name__ == "__main__":
    main()

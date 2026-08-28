#!/usr/bin/env python3
"""CandidateSetHorizonCorrector 完整 Smoke 测试"""

import sys
sys.path.insert(0, 'd:/projects/researchProjects/TrafficRobustST/STAnchor-BlockMemory')

import torch
import numpy as np
from stanchor.models.downstream import CandidateSetHorizonCorrector
from stanchor.retrieval.retriever import NodeCandidates, AggregationOutput

def test_forward_shape():
    """测试1: Forward 输出 shape 正确"""
    print("\n" + "="*80)
    print("Test 1: Forward output shape correct")
    print("="*80)

    B, H, N, K, C = 4, 12, 207, 5, 1
    T = 12

    model = CandidateSetHorizonCorrector(
        context_length=T,
        horizon=H,
        channels=C,
        hidden_dim=256,
        state_dim=256,
        attention_heads=4,
        base_logit_init_bias=1.0
    )

    # 构造输入
    history = torch.randn(B, T, N, C)
    base = torch.randn(B, H, N, C)
    candidates_future = torch.randn(B, H, N, K, C)
    candidates_mask = torch.ones(B, H, N, K, C, dtype=torch.bool)

    # 构造 NodeCandidates
    candidates = NodeCandidates(
        event_ids=torch.randint(0, 1000, (B, N, K)),
        weights=torch.softmax(torch.randn(B, N, K), dim=-1),
        shape_scores=torch.randn(B, N, K),
        level_distances=torch.rand(B, N, K),
        total_scores=torch.randn(B, N, K),
        valid=torch.ones(B, N, K, dtype=torch.bool)
    )

    # 构造 AggregationOutput
    aggregation = AggregationOutput(
        prediction=torch.randn(B, H, N, C),
        variance=torch.rand(B, H, N, C),
        valid=torch.ones(B, H, N, C, dtype=torch.bool),
        candidate_futures=candidates_future,
        candidate_masks=candidates_mask
    )

    # Forward
    final, historical_mass, contributions, learned_memory = model(
        history=history,
        base=base,
        memory=None,
        features=None,
        memory_valid=None,
        candidates=candidates,
        aggregation=aggregation
    )

    # 验证
    assert final.shape == (B, H, N, C), f"Expected {(B, H, N, C)}, got {final.shape}"
    assert historical_mass.shape == (B, H, N, 1), f"Expected {(B, H, N, 1)}, got {historical_mass.shape}"
    assert torch.isfinite(final).all(), "Final output contains NaN/Inf"
    assert torch.isfinite(historical_mass).all(), "Historical mass contains NaN/Inf"

    print(f"[OK] Final shape: {final.shape}")
    print(f"[OK] Historical mass shape: {historical_mass.shape}")
    print(f"[OK] All outputs finite")
    print(f"[OK] Historical mass range: [{historical_mass.min():.4f}, {historical_mass.max():.4f}]")

    return True

def test_attention_shape():
    """测试2: Attention shape 为 [B,H,N,K+1]"""
    print("\n" + "="*80)
    print("Test 2: Attention shape is [B,H,N,K+1]")
    print("="*80)

    B, H, N, K, C = 4, 12, 207, 5, 1
    T = 12

    model = CandidateSetHorizonCorrector(
        context_length=T,
        horizon=H,
        channels=C,
        hidden_dim=256,
        state_dim=256,
        attention_heads=4,
        base_logit_init_bias=1.0
    )

    # 构造输入
    history = torch.randn(B, T, N, C)
    base = torch.randn(B, H, N, C)
    candidates_future = torch.randn(B, H, N, K, C)
    candidates_mask = torch.ones(B, H, N, K, C, dtype=torch.bool)

    candidates = NodeCandidates(
        event_ids=torch.randint(0, 1000, (B, N, K)),
        weights=torch.softmax(torch.randn(B, N, K), dim=-1),
        shape_scores=torch.randn(B, N, K),
        level_distances=torch.rand(B, N, K),
        total_scores=torch.randn(B, N, K),
        valid=torch.ones(B, N, K, dtype=torch.bool)
    )

    aggregation = AggregationOutput(
        prediction=torch.randn(B, H, N, C),
        variance=torch.rand(B, H, N, C),
        valid=torch.ones(B, H, N, C, dtype=torch.bool),
        candidate_futures=candidates_future,
        candidate_masks=candidates_mask
    )

    # Forward
    _ = model(
        history=history,
        base=base,
        memory=None,
        features=None,
        memory_valid=None,
        candidates=candidates,
        aggregation=aggregation
    )

    # 检查 attention
    attention = model.current_attention
    assert attention is not None, "Attention not recorded"
    assert attention.shape == (B, H, N, K+1), f"Expected {(B, H, N, K+1)}, got {attention.shape}"

    # 检查 attention 和为 1
    attn_sum = attention.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention sum != 1"

    # 检查 Base attention 在合理范围内
    base_attn = attention[..., -1]  # 最后一个是 Base
    print(f"[OK] Attention shape: {attention.shape}")
    print(f"[OK] Attention sum to 1: {attn_sum.mean():.6f}")
    print(f"[OK] Base attention mean: {base_attn.mean():.4f}")
    print(f"[OK] Base attention std: {base_attn.std():.4f}")
    print(f"[OK] Base attention range: [{base_attn.min():.4f}, {base_attn.max():.4f}]")

    return True

def test_no_candidate_fallback():
    """测试3: 无有效候选时回退到 Base"""
    print("\n" + "="*80)
    print("Test 3: Fallback to Base when no valid candidates")
    print("="*80)

    B, H, N, K, C = 4, 12, 207, 5, 1
    T = 12

    model = CandidateSetHorizonCorrector(
        context_length=T,
        horizon=H,
        channels=C,
        hidden_dim=256,
        state_dim=256,
        attention_heads=4,
        base_logit_init_bias=1.0
    )

    # 构造输入
    history = torch.randn(B, T, N, C)
    base = torch.randn(B, H, N, C)
    candidates_future = torch.randn(B, H, N, K, C)

    # 所有历史候选无效
    candidates_mask = torch.zeros(B, H, N, K, C, dtype=torch.bool)

    candidates = NodeCandidates(
        event_ids=torch.randint(0, 1000, (B, N, K)),
        weights=torch.zeros(B, N, K),  # 全0权重
        shape_scores=torch.zeros(B, N, K),
        level_distances=torch.zeros(B, N, K),
        total_scores=torch.zeros(B, N, K),
        valid=torch.zeros(B, N, K, dtype=torch.bool)  # 全部无效
    )

    aggregation = AggregationOutput(
        prediction=base.clone(),  # 无候选时应该等于 base
        variance=torch.zeros(B, H, N, C),
        valid=torch.ones(B, H, N, C, dtype=torch.bool),
        candidate_futures=candidates_future,
        candidate_masks=candidates_mask
    )

    # Forward
    final, historical_mass, _, _ = model(
        history=history,
        base=base,
        memory=None,
        features=None,
        memory_valid=None,
        candidates=candidates,
        aggregation=aggregation
    )

    # 验证：输出应该等于 Base
    diff = (final - base).abs().max()
    attention = model.current_attention
    base_attn = attention[..., -1]

    print(f"[OK] Max diff between final and base: {diff:.8f}")
    print(f"[OK] Base attention mean: {base_attn.mean():.6f}")
    print(f"[OK] Historical mass mean: {historical_mass.mean():.6f}")

    # 允许小的数值误差（由于 softmax 计算）
    assert diff < 1e-3, f"Final != Base (diff={diff:.6f})"
    assert base_attn.mean() > 0.99, f"Base attention should be ~1.0, got {base_attn.mean():.4f}"
    assert historical_mass.mean() < 0.01, f"Historical mass should be ~0.0, got {historical_mass.mean():.4f}"

    return True

def test_backward():
    """测试4: Backward 梯度正确"""
    print("\n" + "="*80)
    print("Test 4: Backward gradients correct")
    print("="*80)

    B, H, N, K, C = 4, 12, 207, 5, 1
    T = 12

    model = CandidateSetHorizonCorrector(
        context_length=T,
        horizon=H,
        channels=C,
        hidden_dim=256,
        state_dim=256,
        attention_heads=4,
        base_logit_init_bias=1.0
    )

    # 构造输入
    history = torch.randn(B, T, N, C)
    base = torch.randn(B, H, N, C)
    candidates_future = torch.randn(B, H, N, K, C)
    candidates_mask = torch.ones(B, H, N, K, C, dtype=torch.bool)
    target = torch.randn(B, H, N, C)

    candidates = NodeCandidates(
        event_ids=torch.randint(0, 1000, (B, N, K)),
        weights=torch.softmax(torch.randn(B, N, K), dim=-1),
        shape_scores=torch.randn(B, N, K),
        level_distances=torch.rand(B, N, K),
        total_scores=torch.randn(B, N, K),
        valid=torch.ones(B, N, K, dtype=torch.bool)
    )

    aggregation = AggregationOutput(
        prediction=torch.randn(B, H, N, C),
        variance=torch.rand(B, H, N, C),
        valid=torch.ones(B, H, N, C, dtype=torch.bool),
        candidate_futures=candidates_future,
        candidate_masks=candidates_mask
    )

    # Forward
    final, _, _, _ = model(
        history=history,
        base=base,
        memory=None,
        features=None,
        memory_valid=None,
        candidates=candidates,
        aggregation=aggregation
    )

    # Loss
    loss = (final - target).abs().mean()

    # Backward
    loss.backward()

    # 检查梯度
    params_with_grad = 0
    params_without_grad = 0
    nan_grad_params = []

    for name, param in model.named_parameters():
        if param.grad is None:
            params_without_grad += 1
            print(f"  [WARN] No grad: {name}")
        elif not torch.isfinite(param.grad).all():
            nan_grad_params.append(name)
            print(f"  [ERROR] NaN/Inf grad: {name}")
        else:
            params_with_grad += 1

    print(f"[OK] Parameters with grad: {params_with_grad}")
    print(f"[OK] Parameters without grad: {params_without_grad}")

    assert params_without_grad == 0, f"{params_without_grad} parameters have no gradient"
    assert len(nan_grad_params) == 0, f"NaN/Inf gradients in: {nan_grad_params}"
    assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss}"

    print(f"[OK] Loss: {loss.item():.6f}")
    print(f"[OK] All gradients finite")

    return True

def test_parameter_count():
    """测试5: 参数量符合预期"""
    print("\n" + "="*80)
    print("Test 5: Parameter count in target range (25-30w)")
    print("="*80)

    model = CandidateSetHorizonCorrector(
        context_length=12,
        horizon=12,
        channels=1,
        hidden_dim=256,
        state_dim=256,
        attention_heads=4,
        base_logit_init_bias=1.0
    )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"[OK] Total parameters: {total_params:,} ({total_params/10000:.2f}w)")

    assert 250000 <= total_params <= 300000, f"Params {total_params} not in [250000, 300000]"

    return True

def main():
    print("="*80)
    print("CandidateSetHorizonCorrector Smoke Test Suite")
    print("="*80)

    tests = [
        ("Forward Shape", test_forward_shape),
        ("Attention Shape", test_attention_shape),
        ("No Candidate Fallback", test_no_candidate_fallback),
        ("Backward Gradients", test_backward),
        ("Parameter Count", test_parameter_count),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n[PASS] {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_name} FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80)

    if failed == 0:
        print("[OK] All smoke tests passed!")
        return 0
    else:
        print("[ERROR] Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())

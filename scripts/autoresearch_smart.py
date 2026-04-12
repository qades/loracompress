#!/usr/bin/env python3
"""
Smart autoresearch with high-LR exploration and dimension correlation analysis.

Features:
- Top-down LR search (start high, detect divergence early)
- Rank vs dimension correlation analysis
- Early stopping on divergence
"""
import sys
import os
import time
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc


def train_with_divergence_check(target_weight, rank, lr, max_epochs=200, 
                                divergence_threshold=10.0, device='cpu'):
    """
    Train with early divergence detection.
    
    Returns:
        error: Final relative error
        epochs_trained: How many epochs before stop
        diverged: Whether training diverged
        history: Loss history
    """
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    history = []
    initial_loss = None
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        
        current = loss.item()
        history.append(current)
        
        # Check for initial loss
        if initial_loss is None:
            initial_loss = current
        
        # Divergence check: loss increased 10x from initial
        if current > initial_loss * divergence_threshold:
            return None, epoch, True, history
        
        # Divergence check: NaN
        if not np.isfinite(current):
            return None, epoch, True, history
        
        loss.backward()
        optimizer.step()
    
    # Compute final error
    with torch.no_grad():
        W_final = torch.matmul(B, A)
        final_mse = F.mse_loss(W_final, target).item()
        rel_error = (final_mse ** 0.5) / torch.norm(target).item() * 100
    
    return rel_error, max_epochs, False, history


def find_max_lr(target_weight, rank=8, device='cpu'):
    """
    Find maximum stable learning rate by testing progressively higher.
    
    Keeps going up until we find divergence, then back off.
    """
    print("\n" + "="*60)
    print("Finding Maximum Stable Learning Rate")
    print("="*60)
    
    # Start high and keep going higher if stable
    lr_tests = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    results = []
    last_stable_lr = None
    
    for lr in lr_tests:
        print(f"  Testing LR={lr:>6.2f}: ", end='', flush=True)
        
        error, epochs, diverged, history = train_with_divergence_check(
            target_weight, rank, lr, max_epochs=50, device=device
        )
        
        if diverged:
            print(f"DIVERGED at epoch {epochs}")
            results.append({'lr': lr, 'error': None, 'diverged': True, 'epochs': epochs})
            # Stop here - we found the limit
            break
        else:
            print(f"stable, error={error:.4f}%")
            results.append({'lr': lr, 'error': error, 'diverged': False, 'epochs': epochs})
            last_stable_lr = lr
    
    if last_stable_lr is None:
        # Even 0.1 diverged, try lower
        print(f"\n⚠ High LRs unstable, testing lower...")
        for lr in [0.03, 0.01, 0.003]:
            print(f"  Testing LR={lr:>6.0e}: ", end='', flush=True)
            error, epochs, diverged, _ = train_with_divergence_check(
                target_weight, rank, lr, max_epochs=50, device=device
            )
            if not diverged:
                print(f"stable, error={error:.4f}%")
                last_stable_lr = lr
                break
            else:
                print(f"DIVERGED")
    
    print(f"\n✓ Maximum stable LR: {last_stable_lr}")
    return last_stable_lr, results


def analyze_dimension_correlation(target_weight, max_lr, device='cpu'):
    """
    Analyze how optimal rank depends on matrix dimensions.
    
    Test ranks proportional to matrix dimensions.
    """
    d, k = target_weight.shape
    min_dim = min(d, k)
    
    print("\n" + "="*60)
    print(f"Dimension Analysis: {d}×{k} (min={min_dim})")
    print("="*60)
    
    # Test ranks: small fraction to larger fraction of min_dim
    rank_tests = [
        ('tiny', max(4, int(min_dim * 0.007))),      # ~0.7%
        ('small', max(8, int(min_dim * 0.014))),     # ~1.4%
        ('medium', max(16, int(min_dim * 0.028))),   # ~2.8%
        ('large', max(32, int(min_dim * 0.056))),    # ~5.6%
    ]
    
    print(f"\nTesting ranks relative to min dimension ({min_dim}):")
    results = []
    
    for name, rank in rank_tests:
        if rank > min_dim:
            print(f"  {name:8s} (rank={rank:3d}): SKIPPED (rank > dim)")
            continue
            
        print(f"  {name:8s} (rank={rank:3d}, {rank/min_dim*100:.1f}%): ", end='', flush=True)
        
        error, epochs, diverged, _ = train_with_divergence_check(
            target_weight, rank, max_lr, max_epochs=200, device=device
        )
        
        if diverged:
            print(f"DIVERGED")
            results.append({'name': name, 'rank': rank, 'error': None, 'pct': rank/min_dim*100})
        else:
            compression = (d * k) / (rank * (d + k))
            print(f"error={error:.4f}%, ratio={compression:.1f}x")
            results.append({
                'name': name, 'rank': rank, 'error': error, 
                'pct': rank/min_dim*100, 'compression': compression
            })
    
    # Find best
    valid = [r for r in results if r['error'] is not None]
    if valid:
        best = min(valid, key=lambda x: x['error'] * (1 + 0.01 * x['rank']))
        print(f"\n✓ Best: {best['name']} rank={best['rank']} ({best['pct']:.1f}% of min_dim)")
        return best, results
    else:
        print("\n✗ All diverged!")
        return None, results


def test_across_layers(model_name, module_type, layer_indices, max_lr, rank, device='cpu'):
    """
    Test the same config across different layers to verify consistency.
    """
    from transformers import AutoModelForCausalLM
    
    print("\n" + "="*60)
    print(f"Cross-Layer Consistency Test: {module_type}")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
    ).to(device)
    
    results = []
    
    for layer_idx in layer_indices:
        target_name = f"model.layers.{layer_idx}.self_attn.{module_type}.weight"
        if module_type in ['gate_proj', 'up_proj', 'down_proj']:
            target_name = f"model.layers.{layer_idx}.mlp.{module_type}.weight"
        
        if target_name not in dict(model.named_parameters()):
            continue
        
        target_weight = dict(model.named_parameters())[target_name].data
        d, k = target_weight.shape
        
        print(f"  Layer {layer_idx:2d} ({d}×{k}): ", end='', flush=True)
        
        error, epochs, diverged, _ = train_with_divergence_check(
            target_weight, rank, max_lr, max_epochs=150, device=device
        )
        
        if diverged:
            print(f"DIVERGED")
            results.append({'layer': layer_idx, 'error': None, 'shape': (d, k)})
        else:
            print(f"error={error:.4f}%")
            results.append({'layer': layer_idx, 'error': error, 'shape': (d, k)})
    
    del model
    import gc
    gc.collect()
    
    valid = [r for r in results if r['error'] is not None]
    if valid:
        errors = [r['error'] for r in valid]
        print(f"\n  Mean error: {np.mean(errors):.4f}%")
        print(f"  Std dev: {np.std(errors):.4f}%")
        print(f"  Range: {min(errors):.4f}% - {max(errors):.4f}%")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart autoresearch with high-LR exploration")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=15)
    parser.add_argument("--module", default="q_proj")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test-layers", nargs='+', type=int, default=[0, 5, 10, 15, 20, 25],
                        help="Layers to test for consistency")
    parser.add_argument("--output", default="./autoresearch_smart.json")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Smart Autoresearch: High-LR & Dimension Analysis")
    print("="*60)
    
    # Load model
    from transformers import AutoModelForCausalLM
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
    ).to(args.device)
    
    # Get primary test layer
    target_name = f"model.layers.{args.layer_idx}.self_attn.{args.module}.weight"
    if args.module in ['gate_proj', 'up_proj', 'down_proj']:
        target_name = f"model.layers.{args.layer_idx}.mlp.{args.module}.weight"
    
    target_weight = dict(model.named_parameters())[target_name].data
    d, k = target_weight.shape
    
    print(f"Primary test: {target_name}")
    print(f"Shape: {d}×{k}")
    
    del model
    gc.collect()
    
    total_start = time.time()
    
    # PHASE 1: Find max LR
    max_lr, lr_results = find_max_lr(target_weight, device=args.device)
    
    # PHASE 2: Dimension correlation
    dim_result, dim_results = analyze_dimension_correlation(
        target_weight, max_lr, device=args.device
    )
    
    # PHASE 3: Cross-layer consistency
    if dim_result:
        cross_results = test_across_layers(
            args.model, args.module, args.test_layers,
            max_lr, dim_result['rank'], device=args.device
        )
    else:
        cross_results = []
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*60)
    print("SMART AUTORESEARCH SUMMARY")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Max stable LR: {max_lr:.0e}")
    if dim_result:
        print(f"  Optimal rank: {dim_result['rank']} ({dim_result['pct']:.1f}% of min_dim)")
        print(f"  Expected error: {dim_result['error']:.4f}%")
        print(f"  Compression: {dim_result['compression']:.1f}x")
    
    print(f"\nTiming: {total_time:.1f}s")
    
    # Save results
    results = {
        'max_lr': max_lr,
        'lr_search': lr_results,
        'dimension_analysis': dim_results,
        'cross_layer': cross_results,
        'optimal_config': dim_result,
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

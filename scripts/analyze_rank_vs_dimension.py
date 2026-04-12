#!/usr/bin/env python3
"""
Analyze if optimal rank correlates with layer dimensions.

Tests different layer types:
- Small: Attention layers (e.g., 576×576)
- Medium: MLP intermediate (e.g., 576×1536)
- Large: MLP output or embeddings (e.g., 1536×576 or bigger)

For each, finds optimal rank and checks correlation.
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


def train_layer(target_weight, rank, lr=0.03, max_epochs=500, 
                patience=100, device='cpu'):
    """Train with universal recipe, return final error."""
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        
        if not torch.isfinite(loss):
            return None, epoch, True
        
        loss.backward()
        optimizer.step()
        
        current = loss.item()
        if current < best_loss - 1e-8:
            best_loss = current
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    rel_error = (best_loss ** 0.5) / torch.norm(target).item() * 100
    compression = (d * k) / (rank * (d + k))
    
    return rel_error, epoch, False, compression


def find_optimal_rank(target_weight, layer_name, device='cpu'):
    """Find optimal rank for a given layer."""
    d, k = target_weight.shape
    min_dim = min(d, k)
    total_params = d * k
    
    print(f"\n{'='*70}")
    print(f"Layer: {layer_name}")
    print(f"Shape: {d}×{k}, min_dim={min_dim}, params={total_params:,}")
    print(f"{'='*70}")
    
    # Test ranks from tiny to large (as % of min_dim)
    rank_tests = [
        ('tiny', max(2, int(min_dim * 0.0035))),   # ~0.35%
        ('small', max(4, int(min_dim * 0.007))),   # ~0.7%
        ('medium', max(8, int(min_dim * 0.014))),  # ~1.4%
        ('large', max(16, int(min_dim * 0.028))),  # ~2.8%
        ('xlarge', max(32, int(min_dim * 0.056))), # ~5.6%
        ('huge', max(64, int(min_dim * 0.11))),    # ~11%
    ]
    
    results = []
    for name, rank in rank_tests:
        if rank > min_dim:
            print(f"  {name:8s} (rank={rank:4d}): SKIPPED (rank > min_dim)")
            continue
        
        print(f"  {name:8s} (rank={rank:4d}, {rank/min_dim*100:5.2f}%): ", end='', flush=True)
        
        start = time.time()
        error, epochs, diverged, compression = train_layer(
            target_weight, rank, device=device
        )
        elapsed = time.time() - start
        
        if diverged:
            print(f"DIVERGED")
            results.append({
                'name': name, 'rank': rank, 'pct': rank/min_dim*100,
                'error': None, 'diverged': True
            })
        else:
            print(f"error={error:.4f}%, ratio={compression:.1f}x, epochs={epochs}, time={elapsed:.1f}s")
            results.append({
                'name': name, 'rank': rank, 'pct': rank/min_dim*100,
                'error': error, 'epochs': epochs, 'compression': compression,
                'time': elapsed
            })
    
    # Find optimal (balance error vs compression)
    valid = [r for r in results if r['error'] is not None]
    if not valid:
        return None, results
    
    # Score: penalize both high error AND high rank
    for r in valid:
        r['score'] = r['error'] * (1 + 0.05 * r['rank'] / 4)
    
    best = min(valid, key=lambda x: x['score'])
    
    print(f"\n  ✓ Optimal rank: {best['rank']} ({best['pct']:.2f}% of min_dim)")
    print(f"    Error: {best['error']:.4f}%, Compression: {best['compression']:.1f}x")
    
    return best, results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze rank vs dimension correlation")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="./rank_dimension_analysis.json")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Rank vs Dimension Correlation Analysis")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Testing layers: attention (small), MLP up (medium), MLP down (large)")
    print("="*70)
    
    # Load model
    from transformers import AutoModelForCausalLM
    import gc
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
    ).to(args.device)
    
    layer_idx = args.layer_idx
    
    # Test different layer types
    test_configs = [
        ("Small (Attention q_proj)", f"model.layers.{layer_idx}.self_attn.q_proj.weight"),
        ("Medium (MLP gate_proj)", f"model.layers.{layer_idx}.mlp.gate_proj.weight"),
        ("Large (MLP down_proj)", f"model.layers.{layer_idx}.mlp.down_proj.weight"),
    ]
    
    all_results = {}
    layer_summaries = []
    
    for desc, weight_name in test_configs:
        if weight_name not in dict(model.named_parameters()):
            print(f"\n⚠ {weight_name} not found, skipping...")
            continue
        
        target_weight = dict(model.named_parameters())[weight_name].data
        best, results = find_optimal_rank(target_weight, desc, args.device)
        
        all_results[desc] = {
            'shape': list(target_weight.shape),
            'best': best,
            'all_results': results,
        }
        
        if best:
            d, k = target_weight.shape
            layer_summaries.append({
                'name': desc,
                'shape': (d, k),
                'min_dim': min(d, k),
                'total_params': d * k,
                'optimal_rank': best['rank'],
                'rank_pct': best['pct'],
                'error': best['error'],
                'compression': best['compression'],
            })
        
        # Free memory
        gc.collect()
    
    del model
    gc.collect()
    
    # Analysis
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    if len(layer_summaries) >= 2:
        print(f"\n{'Layer':<25} {'Shape':<15} {'Min Dim':<10} {'Opt Rank':<10} {'Rank %':<10} {'Error':<10}")
        print("-"*80)
        for s in layer_summaries:
            print(f"{s['name']:<25} {str(s['shape']):<15} {s['min_dim']:<10} "
                  f"{s['optimal_rank']:<10} {s['rank_pct']:>8.2f}%  {s['error']:>8.4f}%")
        
        # Check correlation
        min_dims = [s['min_dim'] for s in layer_summaries]
        opt_ranks = [s['optimal_rank'] for s in layer_summaries]
        rank_pcts = [s['rank_pct'] for s in layer_summaries]
        
        print(f"\nCorrelation Analysis:")
        print(f"  Min dims: {min_dims}")
        print(f"  Opt ranks: {opt_ranks}")
        print(f"  Rank %s: {[f'{p:.2f}%' for p in rank_pcts]}")
        
        # Simple trend check
        if len(set(min_dims)) > 1 and len(set(opt_ranks)) > 1:
            # Check if larger dim -> larger rank
            dim_rank_corr = np.corrcoef(min_dims, opt_ranks)[0, 1] if len(min_dims) > 2 else None
            
            if dim_rank_corr is not None:
                print(f"\n  Dimension-Rank correlation: {dim_rank_corr:.3f}")
                if dim_rank_corr > 0.5:
                    print(f"  → Strong positive correlation: larger layers need higher ranks!")
                elif dim_rank_corr < -0.5:
                    print(f"  → Negative correlation (unexpected)")
                else:
                    print(f"  → Weak correlation: optimal rank % is relatively constant")
            
            # Check if rank % is consistent
            pct_std = np.std(rank_pcts)
            print(f"\n  Rank % std dev: {pct_std:.2f}%")
            if pct_std < 2.0:
                print(f"  → Rank % is very consistent across layer sizes!")
                print(f"  → Recommendation: Use fixed % of min_dim (e.g., {np.mean(rank_pcts):.1f}%)")
            else:
                print(f"  → Rank % varies by layer size")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'layer_summaries': layer_summaries,
            'detailed_results': all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    if layer_summaries:
        avg_pct = np.mean([s['rank_pct'] for s in layer_summaries])
        print(f"\nBased on analysis:")
        print(f"  Average optimal rank: {avg_pct:.2f}% of min dimension")
        print(f"\nFor new layers, use:")
        print(f"  rank = max(4, int(min(d, k) * {avg_pct/100:.4f}))")
        print(f"\nExample for 576×576: rank ≈ {int(576 * avg_pct / 100)}")
        print(f"Example for 576×1536: rank ≈ {int(576 * avg_pct / 100)}")


if __name__ == "__main__":
    main()

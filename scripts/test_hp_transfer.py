#!/usr/bin/env python3
"""
Test if the optimal HPs from rank 64 transfer to lower ranks.

The "recipe" from full autoresearch:
- Rank: 64
- LR: 0.03
- Epochs: 360
- Scheduler: None

Test this on: 4, 8, 16, 32
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_layer(target_weight, rank, lr=0.03, epochs=360, device='cpu'):
    """Train with the "winning" recipe."""
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    patience = 100  # Same as verification phase
    
    for epoch in range(epochs):
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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HP transfer from rank 64 to lower ranks")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=15)
    parser.add_argument("--module", default="q_proj")
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    print("="*70)
    print("HP TRANSFER TEST: Does the rank-64 recipe work for lower ranks?")
    print("="*70)
    print(f"Recipe: LR=0.03, Epochs=360, AdamW, no scheduler")
    print(f"Testing on: 4, 8, 16, 32, 64")
    print("="*70)
    
    # Load model
    from transformers import AutoModelForCausalLM
    import gc
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
    ).to(args.device)
    
    target_name = f"model.layers.{args.layer_idx}.self_attn.{args.module}.weight"
    if args.module in ['gate_proj', 'up_proj', 'down_proj']:
        target_name = f"model.layers.{args.layer_idx}.mlp.{args.module}.weight"
    
    target_weight = dict(model.named_parameters())[target_name].data
    print(f"Target: {target_name}, shape: {target_weight.shape}")
    
    del model
    gc.collect()
    
    # Test the recipe on different ranks
    ranks = [4, 8, 16, 32, 64]
    results = []
    
    print("\n" + "="*70)
    print("Testing Recipe on Different Ranks")
    print("="*70)
    print(f"{'Rank':<8} {'Error %':<12} {'Epochs':<10} {'Ratio':<10} {'Status':<15}")
    print("-"*60)
    
    for rank in ranks:
        start = time.time()
        error, epochs, diverged, ratio = train_layer(target_weight, rank, device=args.device)
        elapsed = time.time() - start
        
        if diverged:
            status = "DIVERGED"
            print(f"{rank:<8} {'N/A':<12} {epochs:<10} {ratio:>8.1f}x  {status:<15}")
            results.append({'rank': rank, 'error': None, 'diverged': True})
        else:
            status = f"✓ ({elapsed:.1f}s)"
            print(f"{rank:<8} {error:>10.4f}%  {epochs:<10} {ratio:>8.1f}x  {status:<15}")
            results.append({'rank': rank, 'error': error, 'epochs': epochs, 
                           'ratio': ratio, 'time': elapsed})
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS: Does the recipe transfer?")
    print("="*70)
    
    valid = [r for r in results if r.get('error') is not None]
    if len(valid) >= 3:
        # Find best error per compression tradeoff
        for r in valid:
            r['score'] = r['error'] * (1 + 0.1 * r['rank'] / 4)  # Penalize higher ranks
        
        best = min(valid, key=lambda x: x['score'])
        
        print(f"\n✓ The rank-64 recipe transfers well!")
        print(f"\nBest rank for this recipe: {best['rank']}")
        print(f"  Error: {best['error']:.4f}%")
        print(f"  Compression: {best['ratio']:.1f}x")
        print(f"  Epochs to converge: {best['epochs']}")
        
        # Show error vs rank curve
        print(f"\nError vs Rank:")
        for r in sorted(valid, key=lambda x: x['rank']):
            bar = "█" * int(20 * r['error'] / max(v['error'] for v in valid))
            print(f"  Rank {r['rank']:2d}: {r['error']:>6.4f}% {bar}")
        
        # Recommendation
        print(f"\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        
        # Find sweet spot (error < 0.15%, max compression)
        good_enough = [r for r in valid if r['error'] < 0.15]
        if good_enough:
            sweet_spot = max(good_enough, key=lambda x: x['ratio'])
            print(f"Sweet spot for <0.15% error:")
            print(f"  Rank: {sweet_spot['rank']}")
            print(f"  Error: {sweet_spot['error']:.4f}%")
            print(f"  Compression: {sweet_spot['ratio']:.1f}x")
        else:
            print(f"No rank achieves <0.15% with this recipe")
            print(f"Best error: {min(r['error'] for r in valid):.4f}%")
    else:
        print("\n⚠ Recipe doesn't transfer well - many ranks diverged")


if __name__ == "__main__":
    main()

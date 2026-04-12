#!/usr/bin/env python3
"""
Fine-grained LR search to find optimal (not just max stable).

Searches for LR that minimizes final error, not just converges.
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


def train_to_convergence(target_weight, rank, lr, max_epochs=500, 
                         patience=50, tol=1e-8, device='cpu'):
    """Train until convergence, return final error."""
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
            return None, epoch, True  # Diverged
        
        loss.backward()
        optimizer.step()
        
        current = loss.item()
        if current < best_loss - tol:
            best_loss = current
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    rel_error = (best_loss ** 0.5) / torch.norm(target).item() * 100
    return rel_error, epoch, False


def find_optimal_lr(target_weight, rank=4, device='cpu'):
    """
    Comprehensive LR search to find minimum error.
    
    Phase 1: Wide coarse search
    Phase 2: Fine search around best
    Phase 3: Verify optimum
    """
    print("="*70)
    print("PHASE 1: Wide Coarse Search")
    print("="*70)
    
    # Very wide range - this is NOT normal training
    coarse_lrs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5]
    
    results = []
    for lr in coarse_lrs:
        print(f"  LR={lr:>6.3f}: ", end='', flush=True)
        start = time.time()
        
        error, epochs, diverged = train_to_convergence(
            target_weight, rank, lr, max_epochs=300, patience=30, device=device
        )
        
        elapsed = time.time() - start
        
        if diverged:
            print(f"DIVERGED at epoch {epochs}")
            results.append({'lr': lr, 'error': None, 'diverged': True})
        else:
            print(f"error={error:.4f}%, epochs={epochs}, time={elapsed:.1f}s")
            results.append({'lr': lr, 'error': error, 'epochs': epochs, 'time': elapsed})
    
    # Find best from coarse
    valid = [r for r in results if r['error'] is not None]
    if not valid:
        print("\n✗ All diverged!")
        return None, results
    
    best_coarse = min(valid, key=lambda x: x['error'])
    print(f"\n✓ Best from coarse: LR={best_coarse['lr']}, error={best_coarse['error']:.4f}%")
    
    # Phase 2: Fine search around best
    print("\n" + "="*70)
    print("PHASE 2: Fine Search Around Optimum")
    print("="*70)
    
    best_lr = best_coarse['lr']
    # Test points around best: 0.5x, 0.7x, 1x, 1.4x, 2x
    fine_multipliers = [0.5, 0.7, 1.0, 1.4, 2.0]
    fine_lrs = [best_lr * m for m in fine_multipliers if best_lr * m <= 0.9]  # Cap at 0.9
    
    fine_results = []
    for lr in fine_lrs:
        print(f"  LR={lr:>6.4f}: ", end='', flush=True)
        start = time.time()
        
        error, epochs, diverged = train_to_convergence(
            target_weight, rank, lr, max_epochs=400, patience=50, device=device
        )
        
        elapsed = time.time() - start
        
        if diverged:
            print(f"DIVERGED")
            fine_results.append({'lr': lr, 'error': None, 'diverged': True})
        else:
            print(f"error={error:.4f}%, epochs={epochs}, time={elapsed:.1f}s")
            fine_results.append({'lr': lr, 'error': error, 'epochs': epochs, 'time': elapsed})
    
    # Find absolute best
    all_valid = [r for r in results + fine_results if r['error'] is not None]
    absolute_best = min(all_valid, key=lambda x: x['error'])
    
    print(f"\n✓ ABSOLUTE BEST: LR={absolute_best['lr']:.4f}, error={absolute_best['error']:.4f}%")
    
    # Phase 3: Verify with longer training
    print("\n" + "="*70)
    print("PHASE 3: Verify Optimum (Extended Training)")
    print("="*70)
    
    print(f"Retraining with LR={absolute_best['lr']:.4f} for up to 1000 epochs...")
    final_error, final_epochs, diverged = train_to_convergence(
        target_weight, rank, absolute_best['lr'], 
        max_epochs=1000, patience=100, device=device
    )
    
    if diverged:
        print(f"  Warning: Diverged during verification!")
    else:
        print(f"  Verified: error={final_error:.4f}%, epochs={final_epochs}")
        absolute_best['verified_error'] = final_error
        absolute_best['verified_epochs'] = final_epochs
    
    return absolute_best, results + fine_results


def analyze_lr_vs_error(all_results, output_file='lr_analysis.png'):
    """Plot LR vs error curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        valid = [r for r in all_results if r['error'] is not None]
        lrs = [r['lr'] for r in valid]
        errors = [r['error'] for r in valid]
        
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, errors, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Relative Error (%)', fontsize=12)
        plt.title('Learning Rate vs Final Error', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved to {output_file}")
    except ImportError:
        print("\n(matplotlib not available for plotting)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-grained LR optimization")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=15)
    parser.add_argument("--module", default="q_proj")
    parser.add_argument("--rank", type=int, default=4, help="Rank to test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="./lr_optimization.json")
    parser.add_argument("--plot", action="store_true", help="Generate plot")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Fine-Grained Learning Rate Optimization")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer_idx}, Module: {args.module}")
    print(f"Rank: {args.rank}")
    print()
    print("Searching for LR that minimizes error (not just converges)")
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
    print(f"Selected: {target_name}, shape: {target_weight.shape}")
    
    del model
    gc.collect()
    
    # Run optimization
    total_start = time.time()
    best, all_results = find_optimal_lr(target_weight, rank=args.rank, device=args.device)
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    if best:
        print(f"\n✓ OPTIMAL CONFIGURATION:")
        print(f"  Learning rate: {best['lr']:.6f}")
        print(f"  Final error: {best.get('verified_error', best['error']):.6f}%")
        print(f"  Training epochs: {best.get('verified_epochs', best['epochs'])}")
        print(f"  Time: {total_time:.1f}s")
        
        # Table of all results
        print(f"\n  All Results (sorted by error):")
        valid = [r for r in all_results if r['error'] is not None]
        valid.sort(key=lambda x: x['error'])
        print(f"  {'LR':<12} {'Error %':<12} {'Epochs':<10}")
        print(f"  {'-'*34}")
        for r in valid[:10]:  # Top 10
            marker = " <-- BEST" if r['lr'] == best['lr'] else ""
            print(f"  {r['lr']:<12.6f} {r['error']:>10.4f}%  {r['epochs']:<10}{marker}")
    else:
        print("\n✗ No stable configuration found!")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'optimal': best,
            'all_results': all_results,
            'total_time': total_time,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Plot if requested
    if args.plot:
        analyze_lr_vs_error(all_results)


if __name__ == "__main__":
    main()

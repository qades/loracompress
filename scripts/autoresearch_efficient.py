#!/usr/bin/env python3
"""
Efficient staged autoresearch for single layer.

Stage 1: Learning rate (quick, 50 epochs)
Stage 2: Epochs to convergence (with best LR)
Stage 3: Rank (with optimal training config)
"""
import sys
import os
import time
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F


def quick_train(target_weight, rank, lr, epochs, device='cpu'):
    """Quick training for screening."""
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    best_loss = float('inf')
    
    for _ in range(epochs):
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    rel_error = (best_loss ** 0.5) / torch.norm(target).item() * 100
    return rel_error


def find_optimal_lr(target_weight, rank=8, lrs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 
                    quick_epochs=50, device='cpu'):
    """Stage 1: Find best learning rate quickly."""
    print("\n" + "="*60)
    print("STAGE 1: Learning Rate Screening (50 epochs each)")
    print("="*60)
    
    results = []
    for lr in lrs:
        print(f"  LR {lr:>7.0e}: ", end='', flush=True)
        start = time.time()
        error = quick_train(target_weight, rank, lr, quick_epochs, device)
        elapsed = time.time() - start
        results.append({'lr': lr, 'error': error, 'time': elapsed})
        print(f"error={error:.4f}%, time={elapsed:.1f}s")
    
    best = min(results, key=lambda x: x['error'])
    print(f"\n✓ Best LR: {best['lr']:.0e} (error={best['error']:.4f}%)")
    return best['lr'], results


def find_convergence_epoch(target_weight, rank, lr, device='cpu'):
    """Stage 2: Find how many epochs needed to converge."""
    print("\n" + "="*60)
    print(f"STAGE 2: Convergence Analysis (rank={rank}, lr={lr:.0e})")
    print("="*60)
    
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    patience = 30
    history = []
    
    for epoch in range(500):  # Max 500, but will stop early
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        loss.backward()
        optimizer.step()
        
        current = loss.item()
        history.append(current)
        
        if current < best_loss - 1e-8:
            best_loss = current
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"  Converged at epoch {epoch+1}")
            break
    else:
        print(f"  Reached max epochs (500), not fully converged")
    
    final_error = (best_loss ** 0.5) / torch.norm(target).item() * 100
    print(f"  Final error: {final_error:.4f}%")
    
    # Suggest epochs for next stage (with margin)
    suggested_epochs = min(epoch + 50, 300)
    print(f"  Suggest using {suggested_epochs} epochs for rank tests")
    
    return suggested_epochs, final_error, history


def test_ranks(target_weight, lr, epochs, ranks=[4, 8, 16, 32], device='cpu'):
    """Stage 3: Test different ranks with optimal config."""
    print("\n" + "="*60)
    print(f"STAGE 3: Rank Analysis (lr={lr:.0e}, {epochs} epochs)")
    print("="*60)
    
    d, k = target_weight.shape
    results = []
    
    for rank in ranks:
        print(f"  Rank {rank:3d}: ", end='', flush=True)
        start = time.time()
        
        error = quick_train(target_weight, rank, lr, epochs, device)
        compression = (d * k) / (rank * (d + k))
        elapsed = time.time() - start
        
        results.append({
            'rank': rank,
            'error': error,
            'compression': compression,
            'time': elapsed,
        })
        
        print(f"error={error:.4f}%, ratio={compression:.1f}x, time={elapsed:.1f}s")
    
    # Find best (lowest error with reasonable compression)
    best = min(results, key=lambda x: x['error'] * (1 + 0.02 * x['rank']))
    print(f"\n✓ Best rank: {best['rank']} (error={best['error']:.4f}%, ratio={best['compression']:.1f}x)")
    
    return results, best


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Efficient staged autoresearch")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=15)
    parser.add_argument("--module", default="q_proj")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="./autoresearch_efficient.json")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Efficient Staged Autoresearch")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer_idx}, Module: {args.module}")
    print()
    print("Strategy: LR → Epochs → Rank")
    print("="*60)
    
    # Load model
    from transformers import AutoModelForCausalLM
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
    ).to(args.device)
    
    target_name = f"model.layers.{args.layer_idx}.self_attn.{args.module}.weight"
    if args.module in ['gate_proj', 'up_proj', 'down_proj']:
        target_name = f"model.layers.{args.layer_idx}.mlp.{args.module}.weight"
    
    target_weight = dict(model.named_parameters())[target_name].data
    print(f"Selected: {target_name}")
    print(f"Shape: {target_weight.shape}")
    
    del model
    import gc
    gc.collect()
    
    total_start = time.time()
    
    # STAGE 1: Learning Rate
    best_lr, lr_results = find_optimal_lr(target_weight, device=args.device)
    
    # STAGE 2: Convergence Epochs
    optimal_epochs, conv_error, history = find_convergence_epoch(
        target_weight, rank=8, lr=best_lr, device=args.device
    )
    
    # STAGE 3: Rank Analysis
    rank_results, best_rank = test_ranks(
        target_weight, lr=best_lr, epochs=optimal_epochs, device=args.device
    )
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\nOptimal Configuration:")
    print(f"  Rank: {best_rank['rank']}")
    print(f"  Learning rate: {best_lr:.0e}")
    print(f"  Epochs: {optimal_epochs}")
    print(f"  Expected error: {best_rank['error']:.4f}%")
    print(f"  Compression: {best_rank['compression']:.1f}x")
    
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Save results
    results = {
        'model': args.model,
        'layer': target_name,
        'best_config': {
            'rank': best_rank['rank'],
            'lr': best_lr,
            'epochs': optimal_epochs,
            'error': best_rank['error'],
            'compression': best_rank['compression'],
        },
        'lr_screening': lr_results,
        'convergence': {
            'epochs': optimal_epochs,
            'error': conv_error,
        },
        'rank_analysis': rank_results,
        'total_time': total_time,
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Verdict
    print("\n" + "="*60)
    if best_rank['error'] < 0.3:
        print("✓ EXCELLENT: Low error, model reproduction very feasible!")
    elif best_rank['error'] < 1.0:
        print("~ GOOD: Acceptable error for many use cases")
    else:
        print("⚠ HIGH ERROR: May need different approach")
    print("="*60)


if __name__ == "__main__":
    main()

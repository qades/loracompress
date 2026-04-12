#!/usr/bin/env python3
"""
Thorough autoresearch for single layer LoRA compression.

Tests:
- Rank (4, 8, 16, 32, 64)
- Epochs (for convergence analysis)
- Learning rate
- Final error verification
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


def train_until_convergence(target_weight, rank, lr=1e-3, max_epochs=500, 
                            patience=50, tol=1e-6, device='cpu', verbose=False):
    """
    Train until convergence with early stopping.
    
    Returns:
        best_loss: Final MSE
        history: Training history
        epochs_trained: Actual epochs before convergence
    """
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    best_loss = float('inf')
    best_A, best_B = None, None
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        history.append(current_loss)
        
        if current_loss < best_loss - tol:
            best_loss = current_loss
            best_A = A.detach().clone()
            best_B = B.detach().clone()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"    Converged at epoch {epoch+1}")
            break
    
    # Compute final metrics
    with torch.no_grad():
        W_final = torch.matmul(best_B, best_A)
        final_mse = F.mse_loss(W_final, target).item()
        rel_error = (final_mse ** 0.5) / torch.norm(target).item() * 100
    
    return {
        'loss': final_mse,
        'rel_error': rel_error,
        'epochs_trained': len(history),
        'history': history,
        'A': best_A,
        'B': best_B,
    }


def full_autoresearch(target_weight, device='cpu'):
    """
    Comprehensive autoresearch on a single layer.
    """
    d, k = target_weight.shape
    print(f"\nLayer shape: {target_weight.shape}")
    print(f"Parameters: {d*k:,}")
    print(f"Matrix rank (max possible): {min(d, k)}")
    print()
    
    # Test 1: Rank vs Quality (with sufficient training)
    print("="*60)
    print("Test 1: Rank vs Compression Quality")
    print("="*60)
    
    ranks = [4, 8, 16, 32, 64]
    rank_results = []
    
    for rank in ranks:
        print(f"\nRank {rank:3d}: ", end='', flush=True)
        start = time.time()
        
        result = train_until_convergence(
            target_weight, rank, 
            lr=1e-3, max_epochs=1000, patience=100, tol=1e-7,
            device=device, verbose=False
        )
        
        elapsed = time.time() - start
        compression = (d * k) / (rank * (d + k))
        
        rank_results.append({
            'rank': rank,
            'error': result['rel_error'],
            'compression': compression,
            'epochs': result['epochs_trained'],
            'time': elapsed,
            'mse': result['loss'],
        })
        
        print(f"error={result['rel_error']:.4f}%, compression={compression:.1f}x, "
              f"epochs={result['epochs_trained']}, time={elapsed:.1f}s")
    
    # Test 2: Learning Rate Sensitivity (for best rank)
    print("\n" + "="*60)
    print("Test 2: Learning Rate Sensitivity")
    print("="*60)
    
    # Find best rank from previous test (balance of error and compression)
    best_rank_result = min(rank_results, key=lambda x: x['error'] * (1 + 0.05 * x['rank']))
    best_rank = best_rank_result['rank']
    
    print(f"Testing on rank={best_rank} (best balance)")
    
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    lr_results = []
    
    for lr in lrs:
        print(f"  LR {lr:.0e}: ", end='', flush=True)
        start = time.time()
        
        result = train_until_convergence(
            target_weight, best_rank,
            lr=lr, max_epochs=500, patience=50, tol=1e-7,
            device=device, verbose=False
        )
        
        elapsed = time.time() - start
        lr_results.append({
            'lr': lr,
            'error': result['rel_error'],
            'epochs': result['epochs_trained'],
            'time': elapsed,
        })
        
        print(f"error={result['rel_error']:.4f}%, epochs={result['epochs_trained']}, time={elapsed:.1f}s")
    
    # Test 3: Final Convergence Analysis (best config)
    print("\n" + "="*60)
    print("Test 3: Final Convergence Verification")
    print("="*60)
    
    best_lr = min(lr_results, key=lambda x: x['error'])['lr']
    
    print(f"Final training: rank={best_rank}, lr={best_lr:.0e}")
    print("Training for up to 2000 epochs with patience=200...")
    
    final_result = train_until_convergence(
        target_weight, best_rank,
        lr=best_lr, max_epochs=2000, patience=200, tol=1e-8,
        device=device, verbose=True
    )
    
    print(f"\nFinal Results:")
    print(f"  MSE: {final_result['loss']:.6e}")
    print(f"  Relative error: {final_result['rel_error']:.6f}%")
    print(f"  Epochs to converge: {final_result['epochs_trained']}")
    print(f"  Compression ratio: {(d*k)/(best_rank*(d+k)):.1f}x")
    
    # Summary
    print("\n" + "="*60)
    print("AUTORESEARCH SUMMARY")
    print("="*60)
    
    print("\nRank Analysis:")
    print(f"{'Rank':<8} {'Error %':<12} {'Compression':<12} {'Epochs':<10}")
    print("-" * 50)
    for r in rank_results:
        marker = " <--" if r['rank'] == best_rank else ""
        print(f"{r['rank']:<8} {r['error']:>10.4f}%  {r['compression']:>10.1f}x  {r['epochs']:<10}{marker}")
    
    print("\nLearning Rate Analysis:")
    print(f"{'LR':<12} {'Error %':<12} {'Epochs':<10}")
    print("-" * 40)
    for lr in lr_results:
        marker = " <--" if lr['lr'] == best_lr else ""
        print(f"{lr['lr']:<12.0e} {lr['error']:>10.4f}%  {lr['epochs']:<10}{marker}")
    
    return {
        'best_rank': best_rank,
        'best_lr': best_lr,
        'final_error': final_result['rel_error'],
        'final_mse': final_result['loss'],
        'compression_ratio': (d*k)/(best_rank*(d+k)),
        'rank_results': rank_results,
        'lr_results': lr_results,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Thorough single layer autoresearch")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=15)
    parser.add_argument("--module", default="q_proj")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="./autoresearch_single_layer.json")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Thorough Single Layer Autoresearch")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer_idx}")
    print(f"Module: {args.module}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load model and extract layer
    from transformers import AutoModelForCausalLM
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
    ).to(args.device)
    
    # Find the specific layer
    target_name = f"model.layers.{args.layer_idx}.self_attn.{args.module}.weight"
    if args.module in ['gate_proj', 'up_proj', 'down_proj']:
        target_name = f"model.layers.{args.layer_idx}.mlp.{args.module}.weight"
    
    weight_dict = dict(model.named_parameters())
    if target_name not in weight_dict:
        print(f"Available layers:")
        for name in weight_dict.keys():
            if 'proj' in name and 'weight' in name:
                print(f"  {name}")
        raise ValueError(f"Layer {target_name} not found")
    
    target_weight = weight_dict[target_name].data
    
    print(f"Selected: {target_name}")
    
    # Clean up
    del model
    import gc
    gc.collect()
    
    # Run autoresearch
    results = full_autoresearch(target_weight, args.device)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print key finding
    print("\n" + "="*60)
    print("KEY FINDING")
    print("="*60)
    print(f"Optimal configuration:")
    print(f"  Rank: {results['best_rank']}")
    print(f"  Learning rate: {results['best_lr']:.0e}")
    print(f"  Final error: {results['final_error']:.6f}%")
    print(f"  Compression: {results['compression_ratio']:.1f}x")
    
    if results['final_error'] < 0.5:
        print("\n✓ ERROR IS LOW - Model reproduction should work!")
    elif results['final_error'] < 2.0:
        print("\n~ ERROR IS MODERATE - May work with careful tuning")
    else:
        print("\n✗ ERROR IS HIGH - Need higher rank or more epochs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test LoRA compression on a SINGLE layer.

This allows us to:
1. Test if weight reproduction works efficiently
2. Find optimal rank for different layer sizes
3. Benchmark without processing the full model
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


def analyze_model_layers(model_name="HuggingFaceTB/SmolLM2-135M", device='cpu'):
    """Analyze all layers in the model and categorize by size."""
    from transformers import AutoModelForCausalLM
    
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).to(device)
    
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                      'gate_proj', 'up_proj', 'down_proj']
    
    layers = []
    for name, param in model.named_parameters():
        if not any(x in name for x in target_modules) or 'weight' not in name:
            continue
        
        d, k = param.shape
        # Parse name: model.layers.0.self_attn.q_proj.weight
        # parts: ['model', 'layers', '0', 'self_attn', 'q_proj', 'weight']
        parts = name.split('.')
        layer_idx = int(parts[2]) if 'layers.' in name and len(parts) > 2 and parts[2].isdigit() else -1
        module_type = parts[-2] if len(parts) >= 2 else 'unknown'
        
        layers.append({
            'name': name,
            'shape': (d, k),
            'params': d * k,
            'type': module_type,  # q_proj, k_proj, etc.
            'layer_idx': layer_idx,
        })
    
    # Sort by size
    layers.sort(key=lambda x: x['params'])
    
    return model, layers


def test_layer_compression(target_weight, rank, epochs=100, lr=1e-3, device='cpu', verbose=False):
    """
    Test compression of a single weight matrix.
    
    Returns:
        loss: Final MSE
        rel_error: Relative error (%)
        compression_ratio: Storage savings
    """
    d, k = target_weight.shape
    
    # Convert target to float32 for training (handles BFloat16 models)
    target = target_weight.float().to(device)
    
    # Initialize LoRA in float32
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    # Compute metrics
    frob_norm_target = torch.norm(target).item()
    rel_error = (best_loss ** 0.5) / frob_norm_target * 100
    
    original_params = d * k
    compressed_params = rank * (d + k)
    compression_ratio = original_params / compressed_params
    
    return {
        'loss': best_loss,
        'rel_error': rel_error,
        'compression_ratio': compression_ratio,
        'original_params': original_params,
        'compressed_params': compressed_params,
    }


def rank_autoresearch_single_layer(target_weight, rank_candidates=[4, 8, 16, 32, 64], 
                                    epochs=100, lr=1e-3, device='cpu'):
    """
    Test different ranks on a single layer to find optimal.
    
    Returns best rank based on error vs compression tradeoff.
    """
    print(f"\nTesting {len(rank_candidates)} different ranks...")
    print(f"Target shape: {target_weight.shape}")
    print(f"Epochs per test: {epochs}")
    print()
    
    results = []
    
    for rank in rank_candidates:
        print(f"  Rank {rank:3d}: ", end='', flush=True)
        start = time.time()
        
        result = test_layer_compression(target_weight, rank, epochs, lr, device)
        result['rank'] = rank
        result['time'] = time.time() - start
        
        results.append(result)
        
        print(f"error={result['rel_error']:.2f}%, ratio={result['compression_ratio']:.1f}x, "
              f"time={result['time']:.1f}s")
    
    # Find best rank (lowest error with reasonable compression)
    # Score = error * (1 + 0.1 * log2(rank/4)) - penalize higher ranks slightly
    for r in results:
        r['score'] = r['rel_error'] * (1 + 0.1 * np.log2(max(r['rank'], 4) / 4))
    
    best = min(results, key=lambda x: x['score'])
    
    return results, best


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LoRA on single layer")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=None, 
                        help="Layer index (default: middle layer)")
    parser.add_argument("--module", default="q_proj",
                        choices=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                'gate_proj', 'up_proj', 'down_proj'],
                        help="Which module to test")
    parser.add_argument("--ranks", nargs='+', type=int, default=[4, 8, 16, 32],
                        help="Ranks to test")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--list-layers", action="store_true",
                        help="List all available layers and exit")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Single Layer LoRA Test")
    print("=" * 80)
    
    # Analyze model
    model, layers = analyze_model_layers(args.model, args.device)
    
    if args.list_layers:
        print(f"\nAll {len(layers)} layers in {args.model}:")
        print(f"{'Name':<60} {'Shape':<20} {'Params':<12} {'Type':<15}")
        print("-" * 110)
        for l in layers:
            print(f"{l['name']:<60} {str(l['shape']):<20} {l['params']:>10,}  {l['type']:<15}")
        return
    
    # Select layer
    if args.layer_idx is None:
        # Pick a middle layer (layer 15 of 30)
        middle_layers = [l for l in layers if l['layer_idx'] == 15 and l['type'] == args.module]
        if not middle_layers:
            # Fallback: pick first matching module type
            middle_layers = [l for l in layers if l['type'] == args.module]
        selected = middle_layers[0] if middle_layers else layers[len(layers)//2]
    else:
        matching = [l for l in layers if l['layer_idx'] == args.layer_idx and l['type'] == args.module]
        selected = matching[0] if matching else layers[0]
    
    print(f"\nSelected layer: {selected['name']}")
    print(f"Shape: {selected['shape']}")
    print(f"Parameters: {selected['params']:,}")
    print(f"Type: {selected['type']}")
    
    # Extract the weight
    weight_dict = dict(model.named_parameters())
    target_weight = weight_dict[selected['name']].data
    
    # Clean up model to save memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run autoresearch on ranks
    print("\n" + "=" * 80)
    print("Rank Autoresearch")
    print("=" * 80)
    
    results, best = rank_autoresearch_single_layer(
        target_weight,
        rank_candidates=args.ranks,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"\n{'Rank':<8} {'Error %':<12} {'Ratio':<10} {'Score':<10} {'Params':<15}")
    print("-" * 65)
    for r in results:
        marker = " <-- BEST" if r['rank'] == best['rank'] else ""
        print(f"{r['rank']:<8} {r['rel_error']:>10.2f}%  {r['compression_ratio']:>8.1f}x  "
              f"{r['score']:>8.2f}  {r['compressed_params']:>12,}{marker}")
    
    print(f"\n✓ Best rank: {best['rank']}")
    print(f"  Relative error: {best['rel_error']:.2f}%")
    print(f"  Compression: {best['compression_ratio']:.1f}x")
    print(f"  Storage: {best['compressed_params']:,} params (vs {best['original_params']:,} original)")
    
    # Save results
    output = {
        'layer_name': selected['name'],
        'layer_shape': selected['shape'],
        'all_results': results,
        'best_rank': best['rank'],
        'best_error': best['rel_error'],
        'best_compression': best['compression_ratio'],
    }
    
    os.makedirs('./single_layer_results', exist_ok=True)
    with open(f'./single_layer_results/{selected["name"].replace(".", "_")}.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to ./single_layer_results/")


if __name__ == "__main__":
    main()

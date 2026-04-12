#!/usr/bin/env python3
"""
FULL hyperparameter autoresearch for LoRA weight reproduction.

Searches:
- Rank (4, 8, 16, 32, 64, 128)
- Initial learning rate
- LR scheduler type (cosine, exponential, step, plateau)
- Optimizer (AdamW, SGD with momentum)
- Weight decay
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
from typing import Dict, Optional, Callable


def train_with_config(target_weight, rank, lr, epochs, optimizer_name='adamw',
                      scheduler_name=None, scheduler_params=None, weight_decay=0.0,
                      patience=50, device='cpu', verbose=False):
    """
    Train with full hyperparameter configuration.
    
    Returns dict with results.
    """
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    # Initialize LoRA
    A = nn.Parameter(torch.randn(rank, k, device=device, dtype=torch.float32) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device, dtype=torch.float32) * 0.01)
    
    # Optimizer
    params = [A, B]
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Scheduler
    scheduler = None
    if scheduler_name == 'cosine':
        T_max = scheduler_params.get('T_max', epochs)
        eta_min = scheduler_params.get('eta_min', lr * 0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name == 'exponential':
        gamma = scheduler_params.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name == 'step':
        step_size = scheduler_params.get('step_size', epochs // 3)
        gamma = scheduler_params.get('gamma', 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'plateau':
        factor = scheduler_params.get('factor', 0.5)
        patience_sch = scheduler_params.get('patience', 20)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience_sch
        )
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        
        if not torch.isfinite(loss):
            return {'error': None, 'diverged': True, 'epochs': epoch}
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        history.append(current_loss)
        
        # Scheduler step
        if scheduler is not None:
            if scheduler_name == 'plateau':
                scheduler.step(current_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if current_loss < best_loss - 1e-8:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    # Compute final metrics
    rel_error = (best_loss ** 0.5) / torch.norm(target).item() * 100
    compression = (d * k) / (rank * (d + k))
    
    return {
        'error': rel_error,
        'mse': best_loss,
        'epochs': len(history),
        'compression': compression,
        'diverged': False,
    }


def coarse_search(target_weight, device='cpu'):
    """Phase 1: Wide coarse search to find promising regions."""
    print("\n" + "="*70)
    print("PHASE 1: COARSE SEARCH")
    print("="*70)
    
    configs = [
        # (rank, lr, epochs, scheduler, sched_params, optimizer, wd)
        (4, 0.03, 200, None, {}, 'adamw', 0.0),
        (4, 0.1, 200, None, {}, 'adamw', 0.0),
        (4, 0.03, 300, 'cosine', {'T_max': 200}, 'adamw', 0.0),
        (4, 0.1, 300, 'cosine', {'T_max': 200}, 'adamw', 0.0),
        (8, 0.03, 200, None, {}, 'adamw', 0.0),
        (8, 0.1, 400, None, {}, 'adamw', 0.0),
        (8, 0.03, 300, 'cosine', {'T_max': 200}, 'adamw', 0.0),
        (16, 0.03, 300, None, {}, 'adamw', 0.0),
        (16, 0.03, 400, 'cosine', {'T_max': 300}, 'adamw', 0.0),
        (32, 0.03, 400, None, {}, 'adamw', 0.0),
    ]
    
    results = []
    for rank, lr, epochs, sched, sched_p, opt, wd in configs:
        desc = f"r={rank}, lr={lr:.0e}, ep={epochs}"
        if sched:
            desc += f", {sched}"
        print(f"  {desc:<35}: ", end='', flush=True)
        
        start = time.time()
        result = train_with_config(
            target_weight, rank, lr, epochs=epochs, optimizer_name=opt,
            scheduler_name=sched, scheduler_params=sched_p, weight_decay=wd,
            patience=min(50, epochs//4), device=device
        )
        elapsed = time.time() - start
        
        if result['diverged']:
            print(f"DIVERGED at epoch {result['epochs']}")
        else:
            print(f"error={result['error']:.4f}%, ratio={result['compression']:.1f}x, "
                  f"epochs={result['epochs']}, time={elapsed:.1f}s")
            result['config'] = {'rank': rank, 'lr': lr, 'scheduler': sched, 'opt': opt, 'wd': wd}
            result['time'] = elapsed
            results.append(result)
    
    # Find best
    if results:
        best = min(results, key=lambda x: x['error'])
        print(f"\n✓ Best from coarse: error={best['error']:.4f}%")
        print(f"  Config: {best['config']}")
        return best, results
    else:
        return None, []


def fine_search(target_weight, best_coarse, device='cpu'):
    """Phase 2: Fine search around best configuration."""
    print("\n" + "="*70)
    print("PHASE 2: FINE SEARCH")
    print("="*70)
    
    base = best_coarse['config']
    base_epochs = base.get('epochs', 300)
    
    # Generate variations
    variations = [
        # Epochs variations
        {**base, 'epochs': int(base_epochs * 0.7)},
        {**base, 'epochs': int(base_epochs * 1.5)},
        {**base, 'epochs': int(base_epochs * 2.0)},
        # LR variations
        {**base, 'lr': base['lr'] * 0.5},
        {**base, 'lr': base['lr'] * 1.4},
        # Weight decay variations
        {**base, 'wd': 0.01},
        {**base, 'wd': 0.001},
        # Different schedulers
        {**base, 'scheduler': 'exponential', 'sched_params': {'gamma': 0.98}},
        {**base, 'scheduler': 'step', 'sched_params': {'step_size': base_epochs//3, 'gamma': 0.5}},
        {**base, 'scheduler': 'plateau', 'sched_params': {'factor': 0.5, 'patience': 30}},
        # Optimizer variations
        {**base, 'opt': 'adam'},
        {**base, 'opt': 'sgd'},
    ]
    
    # Also test nearby ranks with adjusted epochs
    for rank in [max(4, base['rank'] // 2), base['rank'] * 2]:
        if rank <= 128:
            # Higher rank may need more epochs
            adjusted_epochs = int(base_epochs * (1 + 0.2 * (rank / base['rank'] - 1)))
            variations.append({**base, 'rank': rank, 'epochs': adjusted_epochs})
    
    results = []
    for var in variations[:15]:  # Limit to 15
        epochs = var.get('epochs', 400)
        desc = f"r={var['rank']}, lr={var['lr']:.2e}, ep={epochs}"
        if var.get('scheduler'):
            desc += f", {var['scheduler']}"
        if var.get('wd', 0) > 0:
            desc += f", wd={var['wd']}"
        print(f"  {desc:<40}: ", end='', flush=True)
        
        start = time.time()
        result = train_with_config(
            target_weight, var['rank'], var['lr'], epochs=epochs,
            optimizer_name=var.get('opt', 'adamw'),
            scheduler_name=var.get('scheduler'),
            scheduler_params=var.get('sched_params', {}),
            weight_decay=var.get('wd', 0.0),
            patience=min(80, epochs//5), device=device
        )
        elapsed = time.time() - start
        
        if result['diverged']:
            print(f"DIVERGED")
        else:
            print(f"error={result['error']:.4f}%, epochs={result['epochs']}")
            result['config'] = var
            result['time'] = elapsed
            results.append(result)
    
    if results:
        all_results = [best_coarse] + results
        best = min(all_results, key=lambda x: x['error'])
        print(f"\n✓ ABSOLUTE BEST: error={best['error']:.4f}%")
        print(f"  Config: {best['config']}")
        return best, all_results
    else:
        return best_coarse, [best_coarse]


def verify_and_benchmark(target_weight, best_config, device='cpu'):
    """Phase 3: Extended training to verify and benchmark."""
    print("\n" + "="*70)
    print("PHASE 3: VERIFICATION & BENCHMARK")
    print("="*70)
    
    cfg = best_config['config']
    # Use 3x the epochs from best config for verification, up to 3000
    verify_epochs = min(3000, cfg.get('epochs', 400) * 3)
    print(f"Extended training with best config for up to {verify_epochs} epochs...")
    print(f"Config: {cfg}")
    
    start = time.time()
    result = train_with_config(
        target_weight, cfg['rank'], cfg['lr'], epochs=verify_epochs,
        optimizer_name=cfg.get('opt', 'adamw'),
        scheduler_name=cfg.get('scheduler'),
        scheduler_params=cfg.get('sched_params', {}),
        weight_decay=cfg.get('wd', 0.0),
        patience=200, device=device, verbose=True
    )
    elapsed = time.time() - start
    
    print(f"\n✓ VERIFIED RESULTS:")
    print(f"  Final error: {result['error']:.6f}%")
    print(f"  MSE: {result['mse']:.6e}")
    print(f"  Epochs to converge: {result['epochs']}")
    print(f"  Compression: {result['compression']:.1f}x")
    print(f"  Training time: {elapsed:.1f}s")
    
    result['config'] = cfg
    result['time'] = elapsed
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Full hyperparameter autoresearch")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--layer-idx", type=int, default=15)
    parser.add_argument("--module", default="q_proj")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="./autoresearch_full.json")
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL HYPERPARAMETER AUTORESEARCH")
    print("="*70)
    print("Searching: rank, LR, scheduler, optimizer, weight decay")
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
    print(f"Target: {target_name}")
    print(f"Shape: {target_weight.shape}")
    
    del model
    gc.collect()
    
    total_start = time.time()
    
    # Run phases
    best_coarse, coarse_results = coarse_search(target_weight, args.device)
    
    if best_coarse is None:
        print("\n✗ All coarse configs diverged!")
        return
    
    best_fine, fine_results = fine_search(target_weight, best_coarse, args.device)
    final_result = verify_and_benchmark(target_weight, best_fine, args.device)
    
    total_time = time.time() - total_start
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    cfg = final_result['config']
    print(f"\n🏆 RECORD BREAKING CONFIGURATION:")
    print(f"  Rank: {cfg['rank']}")
    print(f"  Learning rate: {cfg['lr']:.6f}")
    print(f"  Epochs: {cfg.get('epochs', 'N/A')}")
    print(f"  Scheduler: {cfg.get('scheduler', 'none')}")
    print(f"  Optimizer: {cfg.get('opt', 'adamw')}")
    print(f"  Weight decay: {cfg.get('wd', 0.0)}")
    print(f"\n  Final error: {final_result['error']:.6f}% ⭐")
    print(f"  Compression: {final_result['compression']:.1f}x")
    print(f"  Training time: {final_result['time']:.1f}s")
    print(f"  Total search time: {total_time:.1f}s")
    
    # Save
    with open(args.output, 'w') as f:
        json.dump({
            'final_result': final_result,
            'all_results': coarse_results + fine_results,
            'total_time': total_time,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

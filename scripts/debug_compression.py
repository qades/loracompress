#!/usr/bin/env python3
"""
Debug why compression is stuck at ~35% error.
Tests various hypotheses:
1. Is the target weight properly normalized?
2. Is the optimization converging at all?
3. Are we hitting numerical issues?
4. Is the rank actually sufficient?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt


def compute_l1_error(W_approx, target):
    """L1 relative error: mean(|diff|) / mean(|target|) * 100"""
    l1_error = torch.mean(torch.abs(W_approx - target)).item()
    mean_abs_target = torch.mean(torch.abs(target)).item()
    return (l1_error / mean_abs_target * 100) if mean_abs_target > 0 else float('inf')


def svd_reconstruction_error(target, rank):
    """Compute best possible error via SVD (theoretical lower bound)."""
    U, S, Vh = torch.linalg.svd(target, full_matrices=False)
    # Reconstruct with top 'rank' singular values
    S_rank = torch.zeros_like(S)
    S_rank[:rank] = S[:rank]
    W_svd = (U @ torch.diag(S_rank) @ Vh)
    error = compute_l1_error(W_svd, target)
    
    # Also compute explained variance
    total_energy = torch.sum(S ** 2).item()
    rank_energy = torch.sum(S[:rank] ** 2).item()
    explained_variance = rank_energy / total_energy * 100
    
    return error, explained_variance, S.numpy()


def train_lora_detailed(target, rank, lr=0.01, epochs=2000, device='cpu', log_every=100):
    """Train with detailed logging."""
    d, k = target.shape
    target = target.float().to(device)
    
    # Initialize
    A = nn.Parameter(torch.randn(rank, k, device=device) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    losses = []
    l1_errors = []
    
    print(f"\nTraining rank={rank}, lr={lr}, epochs={epochs}")
    print(f"Target shape: {target.shape}, mean={target.mean():.6f}, std={target.std():.6f}")
    print(f"Target abs mean: {torch.mean(torch.abs(target)):.6f}")
    print("-" * 60)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        W_approx = B @ A
        loss = F.mse_loss(W_approx, target)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                l1_err = compute_l1_error(W_approx, target)
                l1_errors.append(l1_err)
                grad_norm_A = A.grad.norm().item() if A.grad is not None else 0
                grad_norm_B = B.grad.norm().item() if B.grad is not None else 0
                
                print(f"Epoch {epoch:4d}: MSE={loss.item():.6f}, L1={l1_err:.2f}%, "
                      f"|grad_A|={grad_norm_A:.6f}, |grad_B|={grad_norm_B:.6f}")
    
    return losses, l1_errors, A.detach(), B.detach()


def diagnose():
    """Run full diagnostic."""
    print("=" * 70)
    print("COMPRESSION DIAGNOSTIC")
    print("=" * 70)
    
    # Load model
    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        'HuggingFaceTB/SmolLM2-135M',
        trust_remote_code=True
    )
    
    # Find q_proj layer
    target_weight = None
    layer_name = None
    for name, param in model.named_parameters():
        if 'q_proj' in name and param.shape == (576, 576):
            target_weight = param.data
            layer_name = name
            break
    
    if target_weight is None:
        print("ERROR: Could not find 576x576 q_proj layer!")
        return
    
    print(f"Found layer: {layer_name}")
    print(f"Shape: {target_weight.shape}")
    print(f"Dtype: {target_weight.dtype}")
    print(f"Device: {target_weight.device}")
    
    # Analyze target weight
    print("\n" + "=" * 70)
    print("[2] TARGET WEIGHT ANALYSIS")
    print("=" * 70)
    
    w = target_weight.float()
    print(f"Mean: {w.mean():.6f}")
    print(f"Std: {w.std():.6f}")
    print(f"Min: {w.min():.6f}")
    print(f"Max: {w.max():.6f}")
    print(f"Mean absolute: {torch.mean(torch.abs(w)):.6f}")
    print(f"Frobenius norm: {torch.norm(w):.2f}")
    
    # Check for outliers / sparsity
    abs_w = torch.abs(w)
    print(f"\nSparsity analysis:")
    print(f"  Values < 0.01: {(abs_w < 0.01).float().mean() * 100:.1f}%")
    print(f"  Values < 0.1:  {(abs_w < 0.1).float().mean() * 100:.1f}%")
    print(f"  Values > 1.0:  {(abs_w > 1.0).float().mean() * 100:.1f}%")
    print(f"  Values > 10.0: {(abs_w > 10.0).float().mean() * 100:.1f}%")
    
    # Spectral analysis (SVD)
    print("\n" + "=" * 70)
    print("[3] SVD ANALYSIS (Theoretical Limits)")
    print("=" * 70)
    
    for rank in [4, 8, 16, 32, 64, 128]:
        err, var_explained, _ = svd_reconstruction_error(w, rank)
        print(f"Rank {rank:3d}: L1 error = {err:6.2f}%, Explained variance = {var_explained:.1f}%")
    
    # Test actual training
    print("\n" + "=" * 70)
    print("[4] TRAINING TESTS")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with different learning rates
    test_configs = [
        {'rank': 16, 'lr': 0.001, 'epochs': 500},
        {'rank': 16, 'lr': 0.01, 'epochs': 500},
        {'rank': 16, 'lr': 0.1, 'epochs': 500},
        {'rank': 64, 'lr': 0.01, 'epochs': 1000},
    ]
    
    results = []
    for config in test_configs:
        losses, errors, A_final, B_final = train_lora_detailed(
            w, **config, device=device, log_every=100
        )
        final_l1 = errors[-1] if errors else float('inf')
        results.append({
            'config': config,
            'final_l1': final_l1,
            'losses': losses,
            'errors': errors
        })
        print(f"\n  → Final L1 error: {final_l1:.2f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("[5] SUMMARY")
    print("=" * 70)
    
    print("\nSVD theoretical minimums vs achieved:")
    for r in results:
        rank = r['config']['rank']
        svd_err, _, _ = svd_reconstruction_error(w, rank)
        print(f"  Rank {rank}: SVD={svd_err:.2f}% | LoRA={r['final_l1']:.2f}% | Gap={r['final_l1']-svd_err:.2f}%")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("[6] DIAGNOSIS")
    print("=" * 70)
    
    # Check if we're close to SVD or far off
    rank16_svd, _, _ = svd_reconstruction_error(w, 16)
    rank16_achieved = next(r['final_l1'] for r in results if r['config']['rank'] == 16 and r['config']['lr'] == 0.01)
    
    if rank16_achieved > rank16_svd * 2:
        print(f"\n❌ PROBLEM: Rank-16 LoRA error ({rank16_achieved:.1f}%) is much worse than SVD ({rank16_svd:.1f}%)")
        print("   This indicates optimization is NOT converging properly.")
        print("\nPossible causes:")
        print("  1. Learning rate too low/high")
        print("  2. Initialization too small (0.01 std)")
        print("  3. Not enough epochs")
        print("  4. Numerical precision issues")
        print("  5. Weight matrix has unusual structure")
    else:
        print(f"\n✓ Optimization is working reasonably well.")
        print(f"  Rank-16: SVD={rank16_svd:.1f}%, LoRA={rank16_achieved:.1f}%")
    
    # Check weight matrix condition
    print("\n" + "=" * 70)
    print("[7] RECOMMENDATIONS")
    print("=" * 70)
    
    # Analyze what's needed for 5% error
    for rank in [64, 128, 256]:
        if rank <= min(w.shape):
            err, var_exp, _ = svd_reconstruction_error(w, rank)
            status = "✓" if err < 5.0 else "✗"
            print(f"{status} Rank {rank}: SVD error = {err:.2f}%")
    
    print("\nTo achieve <5% error:")
    err_64, _, _ = svd_reconstruction_error(w, 64)
    if err_64 > 5.0:
        print(f"  ⚠️  Even rank-64 SVD only achieves {err_64:.2f}%!")
        print(f"     This weight matrix is inherently hard to compress.")
        print(f"     You need rank > 64 OR relax your target.")
    else:
        print(f"  Rank-64 SVD can achieve {err_64:.2f}%.")
        print(f"  If LoRA can't match this, check optimization hyperparameters.")


if __name__ == '__main__':
    diagnose()

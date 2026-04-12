#!/usr/bin/env python3
"""
Autoresearch with L1 error metric to achieve <5% error target.
Benchmark: Q4_K_M has ~3.4% loss for 70-75% size reduction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import json
import time
import os
from transformers import AutoModelForCausalLM


def compute_l1_error(W_approx, target):
    """L1 relative error: mean(|diff|) / mean(|target|) * 100"""
    l1_error = torch.mean(torch.abs(W_approx - target)).item()
    mean_abs_target = torch.mean(torch.abs(target)).item()
    return (l1_error / mean_abs_target * 100) if mean_abs_target > 0 else float('inf')


def train_lora_layer_advanced(target_weight, rank, lr, epochs, device='cpu', patience=100,
                               scheduler_type=None, warmup_epochs=0,
                               noise_mode='none', noise_std=0.0, noise_every=50,
                               adaptive_noise=True, detect_traps=True):
    """Advanced LoRA training with noise injection and trap detection.
    
    Args:
        noise_mode: 'none', 'parameter' (perturb weights), 'gradient' (Langevin), 
                    'weight_average' (SWA-style)
        noise_std: Noise standard deviation
        noise_every: Add noise every N epochs  
        adaptive_noise: Increase noise when stuck, decrease when improving
        detect_traps: Detect and escape local minima/traps
    """
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    
    A = nn.Parameter(torch.randn(rank, k, device=device) * 0.01)
    B = nn.Parameter(torch.randn(d, rank, device=device) * 0.01)
    optimizer = torch.optim.AdamW([A, B], lr=lr)
    
    # Setup scheduler
    scheduler = None
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    elif scheduler_type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    
    # State tracking
    best_loss = float('inf')
    best_A, best_B = None, None
    best_epoch = 0
    epochs_no_improve = 0
    current_noise_std = noise_std
    
    # Trap detection state
    loss_history = []
    trap_detected = False
    escape_attempts = 0
    max_escapes = 3
    
    # Weight averaging (SWA) state
    if noise_mode == 'weight_average':
        swa_A = torch.zeros_like(A)
        swa_B = torch.zeros_like(B)
        swa_count = 0
        swa_start = epochs // 3  # Start averaging after 1/3 of training
    
    for epoch in range(epochs):
        # Warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / warmup_epochs
        
        # Trap detection: check if stuck in plateau
        if detect_traps and len(loss_history) >= 50:
            recent_losses = loss_history[-50:]
            loss_variance = torch.tensor(recent_losses).var().item()
            
            # Low variance + no improvement = plateau trap
            if loss_variance < 1e-8 and epochs_no_improve > 50 and escape_attempts < max_escapes:
                trap_detected = True
                escape_attempts += 1
                print(f"    [Epoch {epoch}] Trap detected (plateau), escaping (attempt {escape_attempts}/{max_escapes})...")
                
                # Escape strategy: reset to best + large perturbation
                with torch.no_grad():
                    A.copy_(best_A + torch.randn_like(best_A) * 0.1)
                    B.copy_(best_B + torch.randn_like(best_B) * 0.1)
                
                # Boost learning rate temporarily
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 3
                
                epochs_no_improve = 0
                loss_history = []
        
        # Apply noise based on mode
        if noise_mode != 'none' and epoch > 0 and epoch % noise_every == 0:
            with torch.no_grad():
                if noise_mode == 'parameter':
                    # Perturb parameters directly
                    A.add_(torch.randn_like(A) * current_noise_std)
                    B.add_(torch.randn_like(B) * current_noise_std)
                    
                elif noise_mode == 'weight_average':
                    # SWA: update running average (done after optimizer step)
                    pass
            
            # Adaptive noise: increase if stuck, decrease if improving
            if adaptive_noise:
                if epochs_no_improve > 20:
                    current_noise_std = min(current_noise_std * 1.2, noise_std * 5)
                else:
                    current_noise_std = max(current_noise_std * 0.9, noise_std * 0.1)
        
        # Forward pass
        optimizer.zero_grad()
        W_approx = torch.matmul(B, A)
        loss = F.mse_loss(W_approx, target)
        
        if not torch.isfinite(loss):
            # Divergence detected - recover from best checkpoint
            if best_A is not None:
                print(f"    [Epoch {epoch}] Divergence detected, restoring best checkpoint...")
                with torch.no_grad():
                    A.copy_(best_A)
                    B.copy_(best_B)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * 0.5  # Reduce LR
                continue
            else:
                return float('inf'), 0
        
        # Backward pass
        loss.backward()
        
        # Gradient noise (Langevin dynamics)
        if noise_mode == 'gradient' and epoch % noise_every == 0:
            with torch.no_grad():
                A.grad.add_(torch.randn_like(A.grad) * current_noise_std)
                B.grad.add_(torch.randn_like(B.grad) * current_noise_std)
        
        optimizer.step()
        
        # Weight averaging (SWA)
        if noise_mode == 'weight_average' and epoch >= swa_start:
            with torch.no_grad():
                swa_A = (swa_A * swa_count + A) / (swa_count + 1)
                swa_B = (swa_B * swa_count + B) / (swa_count + 1)
                swa_count += 1
        
        # Step scheduler after warmup
        if scheduler and (warmup_epochs == 0 or epoch >= warmup_epochs):
            scheduler.step()
        
        # Track best
        current = loss.item()
        loss_history.append(current)
        
        if current < best_loss - 1e-9:
            best_loss = current
            best_A = A.detach().clone()
            best_B = B.detach().clone()
            best_epoch = epoch
            epochs_no_improve = 0
            trap_detected = False
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epoch >= 200 and epochs_no_improve >= patience:
            break
    
    # Use SWA weights if that mode was active
    if noise_mode == 'weight_average' and swa_count > 0:
        final_A, final_B = swa_A, swa_B
    else:
        final_A, final_B = best_A, best_B
    
    # Compute final L1 error
    with torch.no_grad():
        W_best = torch.matmul(final_B, final_A)
        l1_error = compute_l1_error(W_best, target)
    
    return l1_error, epoch + 1


# Keep simple version for backward compatibility
def train_lora_layer(target_weight, rank, lr, epochs, device='cpu', patience=100,
                     scheduler_type=None, warmup_epochs=0,
                     noise_std=0.0, noise_every=50, noise_decay=0.9):
    """Simple training (backward compatible)."""
    noise_mode = 'parameter' if noise_std > 0 else 'none'
    return train_lora_layer_advanced(
        target_weight, rank, lr, epochs, device, patience,
        scheduler_type, warmup_epochs,
        noise_mode=noise_mode, noise_std=noise_std, noise_every=noise_every
    )


def objective(trial, target_weight, target_quality=5.0, noise_std=0.0, noise_every=50,
              advanced_mode=False):
    """Optuna objective: minimize error + penalize compute effort"""
    rank = trial.suggest_int('rank', 4, 64, log=True)
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    epochs = trial.suggest_int('epochs', 200, 2000, log=True)
    
    # Scheduler options
    scheduler_type = trial.suggest_categorical('scheduler', [None, 'cosine', 'linear', 'exponential'])
    warmup = trial.suggest_int('warmup_epochs', 0, 100) if scheduler_type else 0
    
    # Convert 'none' string back to None for categorical
    if scheduler_type == 'none':
        scheduler_type = None
    
    # Advanced mode options
    if advanced_mode:
        noise_mode = trial.suggest_categorical('noise_mode', ['none', 'parameter', 'gradient', 'weight_average'])
        adaptive_noise = trial.suggest_categorical('adaptive_noise', [True, False])
        detect_traps = trial.suggest_categorical('detect_traps', [True, False])
        noise_std_adv = trial.suggest_float('noise_std', 1e-5, 0.01, log=True)
    else:
        noise_mode = 'parameter' if noise_std > 0 else 'none'
        adaptive_noise = True
        detect_traps = True
        noise_std_adv = noise_std
    
    start_time = time.time()
    
    if advanced_mode:
        error, actual_epochs = train_lora_layer_advanced(
            target_weight, rank, lr, epochs,
            scheduler_type=scheduler_type, warmup_epochs=warmup,
            noise_mode=noise_mode, noise_std=noise_std_adv, noise_every=noise_every,
            adaptive_noise=adaptive_noise, detect_traps=detect_traps
        )
    else:
        error, actual_epochs = train_lora_layer(
            target_weight, rank, lr, epochs, 
            scheduler_type=scheduler_type, warmup_epochs=warmup,
            noise_std=noise_std, noise_every=noise_every
        )
    
    train_time = time.time() - start_time
    
    if not torch.isfinite(torch.tensor(error)):
        return float('inf')
    
    # Compression ratio: (d*k) / (rank*(d+k))
    d, k = target_weight.shape
    compression = (d * k) / (rank * (d + k))
    size_reduction = 1 - 1/compression
    
    # Calculate effort metrics
    # FLOPs ~ rank * (d + k) * actual_epochs (matmul cost per epoch)
    compute_effort = rank * (d + k) * actual_epochs
    
    # Multi-objective with EFFICIENCY bonus
    if error <= target_quality:
        # Good quality: balance compression vs compute effort
        # Prefer: high compression, low rank, few epochs, fast wall time
        quality_score = error / target_quality  # 0 to 1
        
        # Efficiency bonus: lower rank/epochs/time = better
        # Normalize by typical values (rank=32, epochs=1000, time=10s)
        rank_efficiency = 32 / rank  # Higher is better (lower rank)
        epoch_efficiency = 1000 / actual_epochs  # Higher is better (fewer epochs)
        time_efficiency = 10.0 / max(train_time, 0.1)  # Higher is better (faster)
        
        # Combined score: prefer good quality + efficient compute
        # Compression matters most, then quality, then efficiency
        score = -compression * 10  # Maximize compression
        score += quality_score * 5  # Penalize worse quality
        score -= rank_efficiency * 0.5  # Bonus for low rank
        score -= epoch_efficiency * 0.3  # Bonus for fast convergence
        
        trial.set_user_attr('efficiency_bonus', 
                           rank_efficiency + epoch_efficiency)
    else:
        # Bad quality: heavy penalty proportional to how far over
        score = 1000 + (error - target_quality) * 100
        trial.set_user_attr('efficiency_bonus', 0)
    
    trial.set_user_attr('error', error)
    trial.set_user_attr('compression', compression)
    trial.set_user_attr('size_reduction', size_reduction)
    trial.set_user_attr('actual_epochs', actual_epochs)
    trial.set_user_attr('train_time', train_time)
    trial.set_user_attr('compute_effort', compute_effort)
    trial.set_user_attr('scheduler', scheduler_type)
    trial.set_user_attr('warmup', warmup)
    
    # Store advanced mode parameters if used
    if advanced_mode:
        trial.set_user_attr('noise_mode', noise_mode if advanced_mode else 'none')
        trial.set_user_attr('adaptive_noise', adaptive_noise if advanced_mode else False)
        trial.set_user_attr('detect_traps', detect_traps if advanced_mode else False)
    
    return score


def load_previous_results(target_quality):
    """Load results from less strict runs to warm-start search."""
    previous_files = []
    if target_quality <= 3.0 and os.path.exists('autoresearch_l1_results.json'):
        # Loading 5% results to help 3% search
        previous_files.append('autoresearch_l1_results.json')
    
    good_configs = []
    for filepath in previous_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            # Get configs that were decent quality
            for r in data.get('all_results', []):
                if r['error'] < target_quality * 1.5:  # Within 50% of target
                    good_configs.append(r)
            print(f"  Loaded {len(good_configs)} good configs from {filepath}")
        except Exception as e:
            print(f"  Could not load {filepath}: {e}")
    
    return good_configs


def run_autoresearch(layer_shape=(576, 576), n_trials=50, target_quality=5.0,
                     noise_std=0.0, noise_every=50, advanced_mode=False):
    """Run hyperparameter search for target layer shape."""
    print("="*70)
    print("L1 Quality Autoresearch")
    print("="*70)
    print(f"Target shape: {layer_shape}")
    print(f"Target quality: <{target_quality}% L1 error")
    print(f"Benchmark: Q4_K_M = 3.4% error, 70-75% size reduction")
    if noise_std > 0:
        print(f"Experimental: noise_std={noise_std}, noise_every={noise_every}")
    if advanced_mode:
        print("ADVANCED MODE: trap detection + adaptive noise + multiple noise strategies")
    print("="*70)
    
    # Load previous results for warm-start
    print("\nChecking for previous results...")
    warm_start_configs = load_previous_results(target_quality)
    
    # Get real weight from model
    print("\nLoading model to extract sample weight...")
    model = AutoModelForCausalLM.from_pretrained(
        'HuggingFaceTB/SmolLM2-135M', trust_remote_code=True
    )
    
    # Find a q_proj layer with target shape
    target_weight = None
    for name, param in model.named_parameters():
        if 'q_proj' in name and param.shape == layer_shape:
            target_weight = param.data
            print(f"Using real weight: {name} {param.shape}")
            break
    
    if target_weight is None:
        print(f"No layer with shape {layer_shape} found, using synthetic")
        torch.manual_seed(42)
        target_weight = torch.randn(layer_shape) * 0.1
    
    # Run Optuna study
    print(f"\nRunning {n_trials} trials...")
    
    # Use different study name for different targets to allow parallel results
    study_name = f"l1_quality_{target_quality}pct"
    storage = f"sqlite:///optuna_{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True
    )
    
    # Enqueue warm-start configs from previous runs
    if warm_start_configs:
        print(f"  Enqueueing {min(len(warm_start_configs), 10)} warm-start trials...")
        for config in warm_start_configs[:10]:  # Top 10
            sched = config.get('scheduler')
            # Handle None vs 'none'
            if sched is None:
                sched = 'none'
            params = {
                'rank': config['rank'],
                'lr': config['lr'],
                'epochs': config['epochs'],
                'scheduler': sched,
                'warmup_epochs': config.get('warmup', 0) if sched != 'none' else 0,
            }
            study.enqueue_trial(params, skip_if_exists=True)
    
    def wrapped_objective(trial):
        return objective(trial, target_weight, target_quality, noise_std, noise_every, advanced_mode)
    
    study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get all results
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            result = {
                'rank': trial.params.get('rank'),
                'lr': trial.params.get('lr'),
                'epochs': trial.params.get('epochs'),
                'error': trial.user_attrs.get('error'),
                'compression': trial.user_attrs.get('compression'),
                'size_reduction': trial.user_attrs.get('size_reduction'),
                'actual_epochs': trial.user_attrs.get('actual_epochs'),
                'score': trial.value,
                'scheduler': trial.user_attrs.get('scheduler'),
            }
            # Add advanced mode fields if present
            if 'noise_mode' in trial.user_attrs:
                result['noise_mode'] = trial.user_attrs.get('noise_mode')
                result['adaptive_noise'] = trial.user_attrs.get('adaptive_noise')
                result['detect_traps'] = trial.user_attrs.get('detect_traps')
            results.append(result)
    
    # Sort by error (best quality first)
    results.sort(key=lambda x: x['error'])
    
    # Sort by composite score: quality + efficiency
    # Best = lowest error, highest compression, lowest effort
    def composite_score(r):
        quality = r['error'] / target_quality
        compression_bonus = -r['compression'] / 50  # Normalize
        effort_penalty = (r['rank'] / 32) * (r['actual_epochs'] / 1000)  # Lower is better
        return quality + compression_bonus + effort_penalty
    
    results_by_efficiency = sorted(results, key=composite_score)
    
    # Display results
    print("\n" + "="*70)
    print("TOP 10 BY QUALITY (error)")
    print("="*70)
    print(f"{'Rank':>6} {'LR':>8} {'Sched':>10} {'Epochs':>8} {'Time':>8} {'Error%':>8} {'Compr':>8}")
    print("-"*70)
    
    for r in results[:10]:
        if r['error'] <= target_quality:
            marker = "✓"
        else:
            marker = "✗"
        sched = r.get('scheduler') or 'none'
        time_str = f"{r.get('train_time', 0):.1f}s"
        print(f"{r['rank']:>6} {r['lr']:>8.4f} {sched:>10} {r['actual_epochs']:>8} {time_str:>8} {r['error']:>7.2f}{marker} {r['compression']:>7.1f}x")
    
    print("\n" + "="*70)
    print("TOP 10 BY EFFICIENCY (quality + speed)")
    print("="*70)
    print(f"{'Rank':>6} {'LR':>8} {'Sched':>10} {'Epochs':>8} {'Effort':>10} {'Error%':>8} {'Compr':>8}")
    print("-"*70)
    
    for r in results_by_efficiency[:10]:
        if r['error'] <= target_quality:
            marker = "✓"
        else:
            marker = "✗"
        sched = r.get('scheduler') or 'none'
        effort = r.get('compute_effort', r['rank'] * r['actual_epochs']) / 1e6
        print(f"{r['rank']:>6} {r['lr']:>8.4f} {sched:>10} {r['actual_epochs']:>8} {effort:>9.2f}M {r['error']:>7.2f}{marker} {r['compression']:>7.1f}x")
    
    # Find best configs under different thresholds
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATIONS")
    print("="*70)
    
    for threshold in [3.0, 4.0, 5.0]:
        good = [r for r in results if r['error'] <= threshold]
        if good:
            # Best compression
            best_compression = max(good, key=lambda x: x['compression'])
            sched_str = f", Scheduler={best_compression.get('scheduler') or 'none'}"
            print(f"\nFor <{threshold}% error (BEST COMPRESSION):")
            print(f"  Rank={best_compression['rank']}, LR={best_compression['lr']:.4f}, Epochs={best_compression['epochs']}{sched_str}")
            print(f"  → Error={best_compression['error']:.2f}%, Compression={best_compression['compression']:.1f}x")
            
            # Most efficient (low rank + few epochs)
            def efficiency_score(r):
                return r['rank'] * r['actual_epochs']  # Lower is better
            most_efficient = min(good, key=efficiency_score)
            eff_sched_str = f", Scheduler={most_efficient.get('scheduler') or 'none'}"
            print(f"\nFor <{threshold}% error (MOST EFFICIENT):")
            print(f"  Rank={most_efficient['rank']}, LR={most_efficient['lr']:.4f}, Epochs={most_efficient['epochs']}{eff_sched_str}")
            print(f"  → Error={most_efficient['error']:.2f}%, Time={most_efficient.get('train_time', 0):.1f}s, Effort={efficiency_score(most_efficient)/1e6:.1f}M")
    
    # Save results
    output = {
        'target_shape': layer_shape,
        'target_quality': target_quality,
        'n_trials': n_trials,
        'best_config': {
            'rank': study.best_params.get('rank'),
            'lr': study.best_params.get('lr'),
            'epochs': study.best_params.get('epochs'),
            'error': study.best_trial.user_attrs.get('error'),
            'compression': study.best_trial.user_attrs.get('compression'),
        },
        'all_results': results,
    }
    
    with open('autoresearch_l1_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to autoresearch_l1_results.json")
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--target-quality', type=float, default=5.0,
                        help='Target L1 error % (default: 5.0)')
    parser.add_argument('--shape', type=int, nargs=2, default=[576, 576],
                        help='Layer shape (default: 576 576)')
    # Noise parameters (experimental)
    parser.add_argument('--noise-std', type=float, default=0.0,
                        help='Add Gaussian noise to params every N epochs (default: 0 = off)')
    parser.add_argument('--noise-every', type=int, default=50,
                        help='Add noise every N epochs (default: 50)')
    # Advanced mode
    parser.add_argument('--advanced', action='store_true',
                        help='Enable advanced mode: trap detection, adaptive noise, multiple strategies')
    args = parser.parse_args()
    
    if args.noise_std > 0:
        print(f"\n⚠️  EXPERIMENTAL: Adding noise std={args.noise_std} every {args.noise_every} epochs")
        print("   This may help escape local minima but could also slow convergence\n")
    
    if args.advanced:
        print("\n🔬 ADVANCED MODE ENABLED")
        print("   Features: trap detection, adaptive noise, SWA, gradient noise\n")
    
    run_autoresearch(
        layer_shape=tuple(args.shape),
        n_trials=args.n_trials,
        target_quality=args.target_quality,
        noise_std=args.noise_std,
        noise_every=args.noise_every,
        advanced_mode=args.advanced
    )


if __name__ == '__main__':
    main()

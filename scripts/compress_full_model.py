#!/usr/bin/env python3
"""
Compress full model with layer-adaptive ranks.

Uses optimal rank per layer type based on our analysis.
"""
import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_optimal_rank(shape, module_type):
    """Get optimal rank based on layer type."""
    d, k = shape
    min_dim = min(d, k)
    
    if module_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        pct = 0.028  # Attention: 2.8%
    elif module_type in ['gate_proj', 'up_proj', 'down_proj']:
        pct = 0.007  # MLP: 0.7%
    else:
        pct = 0.014
    
    rank = max(4, int(min_dim * pct))
    
    # Round to nice numbers
    if rank <= 8:
        rank = max(4, rank)
    elif rank <= 16:
        rank = 16
    elif rank <= 32:
        rank = 32
    else:
        rank = min(64, (rank // 16) * 16)
    
    return min(rank, min_dim)


def compute_l1_error(W_approx, target):
    """Compute L1 relative error: mean(|diff|) / mean(|target|) * 100"""
    l1_error = torch.mean(torch.abs(W_approx - target)).item()
    mean_abs_target = torch.mean(torch.abs(target)).item()
    return (l1_error / mean_abs_target * 100) if mean_abs_target > 0 else float('inf')


def verify_layer_quality(layer_file, target_weight, device='cpu'):
    """Load saved layer and compute actual L1 error."""
    try:
        layer_data = torch.load(layer_file, map_location=device)
        if layer_data.get('type') != 'lora':
            return None, None
        
        A = layer_data['A']
        B = layer_data['B']
        W_approx = torch.matmul(B, A)
        
        target = target_weight.float().to(device)
        actual_error = compute_l1_error(W_approx, target)
        actual_rank = layer_data.get('rank', A.shape[0])
        return actual_error, actual_rank
    except Exception as e:
        print(f"  Warning: Could not verify layer: {e}")
        return None, None


def compress_weight_matrix(target_weight, rank, lr=0.03, max_epochs=1000, 
                           patience=150, min_epochs=100, target_error=0.15,
                           max_retries=1, rank_increase_on_fail=True, device='cpu'):
    """
    Compress a single weight matrix using LoRA.
    
    Args:
        min_epochs: Minimum epochs (reduced from 200 to 100 for faster convergence)
        target_error: Target error threshold
        max_retries: Number of retries with rank increase if target not met
        rank_increase_on_fail: If True, increase rank instead of just epochs on failure
    """
    d, k = target_weight.shape
    target = target_weight.float().to(device)
    current_rank = rank
    
    for attempt in range(max_retries + 1):
        A = nn.Parameter(torch.randn(current_rank, k, device=device, dtype=torch.float32) * 0.01)
        B = nn.Parameter(torch.randn(d, current_rank, device=device, dtype=torch.float32) * 0.01)
        optimizer = torch.optim.AdamW([A, B], lr=lr)
        
        best_loss = float('inf')
        best_A, best_B = None, None
        epochs_without_improvement = 0
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            W_approx = torch.matmul(B, A)
            loss = F.mse_loss(W_approx, target)
            
            if not torch.isfinite(loss):
                return None, None, None, True, current_rank
            
            loss.backward()
            optimizer.step()
            
            current = loss.item()
            if current < best_loss - 1e-8:
                best_loss = current
                best_A = A.detach().clone()
                best_B = B.detach().clone()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epoch >= min_epochs and epochs_without_improvement >= patience:
                break
        
        # Compute L1 relative error using shared function
        with torch.no_grad():
            W_best = torch.matmul(best_B, best_A)
            final_error = compute_l1_error(W_best, target)
        
        # Check if target met
        if final_error <= target_error or attempt >= max_retries:
            return best_A, best_B, final_error, False, current_rank
        
        # Strategy: increase rank instead of just epochs (more effective)
        if rank_increase_on_fail and current_rank < min(d, k) // 2:
            old_rank = current_rank
            current_rank = min(current_rank + 4, min(d, k) // 2)
            print(f"  ! Error {final_error:.4f}% > target {target_error:.4f}%, increasing rank {old_rank} -> {current_rank}")
        else:
            print(f"  ! Error {final_error:.4f}% > target {target_error:.4f}%, retry {attempt+1}/{max_retries}")
    
    return best_A, best_B, final_error, False, current_rank


def compress_model(model_name="HuggingFaceTB/SmolLM2-135M", device='cpu', 
                   output_dir='./compressed_model',
                   resume=False, max_error=None, min_compression=None,
                   only_modules=None, exclude_modules=None,
                   limit_layers=None, dry_run=False):
    """Compress entire model layer by layer with incremental support.
    
    Args:
        resume: If True, load existing metadata and skip already-compressed good layers
        max_error: If set, re-compress layers with error > this
        min_compression: If set, re-compress layers with compression < this
        only_modules: List of module types to process (e.g., ['q_proj', 'k_proj'])
        exclude_modules: List of module types to skip
        limit_layers: Max number of layers to process (for testing)
        dry_run: Show what would be done without doing it
    """
    print("="*70)
    print("Full Model Compression" + (" (RESUME MODE)" if resume else ""))
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    if resume:
        print(f"Resume: enabled")
        if max_error:
            print(f"Re-compress if error > {max_error}%")
        if min_compression:
            print(f"Re-compress if compression < {min_compression}x")
    if only_modules:
        print(f"Only modules: {only_modules}")
    if exclude_modules:
        print(f"Exclude modules: {exclude_modules}")
    if limit_layers:
        print(f"Limit: {limit_layers} layers")
    if dry_run:
        print("*** DRY RUN - No actual compression ***")
    print("="*70)
    
    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    os.makedirs(output_dir, exist_ok=True)
    layers_dir = os.path.join(output_dir, 'layers')
    os.makedirs(layers_dir, exist_ok=True)
    
    # Load existing metadata if resuming
    existing_metadata = {}
    if resume:
        metadata_path = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            print(f"\nLoaded existing metadata: {len(existing_metadata.get('layer_metadata', {}))} layers")
    
    if not dry_run:
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Get model config
        config = model.config
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    # Determine which modules to process
    target_modules = only_modules if only_modules else ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                         'gate_proj', 'up_proj', 'down_proj']
    if exclude_modules:
        target_modules = [m for m in target_modules if m not in exclude_modules]
    
    compressed_layers = {}
    layer_metadata = {}
    total_original = 0
    total_compressed = 0
    layers_skipped = 0
    layers_recompressed = 0
    start_time = time.time()
    
    # Collect all layers to process first
    layers_to_process = []
    for name, param in model.named_parameters():
        if not any(x in name for x in target_modules) or 'weight' not in name:
            continue
        layers_to_process.append((name, param))
    
    if limit_layers:
        layers_to_process = layers_to_process[:limit_layers]
    
    print(f"\nFound {len(layers_to_process)} layers to evaluate...")
    
    for name, param in layers_to_process:
        shape = param.shape
        d, k = shape
        module_type = name.split('.')[-2]
        rank = get_optimal_rank(shape, module_type)
        
        # Check if we should skip this layer (resume mode)
        existing_layer = existing_metadata.get('layer_metadata', {}).get(name)
        skip_reason = None
        
        if resume and existing_layer:
            # Verify actual quality by loading and computing error
            layer_file = existing_layer.get('file')
            actual_error = None
            actual_rank = None
            
            if layer_file and os.path.exists(layer_file):
                actual_error, actual_rank = verify_layer_quality(layer_file, param.data, device)
            
            # Fall back to stored error if verification fails
            if actual_error is None:
                actual_error = existing_layer.get('error', float('inf'))
                actual_rank = existing_layer.get('rank', rank)
            
            existing_compression = existing_layer.get('compression', 1.0)
            
            # Check quality thresholds using ACTUAL verified error
            if existing_layer.get('type') == 'lora':
                needs_recompress = False
                
                if max_error is not None and actual_error > max_error:
                    needs_recompress = True
                    skip_reason = None
                    layers_recompressed += 1
                elif min_compression is not None and existing_compression < min_compression:
                    needs_recompress = True
                    skip_reason = None
                    layers_recompressed += 1
                
                if not needs_recompress:
                    skip_reason = f"quality_ok(error={actual_error:.2f}%, compression={existing_compression:.1f}x)"
            elif existing_layer.get('type') == 'failed':
                skip_reason = None  # Always retry failed layers
            else:
                skip_reason = "existing"
            
            if skip_reason:
                layers_skipped += 1
                print(f"\nSkipping: {name} ({skip_reason})")
                # Update metadata with verified error
                existing_layer['error'] = actual_error
                existing_layer['rank'] = actual_rank
                compressed_layers[name] = existing_layer
                layer_metadata[name] = existing_layer
                if existing_layer.get('type') == 'lora':
                    total_original += d * k
                    total_compressed += actual_rank * (d + k)
                continue
            else:
                print(f"\nRe-compressing: {name} (actual_error={actual_error:.2f}% > threshold)")
        else:
            print(f"\nCompressing: {name}")
        
        print(f"  Shape: {d}×{k}, Rank: {rank}")
        
        if dry_run:
            continue
        
        # Adaptive settings based on layer type
        # Using L1 relative error (mean|diff|/mean|target|) * 100
        # < 5% is good, < 2% is excellent for these ranks
        if module_type in ['k_proj', 'v_proj']:
            # Small layers (192×576) - hard to compress
            base_target_error = 8.0  # L1: ~8% (was 0.30% RMSE)
            max_retries = 2
            use_epochs = 500
        elif module_type in ['q_proj', 'o_proj']:
            base_target_error = 5.0  # L1: ~5% (was 0.18% RMSE)
            max_retries = 1
            use_epochs = 600
        else:
            base_target_error = 3.0  # L1: ~3% (was 0.12% RMSE)
            max_retries = 1
            use_epochs = 350
        
        # If re-compressing for quality improvement, use stricter target
        # (whichever is lower: base target or the threshold that triggered re-compression)
        if resume and (max_error is not None):
            target_error = min(base_target_error, max_error)
        else:
            target_error = base_target_error
        
        A, B, error, diverged, final_rank = compress_weight_matrix(
            param.data, rank, max_epochs=use_epochs,
            target_error=target_error, max_retries=max_retries,
            rank_increase_on_fail=True, device=device
        )
        
        # Create safe filename
        safe_name = name.replace('.', '_')
        layer_file = os.path.join(layers_dir, f"{safe_name}.pt")
        
        if diverged:
            print(f"  ✗ FAILED - will use original weights")
            layer_info = {
                'type': 'failed',
                'name': name,
                'shape': list(shape),
                'rank': rank,
                'error': None,
                'compression': 1.0,
                'module_type': module_type,
            }
        else:
            compression = (d * k) / (final_rank * (d + k))
            rank_increased = final_rank != rank
            rank_str = f" (rank {rank}→{final_rank})" if rank_increased else ""
            print(f"  ✓ Error: {error:.4f}%, Compression: {compression:.1f}x{rank_str}")
            
            # Save individual layer file
            layer_data = {
                'type': 'lora',
                'A': A.cpu(),
                'B': B.cpu(),
                'rank': final_rank,
                'requested_rank': rank,
                'shape': list(shape),
                'error': error,
                'compression': compression,
                'module_type': module_type,
            }
            torch.save(layer_data, layer_file)
            
            layer_info = {
                'type': 'lora',
                'name': name,
                'shape': list(shape),
                'module_type': module_type,
                'rank': final_rank,
                'requested_rank': rank,
                'error': error,
                'compression': compression,
                'target_error': target_error,
                'file': layer_file,
            }
            total_original += d * k
            total_compressed += final_rank * (d + k)
        
        compressed_layers[name] = layer_info
        layer_metadata[name] = layer_info
        
        # Save incremental metadata after each layer
        elapsed = time.time() - start_time
        incremental_metadata = {
            'model_name': model_name,
            'layers_completed': len(compressed_layers),
            'layers_skipped': layers_skipped,
            'layers_recompressed': layers_recompressed,
            'total_original_params': total_original,
            'total_compressed_params': total_compressed,
            'compression_ratio': total_original / total_compressed if total_compressed > 0 else 1.0,
            'compression_time': elapsed,
            'layer_metadata': layer_metadata,
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(incremental_metadata, f, indent=2)
    
    if dry_run:
        print("\n*** DRY RUN COMPLETE - No files modified ***")
        return None
    
    # Also save combined file for convenience
    print("\nSaving combined compressed weights...")
    combined_data = {}
    for name, info in compressed_layers.items():
        if info['type'] == 'lora' and 'file' in info:
            layer_data = torch.load(info['file'])
            combined_data[name] = layer_data
    torch.save(combined_data, os.path.join(output_dir, 'compressed_weights.pt'))
    
    # Final metadata
    elapsed = time.time() - start_time
    final_metadata = {
        'model_name': model_name,
        'num_layers': len(compressed_layers),
        'layers_skipped': layers_skipped,
        'layers_recompressed': layers_recompressed,
        'total_original_params': total_original,
        'total_compressed_params': total_compressed,
        'compression_ratio': total_original / total_compressed if total_compressed > 0 else 1.0,
        'compression_time': elapsed,
        'layer_metadata': layer_metadata,
        'status': 'complete',
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(final_metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("Compression Complete!")
    print("="*70)
    if resume:
        print(f"Layers skipped (good quality): {layers_skipped}")
        print(f"Layers re-compressed: {layers_recompressed}")
    print(f"Original: {total_original/1e6:.1f}M params")
    print(f"Compressed: {total_compressed/1e6:.1f}M params")
    print(f"Ratio: {final_metadata['compression_ratio']:.1f}x")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {output_dir}")
    print(f"  - Individual layers: {layers_dir}/")
    print(f"  - Combined file: compressed_weights.pt")
    print(f"  - Metadata: metadata.json")
    
    return final_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compress model with LoRA. Supports incremental/resume mode."
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="./compressed_model")
    
    # Resume/incremental options
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing compressed model, skip good layers")
    parser.add_argument("--max-error", type=float, default=None,
                        help="Re-compress layers with error > this (requires --resume)")
    parser.add_argument("--min-compression", type=float, default=None,
                        help="Re-compress layers with compression < this (requires --resume)")
    
    # Filtering options
    parser.add_argument("--only", type=str, default=None,
                        help="Only process these modules (comma-separated, e.g., 'q_proj,k_proj')")
    parser.add_argument("--exclude", type=str, default=None,
                        help="Exclude these modules (comma-separated)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N layers (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    
    args = parser.parse_args()
    
    # Parse module filters
    only_modules = args.only.split(',') if args.only else None
    exclude_modules = args.exclude.split(',') if args.exclude else None
    
    compress_model(
        args.model, args.device, args.output_dir,
        resume=args.resume,
        max_error=args.max_error,
        min_compression=args.min_compression,
        only_modules=only_modules,
        exclude_modules=exclude_modules,
        limit_layers=args.limit,
        dry_run=args.dry_run
    )

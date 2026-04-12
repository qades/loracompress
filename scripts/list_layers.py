#!/usr/bin/env python3
"""
List all layers with predicted optimal rank for compression.

Based on analysis:
- Attention (q/k/v/o_proj): rank = max(4, int(min_dim * 0.028))  # ~2.8%
- MLP (gate/up/down_proj): rank = max(4, int(min_dim * 0.007))   # ~0.7%
"""
import sys
import os
import signal
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Handle broken pipe gracefully
def handle_sigpipe(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGPIPE, handle_sigpipe)


def predict_optimal_rank(shape, module_type):
    """Predict optimal rank and expected error based on layer type and dimensions."""
    d, k = shape
    min_dim = min(d, k)
    
    # Based on our analysis
    if module_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        # Attention layers need higher rank for same error
        pct = 0.028  # 2.8%
        base_rank = int(min_dim * pct)
    elif module_type in ['gate_proj', 'up_proj', 'down_proj']:
        # MLP layers compress better
        pct = 0.007  # 0.7%
        base_rank = int(min_dim * pct)
    else:
        # Default
        pct = 0.014  # 1.4%
        base_rank = int(min_dim * pct)
    
    # Round to nearest power of 2 or multiple of 4, within bounds
    rank = max(4, min(base_rank, min_dim // 2))
    
    # Round to nice number
    if rank <= 8:
        rank = rank
    elif rank <= 16:
        rank = 16
    elif rank <= 32:
        rank = 32
    elif rank <= 64:
        rank = 64
    else:
        rank = (rank // 16) * 16
    
    compression = (d * k) / (rank * (d + k))
    
    # Estimate error based on experimental data
    error = estimate_error(shape, module_type, rank)
    
    return rank, pct * 100, compression, error


def estimate_error(shape, module_type, rank):
    """
    Estimate compression error based on experimental results.
    
    From our experiments:
    - q_proj (576x576): rank 16 -> ~0.15%, rank 4 -> ~0.17%
    - k/v_proj (192x576): rank 5 -> ~0.15%
    - MLP (1536x576): rank 4 -> ~0.10%, rank 64 -> ~0.085%
    """
    d, k = shape
    min_dim = min(d, k)
    rank_pct = rank / min_dim
    
    if module_type in ['q_proj', 'o_proj']:
        # 576x576 matrices
        # rank 16 (2.8%) -> 0.15%, rank 4 (0.7%) -> 0.17%
        base_error = 0.18
        improvement = rank_pct / 0.028 * 0.03  # Higher rank -> lower error
        error = max(0.10, base_error - improvement)
    elif module_type in ['k_proj', 'v_proj']:
        # 192x576 matrices - smaller, easier
        # rank 5 (2.6%) -> ~0.15%
        base_error = 0.16
        improvement = rank_pct / 0.026 * 0.02
        error = max(0.08, base_error - improvement)
    elif module_type in ['gate_proj', 'up_proj']:
        # 1536x576 - MLP up-project
        # rank 4 (0.7%) -> ~0.10%
        base_error = 0.11
        improvement = rank_pct / 0.007 * 0.02
        error = max(0.07, base_error - improvement)
    elif module_type == 'down_proj':
        # 576x1536 - MLP down-project
        # Similar to gate_proj
        base_error = 0.11
        improvement = rank_pct / 0.007 * 0.02
        error = max(0.07, base_error - improvement)
    else:
        error = 0.15  # Default estimate
    
    return error


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="List model layers with compression info")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit output lines")
    
    args = parser.parse_args()
    
    try:
        from transformers import AutoModelForCausalLM
        
        print("Loading model...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True,
        ).to(args.device)
        
        # Collect all relevant layers
        layers = []
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                          'gate_proj', 'up_proj', 'down_proj']
        
        for name, param in model.named_parameters():
            if not any(x in name for x in target_modules) or 'weight' not in name:
                continue
            
            shape = tuple(param.shape)
            d, k = shape
            params = d * k
            
            # Parse type
            parts = name.split('.')
            module_type = parts[-2] if len(parts) >= 2 else 'unknown'
            layer_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else -1
            
            # Predict optimal rank and error
            opt_rank, pct, compression, error = predict_optimal_rank(shape, module_type)
            
            layers.append({
                'name': name,
                'shape': shape,
                'params': params,
                'type': module_type,
                'layer_idx': layer_idx,
                'opt_rank': opt_rank,
                'rank_pct': pct,
                'compression': compression,
                'error': error,
            })
        
        # Sort by layer index then type
        type_order = {'q_proj': 0, 'k_proj': 1, 'v_proj': 2, 'o_proj': 3,
                      'gate_proj': 4, 'up_proj': 5, 'down_proj': 6}
        layers.sort(key=lambda x: (x['layer_idx'], type_order.get(x['type'], 99)))
        
        # Print header
        print(f"\nAll {len(layers)} layers in {args.model}:")
        print(f"{'Name':<50} {'Shape':<12} {'Params':<10} {'Type':<10} {'Rank':<6} {'Ratio':<7} {'Error':<8}")
        print("-" * 118)
        
        # Print layers
        for i, l in enumerate(layers):
            if args.limit and i >= args.limit:
                print(f"... ({len(layers) - i} more layers)")
                break
            
            try:
                print(f"{l['name']:<50} {str(l['shape']):<12} {l['params']:>9,}  "
                      f"{l['type']:<10} {l['opt_rank']:<6} {l['compression']:>6.1f}x {l['error']:>7.3f}%")
            except BrokenPipeError:
                break
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Group by type
        by_type = {}
        for l in layers:
            t = l['type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(l)
        
        print(f"\nLayer type breakdown:")
        for t, ls in sorted(by_type.items()):
            total_params = sum(l['params'] for l in ls)
            avg_compression = sum(l['compression'] for l in ls) / len(ls)
            avg_error = sum(l['error'] for l in ls) / len(ls)
            max_error = max(l['error'] for l in ls)
            print(f"  {t:<12}: {len(ls):3d} layers, {total_params/1e6:.1f}M params, "
                  f"~{avg_compression:.1f}x compression, ~{avg_error:.2f}% error (max {max_error:.2f}%)")
        
        # Total compression potential
        total_params = sum(l['params'] for l in layers)
        compressed_params = sum(l['params'] / l['compression'] for l in layers)
        overall_ratio = total_params / compressed_params
        
        # Error analysis
        all_errors = [l['error'] for l in layers]
        avg_error = sum(all_errors) / len(all_errors)
        max_error = max(all_errors)
        min_error = min(all_errors)
        
        # Weighted error (by parameter count)
        weighted_error = sum(l['error'] * l['params'] for l in layers) / total_params
        
        print(f"\nOverall compression potential:")
        print(f"  Original: {total_params/1e6:.1f}M parameters")
        print(f"  Compressed: {compressed_params/1e6:.1f}M parameters")
        print(f"  Ratio: {overall_ratio:.1f}x")
        print(f"  Storage saved: {(1 - 1/overall_ratio)*100:.1f}%")
        
        print(f"\nExpected error metrics:")
        print(f"  Average error: {avg_error:.3f}%")
        print(f"  Weighted by params: {weighted_error:.3f}%")
        print(f"  Min/Max: {min_error:.3f}% / {max_error:.3f}%")
        # Manual std dev calculation
        mean_err = avg_error
        variance = sum((e - mean_err) ** 2 for e in all_errors) / len(all_errors)
        std_dev = math.sqrt(variance)
        print(f"  Std dev: {std_dev:.3f}%")
        
        # Cumulative error estimate
        # Rough model: errors compound but not linearly
        # Assume some independence: total_error ≈ sqrt(sum(error_i^2))
        cumulative_error = math.sqrt(sum(e**2 for e in all_errors))
        print(f"  Estimated cumulative: {cumulative_error:.2f}%")
        
        print(f"\nRank selection formula:")
        print(f"  Attention (q/k/v/o): rank = max(4, int(min_dim × 0.028))")
        print(f"  MLP (gate/up/down):  rank = max(4, int(min_dim × 0.007))")
        
    except BrokenPipeError:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

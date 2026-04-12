#!/usr/bin/env python3
"""
Decompress model from LoRA weights with fallback to original.

Can selectively use compressed layers based on quality thresholds.
"""
import sys
import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def decompress_weight(A, B, shape):
    """Reconstruct weight matrix from LoRA."""
    W = torch.matmul(B, A)
    return W


def load_model_with_fallback(compressed_dir, original_model_name=None, 
                             max_error=None, min_compression=None, device='cpu'):
    """
    Load model with compressed layers, fallback to original for missing/low-quality layers.
    
    Args:
        compressed_dir: Directory with compressed layers
        original_model_name: Fallback model name (if None, use from metadata)
        max_error: Skip compressed layers with error > this (use original)
        min_compression: Skip compressed layers with compression < this (use original)
        device: Device to load on
    """
    print("="*70)
    print("Loading Model with Fallback")
    print("="*70)
    print(f"Compressed dir: {compressed_dir}")
    if max_error:
        print(f"Max error threshold: {max_error}%")
    if min_compression:
        print(f"Min compression threshold: {min_compression}x")
    
    # Load metadata
    metadata_path = os.path.join(compressed_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise ValueError(f"No metadata found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_name = original_model_name or metadata['model_name']
    print(f"\nBase model: {model_name}")
    print(f"Compression stats from metadata:")
    print(f"  Layers completed: {metadata.get('layers_completed', metadata.get('num_layers', 'N/A'))}")
    print(f"  Original params: {metadata.get('total_original_params', 0)/1e6:.1f}M")
    print(f"  Compressed params: {metadata.get('total_compressed_params', 0)/1e6:.1f}M")
    print(f"  Ratio: {metadata.get('compression_ratio', 0):.1f}x")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(compressed_dir)
    
    # Create base model (will be modified with compressed weights where available)
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
    ).to(device)
    
    # Track statistics
    stats = {
        'compressed_used': 0,
        'original_fallback': 0,
        'skipped_high_error': 0,
        'skipped_low_compression': 0,
        'not_found': 0,
        'total_error': 0,
    }
    
    # Load compressed layers
    layer_metadata = metadata.get('layer_metadata', {})
    layers_dir = os.path.join(compressed_dir, 'layers')
    
    print(f"\nProcessing {len(layer_metadata)} layers...")
    
    # Warn about partial compression
    total_model_layers = len(list(model.named_parameters()))
    compressed_count = len(layer_metadata)
    if compressed_count < 50:  # SmolLM2-135M has ~210 compressible layers
        print(f"\n⚠️  WARNING: Only {compressed_count} layers compressed!")
        print("   This appears to be a partial/test compression.")
        print("   Results will be garbage - run 'make compress-model' for full compression.")
        print()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in layer_metadata:
                continue
            
            layer_info = layer_metadata[name]
            
            # Check if we should use compressed version
            use_compressed = True
            skip_reason = None
            
            if layer_info['type'] != 'lora':
                use_compressed = False
                skip_reason = 'not_lora'
            elif max_error is not None and layer_info.get('error', 0) > max_error:
                use_compressed = False
                skip_reason = 'high_error'
                stats['skipped_high_error'] += 1
            elif min_compression is not None and layer_info.get('compression', 0) < min_compression:
                use_compressed = False
                skip_reason = 'low_compression'
                stats['skipped_low_compression'] += 1
            elif not os.path.exists(layer_info.get('file', '')):
                use_compressed = False
                skip_reason = 'not_found'
                stats['not_found'] += 1
            
            if use_compressed:
                # Load and decompress
                try:
                    layer_data = torch.load(layer_info['file'], map_location=device)
                    A = layer_data['A']
                    B = layer_data['B']
                    W = decompress_weight(A, B, layer_info['shape'])
                    param.copy_(W)
                    stats['compressed_used'] += 1
                    stats['total_error'] += layer_info.get('error', 0)
                except Exception as e:
                    print(f"  Warning: Failed to load {name}: {e}")
                    stats['original_fallback'] += 1
            else:
                # Keep original weight (already loaded)
                stats['original_fallback'] += 1
                if skip_reason == 'high_error':
                    print(f"  {name}: skipped (error={layer_info.get('error', 0):.3f}% > {max_error}%)")
                elif skip_reason == 'low_compression':
                    print(f"  {name}: skipped (compression={layer_info.get('compression', 0):.1f}x < {min_compression}x)")
    
    model.eval()
    
    # Print summary
    print("\n" + "="*70)
    print("Loading Summary")
    print("="*70)
    print(f"Compressed layers used: {stats['compressed_used']}")
    print(f"Original weights (fallback): {stats['original_fallback']}")
    if max_error:
        print(f"  Skipped (high error): {stats['skipped_high_error']}")
    if min_compression:
        print(f"  Skipped (low compression): {stats['skipped_low_compression']}")
    print(f"  Not found: {stats['not_found']}")
    
    if stats['compressed_used'] > 0:
        avg_error = stats['total_error'] / stats['compressed_used']
        print(f"Average error (compressed): {avg_error:.4f}%")
    
    return model, tokenizer, metadata, stats


def benchmark_models(model, tokenizer, prompts, device='cpu', max_tokens=30):
    """Generate and display outputs for test prompts."""
    print("\n" + "="*70)
    print("Generating Outputs")
    print("="*70)
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            torch.manual_seed(42)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from output
        generated_text = generated[len(prompt):].strip()
        
        print(f"  Output: {generated_text[:100]}...")
        
        results.append({
            'prompt': prompt,
            'generated': generated_text,
        })
    
    return results


def compare_with_original(compressed_model, compressed_tokenizer, 
                          original_model_name, prompts, device='cpu', max_tokens=30):
    """Compare compressed model with original."""
    print("\n" + "="*70)
    print("Comparison: Compressed vs Original")
    print("="*70)
    
    # Load original
    print("Loading original model...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        original_model_name, trust_remote_code=True,
    ).to(device)
    orig_tokenizer = AutoTokenizer.from_pretrained(original_model_name, trust_remote_code=True)
    if orig_tokenizer.pad_token is None:
        orig_tokenizer.pad_token = orig_tokenizer.eos_token
    orig_model.eval()
    
    matches = 0
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Original
        inputs_orig = orig_tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            torch.manual_seed(42)
            output_orig = orig_model.generate(
                **inputs_orig, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=orig_tokenizer.pad_token_id,
            )
        text_orig = orig_tokenizer.decode(output_orig[0], skip_special_tokens=True)
        
        # Compressed
        inputs_comp = compressed_tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            torch.manual_seed(42)
            output_comp = compressed_model.generate(
                **inputs_comp, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=compressed_tokenizer.pad_token_id,
            )
        text_comp = compressed_tokenizer.decode(output_comp[0], skip_special_tokens=True)
        
        match = text_orig == text_comp
        if match:
            matches += 1
        
        print(f"  Original:      {text_orig[:80]}...")
        print(f"  Compressed:    {text_comp[:80]}...")
        print(f"  Match: {'✓ YES' if match else '✗ NO'}")
        
        results.append({
            'prompt': prompt,
            'original': text_orig,
            'compressed': text_comp,
            'match': match,
        })
    
    print(f"\nOverall: {matches}/{len(prompts)} prompts match ({100*matches/len(prompts):.1f}%)")
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed-dir", default="./compressed_model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--max-error", type=float, default=None,
                        help="Skip compressed layers with error > this")
    parser.add_argument("--min-compression", type=float, default=None,
                        help="Skip compressed layers with compression < this")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with original model")
    parser.add_argument("--output", default="./benchmark_results.json")
    
    args = parser.parse_args()
    
    # Load model with fallback
    model, tokenizer, metadata, stats = load_model_with_fallback(
        args.compressed_dir,
        max_error=args.max_error,
        min_compression=args.min_compression,
        device=args.device,
    )
    
    # Test prompts
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In the year 2050,",
        "Once upon a time",
        "The theory of relativity states that",
        "def fibonacci(n):",
        "Machine learning is",
        "The future of AI",
    ]
    
    # Generate outputs
    if args.compare:
        results = compare_with_original(
            model, tokenizer, metadata['model_name'],
            test_prompts, args.device, args.max_tokens
        )
    else:
        results = benchmark_models(
            model, tokenizer, test_prompts, args.device, args.max_tokens
        )
    
    # Save results
    output_data = {
        'metadata': metadata,
        'loading_stats': stats,
        'thresholds': {
            'max_error': args.max_error,
            'min_compression': args.min_compression,
        },
        'results': results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

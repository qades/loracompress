#!/usr/bin/env python3
"""
Decompress model from LoRA weights and benchmark against original.
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


def load_compressed_model(compressed_dir, device='cpu'):
    """
    Load compressed model and reconstruct it.
    
    Returns:
        model: Reconstructed model
        tokenizer
        metadata
    """
    print("="*70)
    print("Loading Compressed Model")
    print("="*70)
    print(f"Directory: {compressed_dir}")
    
    # Load metadata
    with open(os.path.join(compressed_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"\nCompression stats:")
    print(f"  Original: {metadata['total_original_params']/1e6:.1f}M params")
    print(f"  Compressed: {metadata['total_compressed_params']/1e6:.1f}M params")
    print(f"  Ratio: {metadata['compression_ratio']:.1f}x")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(compressed_dir)
    
    # Create model with zeroed weights
    print("\nLoading base model structure...")
    model = AutoModelForCausalLM.from_pretrained(
        metadata['model_name'],
        trust_remote_code=True,
    ).to(device)
    
    # Load compressed weights
    print("Decompressing weights...")
    compressed = torch.load(
        os.path.join(compressed_dir, 'compressed_weights.pt'),
        map_location=device
    )
    
    # Reconstruct each layer
    reconstructed = 0
    total_error = 0
    num_lora = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in compressed:
                continue
            
            data = compressed[name]
            
            if data['type'] == 'lora':
                A = data['A'].to(device)
                B = data['B'].to(device)
                W = decompress_weight(A, B, data['shape'])
                param.copy_(W)
                
                if 'error' in data:
                    total_error += data['error']
                    num_lora += 1
                reconstructed += 1
            else:
                # Original weight (not compressed)
                param.copy_(data['weight'].to(device))
                reconstructed += 1
    
    avg_error = total_error / num_lora if num_lora > 0 else 0
    print(f"  Reconstructed {reconstructed} layers")
    print(f"  Average reconstruction error: {avg_error:.4f}%")
    
    model.eval()
    return model, tokenizer, metadata


def benchmark_models(original_model, reconstructed_model, tokenizer, 
                     prompts, device='cpu', max_tokens=30):
    """
    Benchmark original vs reconstructed model.
    """
    print("\n" + "="*70)
    print("Benchmarking: Original vs Reconstructed")
    print("="*70)
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt[:60]}...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        # Generate from original
        with torch.no_grad():
            torch.manual_seed(42)
            output_orig = original_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)
        
        # Generate from reconstructed
        with torch.no_grad():
            torch.manual_seed(42)
            output_recon = reconstructed_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text_recon = tokenizer.decode(output_recon[0], skip_special_tokens=True)
        
        # Compare
        match = text_orig == text_recon
        
        print(f"  Original:      {text_orig[:100]}...")
        print(f"  Reconstructed: {text_recon[:100]}...")
        print(f"  Match: {'✓ YES' if match else '✗ NO'}")
        
        results.append({
            'prompt': prompt,
            'original': text_orig,
            'reconstructed': text_recon,
            'match': match,
        })
    
    # Summary
    matches = sum(1 for r in results if r['match'])
    print("\n" + "="*70)
    print("Benchmark Summary")
    print("="*70)
    print(f"Total prompts: {len(results)}")
    print(f"Exact matches: {matches}/{len(results)} ({100*matches/len(results):.1f}%)")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed-dir", default="./compressed_model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=30)
    
    args = parser.parse_args()
    
    # Load reconstructed model
    recon_model, tokenizer, metadata = load_compressed_model(
        args.compressed_dir, args.device
    )
    
    # Load original model for comparison
    print("\nLoading original model for comparison...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        metadata['model_name'],
        trust_remote_code=True,
    ).to(args.device)
    orig_model.eval()
    
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
    
    # Benchmark
    results = benchmark_models(
        orig_model, recon_model, tokenizer,
        test_prompts, args.device, args.max_tokens
    )
    
    # Save results
    with open(os.path.join(args.compressed_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.compressed_dir}/benchmark_results.json")


if __name__ == "__main__":
    main()

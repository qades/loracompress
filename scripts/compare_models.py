#!/usr/bin/env python3
"""
Compare two models: base model vs LoRA-augmented model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model_loader import load_base_model, load_lora_weights, merge_lora_weights, get_device
from benchmark import ModelComparator, print_comparison_result
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compare base model with LoRA model")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="Base model name")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA weights")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights before comparison")
    parser.add_argument("--output", default="results/lora_comparison.json", help="Output file")
    args = parser.parse_args()
    
    DEVICE = get_device(args.device)
    
    print(f"Device: {DEVICE}")
    print(f"Loading base model: {args.model}")
    
    # Load base model
    base_model, tokenizer = load_base_model(args.model, DEVICE)
    
    print(f"Loading LoRA model from: {args.lora_path}")
    
    # Load LoRA model
    lora_model, _ = load_base_model(args.model, DEVICE)
    lora_model = load_lora_weights(lora_model, args.lora_path)
    
    if args.merge_lora:
        print("Merging LoRA weights...")
        lora_model = merge_lora_weights(lora_model)
    
    print("Creating comparator...")
    comparator = ModelComparator(base_model, lora_model, tokenizer, DEVICE)
    
    # Test prompts
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In the year 2050,",
        "Once upon a time",
        "The quick brown fox",
        "def fibonacci(n):",
        "The theory of relativity states that",
        "To be or not to be",
        "Machine learning is",
        "The future of AI",
    ]
    
    print("\n" + "=" * 80)
    print("COMPARING BASE MODEL vs LoRA MODEL")
    print("=" * 80)
    
    # Run individual comparisons
    for prompt in test_prompts[:5]:  # Show first 5
        result = comparator.compare(
            prompt=prompt,
            max_new_tokens=30,
            do_sample=False,
            seed=42,
        )
        print_comparison_result(result, prompt)
        print()
    
    # Run full benchmark
    print("\n" + "=" * 80)
    print("FULL BENCHMARK RESULTS")
    print("=" * 80)
    
    benchmark_results = comparator.run_benchmark(
        prompts=test_prompts,
        max_new_tokens=30,
        do_sample=False,
        seed=42,
    )
    
    summary = benchmark_results["summary"]
    print(f"\nTotal Prompts: {summary['total_prompts']}")
    print(f"Exact Matches: {summary['exact_matches']}/{summary['total_prompts']} ({summary['match_rate']*100:.1f}%)")
    print(f"Token Matches: {summary['token_matches']}/{summary['total_prompts']} ({summary['token_match_rate']*100:.1f}%)")
    print(f"Avg Token Similarity: {summary['avg_token_similarity']:.4f}")
    print(f"Avg Logits MSE: {summary['avg_logits_mse']:.6e}")
    print(f"All Tests Pass: {'✓ YES' if summary['all_pass'] else '✗ NO'}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    return 0 if summary['match_rate'] > 0.8 else 1  # Consider success if >80% match


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

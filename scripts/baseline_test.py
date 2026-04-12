#!/usr/bin/env python3
"""
Phase 1: Baseline test - verify that two identical models produce identical outputs.
This establishes that our benchmarking framework works correctly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model_loader import load_model_for_comparison, get_device
from benchmark import ModelComparator, print_comparison_result
import json


def main():
    MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
    DEVICE = get_device("auto")
    
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading two instances of {MODEL_NAME}...")
    
    # Load two identical models
    model_a, model_b, tokenizer = load_model_for_comparison(MODEL_NAME, DEVICE, seed=42)
    
    print("Models loaded. Creating comparator...")
    comparator = ModelComparator(model_a, model_b, tokenizer, DEVICE)
    
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
    ]
    
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE TEST - Identical Models Should Match")
    print("=" * 80)
    
    # Run individual comparisons
    for prompt in test_prompts:
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
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_test.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nResults saved to results/baseline_test.json")
    
    # Return exit code
    return 0 if summary['all_pass'] else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

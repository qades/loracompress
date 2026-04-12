#!/usr/bin/env python3
"""
Full pipeline: Train LoRA and benchmark against base model iteratively.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import json
import subprocess
from datetime import datetime


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
RANK = 16
MAX_ITERATIONS = 20
TARGET_MATCH_RATE = 0.95  # Stop when we achieve 95% match rate


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print('=' * 80)
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def load_latest_results(results_file):
    """Load the latest benchmark results."""
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def main():
    print("=" * 80)
    print("LoRA Weight Reproduction Pipeline")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA Rank: {RANK}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"Target Match Rate: {TARGET_MATCH_RATE * 100}%")
    print("=" * 80)
    
    # Step 1: Run baseline test
    print("\n" + "=" * 80)
    print("STEP 1: BASELINE TEST")
    print("=" * 80)
    
    baseline_cmd = [
        sys.executable,
        "scripts/baseline_test.py"
    ]
    
    if not run_command(baseline_cmd, "Running baseline test..."):
        print("ERROR: Baseline test failed!")
        return 1
    
    # Load baseline results
    with open("results/baseline_test.json", 'r') as f:
        baseline_results = json.load(f)
    
    if not baseline_results["summary"]["all_pass"]:
        print("WARNING: Baseline test did not pass completely!")
        print("This may indicate issues with the benchmarking framework.")
    else:
        print("✓ Baseline test passed - identical models produce identical outputs")
    
    # Step 2: Iterative training and evaluation
    print("\n" + "=" * 80)
    print("STEP 2: ITERATIVE TRAINING")
    print("=" * 80)
    
    best_match_rate = 0.0
    best_checkpoint = None
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'=' * 80}")
        
        # Training parameters that adjust over iterations
        epochs = 5 if iteration <= 3 else 10
        num_samples = 5000 * iteration  # Increase samples over time
        lr = 1e-4 if iteration <= 5 else 5e-5  # Lower LR later
        
        output_dir = f"./lora_checkpoints/iter_{iteration:03d}"
        
        # Train LoRA
        train_cmd = [
            sys.executable,
            "scripts/train_lora.py",
            "--model", MODEL_NAME,
            "--rank", str(RANK),
            "--epochs", str(epochs),
            "--num-samples", str(num_samples),
            "--lr", str(lr),
            "--output-dir", output_dir,
            "--batch-size", "4",
            "--seq-length", "128",
            "--loss-type", "mse",
        ]
        
        if not run_command(train_cmd, f"Training LoRA (Iteration {iteration})..."):
            print(f"WARNING: Training iteration {iteration} failed, continuing...")
            continue
        
        # Compare with base model
        best_model_path = os.path.join(output_dir, "best_model")
        comparison_output = f"results/comparison_iter_{iteration:03d}.json"
        
        compare_cmd = [
            sys.executable,
            "scripts/compare_models.py",
            "--model", MODEL_NAME,
            "--lora-path", best_model_path,
            "--output", comparison_output,
        ]
        
        if not run_command(compare_cmd, f"Comparing models (Iteration {iteration})..."):
            print(f"WARNING: Comparison iteration {iteration} failed, continuing...")
            continue
        
        # Load results
        with open(comparison_output, 'r') as f:
            results = json.load(f)
        
        match_rate = results["summary"]["match_rate"]
        avg_similarity = results["summary"]["avg_token_similarity"]
        avg_mse = results["summary"]["avg_logits_mse"]
        
        print(f"\nIteration {iteration} Results:")
        print(f"  Match Rate: {match_rate * 100:.2f}%")
        print(f"  Avg Token Similarity: {avg_similarity:.4f}")
        print(f"  Avg Logits MSE: {avg_mse:.6e}")
        
        # Track best
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            best_checkpoint = best_model_path
            print(f"  ✓ New best match rate!")
        
        # Check if we've reached target
        if match_rate >= TARGET_MATCH_RATE:
            print(f"\n{'=' * 80}")
            print(f"SUCCESS! Target match rate {TARGET_MATCH_RATE * 100}% achieved!")
            print(f"{'=' * 80}")
            break
        
        # Early stopping if we're not improving
        if iteration >= 5 and match_rate < best_match_rate * 0.9:
            print(f"\nWARNING: Significant degradation detected. Consider adjusting hyperparameters.")
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total Iterations: {iteration}")
    print(f"Best Match Rate: {best_match_rate * 100:.2f}%")
    print(f"Best Checkpoint: {best_checkpoint}")
    
    if best_match_rate >= TARGET_MATCH_RATE:
        print(f"\n✓ HYPOTHESIS CONFIRMED: LoRA rank {RANK} can reproduce base model weights")
        print(f"  with {best_match_rate * 100:.2f}% accuracy!")
    elif best_match_rate >= 0.5:
        print(f"\n~ PARTIAL SUCCESS: LoRA rank {RANK} achieved {best_match_rate * 100:.2f}% match.")
        print(f"  Further training may improve results.")
    else:
        print(f"\n✗ HYPOTHESIS NOT CONFIRMED: Best match rate was only {best_match_rate * 100:.2f}%")
        print(f"  Consider increasing rank, samples, or training epochs.")
    
    # Save final report
    report = {
        "model": MODEL_NAME,
        "rank": RANK,
        "total_iterations": iteration,
        "best_match_rate": best_match_rate,
        "best_checkpoint": best_checkpoint,
        "target_match_rate": TARGET_MATCH_RATE,
        "success": best_match_rate >= TARGET_MATCH_RATE,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open("results/final_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFinal report saved to results/final_report.json")
    
    return 0 if best_match_rate >= TARGET_MATCH_RATE else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

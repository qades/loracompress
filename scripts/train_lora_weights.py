#!/usr/bin/env python3
"""
Train LoRA to directly reproduce base model weights.

This zeros out the base model and trains LoRA such that:
    W_target = BA

This is a direct low-rank matrix approximation problem.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer import train_lora_to_reproduce_weights_directly


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train LoRA to reproduce base model weights directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training (rank 16)
  %(prog)s --model HuggingFaceTB/SmolLM2-135M --rank 16 --epochs 100
  
  # Higher rank for better approximation
  %(prog)s --rank 64 --epochs 200 --lr 1e-3
  
  # Quick test
  %(prog)s --rank 8 --epochs 50
        """
    )
    
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Base model name",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--output-dir",
        default="./lora_checkpoints",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (auto/cuda/cpu). Default: cpu (GPU may hang on ROCm)",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Target modules (default: all linear layers)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: rank=4, epochs=10, only q_proj/v_proj",
    )
    
    args = parser.parse_args()
    
    # Apply quick test settings
    if args.quick_test:
        args.rank = 4
        args.epochs = 10
        args.target_modules = ["q_proj", "v_proj"]  # Only 2 modules for speed
        args.lr = 1e-2  # Higher LR for faster convergence
        print("QUICK TEST MODE ENABLED")
        print(f"  Rank: {args.rank}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Modules: {args.target_modules}")
        print(f"  LR: {args.lr}")
    
    print("=" * 80)
    print("LoRA Weight Reproduction (Direct)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Rank: {args.rank}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print("\nTraining LoRA such that: W_target = BA")
    print("(Base model weights are zeroed out)\n")
    
    lora_model, best_loss = train_lora_to_reproduce_weights_directly(
        model_name=args.model,
        rank=args.rank,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        target_modules=args.target_modules,
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best weight MSE: {best_loss:.6e}")
    print(f"LoRA saved to: {args.output_dir}")
    
    # Compute theoretical rank analysis
    import torch
    print("\nRank Analysis:")
    print(f"  LoRA rank: {args.rank}")
    print(f"  If weight matrices are low-rank (<={args.rank}), perfect reproduction is possible")
    print(f"  If weight matrices are full-rank, this is a best-effort approximation")

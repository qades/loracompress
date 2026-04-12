#!/usr/bin/env python3
"""
Phase 2: Train LoRA to reproduce base model weights.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trainer import train_lora_to_reproduce_base


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train LoRA to reproduce base model weights"
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        default="./lora_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--loss-type",
        default="mse",
        choices=["mse", "kl", "cosine"],
        help="Loss function type",
    )
    
    args = parser.parse_args()
    
    train_lora_to_reproduce_base(
        model_name=args.model,
        rank=args.rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        loss_type=args.loss_type,
    )

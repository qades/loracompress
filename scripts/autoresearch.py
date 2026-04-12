#!/usr/bin/env python3
"""
Run hyperparameter autoresearch to find optimal LoRA training parameters.
"""
import sys
import os
import time

if __name__ == "__main__":
    import argparse
    
    # Parse args first (before heavy imports) so --help works fast
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for LoRA weight reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test mode (fast)
  %(prog)s --n-trials 5 --num-samples 100 --verbose
  
  # Standard search (default)
  %(prog)s --n-trials 20 --num-samples 500
  
  # Full search with caching
  %(prog)s --n-trials 50 --num-samples 2000 --cache-dir /mnt/ssd/cache
  
  # Debug mode (lots of output)
  %(prog)s --verbose --debug
        """
    )
    
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-135M",
        help="Base model name",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials to run (default: 20)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Max epochs per trial",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Training samples per trial (default: 500 for speed)",
    )
    parser.add_argument(
        "--target-match-rate",
        type=float,
        default=0.95,
        help="Target match rate for success",
    )
    parser.add_argument(
        "--output-dir",
        default="./autoresearch_results",
        help="Output directory",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Dataset cache directory (default: OUTPUT_DIR/cache)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device (auto/cuda/cpu). Default: cpu (GPU may hang on ROCm)",
    )
    parser.add_argument(
        "--train-final",
        action="store_true",
        help="Train final model with best parameters",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=20,
        help="Epochs for final training",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for debugging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (even more verbose)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode: 3 trials, 50 samples, 2 epochs (for debugging)",
    )
    
    args = parser.parse_args()
    
    # Apply quick-test settings if requested
    if args.quick_test:
        args.n_trials = 3
        args.num_samples = 50
        args.max_epochs = 2
        args.val_samples = 20
        args.verbose = True
        print("QUICK TEST MODE ENABLED")
        print(f"  Trials: {args.n_trials}")
        print(f"  Samples: {args.num_samples}")
        print(f"  Epochs: {args.max_epochs}")
    
    # Add src to path and import heavy modules AFTER parsing args
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    def log(msg):
        if args.verbose or args.debug:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    
    log("Importing modules...")
    from autoresearch import LoRAHyperparameterSearch, train_with_best_params
    log("Modules imported successfully")
    
    print("=" * 80)
    print("LoRA Hyperparameter Autoresearch")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Trials: {args.n_trials}")
    print(f"Max epochs per trial: {args.max_epochs}")
    print(f"Samples per trial: {args.num_samples}")
    print(f"Target match rate: {args.target_match_rate}")
    print(f"Output: {args.output_dir}")
    if args.cache_dir:
        print(f"Cache: {args.cache_dir}")
    if args.verbose:
        print("Verbose: enabled")
    print("=" * 80)
    
    # Set cache dir default if not provided
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    
    log("Creating LoRAHyperparameterSearch...")
    
    # Get val_samples from args if set (for quick-test mode), otherwise use default
    val_samples = getattr(args, 'val_samples', 200)
    
    searcher = LoRAHyperparameterSearch(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        max_epochs_per_trial=args.max_epochs,
        num_samples=args.num_samples,
        val_samples=val_samples,
        target_match_rate=args.target_match_rate,
        cache_dir=cache_dir,
    )
    log("LoRAHyperparameterSearch created")
    
    log("Starting search...")
    study = searcher.search()
    log("Search completed")
    
    if args.train_final:
        print("\n" + "=" * 80)
        print("Training final model with best parameters...")
        print("=" * 80)
        
        train_with_best_params(
            study,
            model_name=args.model,
            device=args.device,
            num_epochs=args.final_epochs,
            num_samples=args.num_samples * 4,  # More samples for final training
            output_dir=os.path.join(args.output_dir, "best_model"),
        )

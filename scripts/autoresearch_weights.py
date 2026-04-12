#!/usr/bin/env python3
"""
Hyperparameter autoresearch for DIRECT WEIGHT reproduction.

This searches for optimal hyperparameters to train LoRA such that:
    W_target = BA

(where base model weights are zeroed out)
"""
import sys
import os
import time

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for LoRA weight reproduction (direct)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (3 trials, fast)
  %(prog)s --n-trials 3 --rank-max 32 --epochs 50
  
  # Standard search (20 trials)
  %(prog)s --n-trials 20 --rank-max 64 --epochs 100
  
  # Full search (50 trials, thorough)
  %(prog)s --n-trials 50 --rank-max 128 --epochs 200
        """
    )
    
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="Base model")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--rank-min", type=int, default=4, help="Minimum LoRA rank")
    parser.add_argument("--rank-max", type=int, default=64, help="Maximum LoRA rank")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per trial")
    parser.add_argument("--lr-min", type=float, default=1e-4, help="Min learning rate")
    parser.add_argument("--lr-max", type=float, default=1e-2, help="Max learning rate")
    parser.add_argument("--output-dir", default="./autoresearch_weights", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--train-final", action="store_true", help="Train final model with best params")
    parser.add_argument("--final-epochs", type=int, default=200, help="Epochs for final training")
    parser.add_argument("--quick-test", action="store_true", help="Quick test: 3 trials, rank 4-16, 50 epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.n_trials = 3
        args.rank_min = 4
        args.rank_max = 16
        args.epochs = 50
        args.verbose = True
        print("QUICK TEST MODE ENABLED")
        print(f"  Trials: {args.n_trials}")
        print(f"  Rank range: {args.rank_min}-{args.rank_max}")
        print(f"  Epochs: {args.epochs}")
    
    # Now import heavy modules
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    def log(msg):
        if args.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    
    log("Importing modules...")
    try:
        import optuna
        from trainer import train_lora_to_reproduce_weights_directly
        from model_loader import load_base_model
        log("Modules imported successfully")
    except ImportError as e:
        print(f"ERROR: Missing module: {e}")
        print("Install with: pip install --user optuna")
        sys.exit(1)
    
    print("=" * 80)
    print("LoRA Weight Reproduction Autoresearch (Direct)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Trials: {args.n_trials}")
    print(f"Rank range: {args.rank_min}-{args.rank_max}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"LR range: {args.lr_min:.0e}-{args.lr_max:.0e}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load target model once for all trials (to get weight dimensions)
    log("Loading target model to analyze structure...")
    target_model, tokenizer = load_base_model(args.model, args.device, zero_weights=False)
    
    # Count total parameters in target weights
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    total_params = 0
    for name, param in target_model.named_parameters():
        if any(x in name for x in target_modules) and 'weight' in name:
            total_params += param.numel()
    print(f"Target model has {total_params/1e6:.1f}M parameters in linear layers")
    del target_model  # Free memory
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Define search space
    def objective(trial):
        rank = trial.suggest_int("rank", args.rank_min, args.rank_max, step=4)
        lr = trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True)
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: rank={rank}, lr={lr:.2e}")
        print(f"{'='*60}")
        
        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:03d}")
        
        try:
            _, best_loss = train_lora_to_reproduce_weights_directly(
                model_name=args.model,
                rank=rank,
                num_epochs=args.epochs,
                learning_rate=lr,
                output_dir=trial_dir,
                device=args.device,
                target_modules=None,  # Use all default modules
            )
            
            # Report to optuna
            trial.report(best_loss, args.epochs)
            
            print(f"Trial {trial.number} completed: loss={best_loss:.6e}")
            return best_loss
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    # Create study
    log("Creating Optuna study...")
    study = optuna.create_study(
        direction="minimize",
        study_name="lora_weight_reproduction",
    )
    
    # Run optimization
    log(f"Starting {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Print results
    print("\n" + "=" * 80)
    print("AUTORESEARCH COMPLETE")
    print("=" * 80)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best loss: {study.best_trial.value:.6e}")
    print(f"Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    import json
    results = {
        "best_trial": {
            "number": study.best_trial.number,
            "loss": study.best_trial.value,
            "params": study.best_params,
        },
        "all_trials": [
            {"number": t.number, "loss": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    with open(os.path.join(args.output_dir, "search_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/search_results.json")
    
    # Train final model if requested
    if args.train_final:
        print("\n" + "=" * 80)
        print("Training final model with best parameters...")
        print("=" * 80)
        
        best_params = study.best_params
        final_dir = os.path.join(args.output_dir, "best_model")
        
        _, final_loss = train_lora_to_reproduce_weights_directly(
            model_name=args.model,
            rank=best_params["rank"],
            num_epochs=args.final_epochs,
            learning_rate=best_params["learning_rate"],
            output_dir=final_dir,
            device=args.device,
        )
        
        print(f"\nFinal model trained!")
        print(f"Final loss: {final_loss:.6e}")
        print(f"Saved to: {final_dir}")

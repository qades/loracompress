"""
Hyperparameter autoresearch for LoRA weight reproduction.
Uses Optuna for Bayesian optimization of training parameters.
"""
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np

from model_loader import load_base_model, create_lora_model, get_device
from trainer import WeightReproductionTrainer, WeightReproductionDataset
from benchmark import ModelComparator
from torch.utils.data import DataLoader, Subset


@dataclass
class SearchSpace:
    """Defines the hyperparameter search space."""
    
    # LoRA parameters
    rank_min: int = 4
    rank_max: int = 64
    rank_step: int = 4
    
    alpha_min: float = 1.0
    alpha_max: float = 64.0
    
    dropout_min: float = 0.0
    dropout_max: float = 0.1
    
    # Training parameters
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    
    batch_size_options: List[int] = None
    
    seq_length_min: int = 64
    seq_length_max: int = 256
    seq_length_step: int = 64
    
    loss_type_options: List[str] = None
    
    def __post_init__(self):
        if self.batch_size_options is None:
            self.batch_size_options = [2, 4, 8, 16]
        if self.loss_type_options is None:
            self.loss_type_options = ["mse", "kl", "cosine"]


class LoRAHyperparameterSearch:
    """
    Automated hyperparameter search for LoRA weight reproduction.
    
    Uses Optuna to find optimal hyperparameters that minimize the
    difference between LoRA-augmented model and base model outputs.
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        device: str = "auto",
        output_dir: str = "./autoresearch_results",
        n_trials: int = 50,
        max_epochs_per_trial: int = 5,
        num_samples: int = 2000,
        val_samples: int = 200,
        target_match_rate: float = 0.95,
        search_space: Optional[SearchSpace] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = get_device(device)
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.max_epochs_per_trial = max_epochs_per_trial
        self.num_samples = num_samples
        self.val_samples = val_samples
        self.target_match_rate = target_match_rate
        self.search_space = search_space or SearchSpace()
        self.cache_dir = cache_dir or os.path.join(output_dir, "cache")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load base model once for all trials
        print(f"Loading base model: {model_name}")
        print(f"  Device: {self.device}", flush=True)
        if self.device != "cpu":
            import torch
            print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
        self.base_model, self.tokenizer = load_base_model(model_name, self.device)
        self.base_model.eval()
        print(f"  Base model loaded successfully", flush=True)
        
        # Create validation dataset (fixed for fair comparison)
        print(f"Creating validation dataset with {val_samples} samples")
        print(f"  Cache dir: {self.cache_dir}", flush=True)
        self.val_dataset = WeightReproductionDataset(
            tokenizer=self.tokenizer,
            base_model=self.base_model,
            num_samples=val_samples,
            seq_length=128,  # Fixed for validation
            device=self.device,
            pre_generate=True,
            cache_dir=self.cache_dir,
            model_name=model_name,
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=4, shuffle=False)
        
        # Test prompts for final evaluation
        self.test_prompts = [
            "Hello, my name is",
            "The capital of France is",
            "In the year 2050,",
            "Once upon a time",
            "The quick brown fox",
            "def fibonacci(n):",
            "The theory of relativity",
            "To be or not to be",
        ]
    
    def create_trial_config(self, trial: optuna.Trial) -> Dict:
        """Create a configuration for a trial from the search space."""
        ss = self.search_space
        
        config = {
            # LoRA parameters
            "rank": trial.suggest_int("rank", ss.rank_min, ss.rank_max, step=ss.rank_step),
            "alpha": trial.suggest_float("alpha", ss.alpha_min, ss.alpha_max),
            "dropout": trial.suggest_float("dropout", ss.dropout_min, ss.dropout_max),
            
            # Training parameters
            "learning_rate": trial.suggest_float("learning_rate", ss.lr_min, ss.lr_max, log=True),
            "batch_size": trial.suggest_categorical("batch_size", ss.batch_size_options),
            "seq_length": trial.suggest_int("seq_length", ss.seq_length_min, ss.seq_length_max, step=ss.seq_length_step),
            "loss_type": trial.suggest_categorical("loss_type", ss.loss_type_options),
        }
        
        return config
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Returns a score to minimize (lower is better).
        Combines validation loss, MSE, and match rate.
        """
        config = self.create_trial_config(trial)
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: {config}")
        print(f"{'='*60}")
        
        try:
            # Create LoRA model with trial hyperparameters
            lora_model = create_lora_model(
                self.base_model,
                rank=config["rank"],
                alpha=int(config["alpha"]),
                dropout=config["dropout"],
            )
            
            # Create training dataset (with caching)
            train_dataset = WeightReproductionDataset(
                tokenizer=self.tokenizer,
                base_model=self.base_model,
                num_samples=self.num_samples,
                seq_length=config["seq_length"],
                device=self.device,
                pre_generate=True,
                cache_dir=self.cache_dir,
                model_name=self.model_name,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
            )
            
            # Create trainer
            trainer = WeightReproductionTrainer(
                base_model=self.base_model,
                lora_model=lora_model,
                tokenizer=self.tokenizer,
                device=self.device,
                learning_rate=config["learning_rate"],
            )
            
            # Train for max_epochs_per_trial
            best_val_loss = float('inf')
            
            for epoch in range(self.max_epochs_per_trial):
                train_metrics = trainer.train_epoch(train_loader, config["loss_type"])
                val_metrics = trainer.evaluate(self.val_loader, config["loss_type"])
                
                val_loss = val_metrics["loss"]
                val_mse = val_metrics["mse"]
                
                # Report to optuna for pruning
                trial.report(val_loss, epoch)
                
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
                
                best_val_loss = min(best_val_loss, val_loss)
                print(f"  Epoch {epoch+1}: val_loss={val_loss:.6f}, val_mse={val_mse:.6e}")
            
            # Final evaluation: compare with base model on test prompts
            comparator = ModelComparator(self.base_model, lora_model, self.tokenizer, self.device)
            benchmark_results = comparator.run_benchmark(
                prompts=self.test_prompts,
                max_new_tokens=30,
                do_sample=False,
                seed=42,
            )
            
            match_rate = benchmark_results["summary"]["match_rate"]
            avg_mse = benchmark_results["summary"]["avg_logits_mse"]
            avg_similarity = benchmark_results["summary"]["avg_token_similarity"]
            
            print(f"  Final: match_rate={match_rate:.4f}, mse={avg_mse:.6e}, sim={avg_similarity:.4f}")
            
            # Combined score (lower is better)
            # Prioritize match rate, then MSE, then similarity
            # If we achieve target match rate, strongly favor this trial
            if match_rate >= self.target_match_rate:
                score = -1000 + avg_mse  # Large negative bonus for success
            else:
                score = (1 - match_rate) * 100 + avg_mse * 1000 + (1 - avg_similarity) * 10
            
            # Save trial results
            trial_result = {
                "trial_number": trial.number,
                "config": config,
                "metrics": {
                    "val_loss": best_val_loss,
                    "match_rate": match_rate,
                    "avg_mse": avg_mse,
                    "avg_similarity": avg_similarity,
                },
                "score": score,
            }
            
            result_path = os.path.join(self.output_dir, f"trial_{trial.number:03d}.json")
            with open(result_path, "w") as f:
                json.dump(trial_result, f, indent=2)
            
            return score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return float('inf')
    
    def search(self) -> optuna.Study:
        """Run the hyperparameter search."""
        print(f"\n{'='*60}")
        print(f"Starting Hyperparameter Search")
        print(f"Model: {self.model_name}")
        print(f"Trials: {self.n_trials}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Create study with TPE sampler and median pruner
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        
        # Save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study: optuna.Study):
        """Save search results."""
        results = {
            "best_trial": {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_params,
            },
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": t.state.name,
                }
                for t in study.trials
            ],
            "statistics": {
                "n_trials": len(study.trials),
                "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            },
        }
        
        result_path = os.path.join(self.output_dir, "search_results.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Search Complete!")
        print(f"{'='*60}")
        print(f"Best trial: #{study.best_trial.number}")
        print(f"Best score: {study.best_trial.value:.6f}")
        print(f"Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"\nResults saved to: {result_path}")
        
        # Print hyperparameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\nHyperparameter Importance:")
            for key, value in importance.items():
                print(f"  {key}: {value:.4f}")
        except:
            pass


def train_with_best_params(
    study: optuna.Study,
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    device: str = "auto",
    num_epochs: int = 20,
    num_samples: int = 10000,
    output_dir: str = "./lora_checkpoints/autoresearch_best",
):
    """
    Train a final model using the best hyperparameters found.
    """
    from trainer import train_lora_to_reproduce_base
    
    best_params = study.best_params
    
    print(f"\nTraining final model with best parameters:")
    print(json.dumps(best_params, indent=2))
    
    return train_lora_to_reproduce_base(
        model_name=model_name,
        rank=best_params["rank"],
        num_epochs=num_epochs,
        batch_size=best_params["batch_size"],
        num_samples=num_samples,
        seq_length=best_params["seq_length"],
        learning_rate=best_params["learning_rate"],
        output_dir=output_dir,
        device=device,
        loss_type=best_params["loss_type"],
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter search for LoRA training")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--max-epochs", type=int, default=5, help="Max epochs per trial")
    parser.add_argument("--num-samples", type=int, default=2000, help="Training samples per trial")
    parser.add_argument("--output-dir", default="./autoresearch_results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-final", action="store_true", help="Train final model with best params")
    parser.add_argument("--final-epochs", type=int, default=20, help="Epochs for final training")
    
    args = parser.parse_args()
    
    searcher = LoRAHyperparameterSearch(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        max_epochs_per_trial=args.max_epochs,
        num_samples=args.num_samples,
    )
    
    study = searcher.search()
    
    if args.train_final:
        train_with_best_params(
            study,
            model_name=args.model,
            device=args.device,
            num_epochs=args.final_epochs,
            output_dir=os.path.join(args.output_dir, "best_model"),
        )

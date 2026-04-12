# LoRA Hyperparameter Autoresearch Guide

## Overview

The autoresearch module uses **Optuna** with Bayesian optimization to automatically find the best hyperparameters for training LoRA to reproduce base model weights. This is much more efficient than manual tuning.

## Why Autoresearch?

Manual hyperparameter tuning is:
- **Time-consuming**: Each configuration requires full training
- **Suboptimal**: Easy to miss good parameter combinations
- **Non-intuitive**: Interactions between parameters are complex

Autoresearch:
- Uses Bayesian optimization (TPE sampler) to explore efficiently
- Prunes unpromising trials early
- Finds parameter interactions automatically
- Provides importance rankings for each hyperparameter

## Usage

### Quick Start

```bash
make autoresearch
```

This runs 30 trials with 5 epochs each, then trains a final model with the best parameters.

### Custom Search

```bash
python3 scripts/autoresearch.py \
  --n-trials 50 \
  --max-epochs 5 \
  --num-samples 3000 \
  --target-match-rate 0.95 \
  --train-final \
  --final-epochs 30 \
  --output-dir ./my_search
```

### Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-trials` | 30 | Number of hyperparameter combinations to try |
| `--max-epochs` | 5 | Max training epochs per trial |
| `--num-samples` | 2000 | Training samples per trial |
| `--target-match-rate` | 0.95 | Target accuracy to achieve |
| `--train-final` | False | Train final model with best params |
| `--final-epochs` | 20 | Epochs for final model training |

## Search Space

The autoresearch explores:

### LoRA Parameters
- **rank**: 4, 8, 12, 16, 20, ..., 64 (step 4)
- **alpha**: 1.0 to 64.0 (continuous)
- **dropout**: 0.0 to 0.1 (continuous)

### Training Parameters
- **learning_rate**: 1e-5 to 1e-3 (log scale)
- **batch_size**: [2, 4, 8, 16]
- **seq_length**: 64, 128, 192, 256
- **loss_type**: ["mse", "kl", "cosine"]

## How It Works

1. **Trial Creation**: Optuna suggests hyperparameters
2. **Training**: Each trial trains for up to `max_epochs`
3. **Pruning**: Unpromising trials stopped early
4. **Scoring**: Combined validation loss + match rate
5. **Final Training**: Optional full training with best params

## Results

After search completes:

```bash
# Best hyperparameters
cat autoresearch_results/search_results.json | jq '.best_trial.params'

# All trial results
ls autoresearch_results/trial_*.json
```

### Interpreting Results

**Best Trial Score:**
- Negative (< -1000): Achieved target match rate!
- 0-100: Good progress, may need more training
- >100: Struggling, consider expanding search space


## Expected Timing

On AMD Strix Halo (Radeon 8060S):

| Phase | Samples | Estimated Time |
|-------|---------|----------------|
| Dataset generation | 500 | 2-5 minutes |
| Dataset generation | 2000 | 8-15 minutes |
| One training epoch | 500 samples | 1-2 minutes |
| One training epoch | 2000 samples | 4-8 minutes |
| Full trial (5 epochs) | 500 samples | 5-10 minutes |
| Full trial (5 epochs) | 2000 samples | 20-40 minutes |
| 20 trials | 500 samples | 2-4 hours |
| 50 trials | 2000 samples | 15-30 hours |

### If It Seems Stuck

The dataset generation phase has no progress bar initially. You should see:

```
Generating 500 training samples (seq_length=128)...
Generating dataset:   0%|          | 0/500 [00:00<?, ?it/s]
```

Within 30 seconds. If not:

1. **Check GPU activity**: `rocm-smi` in another terminal
2. **Check CPU activity**: `htop` or `top`
3. **Kill and restart**: Ctrl+C, then re-run with fewer samples

### Quick Test Mode

For a fast sanity check (< 10 minutes):

```bash
python3 scripts/autoresearch.py \
  --n-trials 5 \
  --max-epochs 2 \
  --num-samples 100 \
  --val-samples 20
```

## Dataset Caching

To avoid regenerating datasets on every run, the system caches generated datasets to disk.

### How It Works

- **Cache key**: Hash of (model_name, num_samples, seq_length, vocab_size)
- **Cache location**: `OUTPUT_DIR/cache/` (default: `./autoresearch_results/cache/`)
- **Cache format**: Pickled PyTorch tensors

### First Run (Slow)
```
Generating 500 training samples (seq_length=128)...
Generating dataset: 100%|██████████| 500/500 [02:30<00:00, 3.33it/s]
Saving dataset to cache: ./autoresearch_results/cache/...
  Saved 500 samples (45.2 MB)
```

### Subsequent Runs (Fast)
```
Loading dataset from cache: ./autoresearch_results/cache/...
  Loaded 500 samples from cache
```

### Cache Size Estimates

| Samples | Seq Length | Approx Size |
|---------|------------|-------------|
| 100 | 128 | ~9 MB |
| 500 | 128 | ~45 MB |
| 2000 | 128 | ~180 MB |
| 2000 | 256 | ~360 MB |

### Custom Cache Directory

```bash
# Use a shared cache location
python3 scripts/autoresearch.py \
  --cache-dir /mnt/fast_ssd/dataset_cache \
  --output-dir ./results
```

### Clearing Cache

```bash
# Remove all cached datasets
rm -rf ./autoresearch_results/cache/

# Or remove specific model cache
rm -rf ./autoresearch_results/cache/*SmolLM2*
```

### Cache Strategy for Multiple Runs

If running multiple experiments:

```bash
# Use a shared cache for all experiments
export CACHE_DIR="/mnt/cache/lora-reproduction"

# Run 1: Generates and caches
python3 scripts/autoresearch.py \
  --cache-dir $CACHE_DIR \
  --num-samples 2000

# Run 2: Uses cache (much faster)
python3 scripts/autoresearch.py \
  --cache-dir $CACHE_DIR \
  --num-samples 2000 \
  --n-trials 50

# Run 3: Different seq length, only generates what's new
python3 scripts/autoresearch.py \
  --cache-dir $CACHE_DIR \
  --num-samples 2000 \
  --search-space custom_space.py
```

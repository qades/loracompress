# LoRA Weight Reproduction Experiment

This project explores whether a full LLM can be reproduced using LoRA (Low-Rank Adaptation) matrices.

## Hypothesis

Given a base model $W \in \mathbb{R}^{d \times k}$, we want to train LoRA matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ with rank $r=16$ such that:

$$W' = W + BA \approx W_{target}$$

The hypothesis is that sufficiently trained LoRA adapters can approximate the original model weights closely enough to produce identical outputs.

## Architecture

### Phase 1: Baseline Benchmark
- Load two identical instances of `HuggingFaceTB/SmolLM2-135M`
- Verify they produce identical outputs given the same prompts, seeds, and parameters
- Establishes that our comparison framework works correctly

### Phase 2: LoRA Training
- Train LoRA adapters (rank=16) to reproduce base model behavior
- Loss function minimizes MSE between LoRA model logits and base model logits
- Iterative training with increasing sample sizes and epochs

### Phase 3: Validation
- Compare LoRA-augmented model against original base model
- Measure:
  - Exact output match rate
  - Token sequence similarity
  - Logits MSE (Mean Squared Error)
  - Maximum logit difference

## Quick Start

### Using Make

```bash
# Set up environment
make setup

# Run baseline test (Phase 1)
make baseline

# Train LoRA (Phase 2)
make train

# Compare specific checkpoint
make compare CHECKPOINT=./lora_checkpoints/best_model

# Run full pipeline
make full
```

### Using Python directly

```bash
# Phase 1: Baseline test
python3 scripts/baseline_test.py

# Phase 2: Train LoRA
python3 scripts/train_lora.py --rank 16 --epochs 10 --num-samples 10000

# Phase 3: Compare
python3 scripts/compare_models.py --lora-path ./lora_checkpoints/best_model

# Full pipeline
python3 scripts/run_full_pipeline.py
```

### Using Distrobox (Recommended for AMD Strix Halo)

```bash
# Create and enter container
distrobox create --name lora-reproduction \
  --image docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest
distrobox enter lora-reproduction

# Inside container
cd /workspace/lora-reproduction
make full
```

## Project Structure

```
lora-reproduction/
├── src/
│   ├── model_loader.py      # Model loading utilities
│   ├── benchmark.py         # Comparison framework
│   └── trainer.py           # LoRA training pipeline
├── scripts/
│   ├── baseline_test.py     # Phase 1: Baseline test
│   ├── train_lora.py        # Phase 2: Training script
│   ├── compare_models.py    # Phase 3: Comparison script
│   └── run_full_pipeline.py # Full iterative pipeline
├── configs/                 # Configuration files
├── results/                 # Benchmark results (generated)
├── lora_checkpoints/        # Trained LoRA weights (generated)
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container definition
├── Makefile                # Build automation
└── README.md               # This file
```

## Metrics

The benchmark produces the following metrics:

- **Exact Match Rate**: Percentage of prompts producing identical outputs
- **Token Match Rate**: Percentage with identical token sequences
- **Token Similarity**: Jaccard-like similarity between token sequences
- **Logits MSE**: Mean squared error between output logits
- **Logits Max Diff**: Maximum difference in any logit value

## Expected Outcomes

### Baseline (Phase 1)
- Two identical models should achieve 100% exact match rate
- Logits MSE should be effectively zero (< 1e-10)

### Training (Phase 2)
- Training loss should decrease over epochs
- Logits MSE should converge toward zero
- Model should learn to reproduce base behavior

### Validation (Phase 3)
- Success: >95% match rate between LoRA and base model
- Partial: 50-95% match rate (may improve with more training)
- Failure: <50% match rate (hypothesis may need revision)

## Configuration

Training hyperparameters can be adjusted in `scripts/train_lora.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rank` | 16 | LoRA rank |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--num-samples` | 10000 | Training samples |
| `--seq-length` | 128 | Sequence length |
| `--lr` | 1e-4 | Learning rate |
| `--loss-type` | mse | Loss function (mse/kl/cosine) |

## Hardware Support

This project supports:
- **NVIDIA GPUs** (CUDA)
- **AMD GPUs** (ROCm) - Tested on AMD Strix Halo
- **CPU** (slower but functional)

The code automatically detects the available hardware and uses the appropriate backend.

### AMD Strix Halo Setup

Use the provided Distrobox container which has ROCm pre-configured:

```bash
distrobox create --name lora-reproduction \
  --image docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest
distonumber enter lora-reproduction

cd /workspace/lora-reproduction
make full
```

## Hyperparameter Autoresearch

Instead of manually tuning hyperparameters, use the autoresearch feature to find optimal settings:

```bash
# Quick search (30 trials)
make autoresearch

# Custom search
python3 scripts/autoresearch.py \
  --n-trials 50 \
  --max-epochs 5 \
  --num-samples 3000 \
  --target-match-rate 0.95 \
  --train-final \
  --final-epochs 30
```

The autoresearch will:
1. Try different combinations of rank, alpha, learning rate, batch size, etc.
2. Use Bayesian optimization (TPE sampler) to find promising regions
3. Prune unpromising trials early to save time
4. Report the best hyperparameters found
5. Optionally train a final model with the best settings

### Search Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `rank` | 4-64 | LoRA rank |
| `alpha` | 1-64 | LoRA alpha scaling |
| `dropout` | 0.0-0.1 | LoRA dropout |
| `learning_rate` | 1e-5 to 1e-3 | Learning rate (log scale) |
| `batch_size` | 2, 4, 8, 16 | Training batch size |
| `seq_length` | 64-256 | Sequence length |
| `loss_type` | mse, kl, cosine | Loss function |

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA or ROCm support)
- Transformers 4.35+
- PEFT 0.7+
- Optuna 3.4+ (for autoresearch)
- ~4GB GPU memory for SmolLM2-135M

### ⚠️ ROCm GPU Issue (Known Bug)

**Problem**: GPU tensor operations hang indefinitely on ROCm in this container.

**Symptoms**: Process freezes at `tensor.cuda()` or `model.to('cuda')`.

**Workaround**: All scripts now default to **CPU mode**. Training still works, just slower.

```bash
# This works (CPU mode, default)
make train-weights-test

# This may hang (GPU mode)
make train-weights-gpu  # Use at own risk
```

**Check GPU status**:
```bash
python3 scripts/test_env.py
# or
python3 src/gpu_detector.py
```

**Notes**:
- The `bitsandbytes` library is **pre-installed** in the AMD distrobox with ROCm support
- PyTorch ROCm builds use the same `torch.cuda` API for compatibility
- ROCm memory management is handled automatically
- For 8-bit/4-bit quantization, bitsandbytes is available in the container

## License

MIT License - See individual component licenses for dependencies.

## Direct Weight Reproduction (Alternative Approach)

Instead of training on dataset samples, you can train LoRA to directly approximate the weight matrices:

```bash
# Zero out base model, train BA = W_target (CPU mode, default)
make train-weights

# Quick test (< 1 minute on CPU)
make train-weights-test

# Test just the algorithm (no model loading, instant)
make test-logic

# Or with custom parameters
python3 scripts/train_lora_weights.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --rank 64 \
  --epochs 200 \
  --lr 1e-3 \
  --device cpu
```

### Math

**Standard LoRA:**
```
W = W_0 + BA
```
where W_0 is frozen pre-trained weights.

**Weight Reproduction LoRA:**
```
W_0 = 0 (zeroed out)
W = BA ≈ W_target
```

We directly optimize:
```
loss = ||W_target - BA||²
```

### When to Use This

| Approach | Best For | Speed | Accuracy |
|----------|----------|-------|----------|
| Dataset-based | Behavioral matching | Slower | Good for outputs |
| Weight-based | Exact weight reproduction | Faster | Good for weights |

### Expected Results

With rank 16 on SmolLM2-135M:
- **Weight MSE**: ~1e-4 to 1e-6 (depending on matrix rank)
- **Output match**: 80-95% (lower than dataset method for outputs)
- **Training time**: ~5 minutes for 100 epochs

The weight-based method optimizes for matrix approximation, not output behavior. Use dataset-based training if you care more about output similarity.

## Autoresearch for Direct Weight Reproduction

The new **direct weight** autoresearch finds optimal hyperparameters for training LoRA to reproduce weight matrices directly:

```bash
# Quick test (3 trials, rank 4-16, 50 epochs)
make autoresearch-weights-test

# Standard search (20 trials, rank 4-64, 100 epochs)
make autoresearch-weights

# Custom search
python3 scripts/autoresearch_weights.py \
  --n-trials 30 \
  --rank-min 8 \
  --rank-max 128 \
  --epochs 150 \
  --train-final \
  --final-epochs 300
```

### Search Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `rank` | 4-128 (configurable) | LoRA rank - higher = better approximation |
| `learning_rate` | 1e-4 to 1e-2 | Step size for optimization |

### Comparison: Dataset vs Direct Weight

| Feature | Dataset Method | Direct Weight Method |
|---------|---------------|----------------------|
| **Goal** | Match outputs | Match weights |
| **Loss** | MSE(logits) | MSE(weight matrices) |
| **Training data** | Generated samples | None (direct) |
| **Speed** | Slow (needs dataset) | Fast |
| **Best for** | Behavioral cloning | Weight reproduction |
| **Autoresearch** | `make autoresearch` | `make autoresearch-weights` |

### Expected Results

For SmolLM2-135M with rank 64:
- **Weight MSE**: ~1e-5 to 1e-6
- **Relative error**: 0.1-1%
- **Training time**: ~5 minutes on CPU

The direct weight method is **recommended** for your use case since you want to reproduce the actual model weights, not just match outputs.

## Single Layer Testing & Per-Layer Rank Optimization

Test LoRA compression on a **single layer** before processing the full model:

```bash
# Test one layer with multiple ranks (finds optimal rank)
make test-single-layer

# List all layers to pick a specific one
make list-layers

# Test specific layer with custom ranks
python3 scripts/test_single_layer.py \
  --layer-idx 15 \
  --module q_proj \
  --ranks 4 8 16 32 64 \
  --epochs 100
```

### Per-Layer Rank Optimization

Different layers may need different ranks! Attention layers (q/k/v/o_proj) are typically smaller than MLP layers (gate/up/down_proj).

```bash
# Test attention layer (576×576)
python3 scripts/test_single_layer.py --module q_proj --ranks 4 8 16

# Test MLP layer (576×1536 - bigger, may need higher rank)
python3 scripts/test_single_layer.py --module gate_proj --ranks 8 16 32 64
```

### Layer-wise Full Model Compression

Process the entire model layer-by-layer (enables arbitrary model sizes):

```bash
# Standard compression
make compress-model

# Quick test
make compress-model-quick

# Custom settings
python3 scripts/compress_model_layerwise.py \
  --rank 32 \
  --epochs 150 \
  --lr 5e-4 \
  --test
```

**Storage savings example** (rank 16):
- q_proj (576×576): 331k params → 18k params (18x compression)
- gate_proj (576×1536): 884k params → 34k params (26x compression)

### Why This Matters

1. **Memory efficiency**: Process layers sequentially, O(1) memory regardless of model size
2. **Per-layer optimization**: Different layers need different ranks
3. **Storage compression**: Store ~10-50x smaller models
4. **Runtime tradeoff**: Slower inference (compute BA on-the-fly) but massive storage savings

## Thorough Single Layer Autoresearch

For **convergence-verified** results with proper hyperparameter search:

```bash
# Full autoresearch (rank + LR + epochs to convergence)
make autoresearch-single-layer

# Or with custom layer
python3 scripts/autoresearch_single_layer.py \
  --layer-idx 20 \
  --module v_proj \
  --output ./my_results.json
```

### What It Tests

1. **Rank analysis**: Tests ranks 4, 8, 16, 32, 64 with **early stopping** until convergence
2. **Learning rate**: Finds optimal LR for best rank
3. **Final verification**: Retrains best config with longer patience to verify true minimum

### Why 0.17% Might Be Misleading

The quick test (100 epochs) showed 0.17% error, but:
- May not have converged yet
- Learning rate might not be optimal
- Errors compound across layers

The thorough autoresearch:
- Trains until loss stops improving (patience-based early stopping)
- Tests multiple learning rates
- Verifies convergence with extended training

### Expected Outcome

If error stays < 0.5% after thorough testing:
→ **Model reproduction is feasible!**

If error increases to > 2%:
→ Need higher rank or different approach

## Full Model Compression Pipeline

Compress the entire model with layer-adaptive ranks, then benchmark:

```bash
# Full pipeline: compress + decompress + benchmark
make full-pipeline

# Or step by step:
make compress-model      # Compress with adaptive ranks
make benchmark-compressed # Decompress and benchmark
```

### What It Does

1. **Compress**: Each layer gets optimal rank based on type:
   - Attention (q/k/v/o): rank = max(4, int(min_dim × 2.8%))
   - MLP (gate/up/down): rank = max(4, int(min_dim × 0.7%))

2. **Decompress**: Reconstruct full model from LoRA weights

3. **Benchmark**: Compare original vs reconstructed on test prompts

### Expected Results

- **Compression**: ~50x (135M → 2.7M parameters)
- **Per-layer error**: 0.09-0.15%
- **Output match rate**: 80-100% (depending on prompts)

### Output Files

```
compressed_model/
├── config.json              # Model config
├── compressed_weights.pt    # LoRA weights (A, B matrices)
├── metadata.json            # Compression stats
├── benchmark_results.json   # Comparison results
└── tokenizer files
```

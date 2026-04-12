# Makefile for LoRA Reproduction Experiment

.PHONY: help setup baseline train compare full clean

help:
	@echo "LoRA Weight Reproduction Experiment"
	@echo ""
	@echo "=== COMPRESSION (NEW) ==="
	@echo "  compress-model      - Compress full model layer-by-layer"
	@echo "  compress-resume     - Resume, fix low-quality layers (relaxed thresholds)"
	@echo "  compress-improve    - Resume, aggressive quality improvement"
	@echo "  compress-status     - Show current compression status"
	@echo "  compress-only       - Compress only specific modules (e.g., k_proj,v_proj)"
	@echo "  compress-test       - Test compression on first 5 layers"
	@echo "  compress-dry-run    - Preview what would be re-compressed"
	@echo "  decode              - Decompress and benchmark"
	@echo "  decode-filtered     - Use only high-quality compressed layers"
	@echo "  decode-compare      - Compare compressed vs original outputs"
	@echo ""
	@echo "=== GPU / ROCm ==="
	@echo "  fix-gpu             - Check/fix GPU setup for AMD Strix Halo"
	@echo "  fix-gpu-export      - Show env var export commands"
	@echo "  pin-rocm-packages   - Pin firmware/kernel to stable versions"
	@echo ""
	@echo "=== ANALYSIS ==="
	@echo "  list-layers         - List all layers with optimal rank prediction"
	@echo "  test-single-layer   - Quick single layer test (ranks 4,8,16,32)"
	@echo "  autoresearch-single-layer - THOROUGH single layer autoresearch"
	@echo "  autoresearch-l1     - L1 quality search (target <5% error)"
	@echo "  autoresearch-l1-high - L1 quality search (target <3% error)"
	@echo "  autoresearch-l1-advanced - ADVANCED: trap detection + adaptive noise"
	@echo ""
	@echo "=== TRAINING (LEGACY) ==="
	@echo "  setup               - Set up Python environment"
	@echo "  baseline            - Run Phase 1: Baseline test"
	@echo "  train               - Train LoRA model (dataset-based)"
	@echo "  train-weights       - Train LoRA to directly reproduce weights"
	@echo "  clean               - Clean generated files"
	@echo "  docker              - Build Docker image"

setup:
	pip install -r requirements.txt
	mkdir -p results lora_checkpoints

baseline: setup
	python3 scripts/baseline_test.py

train: setup
	python3 scripts/train_lora.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--rank 16 \
		--epochs 10 \
		--batch-size 4 \
		--num-samples 10000 \
		--output-dir ./lora_checkpoints

compare: setup
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Usage: make compare CHECKPOINT=./lora_checkpoints/best_model"; \
		exit 1; \
	fi
	python3 scripts/compare_models.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--lora-path $(CHECKPOINT)

full: setup
	python3 scripts/run_full_pipeline.py

clean:
	rm -rf results/*.json
	rm -rf lora_checkpoints/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker:
	docker build -t lora-reproduction:latest .

distobox:
	@echo "Creating distrobox container..."
	@if command -v distrobox &> /dev/null; then \
		distrobox create --name lora-reproduction --image docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest; \
		distrobox enter lora-reproduction; \
	else \
		echo "distrobox not found. Please install distrobox first."; \
		echo "See: https://github.com/89luca89/distrobox"; \
	fi

# Autoresearch - OLD: dataset-based approach (logit matching)
autoresearch: setup
	python3 scripts/autoresearch.py \
		--n-trials 20 \
		--max-epochs 5 \
		--num-samples 500 \
		--train-final \
		--final-epochs 20

# Autoresearch for DIRECT WEIGHT reproduction (recommended)
autoresearch-weights: setup
	python3 scripts/autoresearch_weights.py \
		--n-trials 20 \
		--rank-min 4 \
		--rank-max 64 \
		--epochs 100 \
		--train-final \
		--final-epochs 200

# Quick test of weight autoresearch
autoresearch-weights-test: setup
	python3 scripts/autoresearch_weights.py \
		--quick-test

# Direct weight reproduction (zero base model, train BA = W_target) - CPU by default
train-weights: setup
	python3 scripts/train_lora_weights.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--rank 16 \
		--epochs 100 \
		--lr 1e-3 \
		--device cpu \
		--output-dir ./lora_checkpoints/weights_reproduction

# Quick test of weight reproduction (< 30 seconds) - CPU
train-weights-test: setup
	python3 scripts/train_lora_weights.py \
		--quick-test \
		--device cpu \
		--output-dir ./lora_checkpoints/weights_test

# Test just the algorithm logic (fast, no model loading)
test-logic: setup
	python3 scripts/test_weight_logic.py

# Test SINGLE layer compression with rank autoresearch
test-single-layer: setup
	python3 scripts/test_single_layer.py \
		--module q_proj \
		--ranks 4 8 16 32 \
		--epochs 100

# Efficient staged autoresearch (LR → Epochs → Rank)
autoresearch-efficient: setup
	python3 scripts/autoresearch_efficient.py \
		--layer-idx 15 \
		--module q_proj \
		--output ./autoresearch_efficient.json

# Smart autoresearch (high-LR exploration + dimension correlation)
autoresearch-smart: setup
	python3 scripts/autoresearch_smart.py \
		--layer-idx 15 \
		--module q_proj \
		--output ./autoresearch_smart.json

# FULL hyperparameter autoresearch (rank + LR + scheduler + optimizer)
autoresearch-full: setup
	python3 scripts/autoresearch_full.py \
		--layer-idx 15 \
		--module q_proj \
		--output ./autoresearch_full.json

# Test if rank-64 recipe transfers to lower ranks
test-hp-transfer: setup
	python3 scripts/test_hp_transfer.py \
		--layer-idx 15 \
		--module q_proj

# Analyze if optimal rank correlates with layer dimensions
analyze-rank-dim: setup
	python3 scripts/analyze_rank_vs_dimension.py \
		--layer-idx 15 \
		--output ./rank_dimension_analysis.json

# THOROUGH autoresearch on single layer (slower, more exhaustive)
autoresearch-single-layer: setup
	python3 scripts/autoresearch_single_layer.py \
		--layer-idx 15 \
		--module q_proj \
		--output ./autoresearch_single_layer.json

# L1 quality autoresearch - find configs for <5% error (Q4_K_M benchmark)
autoresearch-l1: setup
	python3 scripts/autoresearch_l1_quality.py \
		--n-trials 50 \
		--target-quality 5.0

# L1 quality - aggressive <3% error target
autoresearch-l1-high: setup
	python3 scripts/autoresearch_l1_quality.py \
		--n-trials 50 \
		--target-quality 3.0

# L1 quality - experimental with noise injection
autoresearch-l1-noise: setup
	python3 scripts/autoresearch_l1_quality.py \
		--n-trials 30 \
		--target-quality 5.0 \
		--noise-std 0.001 \
		--noise-every 100

# L1 quality - ADVANCED MODE (trap detection + adaptive noise + SWA)
autoresearch-l1-advanced: setup
	python3 scripts/autoresearch_l1_quality.py \
		--n-trials 50 \
		--target-quality 5.0 \
		--advanced

# List all layers with predicted optimal ranks
list-layers: setup
	python3 scripts/list_layers.py --limit 40

# Compress full model with adaptive layer-wise ranks
compress-model: setup
	python3 scripts/compress_full_model.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--device cpu \
		--output-dir ./compressed_model

# Resume compression, only re-compress low quality layers
compress-resume: setup
	python3 scripts/compress_full_model.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--device cpu \
		--output-dir ./compressed_model \
		--resume \
		--max-error 10.0 \
		--min-compression 10.0

# Aggressive quality improvement - tighten error thresholds
compress-improve: setup
	python3 scripts/compress_full_model.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--device cpu \
		--output-dir ./compressed_model \
		--resume \
		--max-error 6.0 \
		--min-compression 15.0

# Preview what would be re-compressed with current thresholds
compress-status:
	@python3 scripts/check_compression_status.py 2>/dev/null || echo "Run compress-model first"

# Compress only specific modules (for targeted improvement)
compress-only: setup
	python3 scripts/compress_full_model.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--device cpu \
		--output-dir ./compressed_model \
		--resume \
		--only k_proj,v_proj

# Test compression on first 5 layers only
compress-test: setup
	python3 scripts/compress_full_model.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--device cpu \
		--output-dir ./compressed_model_test \
		--limit 5

# Dry run to see what would be processed
compress-dry-run: setup
	python3 scripts/compress_full_model.py \
		--model HuggingFaceTB/SmolLM2-135M \
		--output-dir ./compressed_model \
		--resume \
		--max-error 8.0 \
		--dry-run

# Decompress and benchmark
benchmark-compressed: setup
	python3 scripts/decompress_model.py \
		--compressed-dir ./compressed_model \
		--device cpu \
		--max-tokens 30

# New: Decode with filtering and fallback
decode: setup
	python3 scripts/decompress_and_benchmark.py \
		--compressed-dir ./compressed_model \
		--output ./benchmark_results.json

# Decode with quality thresholds (only use high-quality compressed layers)
decode-filtered: setup
	python3 scripts/decompress_and_benchmark.py \
		--compressed-dir ./compressed_model \
		--max-error 10.0 \
		--min-compression 15.0 \
		--output ./benchmark_filtered.json

# Decode and compare with original
decode-compare: setup
	python3 scripts/decompress_and_benchmark.py \
		--compressed-dir ./compressed_model \
		--compare \
		--output ./benchmark_comparison.json

# Full pipeline: compress + decompress + benchmark
full-pipeline: compress-model decode

# Force GPU mode (may hang on ROCm)
train-weights-gpu: setup
	python3 scripts/train_lora_weights.py \
		--quick-test \
		--device cuda \
		--output-dir ./lora_checkpoints/weights_test_gpu

# GPU/ROCm fixes for AMD Strix Halo (gfx1151)
fix-gpu:
	python3 scripts/fix_rocm_gpu.py

fix-gpu-export:
	@echo "Run this to set environment variables:"
	@echo '  eval $$(python3 scripts/fix_rocm_gpu.py --export)'

# Pin ROCm packages to prevent breaking updates
pin-rocm-packages:
	@echo "This will pin kernel and firmware packages to stable versions"
	@echo "Requires sudo. Run: sudo bash scripts/pin_rocm_packages.sh"

# Development helpers
format:
	black src/ scripts/
	isort src/ scripts/

lint:
	flake8 src/ scripts/
	pylint src/ scripts/

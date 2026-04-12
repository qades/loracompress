#!/usr/bin/env python3
"""Quick test to debug model loading."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from model_loader import load_base_model

print("Test 1: Loading model once (no zero)")
start = time.time()
model1, tokenizer = load_base_model("HuggingFaceTB/SmolLM2-135M", "auto", zero_weights=False)
print(f"Total time: {time.time()-start:.1f}s")
print(f"Model device: {next(model1.parameters()).device}")

# Clean up
del model1
import torch
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nTest 2: Loading model with zero_weights")
start = time.time()
model2, _ = load_base_model("HuggingFaceTB/SmolLM2-135M", "auto", zero_weights=True)
print(f"Total time: {time.time()-start:.1f}s")

print("\nAll tests passed!")

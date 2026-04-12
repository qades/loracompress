#!/usr/bin/env python3
"""Test model loading step by step."""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

print("=" * 60)
print("Model Loading Test")
print("=" * 60)

# Test 1: Import transformers
print("\n1. Importing transformers...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
    print("   ✓ transformers imported")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Initialize CUDA first (ROCm workaround)
print("\n2. Initializing CUDA...")
try:
    torch.cuda.set_device(0)
    torch.cuda.init()
    print(f"   ✓ CUDA initialized: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load tokenizer
print("\n3. Loading tokenizer...")
try:
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        trust_remote_code=True
    )
    print(f"   ✓ Tokenizer loaded in {time.time()-start:.1f}s")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Load model config
print("\n4. Loading model config...")
try:
    start = time.time()
    config = AutoConfig.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        trust_remote_code=True
    )
    print(f"   ✓ Config loaded in {time.time()-start:.1f}s")
    print(f"   Model: {config.num_hidden_layers} layers, "
          f"{config.hidden_size} hidden dim")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 5: Load model to CPU first
print("\n5. Loading model to CPU...")
try:
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        trust_remote_code=True,
    )
    elapsed = time.time() - start
    print(f"   ✓ Model loaded to CPU in {elapsed:.1f}s")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {total_params/1e6:.1f}M parameters")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Move to GPU
print("\n6. Moving model to GPU...")
try:
    start = time.time()
    model = model.cuda()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   ✓ Model moved to GPU in {elapsed:.1f}s")
    print(f"   Device: {next(model.parameters()).device}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)

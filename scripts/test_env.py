#!/usr/bin/env python3
"""Test if PyTorch and ROCm are working."""
import sys
import time
import os

print("Testing PyTorch environment...")
print(f"Python: {sys.version}")

# Check for ROCm environment
print("\nEnvironment check:")
print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}")
print(f"  PYTORCH_HIP_ALLOC_CONF: {os.environ.get('PYTORCH_HIP_ALLOC_CONF', 'not set')}")

# Test 1: Import torch
print("\n1. Importing torch...")
try:
    import torch
    print(f"   ✓ torch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ Device count: {torch.cuda.device_count()}")
        print(f"   ✓ Device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Create a tensor
print("\n2. Creating tensor on CPU...")
try:
    x = torch.randn(100, 100)  # Smaller tensor
    print(f"   ✓ Created tensor: {x.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Try GPU with timeout
print("\n3. Testing GPU (with 5 second timeout)...")
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("GPU operation timed out")

try:
    # Set alarm for 5 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    
    print("   Initializing CUDA...")
    torch.cuda.set_device(0)
    
    print("   Moving small tensor to GPU...")
    start = time.time()
    x_gpu = x.cuda()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Cancel alarm if successful
    signal.alarm(0)
    
    print(f"   ✓ Moved to GPU in {elapsed:.2f}s")
    print(f"   ✓ Device: {x_gpu.device}")
    GPU_WORKING = True
    
except TimeoutError:
    print("   ✗ GPU operation timed out after 5s")
    print("   → GPU mode not working, will use CPU fallback")
    GPU_WORKING = False
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("   → GPU mode not working, will use CPU fallback")
    GPU_WORKING = False

# Test 4: CPU computation (always works)
print("\n4. Running CPU computation...")
try:
    start = time.time()
    y = torch.matmul(x, x)
    print(f"   ✓ CPU computation done in {time.time()-start:.2f}s")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
if GPU_WORKING:
    print("✓ GPU mode working!")
else:
    print("⚠ GPU mode NOT working - use --device cpu")
print("=" * 60)

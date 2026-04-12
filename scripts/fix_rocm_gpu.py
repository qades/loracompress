#!/usr/bin/env python3
"""
ROCm GPU Fix Script for AMD Strix Halo (gfx1151)
Sets environment variables and tests GPU functionality.

CRITICAL: Kernel boot parameters must be set separately (requires root + reboot):
    iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856
"""
import os
import sys
import subprocess
import torch


def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_dmesg_errors():
    """Check dmesg for amdgpu errors."""
    try:
        result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
        dmesg = result.stdout.lower()
        
        errors = []
        if 'xnack disabled' in dmesg or 'xnack' in dmesg and 'disabled' in dmesg:
            errors.append("XNACK disabled (check boot params)")
        if 'failed to load firmware' in dmesg:
            errors.append("Firmware load failed")
        if 'amdgpu: probe of' in dmesg and 'failed' in dmesg:
            errors.append("GPU probe failed")
        if 'vram' in dmesg and ('error' in dmesg or 'failed' in dmesg):
            errors.append("VRAM initialization issues")
        if 'irq' in dmesg and 'amdgpu' in dmesg and 'error' in dmesg:
            errors.append("IRQ errors")
            
        # Check for specific firmware version
        if '20251125' in result.stdout:
            errors.append("BUGGY FIRMWARE 20251125 DETECTED - needs upgrade!")
        if '20260110' in result.stdout:
            print(f"    ✓ Good firmware 20260110 detected")
            
        return errors
    except Exception:
        return []


def check_system():
    """Check kernel and firmware versions."""
    print_section("System Checks")
    
    # Kernel version
    try:
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        kernel = result.stdout.strip()
        print(f"  Kernel: {kernel}")
        
        # Parse version
        parts = kernel.split('.')
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        
        if major > 6 or (major == 6 and minor >= 18):
            print(f"    ✓ Kernel 6.18+ (recommended)")
        else:
            print(f"    ⚠ Kernel < 6.18 (may have issues)")
    except Exception as e:
        print(f"    Could not check kernel: {e}")
    
    # Firmware check via dmesg
    errors = check_dmesg_errors()
    if errors:
        print(f"\n  ⚠ Kernel/dmesg errors detected:")
        for err in errors:
            print(f"    - {err}")
    
    # ROCm version
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"    ✓ rocminfo available")
            # Check for gfx1151
            if 'gfx1151' in result.stdout:
                print(f"    ✓ gfx1151 detected in rocminfo")
            elif 'gfx11' in result.stdout:
                print(f"    ⚠ gfx11 detected but not gfx1151 specifically")
        else:
            print(f"    ✗ rocminfo failed")
    except Exception as e:
        print(f"    rocminfo not available: {e}")


def check_current_env():
    """Check current environment variables."""
    print_section("Current Environment")
    
    rocm_vars = [
        'HSA_OVERRIDE_GFX_VERSION',
        'HSA_XNACK',
        'HSA_FORCE_FINE_GRAIN_PCIE',
        'HSA_ENABLE_SDMA',
        'ROCR_VISIBLE_DEVICES',
        'HIP_VISIBLE_DEVICES',
    ]
    
    for var in rocm_vars:
        value = os.environ.get(var, '<not set>')
        print(f"  {var}: {value}")
    
    # Check PyTorch ROCm version
    print("\n  PyTorch Info:")
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device count: {torch.cuda.device_count()}")
        print(f"    Device name: {torch.cuda.get_device_name(0)}")


def set_rocm_env():
    """Set required environment variables for gfx1151."""
    print_section("Setting ROCm Environment Variables")
    
    env_vars = {
        # Force ROCm to recognize gfx1151 architecture
        'HSA_OVERRIDE_GFX_VERSION': '11.5.1',
        
        # Enable XNACK for memory management
        'HSA_XNACK': '1',
        
        # Force fine-grain PCIe (helps with stability)
        'HSA_FORCE_FINE_GRAIN_PCIE': '1',
        
        # Disable SDMA to prevent hangs
        'HSA_ENABLE_SDMA': '0',
        
        # Additional stability flags
        'HSA_DISABLE_CACHE': '1',
        'AMD_SERIALIZE_KERNEL': '3',
        'AMD_SERIALIZE_COPY': '3',
        
        # GPU memory allocation
        'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:512',
    }
    
    for var, value in env_vars.items():
        current = os.environ.get(var)
        if current != value:
            os.environ[var] = value
            print(f"  ✓ Set {var}={value}")
        else:
            print(f"  ✓ {var} already set to {value}")
    
    return env_vars


def test_gpu_simple():
    """Ultra-simple GPU test - just allocate and move tensor."""
    print_section("Testing GPU (Simple Allocation)")
    
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("GPU operation timed out (hang)")
        
        # Set 10 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        
        print("  Testing: torch.tensor([1.0]).cuda()...")
        x = torch.tensor([1.0]).cuda()
        print(f"  ✓ Single element tensor on GPU: {x}")
        
        print("  Testing: torch.randn(10).cuda()...")
        y = torch.randn(10).cuda()
        print(f"  ✓ Small vector on GPU: shape={y.shape}")
        
        signal.alarm(0)  # Cancel timeout
        return True
        
    except TimeoutError as e:
        print(f"  ✗ {e}")
        print("    GPU is hanging - needs firmware/kernel fix")
        return False
    except Exception as e:
        print(f"  ✗ Simple GPU test failed: {e}")
        return False


def test_gpu_basic():
    """Basic GPU functionality test."""
    print_section("Testing GPU (Basic)")
    
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("GPU operation timed out (hang)")
        
        # Set 30 second timeout for whole test
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        # Test 1: CUDA available
        if not torch.cuda.is_available():
            print("  ✗ torch.cuda.is_available() = False")
            return False
        print(f"  ✓ CUDA/ROCm available")
        
        # Test 2: Device properties
        device = torch.cuda.current_device()
        print(f"  ✓ Current device: {device}")
        print(f"    Name: {torch.cuda.get_device_name(device)}")
        print(f"    Capability: {torch.cuda.get_device_capability(device)}")
        
        # Test 3: Simple tensor operation
        print("\n  Testing tensor operations...")
        x = torch.randn(100, 100).cuda()
        y = torch.matmul(x, x.T)
        print(f"  ✓ Matmul on GPU: {y.shape}")
        
        # Test 4: Small training step
        print("\n  Testing training step...")
        model = torch.nn.Linear(100, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for i in range(3):
            x = torch.randn(32, 100).cuda()
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"  ✓ Training step successful")
        
        # Test 5: Synchronize (catches async errors)
        torch.cuda.synchronize()
        print(f"  ✓ Synchronize successful")
        
        signal.alarm(0)  # Cancel timeout
        return True
        
    except TimeoutError as e:
        print(f"  ✗ {e}")
        print("\n  GPU is hanging. Common causes:")
        print("    1. Firmware version 20251125 (known broken)")
        print("    2. Missing XNACK support in kernel")
        print("    3. PyTorch/ROCm version mismatch")
        print("\n  Try updating:")
        print("    sudo apt update && sudo apt install linux-firmware")
        return False
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compression_workload():
    """Test actual compression workload (LoRA training)."""
    print_section("Testing Compression Workload")
    
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Simulate one layer compression
        d, k, rank = 576, 576, 16
        target = torch.randn(d, k).cuda() * 0.1
        
        A = nn.Parameter(torch.randn(rank, k).cuda() * 0.01)
        B = nn.Parameter(torch.randn(d, rank).cuda() * 0.01)
        optimizer = torch.optim.AdamW([A, B], lr=0.03)
        
        print(f"  Compressing {d}×{k} matrix with rank {rank}...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for epoch in range(100):
            optimizer.zero_grad()
            W_approx = torch.matmul(B, A)
            loss = F.mse_loss(W_approx, target)
            loss.backward()
            optimizer.step()
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        
        print(f"  ✓ 100 epochs completed in {elapsed_ms:.1f}ms")
        print(f"    ({elapsed_ms/100:.1f}ms per epoch)")
        
        # Calculate error
        with torch.no_grad():
            W_final = torch.matmul(B, A)
            l1_error = torch.mean(torch.abs(W_final - target)).item()
            mean_abs = torch.mean(torch.abs(target)).item()
            error_pct = l1_error / mean_abs * 100
        
        print(f"  ✓ Final L1 error: {error_pct:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_boot_params_info():
    """Print information about required kernel boot parameters."""
    print_section("CRITICAL: Kernel Boot Parameters")
    print("""
The following boot parameters MUST be set in your bootloader (requires root + reboot):

    iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856

How to set (GRUB example):
    1. Edit /etc/default/grub
    2. Find GRUB_CMDLINE_LINUX_DEFAULT
    3. Add parameters: GRUB_CMDLINE_LINUX_DEFAULT="... iommu=pt amdgpu.gttsize=126976 ttm.pages_limit=32505856"
    4. Run: sudo grub2-mkconfig -o /boot/grub2/grub.cfg
    5. Reboot

Current cmdline (check if params are set):
""")
    try:
        with open('/proc/cmdline') as f:
            cmdline = f.read().strip()
            print(f"    {cmdline[:200]}...")
            
        params = ['iommu=pt', 'amdgpu.gttsize', 'ttm.pages_limit']
        print("\n  Checking for required params:")
        for param in params:
            if param in cmdline:
                print(f"    ✓ {param} found")
            else:
                print(f"    ✗ {param} MISSING - needs boot config!")
    except Exception as e:
        print(f"    Could not read /proc/cmdline: {e}")


def print_pytorch_install_info():
    """Print PyTorch installation instructions."""
    print_section("PyTorch Installation (If GPU Still Not Working)")
    print("""
If GPU still hangs, reinstall PyTorch from AMD's gfx1151-specific repo:

    pip uninstall torch torchvision torchaudio
    pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchvision torchaudio

Current PyTorch install:
""")
    print(f"    {torch.__version__}")
    if 'rocm' in torch.__version__.lower():
        print("    ✓ ROCm build detected")
    else:
        print("    ✗ Not a ROCm build - needs reinstallation!")


def main():
    print("="*70)
    print("  ROCm GPU Fix Script for AMD Strix Halo (gfx1151)")
    print("="*70)
    
    # Check system first
    check_system()
    
    # Check current state
    check_current_env()
    
    # Set environment variables
    env_vars = set_rocm_env()
    
    # Print boot params info (critical!)
    print_boot_params_info()
    
    # Print PyTorch info
    print_pytorch_install_info()
    
    # Run tests
    print_section("Running GPU Tests")
    print("  Tests have timeouts - will not hang forever")
    print()
    
    # Start with ultra-simple test
    simple_ok = test_gpu_simple()
    if not simple_ok:
        print_section("DIAGNOSIS")
        print("  ✗ GPU allocation hangs immediately")
        print("\n  This usually means:")
        print("    1. linux-firmware-20251125 (broken version)")
        print("       Fix: sudo apt install linux-firmware-20260110")
        print("    2. Kernel XNACK not working")
        print("       Fix: Check for 'amdgpu: XNACK disabled' in dmesg")
        print("    3. Firmware not loading correctly")
        print("       Fix: Check 'dmesg | grep -i firmware | grep -i amdgpu'")
        
        # Print export commands
        print("\n  To apply environment variables in current shell:")
        print("    eval $(python3 scripts/fix_rocm_gpu.py --export)")
        return 1
    
    # If simple works, try basic
    basic_ok = test_gpu_basic()
    
    if basic_ok:
        compression_ok = test_compression_workload()
        
        print_section("Summary")
        if compression_ok:
            print("  ✓ GPU is working for compression!")
            print("\n  You can now run:")
            print("    make compress-model  # Will use GPU")
            return 0
        else:
            print("  ⚠ Basic GPU works but compression fails")
            print("  Try setting kernel boot parameters and reboot")
            return 1
    else:
        print_section("Summary")
        print("  ✗ GPU not working for full operations")
        print("\n  Simple allocation works but full test hangs.")
        print("  This suggests firmware or driver issues.")
        
        # Print export commands
        print("\n  To apply environment variables in current shell:")
        print("    eval $(python3 scripts/fix_rocm_gpu.py --export)")
        return 1


if __name__ == '__main__':
    if '--export' in sys.argv:
        # Print export commands for shell
        print('export HSA_OVERRIDE_GFX_VERSION=11.5.1')
        print('export HSA_XNACK=1')
        print('export HSA_FORCE_FINE_GRAIN_PCIE=1')
        print('export HSA_ENABLE_SDMA=0')
        print('export HSA_DISABLE_CACHE=1')
        print('export AMD_SERIALIZE_KERNEL=3')
        print('export AMD_SERIALIZE_COPY=3')
        print('export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512')
    else:
        sys.exit(main())

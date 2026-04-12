"""
GPU detection with automatic fallback to CPU.
Detects if GPU is actually working or hangs.
"""
import torch
import signal
import sys


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("GPU operation timed out")


def check_gpu_working(timeout_seconds: int = 5) -> bool:
    """
    Check if GPU is actually working (not just available).
    
    Args:
        timeout_seconds: How long to wait for GPU operation
        
    Returns:
        True if GPU works, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        # Try GPU operations
        torch.cuda.set_device(0)
        torch.cuda.init()
        x = torch.randn(10, 10)
        x_gpu = x.cuda()
        torch.cuda.synchronize()
        y = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()
        
        # Cancel alarm
        signal.alarm(0)
        
        return True
        
    except TimeoutError:
        print("WARNING: GPU timed out - falling back to CPU", flush=True)
        return False
    except Exception as e:
        print(f"WARNING: GPU error ({e}) - falling back to CPU", flush=True)
        return False


def get_best_device(preferred: str = "auto") -> str:
    """
    Get the best available device, with auto-fallback to CPU.
    
    Args:
        preferred: Preferred device ("auto", "cuda", "cpu")
        
    Returns:
        Device string ("cuda" or "cpu")
    """
    if preferred == "cpu":
        return "cpu"
    
    if preferred == "cuda":
        if check_gpu_working():
            return "cuda"
        else:
            print("WARNING: CUDA requested but not working, using CPU", flush=True)
            return "cpu"
    
    # Auto mode
    if check_gpu_working():
        return "cuda"
    else:
        return "cpu"


if __name__ == "__main__":
    print("Checking GPU status...")
    if check_gpu_working():
        print("✓ GPU is working")
    else:
        print("✗ GPU is not working - use CPU mode")
        sys.exit(1)

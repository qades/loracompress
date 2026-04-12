"""
Model loading utilities for baseline and LoRA models.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel, LoraConfig, get_peft_model
import copy
import os

# Import GPU detector if available
try:
    from gpu_detector import get_best_device
    HAS_GPU_DETECTOR = True
except ImportError:
    HAS_GPU_DETECTOR = False


def get_device(preferred: str = "auto") -> str:
    """
    Get the best available device.
    Works for CUDA (NVIDIA), ROCm (AMD), and CPU.
    Auto-detects if GPU is actually working (not just available).
    """
    if preferred != "auto":
        # If specific device requested, use it (unless it's cuda and broken)
        if preferred == "cuda" and HAS_GPU_DETECTOR:
            return get_best_device("cuda")
        return preferred
    
    # Auto mode with GPU health check
    if HAS_GPU_DETECTOR:
        return get_best_device("auto")
    
    # Fallback: just check availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
        print(f"GPU detected: {device_name}")
        return "cuda"
    
    return "cpu"


def load_base_model(model_name: str, device: str = "auto", dtype=torch.float32, zero_weights: bool = False):
    """
    Load a base model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        device: Device to use
        dtype: Data type for model weights
        zero_weights: If True, zero out all trainable weights (for LoRA reproduction)
    """
    import time
    device = get_device(device)
    
    print(f"  [DEBUG] Loading tokenizer...", flush=True)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  [DEBUG] Tokenizer loaded in {time.time()-start:.1f}s", flush=True)
    
    print(f"  [DEBUG] Loading model from_pretrained...", flush=True)
    start = time.time()
    
    try:
        # ROCm workaround: Set device first, avoid automatic device_map
        if device != "cpu":
            import torch
            torch.cuda.set_device(0)
            torch.cuda.init()
            print(f"  [DEBUG] CUDA initialized", flush=True)
        
        # Load without device_map to avoid potential ROCm issues
        print(f"  [DEBUG] Calling from_pretrained...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        print(f"  [DEBUG] from_pretrained completed in {time.time()-start:.1f}s", flush=True)
        
        # Move to device manually with explicit synchronization
        if device != "cpu":
            print(f"  [DEBUG] Moving model to {device}...", flush=True)
            start = time.time()
            # Use cuda() instead of to() for ROCm compatibility
            model = model.cuda()
            torch.cuda.synchronize()
            print(f"  [DEBUG] Model moved in {time.time()-start:.1f}s", flush=True)
        
    except Exception as e:
        print(f"  [DEBUG] Error during loading: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    if device == "cpu" or not torch.cuda.is_available():
        print(f"  [DEBUG] Moving model to {device}...", flush=True)
        start = time.time()
        model = model.to(device)
        print(f"  [DEBUG] Model moved in {time.time()-start:.1f}s", flush=True)
    
    if zero_weights:
        print("  Zeroing out base model weights (for LoRA reproduction)...", flush=True)
        start = time.time()
        zero_count = 0
        for name, param in model.named_parameters():
            # Zero out all trainable weights, but keep embeddings and norms
            # We zero Linear layer weights but not LayerNorm or Embedding
            if 'weight' in name and any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                             'gate_proj', 'up_proj', 'down_proj',
                                                             'fc', 'dense', 'query', 'key', 'value']):
                param.data.zero_()
                zero_count += 1
                param.requires_grad = False  # Freeze the zeroed weights
        print(f"    Zeroed out {zero_count} weight matrices in {time.time()-start:.1f}s", flush=True)
    
    return model, tokenizer


def load_model_for_comparison(model_name: str, device: str = "auto", seed: int = 42):
    """Load two identical instances of the same model for baseline testing."""
    set_seed(seed)
    model_a, tokenizer = load_base_model(model_name, device)
    
    # Clear cache to ensure independent loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        torch.hip.empty_cache()
    
    set_seed(seed)
    model_b, _ = load_base_model(model_name, device)
    
    return model_a, model_b, tokenizer


def create_lora_model(base_model, rank: int = 16, alpha: int = 16, dropout: float = 0.0, target_modules=None):
    """
    Create a LoRA-wrapped model.
    
    For SmolLM2, we target attention and MLP layers.
    """
    if target_modules is None:
        # Default target modules for most LLMs including SmolLM2
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(base_model, lora_config)
    return model


def load_lora_weights(base_model, lora_path: str):
    """Load LoRA weights onto a base model."""
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model


def merge_lora_weights(lora_model):
    """Merge LoRA weights into base model for inference."""
    return lora_model.merge_and_unload()

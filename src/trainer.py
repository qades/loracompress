"""
Training pipeline to train LoRA to reproduce base model weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
from typing import Optional, List, Dict, Tuple
import os
import json
import hashlib
import pickle
from model_loader import get_device

# Optional: bitsandbytes for 8-bit/4-bit training (pre-installed in ROCm distrobox)
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


class WeightReproductionDataset(Dataset):
    """
    Dataset that generates random inputs and uses base model outputs as targets.
    
    The goal is to train LoRA so that base_model(input) + lora_delta ≈ base_model(input)
    which means lora_delta should learn to reproduce the base model weights.
    
    Supports caching to disk to avoid regenerating data across runs.
    """
    
    def __init__(
        self,
        tokenizer,
        base_model,
        num_samples: int = 10000,
        seq_length: int = 128,
        vocab_size: Optional[int] = None,
        device: str = "cuda",
        cache_dir: Optional[str] = "./dataset_cache",
        pre_generate: bool = True,
        model_name: Optional[str] = None,
        gen_batch_size: int = 8,
    ):
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size or tokenizer.vocab_size
        self.device = device
        self.cache_dir = cache_dir
        self.model_name = model_name or "unknown_model"
        self.gen_batch_size = gen_batch_size
        
        # Generate cache key based on parameters
        self.cache_key = self._generate_cache_key()
        self.cache_path = os.path.join(cache_dir, f"{self.cache_key}.pkl") if cache_dir else None
        
        # Try to load from cache first
        self.cached_data = []
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"Cache file found: {self.cache_path}", flush=True)
            self._load_from_cache()
        elif pre_generate:
            # Pre-generate all data with progress bar
            self._generate_all(batch_size=gen_batch_size)
            # Save to cache if enabled
            if self.cache_path:
                self._save_to_cache()
    
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on dataset parameters."""
        # Create a hash of the key parameters
        key_string = (
            f"{self.model_name}_"
            f"n{self.num_samples}_"
            f"seq{self.seq_length}_"
            f"vocab{self.vocab_size}"
        )
        # Use MD5 for a shorter but unique key
        hash_obj = hashlib.md5(key_string.encode())
        short_hash = hash_obj.hexdigest()[:12]
        return f"{self.model_name.replace('/', '_')}_n{self.num_samples}_seq{self.seq_length}_{short_hash}"
    
    def _load_from_cache(self):
        """Load dataset from cache file."""
        print(f"Loading dataset from cache: {self.cache_path}")
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.cached_data = cache_data['data']
            print(f"  Loaded {len(self.cached_data)} samples from cache")
        except Exception as e:
            print(f"  Warning: Failed to load cache: {e}")
            print(f"  Regenerating dataset...")
            self.cached_data = []
            self._generate_all()
            self._save_to_cache()
    
    def _save_to_cache(self):
        """Save dataset to cache file."""
        if not self.cache_dir:
            return
        
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Saving dataset to cache: {self.cache_path}")
        try:
            cache_data = {
                'data': self.cached_data,
                'metadata': {
                    'num_samples': self.num_samples,
                    'seq_length': self.seq_length,
                    'vocab_size': self.vocab_size,
                    'model_name': self.model_name,
                }
            }
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            # Get file size
            size_mb = os.path.getsize(self.cache_path) / (1024 * 1024)
            print(f"  Saved {len(self.cached_data)} samples ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")
    
    def _generate_all(self, batch_size: int = 8):
        """
        Generate all dataset samples using batched generation.
        
        Args:
            batch_size: Number of samples to generate in parallel (default: 8)
        """
        import time
        start_time = time.time()
        
        print(f"Generating {self.num_samples} training samples (seq_length={self.seq_length})...", flush=True)
        print(f"  Model device: {next(self.base_model.parameters()).device}", flush=True)
        print(f"  Generation batch size: {batch_size}", flush=True)
        
        self.cached_data = []
        self.base_model.eval()
        
        num_batches = (self.num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating dataset"):
                # Calculate batch boundaries
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.num_samples)
                current_batch_size = end_idx - start_idx
                
                # Generate batch
                batch_samples = self._generate_batch(start_idx, current_batch_size)
                self.cached_data.extend(batch_samples)
        
        elapsed = time.time() - start_time
        samples_per_sec = len(self.cached_data) / elapsed
        print(f"  Generated {len(self.cached_data)} samples in {elapsed:.1f}s ({samples_per_sec:.1f} samples/sec)", flush=True)
    
    def _generate_batch(self, start_idx: int, batch_size: int):
        """
        Generate a batch of training samples.
        
        Args:
            start_idx: Starting index for seeding
            batch_size: Number of samples in this batch
            
        Returns:
            List of sample dicts with 'input_ids' and 'targets'
        """
        # Set seed for reproducibility
        torch.manual_seed(start_idx)
        
        # Generate random token IDs for entire batch [batch_size, seq_length]
        input_ids = torch.randint(
            0, self.vocab_size, (batch_size, self.seq_length),
            dtype=torch.long
        )
        
        # Get base model outputs (these are our targets)
        with torch.no_grad():
            inputs = {"input_ids": input_ids.to(self.device)}
            outputs = self.base_model(**inputs, output_hidden_states=False)
            # targets shape: [batch_size, seq_length, vocab_size]
            targets = outputs.logits
        
        # Split batch into individual samples
        samples = []
        for i in range(batch_size):
            samples.append({
                "input_ids": input_ids[i].cpu(),
                "targets": targets[i].cpu(),
            })
        
        return samples
    
    def _generate_sample(self, idx: int):
        """Generate a single training sample."""
        torch.manual_seed(idx)
        
        # Generate random token IDs
        input_ids = torch.randint(
            0, self.vocab_size, (self.seq_length,),
            dtype=torch.long
        )
        
        # Get base model outputs (these are our targets)
        with torch.no_grad():
            inputs = {"input_ids": input_ids.unsqueeze(0).to(self.device)}
            outputs = self.base_model(**inputs, output_hidden_states=False)
            # Target is the logits from base model
            targets = outputs.logits[0]  # [seq_length, vocab_size]
        
        return {
            "input_ids": input_ids,
            "targets": targets.cpu(),
        }
    
    def _load_cache(self):
        """Load pre-generated data from cache."""
        # Implementation for loading cached data
        pass
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.cached_data and idx < len(self.cached_data):
            return self.cached_data[idx]
        return self._generate_sample(idx)


class WeightReproductionTrainer:
    """
    Trainer to optimize LoRA weights to reproduce base model behavior.
    """
    
    def __init__(
        self,
        base_model,
        lora_model,
        tokenizer,
        device: str = "auto",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        self.base_model = base_model
        self.lora_model = lora_model
        self.tokenizer = tokenizer
        self.device = get_device(device)
        
        # Only optimize LoRA parameters
        self.optimizer = torch.optim.AdamW(
            [p for p in lora_model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def compute_loss(
        self,
        lora_logits: torch.Tensor,
        base_logits: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """
        Compute loss between LoRA model outputs and base model outputs.
        
        Args:
            lora_logits: Logits from LoRA model
            base_logits: Target logits from base model
            loss_type: Type of loss ("mse", "kl", "cosine")
        """
        if loss_type == "mse":
            # Mean squared error on logits
            return F.mse_loss(lora_logits, base_logits)
        
        elif loss_type == "kl":
            # KL divergence on probability distributions
            lora_probs = F.log_softmax(lora_logits, dim=-1)
            base_probs = F.softmax(base_logits, dim=-1)
            return F.kl_div(lora_probs, base_probs, reduction="batchmean")
        
        elif loss_type == "cosine":
            # Cosine similarity loss
            lora_flat = lora_logits.view(-1, lora_logits.size(-1))
            base_flat = base_logits.view(-1, base_logits.size(-1))
            return 1 - F.cosine_similarity(lora_flat, base_flat, dim=-1).mean()
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_type: str = "mse",
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.lora_model.train()
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            
            # Forward pass through LoRA model
            outputs = self.lora_model(input_ids=input_ids)
            lora_logits = outputs.logits
            
            # Compute loss
            loss = self.compute_loss(lora_logits, targets, loss_type)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
            
            pbar.set_postfix({"loss": loss.item()})
        
        return {
            "epoch_loss": total_loss / total_samples,
        }
    
    def evaluate(
        self,
        dataloader: DataLoader,
        loss_type: str = "mse",
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.lora_model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                outputs = self.lora_model(input_ids=input_ids)
                lora_logits = outputs.logits
                
                loss = self.compute_loss(lora_logits, targets, loss_type)
                mse = F.mse_loss(lora_logits, targets)
                
                total_loss += loss.item() * input_ids.size(0)
                total_mse += mse.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
        
        return {
            "loss": total_loss / total_samples,
            "mse": total_mse / total_samples,
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save a training checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA weights
        self.lora_model.save_pretrained(path)
        
        # Save training state
        checkpoint = {
            "epoch": epoch,
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, os.path.join(path, "trainer_state.pt"))
    
    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(os.path.join(path, "trainer_state.pt"))
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint


def train_lora_to_reproduce_base(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    rank: int = 16,
    num_epochs: int = 10,
    batch_size: int = 4,
    num_samples: int = 10000,
    seq_length: int = 128,
    learning_rate: float = 1e-4,
    output_dir: str = "./lora_checkpoints",
    device: str = "auto",
    loss_type: str = "mse",
    cache_dir: str = "./dataset_cache",
):
    """
    Main training function to train LoRA to reproduce base model.
    
    Args:
        model_name: HuggingFace model name
        rank: LoRA rank
        num_epochs: Number of training epochs
        batch_size: Training batch size
        num_samples: Number of training samples
        seq_length: Sequence length for training
        learning_rate: Learning rate
        output_dir: Directory to save checkpoints
        device: Device to use (auto/cuda/cpu)
        loss_type: Loss function type
        cache_dir: Directory to cache generated datasets
    """
    from model_loader import load_base_model, create_lora_model, get_device
    
    device = get_device(device)
    print(f"Using device: {device}")
    print(f"Loading base model: {model_name}")
    base_model, tokenizer = load_base_model(model_name, device)
    
    print(f"Creating LoRA model with rank={rank}")
    lora_model = create_lora_model(base_model, rank=rank)
    lora_model.print_trainable_parameters()
    
    # Create dataset (with caching)
    print(f"Creating dataset with {num_samples} samples")
    dataset = WeightReproductionDataset(
        tokenizer=tokenizer,
        base_model=base_model,
        num_samples=num_samples,
        seq_length=seq_length,
        device=device,
        cache_dir=cache_dir,
        model_name=model_name,
    )
    
    # Split into train/val
    train_size = int(0.9 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create trainer
    trainer = WeightReproductionTrainer(
        base_model=base_model,
        lora_model=lora_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate,
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, loss_type)
        print(f"Train loss: {train_metrics['epoch_loss']:.6f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader, loss_type)
        print(f"Val loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6e}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        trainer.save_checkpoint(checkpoint_dir, epoch, {
            "train": train_metrics,
            "val": val_metrics,
        })
        print(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_dir = os.path.join(output_dir, "best_model")
            trainer.save_checkpoint(best_dir, epoch, {
                "train": train_metrics,
                "val": val_metrics,
            })
            print(f"New best model! Saved to {best_dir}")
    
    print("\nTraining complete!")
    return trainer, lora_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="./lora_checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loss-type", default="mse", choices=["mse", "kl", "cosine"])
    parser.add_argument("--cache-dir", default="./dataset_cache")
    
    args = parser.parse_args()
    
    train_lora_to_reproduce_base(
        model_name=args.model,
        rank=args.rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        loss_type=args.loss_type,
        cache_dir=args.cache_dir,
    )


def train_lora_to_reproduce_weights_directly(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    rank: int = 16,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    output_dir: str = "./lora_checkpoints",
    device: str = "auto",
    target_modules: Optional[List[str]] = None,
):
    """
    Train LoRA to directly reproduce base model weights.
    
    This zeros out the base model weights and trains LoRA matrices BA such that:
        W_target = BA
    
    This is a direct low-rank matrix approximation problem.
    
    Args:
        model_name: HuggingFace model name
        rank: LoRA rank
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        output_dir: Directory to save checkpoints
        device: Device to use
        target_modules: List of module names to apply LoRA to
    """
    from model_loader import load_base_model, create_lora_model, get_device
    import copy
    import time
    import torch
    
    device = get_device(device)
    print(f"Using device: {device}", flush=True)
    
    # ROCm workaround: Initialize CUDA first
    if device != "cpu":
        print(f"Initializing CUDA...", flush=True)
        torch.cuda.set_device(0)
        torch.cuda.init()
        print(f"  CUDA initialized: {torch.cuda.get_device_name(0)}", flush=True)
    
    # Load target model (the one we want to reproduce)
    print(f"Loading target model: {model_name}", flush=True)
    start = time.time()
    target_model, tokenizer = load_base_model(model_name, device, zero_weights=False)
    print(f"  Target model loaded in {time.time()-start:.1f}s", flush=True)
    
    # Clone and zero the model (faster than loading twice)
    print(f"Cloning and zeroing model...", flush=True)
    start = time.time()
    zeroed_model = copy.deepcopy(target_model)
    
    # Zero out weights in the cloned model
    zero_count = 0
    for name, param in zeroed_model.named_parameters():
        if 'weight' in name and any(x in name for x in (target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                                           'gate_proj', 'up_proj', 'down_proj'])):
            param.data.zero_()
            zero_count += 1
            param.requires_grad = False
    
    print(f"  Cloned and zeroed {zero_count} weights in {time.time()-start:.1f}s", flush=True)
    
    # Extract target weights
    print(f"Extracting target weights...", flush=True)
    start = time.time()
    target_weights = {}
    for name, param in target_model.named_parameters():
        if any(x in name for x in (target_modules or ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                                                       'gate_proj', 'up_proj', 'down_proj'])):
            target_weights[name] = param.data.clone()
    
    print(f"  Collected {len(target_weights)} target weight matrices in {time.time()-start:.1f}s", flush=True)
    
    # Create LoRA model on top of zeroed base
    print(f"Creating LoRA model with rank={rank}")
    lora_model = create_lora_model(zeroed_model, rank=rank, target_modules=target_modules)
    lora_model.print_trainable_parameters()
    
    # Debug: print some parameter names to understand structure
    if False:  # Set to True to debug
        print("\nDEBUG: Parameter names in lora_model:")
        for name in list(lora_model.named_parameters())[:10]:
            print(f"  {name[0]}")
    
    # Optimizer for LoRA parameters only
    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    
    # Build mapping from target weight names to LoRA module paths
    def get_lora_module(base_name):
        """Get the LoRA module for a given base weight name."""
        # The LoRA model wraps the base model: lora_model.base_model.model.model...
        # Target name: "model.layers.0.self_attn.q_proj.weight"
        # We need to access: lora_model.base_model.model.model.layers[0].self_attn.q_proj
        parts = base_name.replace('.weight', '').split('.')
        # Navigate: lora_model.base_model.model.model (LlamaModel has .layers)
        module = lora_model.base_model.model.model  # This is LlamaModel with .layers
        for part in parts[1:]:  # Skip first 'model' 
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    # Training loop - direct weight matching
    print(f"\nTraining for {num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_weights = 0
        
        optimizer.zero_grad()
        total_computed_loss = 0.0
        
        for name, target_w in target_weights.items():
            # Get the LoRA module
            lora_module = get_lora_module(name)
            
            # The effective weight is: W = W_base + BA
            # Since W_base is zeroed out, W = BA
            # PEFT computes this via lora_module.get_base_layer() or we compute BA manually
            
            # Get LoRA A and B matrices
            lora_A = lora_module.lora_A.default.weight  # [rank, in_features]
            lora_B = lora_module.lora_B.default.weight  # [out_features, rank]
            
            # Compute BA product
            lora_w = torch.matmul(lora_B, lora_A)  # [out_features, in_features]
            
            # Compute MSE loss
            loss = F.mse_loss(lora_w, target_w)
            total_computed_loss += loss
            total_loss += loss.item()
            num_weights += 1
        
        total_computed_loss.backward()
        optimizer.step()
        
        avg_loss = total_loss / num_weights if num_weights > 0 else 0
        
        if (epoch + 1) % 1 == 0 or epoch == 0:  # Print every epoch for quick test
            print(f"Epoch {epoch + 1}/{num_epochs}: avg_weight_mse = {avg_loss:.6e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if (epoch + 1) % 10 == 0:  # Save every 10 epochs
                os.makedirs(output_dir, exist_ok=True)
                lora_model.save_pretrained(os.path.join(output_dir, "best"))
    
    print(f"\nTraining complete! Best weight MSE: {best_loss:.6e}")
    
    # Final save
    lora_model.save_pretrained(output_dir)
    print(f"Saved LoRA weights to {output_dir}")
    
    return lora_model, best_loss

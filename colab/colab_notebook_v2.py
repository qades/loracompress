"""
Google Colab Notebook v2 - Adaptive Autoresearch for LoRA Compression

This file contains the notebook code. To use:
1. Copy contents into a new Colab notebook
2. Run cells in order

Features:
- Two-region LR sampling (70% conservative, 30% aggressive)
- OAT (Ordered Adaptive Testing) - doubles LR until error rises, then refines
- Success-based adaptation
- Layer browser with compressibility analysis
- Higher rank support (up to 512)
"""

# ============================================================================
# CELL 1: Mount Google Drive
# ============================================================================

def cell_1():
    from google.colab import drive
    import os
    
    drive.mount('/content/drive')
    
    DRIVE_BASE = '/content/drive/MyDrive/LoRA_Compress'
    os.makedirs(DRIVE_BASE, exist_ok=True)
    os.makedirs(f'{DRIVE_BASE}/databases', exist_ok=True)
    os.makedirs(f'{DRIVE_BASE}/results', exist_ok=True)
    
    print(f"✅ Drive mounted at: {DRIVE_BASE}")
    return DRIVE_BASE


# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================

def cell_2():
    import os
    os.chdir('/content')
    !git clone https://github.com/qades/loracompress.git
    os.chdir('loracompress')
    !pip install -q transformers torch optuna tqdm
    
    import torch
    print(f"\n🔥 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    return torch


# ============================================================================
# CELL 3: Adaptive Explorer Class
# ============================================================================

class AdaptiveExplorer:
    """Intelligent hyperparameter explorer with OAT (Ordered Adaptive Testing)."""
    
    def __init__(self, lr_min=0.0005, lr_max=0.8, rank_min=16, rank_max=256,
                 epochs_min=200, epochs_max=3000, aggressive_ratio=0.3):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.rank_min = rank_min
        self.rank_max = rank_max
        self.epochs_min = epochs_min
        self.epochs_max = epochs_max
        self.aggressive_ratio = aggressive_ratio
        
        # History tracking
        self.successful_configs = []
        self.failed_configs = []
        self.lr_history = []  # (lr, error)
        self.rank_history = []  # (rank, error)
        
        # Learned boundaries
        self.lr_upper_boundary = None  # LR above this diverges
        self.lr_optimal_region = None  # (low, high) best LRs
        self.rank_effectiveness = {}
        
        # OAT state
        self.oat_phase = 'explore_lr_up'
        self.oat_base_config = None
        self.oat_test_values = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
        self.oat_test_idx = 0
        self.oat_best_lr = None
    
    def sample_lr_two_region(self, trial):
        """Sample LR: 70% conservative (0.0005-0.1), 30% aggressive (0.1-0.8)."""
        import random
        if random.random() < self.aggressive_ratio:
            return trial.suggest_float('lr', 0.1, self.lr_max, log=True)
        else:
            return trial.suggest_float('lr', self.lr_min, 0.1, log=True)
    
    def sample_lr_adaptive(self, trial):
        """Sample LR based on learned boundaries."""
        import random
        if self.lr_optimal_region and random.random() < 0.6:
            low, high = self.lr_optimal_region
            low = max(self.lr_min, low * 0.8)
            high_expansion = high * 1.2
            if self.lr_upper_boundary:
                high_expansion = min(self.lr_upper_boundary * 0.9, high_expansion)
            high = min(self.lr_max, high_expansion)
            return trial.suggest_float('lr', low, high, log=True)
        else:
            return self.sample_lr_two_region(trial)
    
    def sample_rank_adaptive(self, trial):
        """Sample rank with bias toward effective ranks."""
        import random
        if self.rank_effectiveness and random.random() < 0.5:
            sorted_ranks = sorted(self.rank_effectiveness.items(), key=lambda x: x[1])
            good_ranks = [r for r, err in sorted_ranks[:3] if err < 50]
            if good_ranks:
                base_rank = random.choice(good_ranks)
                low = max(self.rank_min, int(base_rank * 0.75))
                high = min(self.rank_max, int(base_rank * 1.25))
                return trial.suggest_int('rank', low, high, log=True)
        return trial.suggest_int('rank', self.rank_min, self.rank_max, log=True)
    
    def update_from_result(self, config, error):
        """Update internal state based on trial result."""
        lr = config.get('lr')
        rank = config.get('rank')
        
        if lr:
            self.lr_history.append((lr, error))
        if rank:
            self.rank_history.append((rank, error))
            self.rank_effectiveness[rank] = error
        
        if error < 50:
            self.successful_configs.append({**config, 'error': error})
        else:
            self.failed_configs.append({**config, 'error': error})
        
        self._update_lr_boundaries()
    
    def _update_lr_boundaries(self):
        """Analyze LR history to find optimal region and upper boundary."""
        if len(self.lr_history) < 5:
            return
        
        sorted_by_lr = sorted(self.lr_history, key=lambda x: x[0])
        best_error = min(e for _, e in sorted_by_lr)
        threshold = best_error * 1.5
        
        # Find upper boundary
        for lr, err in sorted_by_lr:
            if err > threshold * 2:
                self.lr_upper_boundary = lr
                break
        
        # Find optimal region
        good_lrs = [lr for lr, err in sorted_by_lr if err < threshold]
        if good_lrs:
            self.lr_optimal_region = (min(good_lrs), max(good_lrs))
    
    def get_oat_suggestion(self, trial, trial_number):
        """Ordered Adaptive Testing: Systematic boundary exploration."""
        
        if trial_number == 0 or self.oat_base_config is None:
            # Start with baseline
            self.oat_base_config = {'rank': 64, 'lr': 0.01, 'epochs': 500, 'scheduler': 'cosine'}
            self.oat_phase = 'explore_lr_up'
            self.oat_test_values = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
            self.oat_test_idx = 0
            return self.oat_base_config
        
        # Phase 1: Explore LR upward (doubling)
        if self.oat_phase == 'explore_lr_up':
            if len(self.lr_history) >= 2:
                last_two = self.lr_history[-2:]
                # Check if error got significantly worse
                if last_two[1][1] > last_two[0][1] * 1.3:
                    # Found upper boundary, switch to refinement
                    self.lr_upper_boundary = last_two[1][0]
                    good_lr = last_two[0][0]
                    self.oat_best_lr = good_lr
                    self.oat_phase = 'refine_lr_down'
                    # Step down 5% at a time
                    self.oat_test_values = [good_lr * (0.95 ** i) for i in range(1, 10)]
                    self.oat_test_idx = 0
                    print(f"  [OAT] Upper boundary found at lr={self.lr_upper_boundary:.4f}, refining down...")
            
            if self.oat_phase == 'explore_lr_up' and self.oat_test_idx < len(self.oat_test_values):
                lr = self.oat_test_values[self.oat_test_idx]
                self.oat_test_idx += 1
                return {'rank': 64, 'lr': lr, 'epochs': 500, 'scheduler': 'cosine'}
        
        # Phase 2: Refine LR downward (5% steps)
        if self.oat_phase == 'refine_lr_down':
            if len(self.lr_history) >= 2:
                last_err = self.lr_history[-1][1]
                prev_err = self.lr_history[-2][1]
                if last_err > prev_err:
                    # Error went up, found minimum
                    best_lr = self.lr_history[-2][0]
                    self.oat_best_lr = best_lr
                    self.oat_phase = 'explore_rank'
                    self.oat_test_values = [32, 48, 64, 96, 128, 192, 256]
                    self.oat_test_idx = 0
                    print(f"  [OAT] LR minimum found at {best_lr:.4f}, now exploring ranks...")
            
            if self.oat_phase == 'refine_lr_down' and self.oat_test_idx < len(self.oat_test_values):
                lr = self.oat_test_values[self.oat_test_idx]
                self.oat_test_idx += 1
                return {'rank': 64, 'lr': lr, 'epochs': 500, 'scheduler': 'cosine'}
        
        # Phase 3: Explore rank
        if self.oat_phase == 'explore_rank':
            if self.oat_test_idx < len(self.oat_test_values):
                rank = self.oat_test_values[self.oat_test_idx]
                self.oat_test_idx += 1
                best_lr = self.oat_best_lr if self.oat_best_lr else 0.01
                return {'rank': rank, 'lr': best_lr, 'epochs': 500, 'scheduler': 'cosine'}
            else:
                self.oat_phase = 'exploit'
                print("  [OAT] Exploration complete, switching to exploitation...")
        
        # Phase 4: Exploit using learned distributions
        return None  # Signal to use adaptive sampling


# ============================================================================
# CELL 4: Training Function
# ============================================================================

def cell_4(target_weight, device='cuda'):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    target_weight_gpu = target_weight.float().to(device)
    
    def compute_l1_error(W_approx, target):
        l1_error = torch.mean(torch.abs(W_approx - target)).item()
        mean_abs_target = torch.mean(torch.abs(target)).item()
        return (l1_error / mean_abs_target * 100) if mean_abs_target > 0 else float('inf')
    
    def train_lora_layer(target_weight, rank, lr, epochs, device='cuda', 
                         scheduler_type=None, warmup_epochs=0):
        d, k = target_weight.shape
        target = target_weight.float().to(device)
        
        A = nn.Parameter(torch.randn(rank, k, device=device) * 0.01)
        B = nn.Parameter(torch.randn(d, rank, device=device) * 0.01)
        optimizer = torch.optim.AdamW([A, B], lr=lr)
        
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
        elif scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
        
        best_loss = float('inf')
        best_A, best_B = None, None
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            if warmup_epochs > 0 and epoch < warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * (epoch + 1) / warmup_epochs
            
            optimizer.zero_grad()
            W_approx = torch.matmul(B, A)
            loss = F.mse_loss(W_approx, target)
            
            if not torch.isfinite(loss):
                return float('inf'), 0, None, None
            
            loss.backward()
            optimizer.step()
            
            if scheduler and (warmup_epochs == 0 or epoch >= warmup_epochs):
                scheduler.step()
            
            current = loss.item()
            if current < best_loss - 1e-9:
                best_loss = current
                best_A = A.detach().clone()
                best_B = B.detach().clone()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epoch >= 200 and epochs_no_improve >= 100:
                break
        
        if best_A is None:
            return float('inf'), 0, None, None
        
        with torch.no_grad():
            W_best = torch.matmul(best_B, best_A)
            l1_error = compute_l1_error(W_best, target)
        
        return l1_error, epoch + 1, best_A, best_B
    
    return train_lora_layer


# ============================================================================
# Instructions for use in Colab:
# 
# 1. Copy CELL 1 code into first cell, run to mount Drive
# 2. Copy CELL 2 code into second cell, run to install deps
# 3. Copy AdaptiveExplorer class into third cell
# 4. Copy CELL 4 code into fourth cell
# 5. Add your layer loading and Optuna objective code
#
# For the OAT strategy:
# - First few trials: Doubles LR from 0.01 up to 0.64
# - When error rises: Captures upper boundary, steps down 5% at a time
# - After LR refinement: Tests ranks 32, 48, 64, 96, 128, 192, 256
# - Finally: Uses learned distributions for adaptive sampling
# ============================================================================

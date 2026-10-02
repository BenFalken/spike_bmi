"""
Training helpers for snn_training/train_snn.py: early stopping and
checkpoint saving/loading.
"""

import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    
    Args:
        patience: How many epochs to wait after last improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
    """
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current epoch's score (loss or accuracy)
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False
    
    def reset(self):
        """Reset the early stopping counter"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str,
    additional_info: Optional[Dict] = None
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss
        checkpoint_path: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load onto
        
    Returns:
        Dictionary with checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    current_state_dict = model.state_dict()

    # Several neuron-state buffers are LAZILY shaped -- they start empty
    # (shape [0]) and only take their real shape after the model's first
    # forward pass. The checkpoint was saved from a model that had
    # already run, so its buffers are properly shaped; a freshly
    # constructed model being resumed into has NOT run yet, so loading
    # the checkpoint's already-shaped values directly into it fails with
    # a hard shape-mismatch error under strict=False (which only
    # tolerates missing/extra keys, not a shape mismatch on a key present
    # in both). Confirmed directly against a real crash: --tau-syn
    # configs specifically hit this on .i_syn (the tau_syn synaptic-
    # current-filtering buffer), which the original .v_mem-only
    # exclusion never covered.
    #
    # .v_mem/.i_syn/.neuron.b are excluded UNCONDITIONALLY by name --
    # confirmed these suffixes never collide with a genuine, meaningfully
    # -shaped learnable parameter in this project's own models (every
    # Linear layer here is built with bias=False, so no ".b" bias term
    # exists to be confused with ALIF's adaptation-state buffer).
    #
    # .spike_threshold is handled differently, deliberately NOT excluded
    # by name alone: it's a REAL, meaningfully-shaped, always-relevant
    # nn.Parameter for IAF (must be loaded, not skipped), but a lazy
    # state buffer for ALIF specifically. Only excluded here if there's
    # an ACTUAL shape mismatch against the current, fresh model's own
    # value -- catching ALIF's lazy case without silently discarding
    # IAF's genuine, trained threshold values. Verified directly: a
    # genuine architecture mismatch (e.g. different --hidden-dims) still
    # raises a loud RuntimeError here, not silently swallowed -- this
    # only skips keys matching the specific lazy-buffer patterns above.
    lazy_buffer_suffixes = ('.v_mem', '.i_syn', '.neuron.b')

    filtered_state_dict = {}
    skipped_keys = []
    for k, v in model_state_dict.items():
        if k.endswith(lazy_buffer_suffixes):
            skipped_keys.append(k)
            continue
        if k.endswith('.spike_threshold') and k in current_state_dict \
                and current_state_dict[k].shape != v.shape:
            skipped_keys.append(k)
            continue
        filtered_state_dict[k] = v

    if skipped_keys:
        print(f"Skipping {len(skipped_keys)} lazily-shaped state buffer(s) "
              f"(will re-initialize on first forward pass): {skipped_keys}")

    # Load model state
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint
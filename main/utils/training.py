"""
Training utilities for OTM

This module provides training loop functions, evaluation, and checkpointing utilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Dict, Tuple, Optional, Union
import os
import sinabs


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    use_amp: bool = False
) -> Tuple[float, Dict]:
    """
    Train the model for one epoch
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        epoch: Current epoch number
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    
    # Initialize AMP if requested
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
    
    for batch_idx, (labels, inputs, targets) in enumerate(pbar):
        # Move data to device and reshape
        inputs = inputs.permute(1, 0, 2, 3, 4).to(device)  # [T, N, C, H, W]
        targets = targets.permute(1, 0, 2).to(device)      # [T, N, 2]
        
        # Reset neuron states
        if hasattr(model, 'reset_states'):
            model.reset_states()
        else:
            sinabs.reset_states(model)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with or without AMP
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)  # [N, T, 2]
                # Permute to match target shape [T, N, 2]
                outputs = outputs.permute(1, 0, 2)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)  # [N, T, 2]
            # Permute to match target shape [T, N, 2]
            outputs = outputs.permute(1, 0, 2)
            loss = criterion(outputs, targets)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    
    metrics = {
        'train_loss': epoch_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    
    return epoch_loss, metrics


def evaluate_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0
) -> Tuple[float, Dict]:
    """
    Evaluate the model on test data
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    running_loss = 0.0
    num_batches = len(test_loader)
    
    # For trajectory prediction, we might want to track additional metrics
    total_position_error = 0.0
    num_samples = 0
    
    pbar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch}", leave=False)
    
    with torch.no_grad():
        for batch_idx, (labels, inputs, targets) in enumerate(pbar):
            # Move data to device and reshape
            inputs = inputs.permute(1, 0, 2, 3, 4).to(device)  # [T, N, C, H, W]
            targets = targets.permute(1, 0, 2).to(device)      # [T, N, 2]
            
            # Reset neuron states
            if hasattr(model, 'reset_states'):
                model.reset_states()
            else:
                sinabs.reset_states(model)
            
            # Forward pass
            outputs = model(inputs)  # [N, T, 2]
            # Permute to match target shape [T, N, 2]
            outputs = outputs.permute(1, 0, 2)
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            
            # Calculate position error (Euclidean distance)
            position_error = torch.sqrt(((outputs - targets) ** 2).sum(dim=-1)).mean()
            total_position_error += position_error.item() * inputs.size(1)
            num_samples += inputs.size(1)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / num_batches
    avg_position_error = total_position_error / num_samples
    
    metrics = {
        'test_loss': epoch_loss,
        'position_error': avg_position_error
    }
    
    return epoch_loss, metrics


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
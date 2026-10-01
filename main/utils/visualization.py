"""
Training-curve plot for snn_training/train_snn.py.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    train_losses: List[float],
    test_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Progress",
    lr_history: Optional[List[float]] = None
):
    """
    Plot training and validation loss curves with optional learning rate
    
    Args:
        train_losses: List of training losses per epoch
        test_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
        title: Title for the plot
        lr_history: Optional list of learning rates per epoch
    """
    # Create figure with subplots if lr_history is provided
    if lr_history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses on first axis
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Validation Loss')
    
    # Mark best validation loss
    best_epoch = np.argmin(test_losses)
    best_loss = test_losses[best_epoch]
    ax1.scatter(best_epoch + 1, best_loss, color='green', s=100, 
                label=f'Best Val Loss: {best_loss:.4f}', zorder=5)
    
    ax1.set_ylabel('Loss')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set y-axis limits if losses are small
    if max(max(train_losses), max(test_losses)) < 0.1:
        ax1.set_ylim(0, 0.1)
    
    # Plot learning rate if provided
    if lr_history:
        ax2.plot(epochs, lr_history, 'g-', linewidth=2)
        ax2.set_ylabel('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Mark LR changes
        for i in range(1, len(lr_history)):
            if lr_history[i] < lr_history[i-1]:
                ax2.axvline(x=i+1, color='orange', linestyle='--', alpha=0.5)
                ax2.text(i+1, lr_history[i], f'LR: {lr_history[i]:.1e}', 
                        rotation=90, verticalalignment='bottom')
    else:
        ax1.set_xlabel('Epoch')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

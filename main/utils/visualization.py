"""
Visualization utilities for OTM

This module provides functions for visualizing spike data, trajectories,
and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from typing import List, Tuple, Optional, Union
# import cv2


def place_cross(image: np.ndarray, x: int, y: int, size: int = 1) -> np.ndarray:
    """
    Place a cross marker at a given position
    
    Args:
        image: Image array to modify
        x: X coordinate of cross center
        y: Y coordinate of cross center
        size: Size of the cross (radius from center)
        
    Returns:
        Modified image with cross
    """
    h, w = image.shape[:2]
    
    # Center pixel
    if 0 <= x < w and 0 <= y < h:
        image[y, x] = 1
    
    # Draw cross arms
    for i in range(1, size + 1):
        # Horizontal line
        if 0 <= x - i < w and 0 <= y < h:
            image[y, x - i] = 1
        if 0 <= x + i < w and 0 <= y < h:
            image[y, x + i] = 1
        
        # Vertical line
        if 0 <= x < w and 0 <= y - i < h:
            image[y - i, x] = 1
        if 0 <= x < w and 0 <= y + i < h:
            image[y + i, x] = 1
    
    return image


def create_trajectory_visualization(
    trajectory: np.ndarray,
    spike_data: torch.Tensor,
    img_size: Tuple[int, int] = (64, 64),
    cross_size: int = 1
) -> List[np.ndarray]:
    """
    Create visualization frames showing trajectory overlaid on spike data
    
    Args:
        trajectory: Array of shape [T, 2] with x,y coordinates
        spike_data: Tensor of shape [T, C, H, W] with spike data
        img_size: Size of output images (width, height)
        cross_size: Size of trajectory marker
        
    Returns:
        List of visualization frames
    """
    frames = []
    T = spike_data.shape[0]
    
    # Create cross marker
    marker_size = 2 * cross_size + 1
    cross_marker = np.zeros((marker_size, marker_size), dtype=np.float32)
    cross_marker = place_cross(cross_marker, cross_size, cross_size, cross_size)
    
    for t in range(T):
        # Create RGB frame
        frame = np.zeros((img_size[1], img_size[0], 3), dtype=np.float32)
        
        # Add spike data (red and blue channels for two polarities)
        if spike_data.shape[1] >= 2:
            frame[:, :, 0] = spike_data[t, 0].cpu().numpy()  # Red channel
            frame[:, :, 2] = spike_data[t, 1].cpu().numpy()  # Blue channel
        else:
            frame[:, :, 0] = spike_data[t, 0].cpu().numpy()
        
        # Add trajectory marker
        x, y = int(trajectory[t, 0] * img_size[0]), int(trajectory[t, 1] * img_size[1])
        
        # Create overlay for cross
        overlay = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
        overlay = place_cross(overlay, x, y, cross_size)
        
        # Add green cross for trajectory
        frame[:, :, 1] = np.maximum(frame[:, :, 1], overlay)
        
        frames.append(frame)
    
    return frames


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


def save_animation(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
    dpi: int = 100
):
    """
    Save a list of frames as an animated GIF
    
    Args:
        frames: List of image frames
        output_path: Path to save the animation
        fps: Frames per second
        dpi: Resolution of the output
    """
    if not frames:
        raise ValueError("No frames to save")
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Initialize with first frame
    im = ax.imshow(frames[0], animated=True)
    ax.axis('off')
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000/fps, blit=True
    )
    
    # Save as GIF
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")


def visualize_spike_raster(
    spike_tensor: torch.Tensor,
    sample_idx: int = 0,
    time_window: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize spike raster plot for a single sample
    
    Args:
        spike_tensor: Tensor of shape [T, N, C, H, W] or [T, C, H, W]
        sample_idx: Which sample to visualize (if batch dimension exists)
        time_window: Optional (start, end) time indices to visualize
        save_path: Optional path to save the plot
    """
    # Handle different tensor shapes
    if spike_tensor.dim() == 5:  # [T, N, C, H, W]
        data = spike_tensor[:, sample_idx]
    else:  # [T, C, H, W]
        data = spike_tensor
    
    T, C, H, W = data.shape
    
    # Apply time window if specified
    if time_window:
        start, end = time_window
        data = data[start:end]
        T = end - start
    
    # Create raster plot
    fig, axes = plt.subplots(C, 1, figsize=(10, 3*C), sharex=True)
    if C == 1:
        axes = [axes]
    
    for c in range(C):
        # Find spike locations
        spike_times, spike_y, spike_x = torch.where(data[:, c] > 0)
        spike_neurons = spike_y * W + spike_x
        
        # Plot spikes
        axes[c].scatter(spike_times.cpu(), spike_neurons.cpu(), s=1, c='black', alpha=0.5)
        axes[c].set_ylabel(f'Channel {c}\nNeuron Index')
        axes[c].set_ylim(0, H*W)
    
    axes[-1].set_xlabel('Time Step')
    plt.suptitle('Spike Raster Plot')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
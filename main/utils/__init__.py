"""
Object Tracking Module (OTM) - Utilities Package

This package contains utility functions for visualization, training, and evaluation.
"""

from .visualization import (
    place_cross,
    create_trajectory_visualization,
    plot_training_curves,
    save_animation
)

from .training import (
    train_epoch,
    evaluate_epoch,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    'place_cross',
    'create_trajectory_visualization',
    'plot_training_curves',
    'save_animation',
    'train_epoch',
    'evaluate_epoch',
    'EarlyStopping',
    'save_checkpoint',
    'load_checkpoint'
]
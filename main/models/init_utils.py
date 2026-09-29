"""
Initialization utilities for SNN models
"""

import torch
import torch.nn as nn
import math

def initialize_snn_model(model: nn.Module, weight_init: str) -> None:
    """
    Initialize SNN model with appropriate weight initialization.
    
    This function applies:
    - Xavier uniform initialization for Conv2d and Linear layers
    - Proper BatchNorm initialization to prevent output explosion
    
    Args:
        model: The SNN model to initialize
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Xavier uniform initialization
            if weight_init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == "kaiming":
                nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.Linear):
            # Xavier uniform initialization
            if weight_init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == "kaiming":
                nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize BatchNorm with smaller variance to prevent explosion
            # Default is weight=1, bias=0 which can cause issues with SNNs
            nn.init.constant_(m.weight, 0.1)  # Smaller weight to reduce output magnitude
            nn.init.zeros_(m.bias)
            # Also initialize running stats
            nn.init.zeros_(m.running_mean)
            nn.init.ones_(m.running_var)

def apply_weight_norm(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            print(f"Applying weight norm on {module}")
            torch.nn.utils.parametrizations.weight_norm(module)

def initialize_batchnorm_conservative(model: nn.Module) -> None:
    """
    Conservative BatchNorm initialization for SNNs.
    
    This sets BatchNorm weights to very small values to ensure
    the output doesn't explode and saturate the neurons.
    
    Args:
        model: The SNN model with BatchNorm layers
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # Very conservative initialization
            nn.init.constant_(m.weight, 0.01)
            nn.init.zeros_(m.bias)

def accumulate_bn_statistics(model: nn.Module, dataloader, device: torch.device, num_batches: int = 10) -> None:
    """
    Accumulate BatchNorm statistics before training.
    
    This helps stabilize BatchNorm layers by computing proper running statistics
    from actual data before training begins.
    
    Args:
        model: The model with BatchNorm layers
        dataloader: DataLoader to compute statistics from
        device: Device to run on
        num_batches: Number of batches to use for statistics
    """
    model.train()  # Ensure BatchNorm is in training mode
    
    with torch.no_grad():
        for i, (_, inputs, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.transpose(0, 1).to(device)
            _ = model(inputs)
    
    # Reset neuron states after accumulation
    if hasattr(model, 'reset_states'):
        model.reset_states()
    else:
        # For models using the original style
        from models.otm_models import reset_vmems
        reset_vmems(model)
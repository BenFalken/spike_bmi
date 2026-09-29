"""
Utilities for sinabs-exodus acceleration
"""

import torch
import time
from sinabs.exodus import conversion


def convert_to_exodus(model):
    """
    Convert a sinabs model to use exodus layers for acceleration.
    
    Args:
        model: A sinabs model with IAF/LIF layers
        
    Returns:
        Model with exodus layers
    """
    try:
        exodus_model = conversion.sinabs_to_exodus(model)
        print("✓ Model converted to Exodus (CUDA-accelerated)")
        return exodus_model
    except Exception as e:
        print(f"⚠ Could not convert to Exodus: {e}")
        print("  Using standard sinabs layers")
        return model


def benchmark_model(model, input_shape, num_iterations=10, device='cuda'):
    """
    Benchmark model performance
    
    Args:
        model: The model to benchmark
        input_shape: Input tensor shape (T, N, C, H, W)
        num_iterations: Number of forward passes
        device: Device to run on
        
    Returns:
        Average time per forward pass
    """
    model = model.to(device)
    model.eval()
    
    # Warmup
    x = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_time = total_time / num_iterations
    
    return avg_time


def compare_performance(standard_model, exodus_model, input_shape, num_iterations=10):
    """
    Compare performance between standard and exodus models
    
    Args:
        standard_model: Model with standard sinabs layers
        exodus_model: Model with exodus layers
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking
    """
    print("\nPerformance Comparison:")
    print("="*50)
    
    # Benchmark standard model
    standard_time = benchmark_model(standard_model, input_shape, num_iterations)
    print(f"Standard sinabs: {standard_time*1000:.2f} ms/forward pass")
    
    # Benchmark exodus model
    exodus_time = benchmark_model(exodus_model, input_shape, num_iterations)
    print(f"Exodus (CUDA):   {exodus_time*1000:.2f} ms/forward pass")
    
    # Calculate speedup
    speedup = standard_time / exodus_time
    print(f"\nSpeedup: {speedup:.2f}x faster with Exodus")
    print("="*50)
    
    return speedup
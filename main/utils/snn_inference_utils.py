"""
Scripts used for SNN analysis exclusively on Speck2f. Some packages (ie: TensorFlow)
are not supported on the laptop connected to Speck, and some dependencies which
exist in test_all_decoders must be isolated and imported as standalone scripts.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from sinabs.activation import MultiSpike, SingleSpike

from models.model_bmi import create_model as create_snn_model
from models.model_bmi import load_model_weights

_FALLBACK_VELOCITY_LO = -280.56
_FALLBACK_VELOCITY_HI = 316.54
_FALLBACK_VELOCITY_MARGIN = 0.05


def load_snn_model(checkpoint_path, num_input_channels=None):
    """Load the SNN architecture + weights via model_bmi.py, using the
    training args saved directly inside the checkpoint.

    Returns (model, checkpoint, velocity_scale) where velocity_scale is
    (v_lo, v_hi, v_margin) read from the checkpoint's own training args if
    present, so denormalization always matches what this model actually
    used.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if 'args' not in checkpoint:
        raise KeyError(
            f"{checkpoint_path}: no 'args' key found. This loader expects checkpoints "
            f"written by the CURRENT train_snn.py (which saves training args directly "
            f"in the checkpoint) -- a checkpoint from the old training script isn't "
            f"compatible with this loader or with model_bmi.py's architecture.")
    train_args = checkpoint['args']

    spike_fn_str = train_args.get('spike_fn')
    spike_fn = MultiSpike if spike_fn_str == 'multi' else \
        SingleSpike if spike_fn_str == 'single' else None

    n_channels = num_input_channels or checkpoint.get('input_shape', [None])[0]
    if n_channels is None:
        # Fall back to inferring directly from the first Linear layer's
        # own weight shape (layers.0.weight: (hidden_dims[0],
        # num_input_channels)).
        state_dict = checkpoint.get('model_state_dict', {})
        if 'layers.0.weight' in state_dict:
            n_channels = state_dict['layers.0.weight'].shape[1]
    if n_channels is None:
        raise ValueError(f"{checkpoint_path}: could not determine num_input_channels "
                          f"(not in checkpoint['input_shape'], not inferable from "
                          f"layers.0.weight, and not passed explicitly)")

    snn_model = create_snn_model(
        use_spikingjelly=train_args.get('use_spikingjelly', False),
        last_layer_reset=train_args.get('last_layer_reset', False),
        weight_init=None,
        spike_fn=spike_fn,
        min_vmem=train_args.get('min_vmem'),
        neuron_type=train_args.get('neuron_type', 'lif'),
        tau_mem=train_args.get('tau_mem', 1.0),
        reset_type=train_args.get('reset_type', 'hard'),
        final_layer_reset_type=train_args.get('final_layer_reset_type'),
        surrogate_grad=train_args.get('surrogate_grad', 'periodic_exponential'),
        use_exodus=train_args.get('use_exodus'),
        use_iaf_squeeze=train_args.get('use_iaf_squeeze', False),
        n_bins=train_args.get('n_bins', 18),
        spike_thresholds=train_args.get('spike_thresholds'),
        temporal_decay_init=train_args.get('temporal_decay_init', 0.8),
        learnable_temporal_decay=train_args.get('learnable_temporal_decay', True),
        temporal_decay_stages=train_args.get('temporal_decay_stages', 1),
        num_input_channels=n_channels,
        hidden_dims=train_args.get('hidden_dims'),
        # Build a synaptic stage only if the checkpoint has trained tau_syn values
        # (train_bmi_no_tau_syn.py checkpoints record a --tau-syn they never
        # used); load_model_weights then restores the trained values.
        tau_syn=(train_args.get('tau_syn') or 1.0)
        if any(k.endswith('.tau_syn') for k in checkpoint['model_state_dict']) else None,
        velocity_lo=train_args.get('velocity_lo', _FALLBACK_VELOCITY_LO),
        velocity_hi=train_args.get('velocity_hi', _FALLBACK_VELOCITY_HI),
        velocity_margin=train_args.get('velocity_margin', _FALLBACK_VELOCITY_MARGIN),
    )

    load_model_weights(snn_model, checkpoint['model_state_dict'],
                        neuron_type=train_args.get('neuron_type', 'lif'),
                        source_description=checkpoint_path)
    snn_model.eval()

    velocity_scale = (
        train_args.get('velocity_lo', _FALLBACK_VELOCITY_LO),
        train_args.get('velocity_hi', _FALLBACK_VELOCITY_HI),
        train_args.get('velocity_margin', _FALLBACK_VELOCITY_MARGIN),
    )
    return snn_model, checkpoint, velocity_scale


def reconstruct_path(anchor, velocities, step_time):
    """
    Reconstruct hand velocity path from starting point (anchor).
    """
    velocities = np.asarray(velocities)
    increments = velocities * step_time
    vel_cumsum = np.cumsum(increments, axis=0)
    return np.vstack([anchor, anchor + vel_cumsum])

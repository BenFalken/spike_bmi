"""
Minimal, standalone extraction of exactly two functions infer_snn_speck.py
(and infer_snn_speck_hardware.py) actually use: load_snn_model() (from
test_all_decoders.py) and reconstruct_path() (from plot_trajectory_grid.py).

WHY THIS EXISTS: test_all_decoders.py imports bmi.decoders at module level
(QRNNDecoder, LSTMDecoder, MLPDecoder, KalmanDecoder, WienerDecoder) --
this project's ANN decoder baselines, entirely unrelated to SNN inference,
which pull in a TensorFlow/Keras dependency chain load_snn_model() itself
never touches. plot_trajectory_grid.py in turn imports a large chunk of
test_all_decoders.py's own machinery (ALL_DECODERS, load_dl_decoder,
run_snn_over_full_test_set, etc) to define its OTHER functions -- none of
which reconstruct_path() needs either. Both functions are copied here
VERBATIM (same logic, same comments explaining WHY each piece of logic
exists) -- this is an extraction, not a rewrite, specifically so nothing
about SNN checkpoint loading or path reconstruction can silently drift
from what test_all_decoders.py/plot_trajectory_grid.py actually do.

If test_all_decoders.py's load_snn_model() or plot_trajectory_grid.py's
reconstruct_path() are ever changed, this file needs the same change
applied here too -- there is no single source of truth between them
anymore, precisely because avoiding the TensorFlow-pulling import chain
requires NOT importing test_all_decoders.py/plot_trajectory_grid.py at
all, not even for these two functions.

_get_model_module() is the SAME case, for the same reason: it exists in
test_all_decoders.py too, and is duplicated here rather than imported --
model_bmi.py/model_hkm.py's own import cost is genuinely small (neither
pulls in TensorFlow/Keras or anything else this file was built to avoid),
so this ISN'T about avoiding a heavy dependency chain the way the two
functions above are. It's duplicated purely to keep this file's own
stated promise: nothing in test_all_decoders.py's SNN-loading logic can
silently drift out from under this file, which importing that logic
FROM test_all_decoders.py (rather than copying it) would risk the moment
that file changes for an unrelated ANN-decoder reason.

Only needs: torch, numpy, sinabs, and models/model_bmi.py or
models/model_hkm.py (chosen dynamically per call, see
_get_model_module()) -- no h5py, sklearn, scipy, bmi.decoders/metrics/
preprocessing, or matplotlib.
"""

import numpy as np
import torch
from sinabs.activation import MultiSpike, SingleSpike

_FALLBACK_VELOCITY_LO = -280.56
_FALLBACK_VELOCITY_HI = 316.54
_FALLBACK_VELOCITY_MARGIN = 0.05


def _get_model_module(experiment):
    """Returns (create_model, load_model_weights) from whichever of
    models.model_bmi / models.model_hkm matches `experiment`. Duplicated
    from test_all_decoders.py's own function of the same name -- see
    module docstring for why this file copies rather than imports."""
    if experiment == "bmi":
        from models.model_bmi import create_model, load_model_weights
    elif experiment == "hkm":
        from models.model_hkm import create_model, load_model_weights
    else:
        raise ValueError(f"experiment must be 'bmi' or 'hkm', got {experiment!r}")
    return create_model, load_model_weights


def load_snn_model(checkpoint_path, experiment, num_input_channels=None):
    """Load the SNN architecture + weights via models.model_bmi or
    models.model_hkm (whichever `experiment` selects -- see
    _get_model_module()), using the training args saved directly inside
    the checkpoint (checkpoint['args'], written by train_bmi.py's
    save_checkpoint() for its 'best' save). No separate summary.txt is
    needed -- see module docstring.

    Returns (model, checkpoint, velocity_scale) where velocity_scale is
    (v_lo, v_hi, v_margin) read from the checkpoint's own training args if
    present, so denormalization always matches what THIS model actually
    used, not a possibly-stale hardcoded constant.
    """
    # NOTE: load_model_weights (the second item _get_model_module()
    # returns) is intentionally UNUSED here -- this function has its own,
    # inline weight-loading logic below (the v_mem/.neuron.v/i_syn
    # filtering), predating test_all_decoders.py's later refactor into a
    # shared load_model_weights() function in model_bmi.py/model_hkm.py.
    # Confirmed to still do the same thing, but this is a real, pre-
    # existing drift this module's own docstring already warns about --
    # worth reconciling on its own, separately from this experiment-
    # selection change, not silently folded into it here.
    create_snn_model, _ = _get_model_module(experiment)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if 'args' not in checkpoint:
        raise KeyError(
            f"{checkpoint_path}: no 'args' key found. This loader expects checkpoints "
            f"written by the CURRENT train_bmi.py (which saves training args directly "
            f"in the checkpoint) -- a checkpoint from the old training script isn't "
            f"compatible with this loader or with model_bmi.py's architecture.")
    train_args = checkpoint['args']

    spike_fn_str = train_args.get('spike_fn')
    spike_fn = MultiSpike if spike_fn_str == 'multi' else \
        SingleSpike if spike_fn_str == 'single' else None

    n_channels = num_input_channels or checkpoint.get('input_shape', [None])[0]
    if n_channels is None:
        raise ValueError(f"{checkpoint_path}: could not determine num_input_channels "
                          f"(not in checkpoint['input_shape'] and not passed explicitly)")

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
        # temporal_decay_stages MUST come from the checkpoint: it was once missing from
        # this call, silently falling back to create_model()'s default of 1 no matter
        # what the checkpoint was trained with.
        temporal_decay_stages=train_args.get('temporal_decay_stages', 1),
        # tau_syn is DELIBERATELY hard-coded to None and train_args['tau_syn'] is
        # deliberately IGNORED. This project's checkpoints come from
        # train_bmi_no_tau_syn.py, which builds its model with tau_syn=None regardless
        # of the --tau-syn value it is given -- so train_args['tau_syn'] is just the
        # recorded CLI argument (1.0), NOT a property of the trained network. Verified
        # two independent ways on a real session (indy_20160407_02): (1) the model built
        # with tau_syn=None reproduces the checkpoint's own best_loss EXACTLY
        # (41.8054 = 41.8054), while building it with tau_syn=1.0 scores 47.85; (2) the
        # Speck2f chip is unaffected by tau_syn entirely (identical DynapcnnNetwork
        # weights/thresholds, chip RMSE 45.04 vs 45.11 across the two settings) because
        # the hardware has no synaptic-current stage to configure. Passing the recorded
        # value through instead builds a DIFFERENT network from the one that was trained
        # and from the one the chip runs. Oscar's test_all_decoders.py load_snn_model()
        # hard-codes None for the same reason -- keep the two in step.
        tau_syn=None,
        num_input_channels=n_channels,
        hidden_dims=train_args.get('hidden_dims'),
        norm_input=train_args.get('norm_input', True),
        velocity_lo=train_args.get('velocity_lo', _FALLBACK_VELOCITY_LO),
        velocity_hi=train_args.get('velocity_hi', _FALLBACK_VELOCITY_HI),
        velocity_margin=train_args.get('velocity_margin', _FALLBACK_VELOCITY_MARGIN),
    )
    # sinabs neuron layers' v_mem (membrane potential) is a lazily-shaped
    # STATE buffer, not a learned parameter -- sinabs only gives it its
    # real shape ([N, layer_width]) the first time forward() actually runs
    # with real data; a freshly-constructed model that's never been run
    # sits at a degenerate placeholder shape ([0]), which load_state_dict
    # refuses to overwrite with the checkpoint's differently-shaped
    # (already-initialized, from training) v_mem -- "size mismatch...
    # torch.Size([1, 512])... torch.Size([0])". We don't want the
    # checkpoint's specific v_mem VALUES anyway: every evaluation path
    # always resets state at the start (reset_state=True on the first
    # trial), so whatever v_mem happened to be at save time is
    # irrelevant. Filter those keys out (both sinabs's .v_mem and
    # spikingjelly's equivalent .neuron.v, in case a --use-spikingjelly
    # checkpoint is ever loaded here) and load everything else --
    # Linear weights, the learned temporal-decay parameter -- normally.
    #
    # .i_syn is the SAME class of issue, confirmed directly against a
    # real failure: a sinabs version on at least one real training/
    # inference environment used for this project registers a LIF
    # layer's synaptic-current state as i_syn, lazily shaped the exact
    # same way v_mem is -- "size mismatch... torch.Size([20, 256])...
    # torch.Size([0])", 20 being that checkpoint's actual --batch-size.
    # Filtered the same way as v_mem, for the same reason: state, not a
    # parameter, and irrelevant given every evaluation path here resets
    # state before using the model anyway.
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.endswith('.v_mem') and not k.endswith('.neuron.v')
                  and not k.endswith('.i_syn')
                  # tau_syn entries (if a checkpoint carries any) are dropped on purpose: the
                  # model above is built with tau_syn=None, so they would otherwise be
                  # reported as "unexpected" keys and abort the load -- see the tau_syn note
                  # in the create_snn_model() call for why their values are never wanted.
                  and not k.endswith('.tau_syn')}
    missing, unexpected = snn_model.load_state_dict(state_dict, strict=False)
    # Confirm nothing OTHER than the expected state buffers is missing --
    # a missing Linear.weight, for instance, would mean a genuine
    # architecture mismatch between this checkpoint and model_bmi.py, not
    # this lazy-state-buffer quirk, and should NOT be silently swallowed.
    #
    # 'neutral_scaled_pred' is ALSO tolerated as missing: a checkpoint
    # trained before decode_output()'s zero-spike fallback fix won't have
    # this buffer in its saved state at all. That's fine, not a
    # mismatch -- create_snn_model() above already constructed it fresh
    # from this checkpoint's own velocity_lo/hi/margin train_args (or this
    # project's standard defaults, for a checkpoint old enough to predate
    # even those being saved), which is the CORRECT value regardless of
    # whether the checkpoint file itself happens to contain it.
    unexpected_missing = [k for k in missing
                           if not (k.endswith('.v_mem') or k.endswith('.neuron.v')
                                   or k.endswith('.i_syn') or k == 'neutral_scaled_pred' or k.endswith('.tau_syn'))]
    if unexpected_missing:
        raise RuntimeError(
            f"{checkpoint_path}: load_state_dict is missing key(s) beyond the "
            f"expected v_mem/v/i_syn state buffers: {unexpected_missing} -- this suggests "
            f"a real architecture mismatch, not the lazy-state-buffer issue this "
            f"filtering is meant to handle.")
    if unexpected:
        raise RuntimeError(
            f"{checkpoint_path}: checkpoint has unexpected key(s) not present in the "
            f"current model_bmi.py architecture: {unexpected}")
    snn_model.eval()

    velocity_scale = (
        train_args.get('velocity_lo', _FALLBACK_VELOCITY_LO),
        train_args.get('velocity_hi', _FALLBACK_VELOCITY_HI),
        train_args.get('velocity_margin', _FALLBACK_VELOCITY_MARGIN),
    )
    return snn_model, checkpoint, velocity_scale


def reconstruct_path(anchor, velocities, step_time):
    """anchor: (2,) true position at trial onset.
    velocities: (m, 2) predicted velocity for each of the m windows after
    the anchor.
    Returns (m+1, 2): anchor followed by the cumulatively integrated path,
    so the output has the same length as the true path over
    [i0, i1] inclusive when velocities = pred[i0:i1] (m = i1 - i0 windows).
    """
    velocities = np.asarray(velocities)
    increments = velocities * step_time
    cum = np.cumsum(increments, axis=0)
    return np.vstack([anchor, anchor + cum])

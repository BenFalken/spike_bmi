"""
Comparison/TEST script for KF, WF, LSTM, QRNN, and SNN decoders -- reads
ALREADY-TRAINED, cached model bundles (produced by eval_wf_decoder.py,
eval_kf_decoder.py, eval_dl_decoders.py, and train_snn.py) and evaluates
them side by side over the SAME stretch of a session's chronological test
set. This script never builds or trains a model itself -- renamed from
eval_all_decoders.py specifically to make that distinction unambiguous:
"eval_*_decoder.py" builds a model AND reports its own metrics;
this file only ever tests what's already been built.

SNN evaluation covers the ENTIRE available test set, not a subset chosen
in advance for efficiency (an earlier revision computed which trials would
be needed for a specific ANN range and only ran those -- see
run_snn_over_full_test_set()'s docstring for what changed and why). Every
trial independently resets (SNN_Speck.forward() resets state
unconditionally on every call now -- an earlier reset_state parameter
that made this optional was tried, found worse, and removed entirely) --
there is only one training mode now (see train_snn.py); 'continuous' and
'chunked' modes were both tried and removed after real comparative
results showed windowed outperforming both.

SNN/ANN alignment compares the SAME stretch of real time on both sides:
SNN trial i's t-th per-timestep prediction lines up with ANN dense-window
row (i-1)*nperseg+t (matches bmi.features.extract()'s lagged-target
convention: row j predicts the target at raw sample j+nperseg). Trial 0
is always excluded from the comparison (structurally: ANN needs nperseg
samples of prior history it doesn't have yet at the very start of the
test set), independent of anything else about how the checkpoint was trained.

calibrate_snn_ann_offset() additionally corrects for a REAL, confirmed
residual offset between the ANN and SNN datasets' own test-split
boundaries: they're typically built by two independent pipeline runs
(dense ~4ms-step ANN windowing vs. much coarser SNN windowing), each
rounding test_frac*N against a different row-count granularity -- even at
the identical nominal --test_frac, this generally does NOT land on the
same real moment. Confirmed in practice (check_ann_snn_data_consistency.py,
run against real data): a session found to be the exact same recording on
both sides, CC=1.0000, but offset by several hundred samples. This is
discovered cheaply (ground-truth velocity only, no model inference)
before the SNN model is ever loaded, and falls back to zero offset with a
printed WARNING (not a silent guess) if no confident match is found.

SNN model loading goes through model_bmi.py (single-purpose feedforward
architecture, no dropout/spike-sparsity mechanisms -- both were tried and
removed, see train_snn.py), via a plain load_state_dict() (sinabs's
lazily-shaped v_mem state buffer is deliberately excluded -- it's state,
not a learned parameter, see load_snn_model()). No separate summary_*.txt
either: train_snn.py's save_checkpoint() stores training args directly in
the checkpoint (checkpoint['args']), so it's self-describing, including
velocity-scaling constants.

Checkpoint path convention:
    {checkpoints_root}/{session_id}/best_model_weights.pth            (full-data model)
    {checkpoints_root}/{session_id}/{N}_min/best_model_weights.pth    (duration-tagged, N = integer minutes)

FIGURES: every figure from the previous revision is unchanged and still
produced (full overlay, rolling RMSE, RMSE bar chart, error boxplot,
scatter+R^2, cumulative loss, error-vs-speed), PLUS a new trajectory-grid
figure (make_test_window_trajectory_grid()): the first --n_segments
(default 16) non-overlapping --segment_samples-wide (default 260 = 4x256ms
= ~1s, matching train_snn.py's --truncation-chunks default -- the same
"unit of analysis" used for the SNN's own gradient truncation) segments of
the comparison range, each showing every decoder's reconstructed 2D
trajectory (predicted velocity integrated from the segment's true starting
position) overlaid against the true path. Reuses
reconstruct_path()/make_trajectory_grid_figure() from
plot_trajectory_grid.py directly.

Everything else (KF/WF/DL cache loading, CI computation, multi-session
driver, duration-sweep mode, npz output) is unchanged.

CLI usage, single session:
    python test_all_decoders.py \
        --input_filepath data/dataset/indy_20160407_02_binning.h5 \
        --model_dir results/model_cache/indy_20160407_02 \
        --feature mua --decoders lstm,qrnn,kf,wf,snn --test_frac 0.1 \
        --snn_checkpoint_path checkpoints/bmi/mua/indy_20160407_02/best_model_weights.pth \
        --snn_dataset_path ./datasets/bmi/mua/indy_20160407_02 \
        --save_path results/indy_20160407_02_comparison.png \
        --metrics_save_path results/indy_20160407_02_metrics.json

CLI usage, all sessions found under a checkpoints root:
    python test_all_decoders.py --multi_session \
        --checkpoints_dir checkpoints/bmi/mua_large_final \
        --dataset_root ../data/bfalkenb/data/dataset/mua \
        --model_cache_root results/model_cache \
        --snn_dataset_root datasets/bmi/mua_large \
        --feature mua --decoders lstm,qrnn,kf,wf,snn --test_frac 0.1 \
        --results_dir results/multi_session \
        --combined_metrics_path results/multi_session/combined_metrics.json

CLI usage, all sessions, SWEEPING training duration (reads the duration-
tagged ANN caches written by eval_dl_decoders.py, PLUS the duration-tagged
SNN checkpoints nested under each session's own checkpoint dir -- see the
docstring above):
    python test_all_decoders.py --multi_session \
        --checkpoints_dir checkpoints/bmi/mua \
        --dataset_root data/dataset \
        --model_cache_root results/model_cache \
        --snn_dataset_root ./datasets/bmi/mua \
        --feature mua --decoders lstm,qrnn,snn --test_frac 0.1 \
        --train_durations 1,2,3,4,5,6,7,8,9,10 \
        --results_dir results/multi_session_durations \
        --combined_metrics_path results/multi_session_durations/combined_metrics.json

You can also just import this module in a notebook and call
`run_all_sessions(...)` or `run_session(...)` directly -- see the bottom of
this file for a minimal notebook-style example.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import os

# Force CPU-only execution, BEFORE anything below (bmi.decoders, in
# particular) has a chance to import TensorFlow -- this MUST happen prior
# to TensorFlow's own first import in this process to take effect, since
# TF initializes its GPU/cuDNN context at import time.
#
# Why: encountered in practice as a cuDNN version mismatch --
#   "Loaded runtime CuDNN library: 9.1.0 but source was compiled with:
#   9.3.0" -- causing FAILED_PRECONDITION errors on the very first GPU op
#   (the MLP decoder's .predict() call), which fails IDENTICALLY for every
#   session in a --multi_session run (same code path, same environment).
#   This is an environment/driver-vs-library version mismatch, not
#   something fixable in this script's logic -- the real fix is aligning
#   TensorFlow's expected cuDNN version with what's actually installed.
#   Forcing CPU sidesteps it entirely: this script only runs INFERENCE on
#   small, already-trained models (MLP/LSTM/QRNN) plus lightweight
#   per-trial SNN forward passes -- GPU throughput matters far less here
#   than it does for training, so CPU-only is a reasonable default rather
#   than a workaround to remove later.
#
# Set EVAL_ALL_DECODERS_USE_GPU=1 in the environment to opt back into GPU
# (e.g. once the cuDNN mismatch is actually resolved) without editing this
# file again.
if os.environ.get("EVAL_ALL_DECODERS_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle as pkl
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import t as t_dist

from bmi.preprocessing import transform_data
from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder, KalmanDecoder, WienerDecoder
from bmi.metrics import pearson_corrcoef

# Hardware-agnostic effective-op/memory-access/energy ESTIMATES (as distinct
# from any wall-clock energy MEASUREMENT elsewhere in this project) -- see
# op_energy_estimate.py's own module docstring for the full methodology
# (Liao et al.'s MAC/ACC + memory-access accounting, Horowitz 2014's
# per-operation energy table) and exactly what's verified vs. approximated
# for each decoder. Computed unconditionally for every kf/wf/mlp/lstm/qrnn
# result below -- no flag to disable, per direct request: this is meant to
# be a permanent part of what this script reports, not an opt-in extra.
from op_energy_estimate import (estimate_ops_kf, estimate_ops_wf, estimate_ops_mlp,
                                 estimate_ops_lstm, estimate_ops_qrnn, finalize_snn_ops)

from sinabs.activation import MultiSpike, SingleSpike
# model_bmi.py vs model_hkm.py is decided per-run, dynamically, via
# --experiment -- see _get_model_module() below. NOT imported at module
# level (the old, hardcoded `from models.model_bmi import ...`): that
# unconditionally used model_bmi's SNN_Speck.forward() regardless of
# --experiment, which has no reset_state argument at all -- silently
# wrong (or a TypeError, once --continuous-snn-test-stream tried to use
# it) for any real hkm run. See model_hkm.py's own module docstring for
# why the two model files are a deliberate fork, not a single shared one.

DL_DECODERS = ('mlp', 'lstm', 'qrnn')
ALL_DECODERS = ('mlp', 'lstm', 'qrnn', 'kf', 'wf', 'snn')

DEFAULT_CI_N_SPLITS = 10

_FALLBACK_VELOCITY_LO = -280.56
_FALLBACK_VELOCITY_HI = 316.54
_FALLBACK_VELOCITY_MARGIN = 0.05


def _get_model_module(experiment):
    """Returns (create_model, load_model_weights) from whichever of
    models.model_bmi / models.model_hkm matches `experiment` -- the
    single point deciding which model file governs a given run, so
    nothing downstream needs its own bmi/hkm branch. Imported HERE,
    lazily, rather than at module level -- deferring the import until
    experiment is actually known (post-argparse) is what makes the
    choice genuinely dynamic rather than fixed at file-load time."""
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
    the checkpoint (checkpoint['args'], written by train_snn.py's
    save_checkpoint() for its 'best' save). No separate summary.txt is
    needed -- see module docstring.

    Returns (model, checkpoint, velocity_scale) where velocity_scale is
    (v_lo, v_hi, v_margin) read from the checkpoint's own training args if
    present, so denormalization always matches what THIS model actually
    used, not a possibly-stale hardcoded constant.
    """
    create_snn_model, load_model_weights = _get_model_module(experiment)
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
        # own weight shape -- see snn_inference_utils.py's own copy of
        # this same fix for the full reasoning (a real bug in
        # train_snn.py's final-checkpoint save, fixed at the source, but
        # not retroactively for already-saved checkpoints).
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
    # See model_bmi.py's load_model_weights() for the full reasoning --
    # sinabs' lazily-shaped state buffers (v_mem/i_syn) need excluding
    # before load_state_dict, and that logic is now a single shared
    # function rather than an independent copy here.
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


def predict_snn_trial(snn_model, input_spikes, velocity_scale, experiment, n_units_expected=None,
                       count_ops=False, reset_state=True):
    """Run the SNN ONCE on one trial, returning EVERY per-timestep (x, y)
    prediction in physical velocity units, shape (n_timesteps, 2) -- not
    just the last one. For bmi, one trial is always exactly one 256ms
    (nperseg-timestep) window; for hkm, one trial is whatever length that
    trial's own .pkl file actually is (see model_hkm.py's own module
    docstring for why).

    reset_state: forwarded to SNN_Speck.forward()'s own reset_state
    argument -- ONLY for experiment='hkm'. model_bmi.py's own
    SNN_Speck.forward() has NO reset_state argument at all (removed once
    the "continuous, never-reset stream" training mode it supported was
    tried, found worse, and dropped -- forward() there ALWAYS resets,
    unconditionally); passing it for experiment='bmi' would raise a
    TypeError, so it's silently dropped from the call for bmi rather than
    forwarded -- every bmi call site still gets exactly the "always
    reset" behavior it always has, whatever reset_state is set to. For
    hkm, default True resets at the start of THIS call, matching bmi's
    own always-reset behavior; the caller (run_snn_over_full_test_set(),
    driven by --continuous-snn-test-stream) is what actually varies this
    across a sequence of trials -- see that function's own docstring.

    count_ops: forwarded to SNN_Speck.forward()'s own count_ops flag
    (present on BOTH model_bmi.py and model_hkm.py) -- when True, also
    returns raw {'mac', 'acc', 'elementwise'} operation counts for this
    trial alongside the predictions, for op_energy_estimate.py's
    finalize_snn_ops().
    """
    if n_units_expected is not None and input_spikes.shape[0] != n_units_expected:
        print(f"  WARNING: trial has {input_spikes.shape[0]} units, checkpoint "
              f"expects {n_units_expected} -- verify --feature matches what "
              f"the SNN was trained on before trusting this comparison.")

    x = input_spikes.T[:, np.newaxis, :]
    x = torch.from_numpy(x.astype(np.float32))

    forward_kwargs = {}
    if count_ops:
        forward_kwargs["count_ops"] = True
    if experiment == "hkm":
        forward_kwargs["reset_state"] = reset_state
    # experiment == "bmi": reset_state deliberately NOT passed at all --
    # see docstring above.

    with torch.no_grad():
        if count_ops:
            y_pred, _, _, op_counts = snn_model(x, **forward_kwargs)
        else:
            y_pred, *_ = snn_model(x, **forward_kwargs)
    y_pred = y_pred.squeeze(1)

    v_lo, v_hi, v_margin = velocity_scale
    y_pred_phys = v_lo + (v_hi - v_lo) * (y_pred.numpy() - v_margin) / (1 - 2 * v_margin)
    if count_ops:
        return y_pred_phys, op_counts
    return y_pred_phys


# build_snn_ann_alignment() -- an earlier revision's trial-selection index
# math, used to compute WHICH trials to run for a specific ANN range
# before running them (an efficiency optimization: skip trials that
# wouldn't be used). Removed: THIS revision runs the SNN across the
# entire available test set unconditionally (see
# run_snn_over_full_test_set()) and trims the result AFTER the fact,
# rather than deciding in advance which trials to bother running -- the
# whole reason that pre-computation existed is gone. The alignment MATH
# itself (SNN trial i's raw sample range vs. ANN's lagged row convention)
# still holds and is now inlined directly in run_snn_over_full_test_set().


def _lagged_correlation(signal_a, signal_b, max_lag):
    """Private mirror of check_data_alignment.py's lagged_correlation() --
    duplicated (not imported) deliberately: this is now core pipeline
    correctness (see calibrate_snn_ann_offset()), and shouldn't silently
    change behavior if that standalone diagnostic script is ever edited
    independently. Kept in sync manually; the two were cross-validated
    together (see check_ann_snn_data_consistency.py's test history).
    Same sign convention: correlation at lag L compares signal_a[t]
    against signal_b[t+L] (positive L: signal_a leads).
    """
    lags = np.arange(-max_lag, max_lag + 1)
    ccs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag > 0:
            a, b = signal_a[:-lag], signal_b[lag:]
        elif lag < 0:
            a, b = signal_a[-lag:], signal_b[:lag]
        else:
            a, b = signal_a, signal_b
        if np.std(a) > 1e-8 and np.std(b) > 1e-8:
            ccs[i] = np.corrcoef(a, b)[0, 1]
    return lags, ccs


def compute_aligned_split(N, test_frac, base_nperseg=65):
    """Computes (n_train, n_test) for the ANN's dense-windowed dataset
    such that n_train is an EXACT MULTIPLE of base_nperseg (256ms at
    4ms/sample -- the SNN's base trial length, whether or not a session
    additionally uses make_large_snn_dataset.py's grouping, since that
    only groups whole numbers of base trials and so preserves alignment
    to base_nperseg).

    Why this works as a genuine fix, not just a plausible-sounding one:
    for the ANN's dense (step=1) windowing, row index IS raw sample index
    directly (each row starts exactly at that raw sample) -- so a train
    boundary that's a multiple of base_nperseg in ROW terms is ALSO a
    multiple of base_nperseg in RAW SAMPLE terms, which is exactly what
    the SNN's own non-overlapping windowing needs to never straddle the
    boundary. The subtler part: for the ANN and SNN sides to land on the
    SAME real moment (not just each internally clean), the split must be
    computed from a quantity BOTH pipelines can derive identically -- the
    raw, UNWINDOWED session length (task_time's own length), not each
    pipeline's own post-windowing row count (which differ hugely: dense
    ANN windowing vs. non-overlapping SNN windowing). This function
    reconstructs that raw length EXACTLY from N via extract()'s own dense-
    windowing convention (N = total_raw_samples - base_nperseg, for
    step=1) -- confirmed by direct test to produce IDENTICAL results to a
    pipeline (like export_snn_pkl.py) that has direct access to the true
    raw session length, across the real session's actual N and 20 random
    session lengths, not just one lucky case.

    export_snn_pkl.py needs the SAME formula applied to its own raw
    session length (len(task_time), which it already has direct access
    to) -- see this file's module docstring for the exact snippet.

    Rounds the train boundary DOWN to the nearest base_nperseg multiple.
    Whether this grows or shrinks the test set relative to the naive
    (unaligned) round(test_frac*N) is NOT guaranteed either direction --
    the two are rounded against different bases (this one includes the
    base_nperseg padding) -- and isn't a property this fix needs to
    guarantee; only exact base_nperseg-alignment and cross-pipeline
    consistency matter here.
    """
    total_raw_samples = N + base_nperseg
    naive_n_test_raw = round(test_frac * total_raw_samples)
    naive_n_train_raw = total_raw_samples - naive_n_test_raw
    n_train = (naive_n_train_raw // base_nperseg) * base_nperseg
    n_test = N - n_train
    return n_train, n_test


def calibrate_snn_ann_offset(snn_dataset_path, y_test_vel, max_lag=2000, min_confidence=0.9,
                              base_nperseg=65):
    """Cheaply (no model inference -- only ground-truth velocity from both
    sides) discovers the actual raw-sample offset between the SNN's own
    test-trial ground truth and the ANN's y_test_vel, so
    run_snn_over_full_test_set() can correct for it via ann_row_offset
    instead of silently assuming zero.

    Why this exists: the ANN dataset and SNN dataset for a session are
    typically built by two independent pipeline runs (dense ~4ms-step
    windowing vs. much coarser SNN windowing), each computing its own
    train/test boundary as round(test_frac * N) against a DIFFERENT row-
    count granularity. Even with the identical nominal --test_frac, this
    generally does not land at the same real moment -- confirmed in
    practice (check_ann_snn_data_consistency.py, run against real data,
    found a session with CC=1.0000 -- a PERFECT match, i.e. genuinely the
    same recording -- at a nonzero, session-specific lag rather than at
    zero). This offset differs per session (it falls out of each
    session's own row count), so it needs to be discovered fresh per
    session rather than hardcoded.

    Only trusts the discovered lag if its correlation clears
    min_confidence (default 0.9) -- a low-confidence "best available" lag
    would be worse than no correction at all, since it would apply a
    wrong-but-plausible-looking shift instead of leaving the (at least
    understood) zero-offset default in place. Returns 0 (with a printed
    WARNING) if no confident match is found at any scanned lag -- this
    means the session's ANN and SNN data sources may not actually
    correspond to the same recording at all; see
    check_ann_snn_data_consistency.py for a full standalone investigation
    of a specific session.

    TWO input conventions, both supported:
    - MULTIPLE small test/{i}.pkl files, each exactly one base_nperseg-
      sized trial (the older convention this function originally
      supported) -- trial 0 skipped entirely, trials 1+ concatenated,
      matching run_snn_over_full_test_set()'s own convention for this case.
    - ONE single, whole, continuous test/0.pkl (the newer convention --
      make_huge_dataset.py's own test-saving behavior, used throughout
      mua_256_group/mua_256_group_uniform). Here there's no "trial 0 to
      skip" in the same sense -- instead, the first base_nperseg
      timesteps of THIS one trial are dropped, matching the ANN's own
      "needs base_nperseg samples of prior history before its first
      row" constraint exactly, just applied within a single long trial
      instead of across many short ones. Confirmed directly this is the
      right substitution, not a guess: with a single trial, "skip trial
      0" and "keep nothing" are the same statement, so SOME distinct
      handling is structurally required here, not optional polish.
    """
    test_dir = os.path.join(snn_dataset_path, 'test')
    files = sorted((f for f in os.listdir(test_dir) if f.endswith('.pkl')),
                    key=lambda f: int(f.split('.')[0]))

    if len(files) == 1:
        with open(os.path.join(test_dir, files[0]), 'rb') as f:
            sample = pkl.load(f)
        snn_vel_concat = sample['velocity'][base_nperseg:]
    elif len(files) >= 2:
        snn_velocities = []
        for fname in files:
            with open(os.path.join(test_dir, fname), 'rb') as f:
                sample = pkl.load(f)
            snn_velocities.append(sample['velocity'])
        # Skip SNN trial 0 -- matches run_snn_over_full_test_set()'s own baseline
        # (zero-offset) convention: naive ann_row_start for trial i is
        # (i-1)*nperseg, so trial 1 onward, concatenated, is what the
        # UNCORRECTED alignment already assumes lines up with ANN row 0.
        snn_vel_concat = np.concatenate(snn_velocities[1:], axis=0)
    else:
        return 0

    n_common = min(len(snn_vel_concat), len(y_test_vel))
    if n_common < 100:
        return 0
    ann_slice = y_test_vel[:n_common, 0]
    snn_slice = snn_vel_concat[:n_common, 0]

    if np.std(ann_slice) < 1e-8 or np.std(snn_slice) < 1e-8:
        return 0

    zero_lag_cc = np.corrcoef(ann_slice, snn_slice)[0, 1]
    if zero_lag_cc > min_confidence:
        return 0  # already aligned -- nothing to correct

    max_lag = min(max_lag, n_common // 4)
    lags, ccs = _lagged_correlation(ann_slice, snn_slice, max_lag)
    peak_idx = np.nanargmax(ccs)  # best POSITIVE match specifically, not just largest magnitude
    best_lag, best_cc = int(lags[peak_idx]), float(ccs[peak_idx])

    if best_cc > min_confidence:
        print(f"  SNN/ANN auto-calibration: zero-lag CC={zero_lag_cc:.4f} was weak, but found "
              f"offset={best_lag} samples ({best_lag * 0.004 * 1000:.0f}ms) with CC={best_cc:.4f} "
              f"-- correcting alignment automatically for this session.")
        # NEGATED, deliberately -- see this function's docstring for the
        # exact, hand-verified derivation of why. _lagged_correlation's
        # convention is "signal_a[t] vs signal_b[t+L]" (signal_a=ann_slice,
        # signal_b=snn_slice here) -- a peak at L means ann_slice[t]
        # matches snn_slice[t+L], i.e. snn_slice[k] matches ann_slice[k-L].
        # The CALLER needs the value USABLE DIRECTLY as start_raw such
        # that y_test_vel[start_raw + k] == snn_slice[k] -- that value is
        # -L, not L. Confirmed with a simple, hand-verifiable synthetic
        # case (a known small shift against a sinusoidal signal) before
        # this fix: the function was returning +L directly, which does
        # NOT satisfy that equation and produces a genuinely MISALIGNED
        # comparison for any session with a nonzero offset -- silently
        # invalidating RMSE/CC for the SNN specifically (not the other
        # decoders, which don't go through this path) on exactly those
        # sessions. Re-run any session whose log showed a nonzero
        # "auto-calibration" offset before trusting its SNN numbers.
        return -best_lag
    else:
        print(f"  WARNING: SNN/ANN auto-calibration found no confident match at any scanned lag "
              f"(best CC={best_cc:.4f} at lag={best_lag}) -- proceeding with UNCORRECTED "
              f"(zero-offset) alignment, which may be wrong. This session's ANN and SNN data may "
              f"not correspond to the same recording at all -- investigate directly with "
              f"check_ann_snn_data_consistency.py before trusting this session's SNN results.")
        return 0



def resolve_snn_checkpoint_for_duration(base_snn_checkpoint_path, duration_minutes):
    """Given the FULL-DATA checkpoint path
    (.../per_session/{session_id}/best_model_weights.pth), derive the
    duration-tagged path ACTUALLY produced by
    run_snn_duration_sweep_array.sbatch:
    {checkpoints_root}/duration_sweep/{session_id}/{N}min/best_model_weights.pth
    -- a SEPARATE top-level root, sibling to per_session/, not a
    subdirectory nested under per_session/{session_id}/ itself (that was
    this function's original, incorrect assumption -- confirmed directly
    against run_snn_duration_sweep_array.sbatch's own real
    CHECKPOINT_DIR construction, and "{N}min" has no underscore there,
    unlike this function's own prior "{N}_min").

    Finds "per_session" wherever it actually occurs in the path (rather
    than assuming a fixed depth) so this still resolves correctly even
    when checkpoint_config_name nests one level deeper after
    {session_id} -- session_id is read as the path component
    immediately after "per_session", not derived by counting levels
    from the end.
    """
    parts = base_snn_checkpoint_path.split(os.sep)
    if "per_session" not in parts:
        raise ValueError(
            f"Cannot resolve a duration-tagged checkpoint from {base_snn_checkpoint_path}: "
            f"expected a 'per_session' path component, matching "
            f"run_snn_duration_sweep_array.sbatch's own convention of a sibling "
            f"'duration_sweep' root at the same level.")
    per_session_idx = parts.index("per_session")
    session_id = parts[per_session_idx + 1]
    checkpoints_root = os.sep.join(parts[:per_session_idx])
    filename = os.path.basename(base_snn_checkpoint_path)
    tag_dir = f"{int(round(duration_minutes))}min"
    return os.path.join(checkpoints_root, "duration_sweep", session_id, tag_dir, filename)


def build_snn_checkpoint_path(checkpoints_root, session_id, duration_minutes=None,
                               checkpoint_config_name=None):
    """{checkpoints_root}/{session_id}/best_model_weights.pth (full-data),
    or {checkpoints_root}/{session_id}/{N}_min/best_model_weights.pth
    (duration-tagged, N = round(duration_minutes)).

    checkpoint_config_name: some sweeps save checkpoints one directory
    level deeper than the flat convention above --
    {checkpoints_root}/{session_id}/{checkpoint_config_name}/
    best_model_weights.pth (e.g. a tau_syn/architecture sweep with
    multiple configs per session, like .../mua_8_group_tausyn_sweep/
    {session_id}/tausyn_8/best_model_weights.pth). None (the default)
    preserves the original, flat behavior exactly -- this is additive,
    not a replacement of the existing convention.
    """
    session_dir = os.path.join(checkpoints_root, session_id)
    if checkpoint_config_name is not None:
        session_dir = os.path.join(session_dir, checkpoint_config_name)
    if duration_minutes is not None:
        return resolve_snn_checkpoint_for_duration(
            os.path.join(session_dir, "best_model_weights.pth"), duration_minutes)
    return os.path.join(session_dir, "best_model_weights.pth")


def _tagged(name, duration_tag):
    return f"{name}_{duration_tag}" if duration_tag else name


def _load_config(bundle_dir, name, feature, test_frac, expected_input_dim=None,
                  duration_minutes=None, duration_tag=None):
    config_path = os.path.join(bundle_dir, f"{_tagged(name, duration_tag)}_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config.get('test_frac') != test_frac:
        raise ValueError(
            f"{name}: cached test_frac={config.get('test_frac')} != requested {test_frac}. "
            f"Re-cache with matching --test_frac before comparing.")
    if config.get('feature') != feature:
        raise ValueError(
            f"{name}: cached feature={config.get('feature')!r} != requested {feature!r}. "
            f"Re-cache with --feature {feature} before comparing.")
    if expected_input_dim is not None and config.get('input_dim') != expected_input_dim:
        raise ValueError(
            f"{name}: cached input_dim={config.get('input_dim')} != dataset's {expected_input_dim}. "
            f"Likely cached from a different dataset file.")
    if duration_minutes is not None:
        cached_minutes = config.get('train_duration_minutes')
        if cached_minutes is None or abs(cached_minutes - duration_minutes) > 1e-6:
            raise ValueError(
                f"{name}: cached train_duration_minutes={cached_minutes} != requested "
                f"{duration_minutes} (tag={duration_tag}). Cache/tag mismatch -- "
                f"check {config_path}.")
    return config


def load_dl_decoder(model_dir, feature, decoder, test_frac, X_raw, y_raw_vel, verbose,
                     duration_minutes=None, duration_tag=None):
    """Returns (y_pred, offset, op_estimate) -- op_estimate is a
    hardware-agnostic effective-op/memory-access/energy ESTIMATE (see
    op_energy_estimate.py), computed against the SAME model and SAME
    actual (scaled, windowed) X this call just ran predict() on -- not a
    separately reloaded copy, so it's guaranteed to reflect exactly what
    produced y_pred, not a slightly different run."""
    bundle_dir = os.path.join(model_dir, feature)
    config = _load_config(bundle_dir, decoder, feature, test_frac, X_raw.shape[-1],
                           duration_minutes=duration_minutes, duration_tag=duration_tag)

    tagged = _tagged(decoder, duration_tag)
    with open(os.path.join(bundle_dir, f"{tagged}_scaler.pkl"), 'rb') as f:
        scaler = pkl.load(f)
    X = scaler.transform(X_raw)

    offset = 0
    if decoder in ('qrnn', 'lstm'):
        X, _ = transform_data(X, y_raw_vel, timesteps=config['timesteps'])
        offset = config['timesteps'] - 1

    model_cls = {'qrnn': QRNNDecoder, 'lstm': LSTMDecoder, 'mlp': MLPDecoder}[decoder]
    model = model_cls(config)
    model.build(input_shape=(None,) + X.shape[1:])
    model.load_weights(os.path.join(bundle_dir, f"{tagged}.weights.h5"))

    y_pred = model.predict(X, batch_size=config['batch_size'], verbose=verbose)

    op_estimator = {'mlp': estimate_ops_mlp, 'lstm': estimate_ops_lstm,
                     'qrnn': estimate_ops_qrnn}[decoder]
    op_estimate = op_estimator(model, X)

    return y_pred, offset, op_estimate


def load_kf_decoder(model_dir, feature, test_frac, X_raw, y_raw_full,
                     duration_minutes=None, duration_tag=None):
    """y_raw_full must be the full 6-column y_task (position/velocity/accel) --
    KF needs y_raw_full[:1, :] as its seed state, matching eval_kf_decoder.py.
    Returns (y_pred_vel, offset, op_estimate) -- see load_dl_decoder()'s
    own docstring for what op_estimate is."""
    bundle_dir = os.path.join(model_dir, feature)
    _load_config(bundle_dir, 'kf', feature, test_frac, X_raw.shape[-1],
                 duration_minutes=duration_minutes, duration_tag=duration_tag)

    tagged = _tagged('kf', duration_tag)
    with open(os.path.join(bundle_dir, f"{tagged}_scaler.pkl"), 'rb') as f:
        scaler = pkl.load(f)
    X = scaler.transform(X_raw)

    with open(os.path.join(bundle_dir, f"{tagged}_model.pkl"), 'rb') as f:
        model = pkl.load(f)

    y_pred_full = model.predict(X, y_raw_full[:1, :])
    y_pred_vel = y_pred_full[:, 2:4]

    # predict() loops range(Z.shape[1]-1) -- (n_test_samples - 1) actual
    # recursion steps, not n_test_samples itself; see estimate_ops_kf()'s
    # own docstring for why that distinction matters to the op count.
    op_estimate = estimate_ops_kf(model, n_timesteps=X.shape[0] - 1)

    return y_pred_vel, 0, op_estimate


def load_wf_decoder(model_dir, feature, test_frac, X_raw, y_raw_vel,
                     duration_minutes=None, duration_tag=None):
    """Returns (y_pred, offset, op_estimate) -- see load_dl_decoder()'s
    own docstring for what op_estimate is."""
    bundle_dir = os.path.join(model_dir, feature)
    config = _load_config(bundle_dir, 'wf', feature, test_frac, X_raw.shape[-1],
                           duration_minutes=duration_minutes, duration_tag=duration_tag)

    tagged = _tagged('wf', duration_tag)
    with open(os.path.join(bundle_dir, f"{tagged}_scaler.pkl"), 'rb') as f:
        scaler = pkl.load(f)
    X = scaler.transform(X_raw)

    X, _ = transform_data(X, y_raw_vel, timesteps=config['timesteps'])
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], order='F')

    with open(os.path.join(bundle_dir, f"{tagged}_model.pkl"), 'rb') as f:
        model = pkl.load(f)

    y_pred = model.predict(X)
    offset = config['timesteps'] - 1

    op_estimate = estimate_ops_wf(model, X)

    return y_pred, offset, op_estimate


COLORS = {'mlp': 'crimson', 'lstm': 'darkorange', 'qrnn': 'seagreen',
          'kf': 'purple', 'wf': 'goldenrod', 'snn': 'royalblue'}
STYLES = {'mlp': '--', 'lstm': '--', 'qrnn': '--', 'kf': '-.', 'wf': '-.', 'snn': ':'}


def rolling_rmse(sq_err_xy, window):
    combined = sq_err_xy.mean(axis=1)
    n = len(combined)
    out = np.full(n, np.nan)
    kernel = np.ones(window) / window
    if n >= window:
        smoothed = np.convolve(combined, kernel, mode='valid')
        out[window - 1:] = np.sqrt(smoothed)
    return out


def _mean_ci_from_values(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n > 1:
        sem = values.std(ddof=1) / np.sqrt(n)
        tcrit = t_dist.ppf(1 - (1 - confidence) / 2, df=n - 1)
        margin = float(tcrit * sem)
    else:
        margin = 0.0
    return mean, mean - margin, mean + margin


def chunked_rmse_ci(sq_err_combined, n_splits=DEFAULT_CI_N_SPLITS, confidence=0.95):
    sq_err_combined = np.asarray(sq_err_combined)
    n = len(sq_err_combined)
    n_splits = max(1, min(n_splits, n))
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    chunk_rmse = np.array([
        np.sqrt(sq_err_combined[bounds[i]:bounds[i + 1]].mean())
        for i in range(n_splits)
    ])
    mean, lo, hi = _mean_ci_from_values(chunk_rmse, confidence)
    return mean, lo, hi, n_splits


def chunked_cc_ci(y_true, y_pred, n_splits=DEFAULT_CI_N_SPLITS, confidence=0.95):
    n = len(y_true)
    n_splits = max(1, min(n_splits, n))
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    chunk_cc = []
    for i in range(n_splits):
        sl = slice(bounds[i], bounds[i + 1])
        # NOT passing multioutput='raw_values' here is deliberate and
        # harmless: this wants (cc_x+cc_y)/2 regardless, and
        # pearson_corrcoef's default (multioutput='uniform_average')
        # already returns exactly that average directly as a scalar --
        # the fallback branch below just duplicates it into both
        # variables so the same averaging arithmetic still works. Same
        # function, same missing kwarg as the OTHER call site below, but
        # genuinely not broken here -- see that one's comment for the
        # call site where this DOES matter.
        cc = pearson_corrcoef(y_true[sl], y_pred[sl])
        if hasattr(cc, '__len__'):
            cc_x, cc_y = float(cc[0]), float(cc[1])
        else:
            cc_x = cc_y = float(cc)
        chunk_cc.append((cc_x + cc_y) / 2)
    mean, lo, hi = _mean_ci_from_values(np.array(chunk_cc), confidence)
    return mean, lo, hi, n_splits


def _suffixed_path(save_path, suffix):
    if not save_path or not suffix:
        return save_path
    root, ext = os.path.splitext(save_path)
    return f"{root}_{suffix}{ext}"


def _save_or_show(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, dpi=150)
        print(f"Saved figure to {path}")
        plt.close(fig)
    else:
        plt.show()


def make_test_window_trajectory_grid(label, y_pos_common, pred_common, segment_samples=260,
                                      n_segments=16, step_time=0.004, save_path=None):
    """First n_segments consecutive, non-overlapping segment_samples-wide
    chunks of the comparison range (default: 260 samples = 4x256ms = ~1s
    each, matching train_snn.py's --truncation-chunks default -- the SAME
    "unit of analysis" used for the SNN's own gradient truncation, not a
    coincidence). For each segment, reconstructs every requested decoder's
    predicted 2D trajectory (integrating predicted velocity, anchored at
    the segment's TRUE starting position) and overlays it against the
    true path -- what each decoder's predictions actually look like as a
    trajectory, on the SAME data, side by side.

    Reuses reconstruct_path()/make_trajectory_grid_figure() from
    plot_trajectory_grid.py directly -- those were built and tested there
    first; no reason to reimplement trajectory logic a second time here.
    No target-position marker: these are fixed-length TIME segments, not
    task trials bounded by a single target, so there's no one well-defined
    target per segment the way plot_trajectory_grid.py's own --trial_mode
    task grouping has.
    """
    from plot_trajectory_grid import make_trajectory_grid_figure

    if y_pos_common is None:
        print("  [skip] trajectory grid: no y_pos_common provided")
        return None

    n_common = len(y_pos_common)
    max_segments = n_common // segment_samples
    n_segments_actual = min(n_segments, max_segments)
    if n_segments_actual < n_segments:
        print(f"  [note] trajectory grid: only {n_segments_actual} full "
              f"{segment_samples}-sample segments available in the comparison range "
              f"({n_common} points total), requested {n_segments}")
    if n_segments_actual == 0:
        print(f"  [skip] trajectory grid: comparison range ({n_common} points) is shorter "
              f"than one segment ({segment_samples} points)")
        return None

    segments = [(i, i * segment_samples, i * segment_samples + segment_samples - 1)
                for i in range(n_segments_actual)]

    fig = make_trajectory_grid_figure(
        segments, y_pos_common, target_pos_common=None, pred_common=pred_common,
        step_time=step_time, session_id=label, n_trials_requested=n_segments_actual,
        colors=COLORS, unit_label='segment')
    _save_or_show(fig, save_path)
    return fig


def make_test_window_velocity_grid(label, y_true_common, pred_common, segment_samples=260,
                                    n_segments=16, save_path=None):
    """SAME segments and SAME 2D-grid visual format as
    make_test_window_trajectory_grid() -- reuses
    make_trajectory_grid_figure() directly, just with reconstruct=False:
    plots the raw predicted (vx, vy) CURVE directly (no integration,
    since velocity doesn't need reconstructing from itself), rather than
    an integrated 2D position path.

    This answers a genuinely different question than the trajectory grid,
    not just the same one from another angle: position INTEGRATES
    velocity, which smooths away a lot of per-timestep error -- a decoder
    can look deceptively good in the trajectory grid even when its raw
    velocity trace is noisy underneath, since small, frequent errors of
    opposing sign partially cancel out under integration. Looking at the
    (vx, vy) curve directly is the less forgiving, more diagnostic
    comparison: does the decoder's OWN per-timestep prediction actually
    track the true signal, not just "does the resulting path end up
    close." Unlike the trajectory grid's reconstructed paths (which are
    anchored to start exactly at the true position), each decoder's
    velocity curve here starts wherever it actually predicted -- there's
    nothing to anchor a velocity curve to in the first place.
    """
    n_common = len(y_true_common)
    max_segments = n_common // segment_samples
    n_segments_actual = min(n_segments, max_segments)
    if n_segments_actual < n_segments:
        print(f"  [note] velocity grid: only {n_segments_actual} full "
              f"{segment_samples}-sample segments available in the comparison range "
              f"({n_common} points total), requested {n_segments}")
    if n_segments_actual == 0:
        print(f"  [skip] velocity grid: comparison range ({n_common} points) is shorter "
              f"than one segment ({segment_samples} points)")
        return None

    from plot_trajectory_grid import make_trajectory_grid_figure

    segments = [(i, i * segment_samples, i * segment_samples + segment_samples - 1)
                for i in range(n_segments_actual)]

    fig = make_trajectory_grid_figure(
        segments, y_true_common, target_pos_common=None, pred_common=pred_common,
        step_time=0.004, session_id=label, n_trials_requested=n_segments_actual,
        colors=COLORS, unit_label='segment', reconstruct=False, space_label='velocity')
    _save_or_show(fig, save_path)
    return fig


def _make_session_figures(args, label, pred_common, sq_err, y_true_common,
                           start_raw, end_raw, n_common, metrics, y_pos_common=None):
    make_test_window_trajectory_grid(
        label, y_pos_common, pred_common,
        segment_samples=getattr(args, 'segment_samples', 260),
        n_segments=getattr(args, 'n_segments', 16),
        save_path=_suffixed_path(args.save_path, 'trajectory_grid'))

    make_test_window_velocity_grid(
        label, y_true_common, pred_common,
        segment_samples=getattr(args, 'segment_samples', 260),
        n_segments=getattr(args, 'n_segments', 16),
        save_path=_suffixed_path(args.save_path, 'velocity_grid'))

    fig1, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for a, col, label_ in zip(ax, [0, 1], ['x', 'y']):
        a.plot(y_true_common[:, col], label='true', color='black', linewidth=1.2)
        for name, y_pred in pred_common.items():
            a.plot(y_pred[:, col], label=f"{name.upper()} pred",
                   color=COLORS.get(name, 'gray'), linestyle=STYLES.get(name, '--'),
                   linewidth=0.9, alpha=0.85)
        a.set_ylabel(f'velocity ({label_})')
        a.legend(loc='upper right', fontsize=7, ncol=2)
    ax[-1].set_xlabel(f'Test sample (relative to raw test row {start_raw})')
    ax[0].set_title(f'{label}: full test-set predictions vs. true velocity')
    fig1.tight_layout()
    _save_or_show(fig1, args.save_path)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    for name in pred_common:
        rr = rolling_rmse(sq_err[name], args.roll_window)
        ax2.plot(rr, label=name.upper(), color=COLORS.get(name, 'gray'), linestyle=STYLES.get(name, '-'))
    ax2.set_xlabel(f'Test sample (relative to raw test row {start_raw})')
    ax2.set_ylabel(f'Rolling RMSE (window={args.roll_window} samples)')
    ax2.set_title(f'{label}: decoder error over time')
    ax2.legend(loc='upper right', fontsize=8)
    fig2.tight_layout()
    _save_or_show(fig2, _suffixed_path(args.save_path, 'rolling_rmse'))

    names = list(pred_common.keys())
    fig3, ax3 = plt.subplots(figsize=(1.4 * len(names) + 2, 4))
    x_pos = np.arange(len(names))
    width = 0.35
    ax3.bar(x_pos - width / 2, [metrics[n]['rmse_x'] for n in names], width, label='RMSE (x)', color='steelblue')
    ax3.bar(x_pos + width / 2, [metrics[n]['rmse_y'] for n in names], width, label='RMSE (y)', color='indianred')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([n.upper() for n in names])
    ax3.set_ylabel('RMSE')
    ax3.set_title(f'{label}: overall RMSE by decoder')
    ax3.legend()
    fig3.tight_layout()
    _save_or_show(fig3, _suffixed_path(args.save_path, 'rmse_bar'))

    fig4, ax4 = plt.subplots(figsize=(1.4 * len(names) + 2, 4))
    box_data = [np.sqrt(sq_err[n].mean(axis=1)) for n in names]
    box_labels = [n.upper() for n in names]
    try:
        # matplotlib >= 3.9 renamed 'labels' to 'tick_labels' on
        # Axes.boxplot() -- try the new name first (confirmed via a real
        # crash in this environment: "got an unexpected keyword argument
        # 'labels'"), fall back to the old one for older matplotlib
        # installs elsewhere (e.g. a different cluster/local machine).
        ax4.boxplot(box_data, tick_labels=box_labels, showfliers=False)
    except TypeError:
        ax4.boxplot(box_data, labels=box_labels, showfliers=False)
    ax4.set_ylabel('Per-sample RMSE (x,y combined)')
    ax4.set_title(f'{label}: error distribution by decoder')
    fig4.tight_layout()
    _save_or_show(fig4, _suffixed_path(args.save_path, 'error_boxplot'))

    n_decoders = len(names)
    fig5, axes5 = plt.subplots(2, n_decoders, figsize=(3.2 * n_decoders, 6.4), squeeze=False)
    for j, name in enumerate(names):
        y_pred = pred_common[name]
        for row, col, label_ in zip([0, 1], [0, 1], ['x', 'y']):
            a = axes5[row][j]
            a.scatter(y_true_common[:, col], y_pred[:, col], s=4, alpha=0.3,
                      color=COLORS.get(name, 'gray'))
            lims = [min(y_true_common[:, col].min(), y_pred[:, col].min()),
                    max(y_true_common[:, col].max(), y_pred[:, col].max())]
            a.plot(lims, lims, color='black', linewidth=0.8, linestyle='--')
            r2 = metrics[name][f'r2_{label_}']
            a.set_title(f'{name.upper()} ({label_}), R\u00b2={r2:.3f}', fontsize=9)
            if row == 1:
                a.set_xlabel('true velocity')
            if j == 0:
                a.set_ylabel('predicted velocity')
    fig5.suptitle(f'{label}: predicted vs. true velocity per decoder')
    fig5.tight_layout()
    _save_or_show(fig5, _suffixed_path(args.save_path, 'scatter_r2'))

    fig6, ax6 = plt.subplots(figsize=(12, 4))
    for name in pred_common:
        per_sample_mse = sq_err[name].mean(axis=1)
        cumulative_rmse = np.sqrt(np.cumsum(per_sample_mse) / np.arange(1, n_common + 1))
        ax6.plot(cumulative_rmse, label=name.upper(), color=COLORS.get(name, 'gray'),
                 linestyle=STYLES.get(name, '-'))
    ax6.set_xlabel(f'Test sample (relative to raw test row {start_raw})')
    ax6.set_ylabel('Cumulative RMSE (running average from sample 0)')
    ax6.set_title(f'{label}: cumulative loss across the test set')
    ax6.legend(loc='upper right', fontsize=8)
    fig6.tight_layout()
    _save_or_show(fig6, _suffixed_path(args.save_path, 'cumulative_loss'))

    true_speed = np.linalg.norm(y_true_common, axis=1)
    n_bins = min(args.speed_bins, max(3, n_common // 20))
    bin_edges = np.quantile(true_speed, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(true_speed, bin_edges[1:-1])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig7, ax7 = plt.subplots(figsize=(9, 4.5))
    for name in pred_common:
        per_sample_rmse = np.sqrt(sq_err[name].mean(axis=1))
        binned_rmse = [per_sample_rmse[bin_idx == b].mean() if np.any(bin_idx == b) else np.nan
                       for b in range(n_bins)]
        ax7.plot(bin_centers, binned_rmse, marker='o', label=name.upper(),
                 color=COLORS.get(name, 'gray'), linestyle=STYLES.get(name, '-'))
    ax7.set_xlabel('True speed, |velocity| (quantile-spaced bins)')
    ax7.set_ylabel('Mean per-sample RMSE within bin')
    ax7.set_title(f'{label}: decoder error vs. true movement speed')
    ax7.legend(loc='upper left', fontsize=8)
    fig7.tight_layout()
    _save_or_show(fig7, _suffixed_path(args.save_path, 'error_vs_speed'))


def run_snn_over_full_test_set(snn_checkpoint_path, snn_dataset_path, experiment,
                                ann_row_offset=0, ann_start_offset=0, ann_len=None, verbose=0,
                                base_nperseg=65, continuous_test_stream=False):
    """Run the SNN across the ENTIRE available test set -- not an
    efficiency-truncated subset, the previous behavior.

    experiment: 'bmi' or 'hkm' -- selects models.model_bmi vs.
    models.model_hkm (see _get_model_module()), and governs whether
    continuous_test_stream means anything at all.

    continuous_test_stream (HKM only -- ALWAYS effectively False for
    bmi, since model_bmi.py's own SNN_Speck.forward() has no reset_state
    argument to vary in the first place): default False resets SNN state
    at the start of EVERY test trial ("each independently resets" -- the
    only mode that ever existed for bmi, since 'continuous' and
    'chunked' training modes were both tried there and removed after
    real comparative results showed windowed outperforming both; the
    direct hkm analogue of how ANN/KF/WF are evaluated, and of how the
    SNN itself trains). Pass True to instead run every test trial as ONE
    continuous, never-reset stream: reset_state=True only for the very
    FIRST trial, False for every trial after, so state carries across
    trial boundaries exactly as if the whole test set were one long
    recording -- despite trial boundaries NOT actually being temporally
    continuous. A genuinely different, harder question from the default
    (does state persisting across a non-physiological boundary corrupt
    decoding right after it), not a stricter version of it -- see
    --continuous-snn-test-stream's own CLI help for the full framing.

    TWO input conventions, both supported (matches
    calibrate_snn_ann_offset()'s own split -- see that function's
    docstring for the full reasoning):

    - MULTIPLE small test/{i}.pkl trials, each exactly one base_nperseg-
      sized window (the older convention). Trial 0's own prediction is
      excluded from the final comparison: ANN's dense (step=1) row j
      predicts the LAGGED target at raw sample j+nperseg (matches
      bmi.features.extract()'s convention), so SNN trial i's t-th
      per-timestep prediction lines up with ANN row (i-1)*nperseg+t --
      trial 0 would need ANN rows that don't exist (negative indices).
      Every trial after 0 is included, covering (n_trials-1)*nperseg
      points -- essentially the whole test set. Trial 0's op/energy
      counts are excluded from snn_op_estimate for the same reason.

    - ONE single, whole, continuous test/0.pkl (make_huge_dataset.py's
      own test-saving convention, used throughout mua_256_group/
      mua_256_group_uniform). Here "exclude trial 0 entirely" would mean
      excluding the ENTIRE test set, since there IS only one trial --
      structurally wrong, not just a missed edge case. Instead, the
      first base_nperseg timesteps of this one trial are dropped
      (matching the ANN's own "needs base_nperseg samples of prior
      history" constraint exactly, just applied within a long trial
      instead of across many short ones), and the remaining
      (T-base_nperseg) per-timestep predictions are what's compared.
      snn_op_estimate here reflects the FULL call (including the dropped
      prefix) -- count_ops returns one lump sum per call, not a per-
      timestep breakdown, so cleanly excluding just the dropped prefix's
      own share isn't possible without deeper changes; see this
      function's own body for exactly where that's noted.

    ann_row_offset: from calibrate_snn_ann_offset() -- the ANN and SNN
    datasets' own test-split boundaries are independently rounded against
    very different row-count granularities and don't generally land on
    the same real moment even at identical --test_frac (confirmed on real
    data, not just in theory -- see that function's docstring).
    ann_start_offset/ann_len: intersect the SNN's own full range with
    whatever range the OTHER (non-SNN) requested decoders can actually
    provide (their own warm-up offsets / available length) -- same role
    these played in the previous revision, just applied by trimming the
    already-computed array rather than by skipping trials during the run.

    Returns (pred_snn, start_raw, end_raw, snn_op_estimate), or
    (None, None, None, None) if there's no usable data, or no overlap
    with the requested range. snn_op_estimate is op_energy_estimate.py's
    finalize_snn_ops() output -- the same {effective_macs, effective_accs,
    memory_accesses, energy_*, per_sample} shape every other decoder's
    own op estimate has.
    """
    test_dir = os.path.join(snn_dataset_path, 'test')
    test_files = sorted((f for f in os.listdir(test_dir) if f.endswith('.pkl')),
                         key=lambda f: int(f.split('.')[0]))
    n_snn_test_trials = len(test_files)
    if n_snn_test_trials == 0:
        print(f"  [skip] snn: no test trials found under {test_dir}")
        return None, None, None, None

    print(f"  Loading SNN model: {snn_checkpoint_path}")
    snn_model, checkpoint, velocity_scale = load_snn_model(snn_checkpoint_path, experiment)
    input_shape = checkpoint.get('input_shape')
    n_units_expected = input_shape[0] if input_shape else None

    if n_snn_test_trials == 1:
        with open(os.path.join(test_dir, test_files[0]), 'rb') as f:
            sample = pkl.load(f)
        trial_len = sample['input_spikes'].shape[1]
        print(f"  Running the SINGLE, continuous test trial ({trial_len} timesteps = "
              f"{trial_len * 0.004 * 1000:.0f}ms), dropping the first {base_nperseg} "
              f"timesteps to match the ANN's own required prior-history window")
        y_pred_trial, op_counts = predict_snn_trial(
            snn_model, sample['input_spikes'], velocity_scale, experiment,
            n_units_expected=n_units_expected, count_ops=True)
        pred_snn = y_pred_trial[base_nperseg:]
        # op_counts reflects the FULL call, including the base_nperseg-timestep
        # prior-context prefix trimmed above -- forward() returns one lump sum
        # per call, not a per-timestep breakdown, so cleanly excluding just the
        # dropped prefix's own contribution isn't possible without a deeper
        # change to count_ops's own granularity. n_samples below uses the FULL
        # trial_len (not len(pred_snn)) to stay consistent with that -- i.e.
        # this is "cost of the whole call," not "cost of only the predictions
        # actually compared," and is documented as such rather than presenting
        # a precision the counts don't actually support.
        snn_op_estimate = finalize_snn_ops(
            op_counts['mac'], op_counts['acc'], op_counts['elementwise'],
            n_samples=trial_len)
    else:
        # continuous_test_stream only ever means something for hkm --
        # for bmi it's silently inert (predict_snn_trial() never forwards
        # reset_state to model_bmi.py's forward() at all, see that
        # function's own docstring), so the print below reflects the
        # REAL, effective mode, not just whichever flag was passed.
        effectively_continuous = continuous_test_stream and experiment == "hkm"
        mode_desc = ("as ONE CONTINUOUS stream (reset only before trial 0)"
                     if effectively_continuous else "each independently reset")
        print(f"  Running all {n_snn_test_trials} test trials, {mode_desc}")
        pred_chunks = []
        total_mac, total_acc, total_elementwise, n_samples_counted = 0, 0.0, 0, 0
        for i, fname in enumerate(test_files):
            with open(os.path.join(test_dir, fname), 'rb') as f:
                sample = pkl.load(f)
            reset_state = True if not effectively_continuous else (i == 0)
            y_pred_trial, op_counts = predict_snn_trial(
                snn_model, sample['input_spikes'], velocity_scale, experiment,
                n_units_expected=n_units_expected, count_ops=True, reset_state=reset_state)
            if i > 0:  # trial 0 always excluded from the comparison -- see docstring --
                # and, for the SAME reason, excluded from the op/energy estimate too:
                # a WHOLE separate trial is cleanly excludable here, unlike the
                # single-trial branch's partial-prefix case above.
                pred_chunks.append(y_pred_trial)
                total_mac += op_counts['mac']
                total_acc += op_counts['acc']
                total_elementwise += op_counts['elementwise']
                n_samples_counted += y_pred_trial.shape[0]
            if verbose and (i + 1) % 50 == 0:
                print(f"    SNN: {i + 1}/{n_snn_test_trials} trials run")
        pred_snn = np.concatenate(pred_chunks, axis=0)
        snn_op_estimate = finalize_snn_ops(
            total_mac, total_acc, total_elementwise, n_samples=n_samples_counted)

    start_raw = ann_row_offset
    end_raw = ann_row_offset + len(pred_snn)

    if start_raw < ann_start_offset:
        trim = ann_start_offset - start_raw
        pred_snn = pred_snn[trim:]
        start_raw = ann_start_offset
    if ann_len is not None and end_raw > ann_len:
        pred_snn = pred_snn[:ann_len - start_raw]
        end_raw = ann_len

    if len(pred_snn) == 0:
        print(f"  [skip] snn: no overlap between the SNN's own available range and the "
              f"other decoders' (ann_start_offset={ann_start_offset}, ann_len={ann_len})")
        return None, None, None, None

    source_desc = "1 continuous trial" if n_snn_test_trials == 1 else f"{n_snn_test_trials - 1} trials"
    print(f"  SNN aligned against ANN rows [{start_raw}, {end_raw}) -- "
          f"{len(pred_snn)} points from {source_desc}")
    return pred_snn, start_raw, end_raw, snn_op_estimate


def run_session(args, session_label=None):
    """Run the full comparison for ONE session. See module docstring for
    the two modes (single duration vs. --train_durations sweep)."""
    label = session_label or os.path.basename(args.input_filepath)
    requested = [d.strip() for d in args.decoders.split(',')]
    unknown = set(requested) - set(ALL_DECODERS)
    if unknown:
        raise ValueError(f"Unknown decoder(s) {unknown}; choose from {ALL_DECODERS}")

    print(f"\n=== Session: {label} ===")
    print(f"Loading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y_task = f['y_task'][()]
        file_n_train_attr = f.attrs.get('n_train')  # see combine_trial_windows_to_ann_h5.py
    y_pos_all = y_task[:, 0:2]  # for the trajectory-grid figure -- reconstruct_path() anchors here
    y_vel_all = y_task[:, 2:4]
    N = X.shape[0]

    # n_train resolution, in priority order:
    #   1. --n_train_override, if explicitly passed (manual control)
    #   2. 'n_train' attr on the input file itself, if present -- written by
    #      combine_trial_windows_to_ann_h5.py for trial-structured data
    #      (e.g. the NWB/Jenkins-Nitschke conversion), where compute_
    #      aligned_split()'s own row-count formula doesn't apply: it
    #      assumes ONE continuous, densely-windowed stream, but a
    #      trial-concatenated file's windows are only locally dense
    #      WITHIN each trial, with real discontinuities between trials
    #      that a naive row-based split could straddle, silently letting
    #      one trial's near-duplicate windows leak across train/test.
    #      That file's own boundary was computed to fall between whole
    #      trials instead, mirroring the identical fix already applied on
    #      the SNN side for the same reason -- reading it back here
    #      rather than recomputing it from N keeps both sides consistent
    #      without the caller needing to pass anything by hand.
    #   3. compute_aligned_split(N, args.test_frac) -- original, unchanged
    #      behavior for a normal, single continuous session.
    n_train_override = getattr(args, 'n_train_override', None)
    if n_train_override is not None:
        n_train = n_train_override
        n_test = N - n_train
        print(f"Using --n_train_override={n_train} (explicit)")
    elif file_n_train_attr is not None:
        n_train = int(file_n_train_attr)
        n_test = N - n_train
        print(f"Using pre-computed n_train={n_train} from the input file's own "
              f"'n_train' attr (trial-aware boundary)")
    else:
        n_train, n_test = compute_aligned_split(N, args.test_frac)
    X_test = X[n_train:]
    y_pos_test = y_pos_all[n_train:]
    y_test_vel = y_vel_all[n_train:]
    y_test_full = y_task[n_train:]
    print(f"Chronological holdout: train={n_train}, test={n_test}")

    train_durations = getattr(args, 'train_durations', '') or ''
    if train_durations:
        duration_minutes_list = [float(x) for x in train_durations.split(',') if x.strip() != '']
    else:
        duration_minutes_list = [None]
    sweeping = duration_minutes_list != [None]

    decoders_this_session = requested
    all_metrics = {}

    for duration_minutes in duration_minutes_list:
        if duration_minutes is not None:
            duration_tag = f"{duration_minutes:g}min"
            print(f"\n--- {label}: duration = {duration_tag} ---")
            decoders_here = decoders_this_session
        else:
            duration_tag = None
            decoders_here = requested

        results = {}
        op_estimates = {}  # decoder name -> op_energy_estimate's result dict; kept SEPARATE
        # from results (which stays a pure {name: (y_pred, offset)} 2-tuple dict) so none
        # of the existing offset/alignment logic below (non_snn_offsets, non_snn_end_
        # candidates, pred_common construction) needs to change shape to accommodate this.
        for decoder in [d for d in decoders_here if d in DL_DECODERS]:
            try:
                print(f"Loading cached {decoder.upper()} model" +
                      (f" ({duration_tag})" if duration_tag else ""))
                y_pred, offset, op_estimate = load_dl_decoder(
                    args.model_dir, args.feature, decoder, args.test_frac, X_test, y_test_vel,
                    args.verbose, duration_minutes=duration_minutes, duration_tag=duration_tag)
                results[decoder] = (y_pred, offset)
                op_estimates[decoder] = op_estimate
            except FileNotFoundError as exc:
                print(f"  [skip] {decoder} ({duration_tag or 'full'}): cache not found ({exc})")

        if 'kf' in decoders_here:
            try:
                print("Loading cached KF model" + (f" ({duration_tag})" if duration_tag else ""))
                y_pred_kf, offset_kf, op_estimate_kf = load_kf_decoder(
                    args.model_dir, args.feature, args.test_frac, X_test, y_test_full,
                    duration_minutes=duration_minutes, duration_tag=duration_tag)
                results['kf'] = (y_pred_kf, offset_kf)
                op_estimates['kf'] = op_estimate_kf
            except FileNotFoundError as exc:
                print(f"  [skip] kf ({duration_tag or 'full'}): cache not found ({exc})")

        if 'wf' in decoders_here:
            try:
                print("Loading cached WF model" + (f" ({duration_tag})" if duration_tag else ""))
                y_pred_wf, offset_wf, op_estimate_wf = load_wf_decoder(
                    args.model_dir, args.feature, args.test_frac, X_test, y_test_vel,
                    duration_minutes=duration_minutes, duration_tag=duration_tag)
                results['wf'] = (y_pred_wf, offset_wf)
                op_estimates['wf'] = op_estimate_wf
            except FileNotFoundError as exc:
                print(f"  [skip] wf ({duration_tag or 'full'}): cache not found ({exc})")

        snn_checkpoint_path_d = None
        if 'snn' in decoders_here:
            base_snn_path = args.snn_checkpoint_path
            if duration_minutes is not None:
                candidate = resolve_snn_checkpoint_for_duration(base_snn_path, duration_minutes)
                if os.path.exists(candidate):
                    snn_checkpoint_path_d = candidate
                else:
                    print(f"  [skip] snn ({duration_tag}): no checkpoint found at {candidate}")
            else:
                if base_snn_path and os.path.exists(base_snn_path):
                    snn_checkpoint_path_d = base_snn_path
                else:
                    print(f"  [skip] snn: checkpoint not found at {base_snn_path}")

        if not results and snn_checkpoint_path_d is None:
            print(f"  [skip] no cached bundles found for duration={duration_tag or 'full'} "
                  f"-- skipping this duration entirely")
            continue

        non_snn_offsets = [offset for _, offset in results.values()]
        max_offset = max(non_snn_offsets) if non_snn_offsets else 0

        non_snn_end_candidates = [n_test]
        non_snn_end_candidates += [offset + len(y_pred) for y_pred, offset in results.values()]
        if args.n_windows is not None:
            non_snn_end_candidates.append(max_offset + args.start_test_idx + args.n_windows)
        non_snn_end = min(non_snn_end_candidates)

        pred_common = {}
        if snn_checkpoint_path_d is not None:
            ann_row_offset = calibrate_snn_ann_offset(args.snn_dataset_path, y_test_vel)
            snn_pred, snn_start_raw, snn_end_raw, snn_op_estimate = run_snn_over_full_test_set(
                snn_checkpoint_path_d, args.snn_dataset_path, args.experiment,
                ann_row_offset=ann_row_offset,
                ann_start_offset=max_offset + args.start_test_idx,
                ann_len=non_snn_end, verbose=args.verbose,
                continuous_test_stream=args.continuous_snn_test_stream)
            if snn_pred is not None:
                start_raw, end_raw = snn_start_raw, snn_end_raw
                pred_common['snn'] = snn_pred
                op_estimates['snn'] = snn_op_estimate
            else:
                start_raw = max_offset + args.start_test_idx
                end_raw = non_snn_end
        else:
            start_raw = max_offset + args.start_test_idx
            end_raw = non_snn_end

        if start_raw >= end_raw:
            raise ValueError(f"--start_test_idx {args.start_test_idx} leaves no valid range "
                              f"(start_raw={start_raw}, end_raw={end_raw}); check offsets/n_windows.")

        n_common = end_raw - start_raw
        print(f"Evaluating over raw test rows [{start_raw}, {end_raw}) "
              f"({n_common} points, max_offset={max_offset})")

        y_true_common = y_test_vel[start_raw:end_raw]
        y_pos_common = y_pos_test[start_raw:end_raw]
        for name, (y_pred, offset) in results.items():
            local_start = start_raw - offset
            pred_common[name] = y_pred[local_start:local_start + n_common]
        if 'snn' in pred_common and len(pred_common['snn']) != n_common:
            raise AssertionError(
                f"snn: aligned prediction length {len(pred_common['snn'])} != n_common {n_common} "
                f"-- this should not happen; check run_snn_over_full_test_set()'s range math.")

        sq_err = {name: (y_pred - y_true_common) ** 2 for name, y_pred in pred_common.items()}

        metrics = {}
        for name, y_pred in pred_common.items():
            rmse = root_mean_squared_error(y_true_common, y_pred)
            rmse_x = np.sqrt(sq_err[name][:, 0].mean())
            rmse_y = np.sqrt(sq_err[name][:, 1].mean())
            # multioutput='raw_values' is REQUIRED here, not optional --
            # pearson_corrcoef's default (multioutput='uniform_average')
            # returns a single float, already averaged across x/y, which
            # is a REAL BUG-PRODUCING mismatch with the code below:
            # hasattr(cc, '__len__') is False for a plain float, so cc_y
            # silently became None unconditionally and cc_x silently
            # became the x/y AVERAGE mislabeled as "x only" -- confirmed
            # against real data (check_snn_continuous_trace.py's
            # independently-computed per-axis CC_x/CC_y averaged to
            # within 0.002 of what this bug was reporting as "cc_x").
            cc = pearson_corrcoef(y_true_common, y_pred, multioutput='raw_values')
            cc_x = float(cc[0]) if hasattr(cc, '__len__') else float(cc)
            cc_y = float(cc[1]) if hasattr(cc, '__len__') else None
            r2_x = float(r2_score(y_true_common[:, 0], y_pred[:, 0]))
            r2_y = float(r2_score(y_true_common[:, 1], y_pred[:, 1]))

            ci_n_splits = getattr(args, 'ci_n_splits', DEFAULT_CI_N_SPLITS)
            combined_mse = sq_err[name].mean(axis=1)
            rmse_ci_mean, rmse_ci_low, rmse_ci_high, n_chunks = chunked_rmse_ci(
                combined_mse, n_splits=ci_n_splits)
            cc_ci_mean, cc_ci_low, cc_ci_high, _ = chunked_cc_ci(
                y_true_common, y_pred, n_splits=ci_n_splits)

            metrics[name] = {'rmse': float(rmse), 'rmse_x': float(rmse_x), 'rmse_y': float(rmse_y),
                              'cc_x': cc_x, 'cc_y': cc_y, 'r2_x': r2_x, 'r2_y': r2_y,
                              'rmse_ci_mean': rmse_ci_mean, 'rmse_ci_low': rmse_ci_low,
                              'rmse_ci_high': rmse_ci_high,
                              'cc_ci_mean': cc_ci_mean, 'cc_ci_low': cc_ci_low,
                              'cc_ci_high': cc_ci_high, 'n_chunks': n_chunks}
            if duration_minutes is not None:
                metrics[name]['train_duration_minutes'] = duration_minutes
            # op_estimate: hardware-agnostic effective-op/memory-access/energy
            # ESTIMATE (see op_energy_estimate.py) -- present for every decoder
            # now, kf/wf/mlp/lstm/qrnn (computed inside their own
            # load_*_decoder() above, against the SAME model+X that produced
            # that decoder's predictions) AND snn (computed INSIDE
            # SNN_Speck.forward() itself via its count_ops flag -- see
            # model_bmi.py -- and finalized by run_snn_over_full_test_set()
            # above, since the SNN's architecture and per-timestep, stateful
            # forward pass didn't fit the same external-estimator pattern the
            # other five decoders use; see op_energy_estimate.py's own SNN
            # section for the full reasoning).
            op_estimate = op_estimates.get(name)
            if op_estimate is not None:
                metrics[name]['op_estimate'] = op_estimate
                ps = op_estimate.get('per_sample')
                if ps:
                    print(f"{name.upper():>5s} | MACs/sample = {ps['effective_macs']:,.0f} | "
                          f"mem accesses/sample = {ps['memory_accesses']:,.0f} | "
                          f"est. energy/sample = [{ps['energy_total_j_low']*1e6:.4f}, "
                          f"{ps['energy_total_j_high']*1e6:.4f}] uJ (SRAM-to-DRAM range)")
                else:
                    print(f"{name.upper():>5s} | op_estimate computed but has no per_sample "
                          f"breakdown (degenerate n_samples=0? check op_estimates['{name}'] "
                          f"directly) -- trial totals: MACs={op_estimate['effective_macs']:,.0f}")
            print(f"{name.upper():>5s} | RMSE = {rmse:.4f} (x={rmse_x:.4f}, y={rmse_y:.4f}) | "
                  f"CC_x = {cc_x:.4f}, CC_y = {cc_y if cc_y is None else f'{cc_y:.4f}'} | "
                  f"R2_x = {r2_x:.4f}, R2_y = {r2_y:.4f} | "
                  f"RMSE_CI = {rmse_ci_mean:.4f} [{rmse_ci_low:.4f}, {rmse_ci_high:.4f}] | "
                  f"CC_CI = {cc_ci_mean:.4f} [{cc_ci_low:.4f}, {cc_ci_high:.4f}] (n={n_chunks})")

        metrics_save_path_d = _suffixed_path(args.metrics_save_path, duration_tag)
        if metrics_save_path_d:
            os.makedirs(os.path.dirname(metrics_save_path_d) or '.', exist_ok=True)
            with open(metrics_save_path_d, 'w') as f:
                json.dump({'session': label, 'duration_tag': duration_tag,
                           'train_duration_minutes': duration_minutes,
                           'start_raw': start_raw, 'end_raw': end_raw,
                           'n_samples': n_common, 'decoders': list(pred_common.keys()),
                           'metrics': metrics}, f, indent=2)
            print(f"Saved summary metrics to {metrics_save_path_d}")

        npz_save_path_d = _suffixed_path(args.npz_save_path, duration_tag)
        if npz_save_path_d:
            os.makedirs(os.path.dirname(npz_save_path_d) or '.', exist_ok=True)
            npz_payload = {'start_raw': start_raw, 'end_raw': end_raw, 'y_true': y_true_common}
            for name in pred_common:
                npz_payload[f'{name}_pred'] = pred_common[name]
                npz_payload[f'{name}_sq_err'] = sq_err[name]
            np.savez(npz_save_path_d, **npz_payload)
            print(f"Saved per-sample predictions and squared error to {npz_save_path_d}")

        if not sweeping:
            _make_session_figures(args, label, pred_common, sq_err, y_true_common,
                                   start_raw, end_raw, n_common, metrics,
                                   y_pos_common=y_pos_common)
        else:
            print(f"  (skipping figures for duration={duration_tag} -- sweeping training "
                  f"durations; use the metrics JSON to plot accuracy vs. duration)")

        if sweeping:
            all_metrics[duration_tag] = metrics
        else:
            all_metrics = metrics

    return all_metrics


def discover_sessions(checkpoints_root, checkpoint_config_name=None):
    """Every subdirectory of checkpoints_root that directly contains a
    best_model_weights.pth is a session (session_id = the subdirectory's
    name). Duration-tagged subfolders (N_min/) are nested one level
    deeper and are NOT themselves sessions.

    checkpoint_config_name: when set, looks one level deeper --
    {checkpoints_root}/{session_id}/{checkpoint_config_name}/
    best_model_weights.pth -- matching build_snn_checkpoint_path()'s own
    same parameter. None (the default) preserves the original, flat
    behavior exactly.
    """
    checkpoints_root = Path(checkpoints_root)
    sessions = []
    if not checkpoints_root.is_dir():
        return sessions
    for p in sorted(checkpoints_root.iterdir()):
        if not p.is_dir():
            continue
        marker = (p / checkpoint_config_name / "best_model_weights.pth"
                   if checkpoint_config_name is not None else p / "best_model_weights.pth")
        if marker.exists():
            sessions.append(p.name)
    return sessions


def build_snn_dataset_path(snn_dataset_root, session_id):
    """ASSUMPTION: SNN pkl test windows are stored per-session as
    {snn_dataset_root}/{session_id}/ (with a 'test' subfolder of {i}.pkl
    files inside). Edit this one function if your actual layout differs."""
    return os.path.join(snn_dataset_root, session_id)


def build_session_args(base_args, session_id):
    """Given the shared/base config (base_args) and a session_id, build a
    per-session args namespace with all the session-specific paths filled
    in:
        input file : {dataset_root}/{session_id}_binning.h5
        model cache: {model_cache_root}/{session_id}/{feature}/...
        snn weights: {checkpoints_dir}/{session_id}/best_model_weights.pth
                     (or one level deeper -- see
                     build_snn_checkpoint_path()'s own checkpoint_config_name)
        snn dataset: see build_snn_dataset_path()
    """
    session_args = SimpleNamespace(**vars(base_args))
    session_args.input_filepath = os.path.join(
        base_args.dataset_root, f"{session_id}_binning.h5")
    session_args.model_dir = os.path.join(base_args.model_cache_root, session_id)
    session_args.snn_checkpoint_path = build_snn_checkpoint_path(
        base_args.checkpoints_dir, session_id,
        checkpoint_config_name=getattr(base_args, 'checkpoint_config_name', None))
    session_args.snn_dataset_path = build_snn_dataset_path(
        base_args.snn_dataset_root, session_id)

    os.makedirs(base_args.results_dir, exist_ok=True)
    session_args.save_path = os.path.join(
        base_args.results_dir, f"{session_id}_comparison.png")
    session_args.metrics_save_path = os.path.join(
        base_args.results_dir, f"{session_id}_metrics.json")
    session_args.npz_save_path = (
        os.path.join(base_args.results_dir, f"{session_id}_per_sample.npz")
        if base_args.save_npz else None)
    return session_args


def run_all_sessions(base_args):
    """Discover every session in base_args.checkpoints_dir and run
    run_session() for each. Sessions that error out are logged and skipped
    rather than aborting the whole batch."""
    sessions = discover_sessions(
        base_args.checkpoints_dir,
        checkpoint_config_name=getattr(base_args, 'checkpoint_config_name', None))
    if not sessions:
        config_note = (f"/{base_args.checkpoint_config_name}"
                        if getattr(base_args, 'checkpoint_config_name', None) else "")
        raise FileNotFoundError(
            f"No {{session_id}}{config_note}/best_model_weights.pth found under "
            f"{base_args.checkpoints_dir}")
    print(f"Discovered {len(sessions)} session(s): {sessions}")

    combined = {}
    for session_id in sessions:
        session_args = build_session_args(base_args, session_id)
        try:
            combined[session_id] = run_session(session_args, session_label=session_id)
        except Exception as exc:  # noqa: BLE001 -- keep the batch going
            print(f"[ERROR] session {session_id} failed: {exc}")
            combined[session_id] = {'error': str(exc)}

    if base_args.combined_metrics_path:
        os.makedirs(os.path.dirname(base_args.combined_metrics_path) or '.', exist_ok=True)
        with open(base_args.combined_metrics_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved combined metrics for {len(sessions)} session(s) to "
              f"{base_args.combined_metrics_path}")

    return combined


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', type=str, required=True, choices=['bmi', 'hkm'],
                         help="Which model module to use: 'bmi' -> models.model_bmi, "
                              "'hkm' -> models.model_hkm (a separate fork -- see that "
                              "file's own docstring for why). Determines whether "
                              "--continuous-snn-test-stream is even meaningful (model_bmi.py's "
                              "SNN_Speck.forward() has no reset_state argument at all; only "
                              "model_hkm.py's does).")
    parser.add_argument('--continuous-snn-test-stream', action='store_true',
                         help="HKM only (--experiment hkm), ignored for bmi. Default: reset "
                              "SNN state at the start of EVERY test trial (the direct HKM "
                              "analogue of how ANN/KF/WF are evaluated -- each is an "
                              "independent example, same as SNN training itself). Pass this "
                              "flag to instead run every test trial as ONE continuous, "
                              "never-reset stream (state carried from the end of trial i into "
                              "the start of trial i+1) -- a genuinely different, harder "
                              "question (does state persisting across a boundary that isn't "
                              "actually temporally continuous corrupt decoding right after "
                              "it), not just a stricter version of the default.")
    parser.add_argument('--feature', type=str, default='sua', choices=['sua', 'mua'])
    parser.add_argument('--decoders', type=str, default='mlp,lstm,qrnn,kf,wf,snn',
                         help=f'Comma-separated list, any of {ALL_DECODERS}')
    parser.add_argument('--test_frac', type=float, default=0.1,
                         help='MUST match all cached ANN models AND export_snn_pkl.py --test_frac')
    parser.add_argument('--n_train_override', type=int, default=None,
                         help='Explicit train-row count, bypassing compute_aligned_split() '
                              'entirely. Takes priority over any n_train attr on the input '
                              'file. For normal, single-continuous-session data, leave unset '
                              '-- only needed for trial-structured data where the row-based '
                              'split formula does not apply (see combine_trial_windows_to_'
                              'ann_h5.py, which writes the file-attr alternative instead).')
    parser.add_argument('--start_test_idx', type=int, default=0,
                         help='Skip this many raw test rows past the point every decoder can start')
    parser.add_argument('--n_windows', type=int, default=None,
                         help='Cap the evaluated range to this many samples. Default = full test set.')
    parser.add_argument('--roll_window', type=int, default=20)
    parser.add_argument('--speed_bins', type=int, default=8)
    parser.add_argument('--segment_samples', type=int, default=260,
                         help='Trajectory-grid segment width, in samples (default 260 = '
                              '4x256ms=~1s, matching train_snn.py --truncation-chunks default)')
    parser.add_argument('--n_segments', type=int, default=16,
                         help='Number of segments shown in the trajectory-grid figure '
                              '(first N segments of the comparison range)')
    parser.add_argument('--ci_n_splits', type=int, default=DEFAULT_CI_N_SPLITS)
    parser.add_argument('--train_durations', type=str, default='',
                         help='Comma-separated list of training durations in minutes. When set, '
                              'each requested decoder (including snn) is loaded from its '
                              'duration-tagged cache/checkpoint for every duration. Figures are '
                              'skipped in this mode.')
    parser.add_argument('--verbose', type=int, default=0)

    parser.add_argument('--input_filepath', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--snn_checkpoint_path', type=str, default=None,
                         help="(single-session mode) Path to this session's best_model_weights.pth")
    parser.add_argument('--snn_dataset_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--metrics_save_path', type=str, default=None)
    parser.add_argument('--npz_save_path', type=str, default=None)

    parser.add_argument('--multi_session', action='store_true')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/bmi/mua',
                         help='Root dir containing one subdirectory per session, each with its own '
                              'best_model_weights.pth (and optional {N}_min/ duration subfolders)')
    parser.add_argument('--checkpoint_config_name', type=str, default=None,
                         help='When a sweep saves checkpoints one level deeper than the flat '
                              '{session_id}/best_model_weights.pth convention (e.g. multiple '
                              'configs per session: {session_id}/{config_name}/'
                              'best_model_weights.pth, as in a tau_syn/architecture sweep), set '
                              'this to that config subdirectory name (e.g. "tausyn_8"). Leave '
                              'unset for the original, flat layout.')
    parser.add_argument('--dataset_root', type=str, default='data/dataset')
    parser.add_argument('--model_cache_root', type=str, default='results/model_cache')
    parser.add_argument('--snn_dataset_root', type=str, default='./datasets/bmi/mua')
    parser.add_argument('--results_dir', type=str, default='results/multi_session')
    parser.add_argument('--combined_metrics_path', type=str,
                         default='results/multi_session/combined_metrics.json')
    parser.add_argument('--save_npz', action='store_true')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    print(args)

    if args.multi_session:
        run_all_sessions(args)
    else:
        missing = [name for name in ('input_filepath', 'model_dir') if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"Single-session mode requires --{missing[0].replace('_', '-')} "
                              f"(or pass --multi_session to run over all sessions).")
        needs_snn_paths = 'snn' in args.decoders.split(',')
        if needs_snn_paths and not (args.snn_checkpoint_path and args.snn_dataset_path):
            raise SystemExit("Single-session mode with 'snn' requested needs both "
                              "--snn_checkpoint_path and --snn_dataset_path.")
        run_session(args)

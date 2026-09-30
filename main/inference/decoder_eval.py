"""
Load trained decoders and evaluate them on one session's chronological test split.

ANN-side decoders (KF, WF, LSTM, QRNN) are the bundles written by the
preprocessing_training/eval_*.py scripts to {model_dir}/{feature}/; the SNN is
a train_snn.py checkpoint evaluated on the session's SNN .pkl test trials,
in PyTorch ('snn') and, on a machine with a Speck2f devkit, on the chip
('speck', full-data model only; see speck.py).

Alignment: the ANN dataset has one row per 4 ms step, and row j predicts the
velocity at raw sample j + nperseg. SNN test trials are back-to-back 65-sample
windows (or one long trial), so SNN timestep t of trial i lines up with ANN
row (i - 1) * nperseg + t; trial 0 (or the first nperseg timesteps of a single
trial) has no ANN counterpart and is dropped (for 'speck' too). calibrate_snn_ann_offset() checks
the two test splits start at the same moment and corrects any residual offset.
Every decoder is then scored over the range of rows they all cover.
"""

import json
import os
import pickle as pkl

import numpy as np
import torch
from scipy.stats import t as t_dist
from sinabs.activation import MultiSpike, SingleSpike
from sklearn.metrics import r2_score, root_mean_squared_error

from bmi.decoders import KalmanDecoder, LSTMDecoder, QRNNDecoder, WienerDecoder  # noqa: F401 (unpickling)
from bmi.metrics import pearson_corrcoef
from bmi.preprocessing import aligned_train_boundary, transform_data
from op_energy_estimate import (estimate_ops_kf, estimate_ops_lstm, estimate_ops_qrnn,
                                estimate_ops_wf, finalize_snn_ops)

DL_DECODERS = ('lstm', 'qrnn')
ANN_DECODERS = ('kf', 'wf') + DL_DECODERS
SNN_DECODERS = ('snn', 'speck')
ALL_DECODERS = ANN_DECODERS + SNN_DECODERS
BASE_NPERSEG = 65          # SNN window length in raw samples (256 ms)
STEP_S = 0.004
DEFAULT_CI_N_SPLITS = 10

# Only used for checkpoints whose args lack velocity scaling.
_FALLBACK_VELOCITY_SCALE = (-280.56, 316.54, 0.05)


# --------------------------------------------------------------------------- #
# Test split
# --------------------------------------------------------------------------- #

def test_split_start(n_rows, test_frac, n_train_attr=None, n_train_override=None):
    """First test row of the ANN dataset: --n_train_override, else the
    dataset's own n_train attribute (trial-structured NWB data), else the
    SNN-aligned chronological boundary."""
    if n_train_override is not None:
        return int(n_train_override)
    if n_train_attr is not None:
        return int(n_train_attr)
    return aligned_train_boundary(n_rows + BASE_NPERSEG, test_frac, BASE_NPERSEG)


# --------------------------------------------------------------------------- #
# ANN-side decoders
# --------------------------------------------------------------------------- #

def _tag(name, duration_tag):
    return f"{name}_{duration_tag}" if duration_tag else name


def load_bundle_config(bundle_dir, name, feature, test_frac, input_dim=None,
                       duration_minutes=None, duration_tag=None):
    """Read {name}[_{tag}]_config.json and check it matches this evaluation."""
    path = os.path.join(bundle_dir, f"{_tag(name, duration_tag)}_config.json")
    with open(path, 'r') as f:
        config = json.load(f)
    checks = [('test_frac', config.get('test_frac'), test_frac),
              ('feature', config.get('feature'), feature)]
    if input_dim is not None:
        checks.append(('input_dim', config.get('input_dim'), input_dim))
    for key, cached, wanted in checks:
        if cached != wanted:
            raise ValueError(f"{path}: {key}={cached!r}, but this evaluation uses {wanted!r}")
    if duration_minutes is not None:
        cached = config.get('train_duration_minutes')
        if cached is None or abs(cached - duration_minutes) > 1e-6:
            raise ValueError(f"{path}: train_duration_minutes={cached}, expected {duration_minutes}")
    return config


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pkl.load(f)


def load_ann_decoder(model_dir, feature, name, test_frac, input_dim, duration_minutes=None,
                     duration_tag=None):
    """Returns (model, scaler, config) for kf/wf/lstm/qrnn. Raises
    FileNotFoundError if the bundle is missing."""
    bundle_dir = os.path.join(model_dir, feature)
    tagged = _tag(name, duration_tag)
    config = load_bundle_config(bundle_dir, name, feature, test_frac, input_dim,
                                duration_minutes, duration_tag)
    scaler = _load_pickle(os.path.join(bundle_dir, f"{tagged}_scaler.pkl"))
    if name in ('kf', 'wf'):
        model = _load_pickle(os.path.join(bundle_dir, f"{tagged}_model.pkl"))
    else:
        model = {'lstm': LSTMDecoder, 'qrnn': QRNNDecoder}[name](config)
        model.build(input_shape=(None, config['timesteps'], config['input_dim']))
        model.load_weights(os.path.join(bundle_dir, f"{tagged}.weights.h5"))
    return model, scaler, config


def predict_ann_decoder(name, model, scaler, config, X_test, y_test_full, verbose=0):
    """Predict velocity for the whole test split.

    Returns (y_pred (n, 2), offset, op_estimate): prediction k is for test
    row k + offset (offset = timesteps - 1 for tap-delay decoders)."""
    X = scaler.transform(X_test)
    y_vel = y_test_full[:, 2:4]
    if name == 'kf':
        y_pred = model.predict(X, y_test_full[:1, :])[:, 2:4]
        return y_pred, 0, estimate_ops_kf(model, n_timesteps=X.shape[0] - 1)
    X_seq, _ = transform_data(X, y_vel, timesteps=config['timesteps'])
    offset = config['timesteps'] - 1
    if name == 'wf':
        X_flat = X_seq.reshape(X_seq.shape[0], X_seq.shape[1] * X_seq.shape[2], order='F')
        return model.predict(X_flat), offset, estimate_ops_wf(model, X_flat)
    y_pred = model.predict(X_seq, batch_size=config['batch_size'], verbose=verbose)
    estimator = {'lstm': estimate_ops_lstm, 'qrnn': estimate_ops_qrnn}[name]
    return y_pred, offset, estimator(model, X_seq)


# --------------------------------------------------------------------------- #
# SNN
# --------------------------------------------------------------------------- #

def _model_module(experiment):
    if experiment == 'bmi':
        import models.model_bmi as module
    elif experiment == 'hkm':
        import models.model_hkm as module
    else:
        raise ValueError(f"experiment must be 'bmi' or 'hkm', got {experiment!r}")
    return module


def load_snn_model(checkpoint_path, experiment, num_input_channels=None):
    """Rebuild a train_snn.py checkpoint from its saved args.

    Returns (model, checkpoint, velocity_scale); velocity_scale is
    (v_lo, v_hi, margin) as used in training."""
    module = _model_module(experiment)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    state_dict = checkpoint['model_state_dict']
    n_channels = (num_input_channels or (checkpoint.get('input_shape') or [None])[0]
                  or state_dict['layers.0.weight'].shape[1])
    spike_fn = {'multi': MultiSpike, 'single': SingleSpike}.get(args.get('spike_fn'))
    lo, hi, margin = _FALLBACK_VELOCITY_SCALE
    velocity_scale = (args.get('velocity_lo', lo), args.get('velocity_hi', hi),
                      args.get('velocity_margin', margin))

    model = module.create_model(
        use_spikingjelly=args.get('use_spikingjelly', False),
        last_layer_reset=args.get('last_layer_reset', False),
        weight_init=None,
        spike_fn=spike_fn,
        min_vmem=args.get('min_vmem'),
        neuron_type=args.get('neuron_type', 'lif'),
        tau_mem=args.get('tau_mem', 1.0),
        reset_type=args.get('reset_type', 'hard'),
        final_layer_reset_type=args.get('final_layer_reset_type'),
        surrogate_grad=args.get('surrogate_grad', 'periodic_exponential'),
        use_exodus=args.get('use_exodus'),
        use_iaf_squeeze=args.get('use_iaf_squeeze', False),
        n_bins=args.get('n_bins', 18),
        spike_thresholds=args.get('spike_thresholds'),
        temporal_decay_init=args.get('temporal_decay_init', 0.8),
        learnable_temporal_decay=args.get('learnable_temporal_decay', True),
        temporal_decay_stages=args.get('temporal_decay_stages', 1),
        num_input_channels=n_channels,
        hidden_dims=args.get('hidden_dims'),
        # A synaptic stage exists only if the checkpoint holds trained tau_syn
        # values (train_bmi_no_tau_syn.py checkpoints record a --tau-syn they
        # never used); load_model_weights then restores the trained values.
        tau_syn=(args.get('tau_syn') or 1.0) if any(k.endswith('.tau_syn') for k in state_dict) else None,
        velocity_lo=velocity_scale[0], velocity_hi=velocity_scale[1],
        velocity_margin=velocity_scale[2],
    )
    module.load_model_weights(model, state_dict, neuron_type=args.get('neuron_type', 'lif'),
                              source_description=checkpoint_path)
    model.eval()
    return model, checkpoint, velocity_scale


def unscale_velocity(v_scaled, velocity_scale):
    lo, hi, margin = velocity_scale
    return lo + (hi - lo) * (v_scaled - margin) / (1 - 2 * margin)


def predict_snn_trial(model, input_spikes, velocity_scale, experiment, reset_state=True):
    """Run one trial (input_spikes (C, T)) and return ((T, 2) velocity,
    op counts). reset_state=False (HKM only) carries state in from the
    previous call."""
    x = torch.from_numpy(input_spikes.T[:, np.newaxis, :].astype(np.float32))
    kwargs = {'count_ops': True}
    if experiment == 'hkm':
        kwargs['reset_state'] = reset_state
    with torch.no_grad():
        y_pred, _, _, op_counts = model(x, **kwargs)
    return unscale_velocity(y_pred.squeeze(1).numpy(), velocity_scale), op_counts


def snn_test_files(snn_dataset_path):
    test_dir = os.path.join(snn_dataset_path, 'test')
    files = sorted((f for f in os.listdir(test_dir) if f.endswith('.pkl')),
                   key=lambda f: int(f.split('.')[0]))
    return [os.path.join(test_dir, f) for f in files]


def _lagged_correlation(a, b, max_lag):
    """Correlation of a[t] with b[t + lag] for lag in [-max_lag, max_lag]."""
    lags = np.arange(-max_lag, max_lag + 1)
    ccs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        x, y = (a[:-lag], b[lag:]) if lag > 0 else (a[-lag:], b[:lag]) if lag < 0 else (a, b)
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            ccs[i] = np.corrcoef(x, y)[0, 1]
    return lags, ccs


def calibrate_snn_ann_offset(snn_dataset_path, y_test_vel, max_lag=2000, min_confidence=0.9):
    """ANN test row that SNN timestep 0 (after the dropped first window)
    corresponds to, found by matching true velocity on both sides.

    Returns 0 when the two already match, or when no lag correlates above
    min_confidence (with a warning: the two datasets may not be the same
    recording)."""
    files = snn_test_files(snn_dataset_path)
    if not files:
        return 0
    if len(files) == 1:
        snn_vel = _load_pickle(files[0])['velocity'][BASE_NPERSEG:]
    else:
        snn_vel = np.concatenate([_load_pickle(f)['velocity'] for f in files[1:]], axis=0)

    n = min(len(snn_vel), len(y_test_vel))
    if n < 100:
        return 0
    ann_x, snn_x = y_test_vel[:n, 0], snn_vel[:n, 0]
    if np.std(ann_x) < 1e-8 or np.std(snn_x) < 1e-8:
        return 0
    zero_lag_cc = np.corrcoef(ann_x, snn_x)[0, 1]
    if zero_lag_cc > min_confidence:
        return 0

    lags, ccs = _lagged_correlation(ann_x, snn_x, min(max_lag, n // 4))
    peak = np.nanargmax(ccs)
    best_lag, best_cc = int(lags[peak]), float(ccs[peak])
    if best_cc > min_confidence:
        # ann[t] matches snn[t + L], so SNN timestep k is ANN row k - L.
        print(f"  SNN/ANN alignment: zero-lag CC={zero_lag_cc:.4f}; correcting by "
              f"{-best_lag} rows (CC={best_cc:.4f})")
        return -best_lag
    print(f"  WARNING: SNN/ANN alignment found no confident lag (best CC={best_cc:.4f} at "
          f"{best_lag}); using zero offset. The ANN and SNN test data may not match.")
    return 0


def predict_snn_test_set(model, velocity_scale, snn_dataset_path, experiment,
                         continuous_stream=False):
    """Run the SNN over every test trial.

    Returns (pred (n, 2), op_estimate), with the first window dropped (see
    module docstring). continuous_stream (HKM only) carries state across
    trials instead of resetting at each one."""
    files = snn_test_files(snn_dataset_path)
    if len(files) == 1:
        y_pred, ops = predict_snn_trial(model, _load_pickle(files[0])['input_spikes'],
                                        velocity_scale, experiment)
        # op counts cover the whole call, including the dropped first window
        return y_pred[BASE_NPERSEG:], finalize_snn_ops(ops['mac'], ops['acc'], ops['elementwise'],
                                                       n_samples=len(y_pred))

    continuous = continuous_stream and experiment == 'hkm'
    preds, mac, acc, elementwise, n_samples = [], 0, 0.0, 0, 0
    for i, path in enumerate(files):
        y_pred, ops = predict_snn_trial(model, _load_pickle(path)['input_spikes'], velocity_scale,
                                        experiment, reset_state=(i == 0) or not continuous)
        if i == 0:
            continue
        preds.append(y_pred)
        mac, acc, elementwise = mac + ops['mac'], acc + ops['acc'], elementwise + ops['elementwise']
        n_samples += len(y_pred)
    return np.concatenate(preds, axis=0), finalize_snn_ops(mac, acc, elementwise, n_samples=n_samples)


def predict_speck_test_set(model, checkpoint, velocity_scale, cfg):
    """predict_snn_test_set() on the Speck devkit: returns (pred, chip
    timing and power), see speck.py."""
    import speck
    device = speck.open_speck(model, checkpoint, cfg.snn_dataset_path, cfg.speck_devkit,
                              cfg.speck_wait_time, cfg.speck_raster_dt)
    try:
        return speck.predict_speck_test_set(model, velocity_scale, cfg.snn_dataset_path, device,
                                            cfg.experiment, cfg.continuous_snn_test_stream)
    finally:
        device.close()


def available_decoders(cfg, decoders, duration_tag, snn_checkpoint_path):
    """The decoders with a trained model for this duration."""
    bundle_dir = os.path.join(cfg.model_dir, cfg.feature)
    snn = bool(snn_checkpoint_path and os.path.exists(snn_checkpoint_path))
    return [d for d in decoders
            if (d in SNN_DECODERS and snn) or (d in ANN_DECODERS and os.path.exists(
                os.path.join(bundle_dir, f"{_tag(d, duration_tag)}_config.json")))]


def duration_checkpoint_path(snn_checkpoint_path, duration_minutes):
    """.../per_session/{session}/best_model_weights.pth ->
    .../duration_sweep/{session}/{N}min/best_model_weights.pth."""
    parts = snn_checkpoint_path.split(os.sep)
    if 'per_session' not in parts:
        return None
    i = parts.index('per_session')
    return os.path.join(os.sep.join(parts[:i]), 'duration_sweep', parts[i + 1],
                        f"{int(round(duration_minutes))}min", parts[-1])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _mean_ci(values, confidence=0.95):
    """(mean, low, high) with a t-distribution confidence interval."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    sem = values.std(ddof=1) / np.sqrt(len(values))
    margin = float(t_dist.ppf(1 - (1 - confidence) / 2, df=len(values) - 1) * sem)
    return mean, mean - margin, mean + margin


def _chunk_bounds(n, n_splits):
    n_splits = max(1, min(n_splits, n))
    return np.linspace(0, n, n_splits + 1).astype(int), n_splits


def decoder_metrics(y_true, y_pred, n_splits=DEFAULT_CI_N_SPLITS):
    """RMSE, per-axis CC and R^2, and CIs over n_splits contiguous chunks."""
    sq_err = (y_pred - y_true) ** 2
    cc = pearson_corrcoef(y_true, y_pred, multioutput='raw_values')
    bounds, n_chunks = _chunk_bounds(len(y_true), n_splits)
    chunks = [slice(bounds[i], bounds[i + 1]) for i in range(n_chunks)]
    rmse_ci = _mean_ci([np.sqrt(sq_err[c].mean(axis=1).mean()) for c in chunks])
    cc_ci = _mean_ci([float(pearson_corrcoef(y_true[c], y_pred[c])) for c in chunks])
    return {
        'rmse': float(root_mean_squared_error(y_true, y_pred)),
        'rmse_x': float(np.sqrt(sq_err[:, 0].mean())),
        'rmse_y': float(np.sqrt(sq_err[:, 1].mean())),
        'cc_x': float(cc[0]), 'cc_y': float(cc[1]),
        'r2_x': float(r2_score(y_true[:, 0], y_pred[:, 0])),
        'r2_y': float(r2_score(y_true[:, 1], y_pred[:, 1])),
        'rmse_ci_mean': rmse_ci[0], 'rmse_ci_low': rmse_ci[1], 'rmse_ci_high': rmse_ci[2],
        'cc_ci_mean': cc_ci[0], 'cc_ci_low': cc_ci[1], 'cc_ci_high': cc_ci[2],
        'n_chunks': n_chunks,
    }


# --------------------------------------------------------------------------- #
# One evaluation (one training duration)
# --------------------------------------------------------------------------- #

def evaluate_decoders(data, cfg, decoders, duration_minutes=None, snn_checkpoint_path=None):
    """Evaluate every available decoder for one training duration.

    data: dict with X_test, y_test_full (6 columns), y_test_vel.
    cfg: namespace with model_dir, feature, test_frac, experiment,
        snn_dataset_path, continuous_snn_test_stream, ci_n_splits, verbose,
        and for 'speck' speck_devkit, speck_wait_time, speck_raster_dt.
    Returns None if no decoder is available, else a dict with the scored
    range (start_raw, end_raw, n_samples), per-decoder 'metrics' (each with
    its 'op_estimate', None for 'speck', which has its measured 'chip'
    timing and power instead), and the aligned arrays under 'arrays'.
    """
    tag = f"{duration_minutes:g}min" if duration_minutes is not None else None
    label = tag or 'full'
    preds, op_estimates, missing = {}, {}, []   # preds: name -> (y_pred, offset)
    for name in (d for d in decoders if d in ANN_DECODERS):
        try:
            model, scaler, config = load_ann_decoder(
                cfg.model_dir, cfg.feature, name, cfg.test_frac, data['X_test'].shape[-1],
                duration_minutes, tag)
        except FileNotFoundError:
            missing.append(name)
            continue
        print(f"  Evaluating {name.upper()} ({label})")
        y_pred, offset, op_estimates[name] = predict_ann_decoder(
            name, model, scaler, config, data['X_test'], data['y_test_full'], cfg.verbose)
        preds[name] = (y_pred, offset)

    n_test = len(data['y_test_vel'])
    start = max((offset for _, offset in preds.values()), default=0)
    end = min([n_test] + [offset + len(y) for y, offset in preds.values()])

    snn_names = [d for d in SNN_DECODERS if d in decoders]
    snn_preds, chip = {}, None
    if snn_names and snn_checkpoint_path and os.path.exists(snn_checkpoint_path):
        model, checkpoint, scale = load_snn_model(snn_checkpoint_path, cfg.experiment)
        if 'snn' in snn_names:
            print(f"  Evaluating SNN ({label}): {snn_checkpoint_path}")
            snn_preds['snn'], op_estimates['snn'] = predict_snn_test_set(
                model, scale, cfg.snn_dataset_path, cfg.experiment, cfg.continuous_snn_test_stream)
        if 'speck' in snn_names:
            print(f"  Evaluating SNN on Speck ({label}): {snn_checkpoint_path}")
            snn_preds['speck'], chip = predict_speck_test_set(model, checkpoint, scale, cfg)
            op_estimates['speck'] = None
        # Both run the same trials, so they share one alignment. Intersect
        # their rows with the ANN decoders' range.
        snn_start = calibrate_snn_ann_offset(cfg.snn_dataset_path, data['y_test_vel'])
        trim = max(0, start - snn_start)
        snn_start += trim
        snn_end = min(end, snn_start + min(len(p) for p in snn_preds.values()) - trim)
        if snn_end > snn_start:
            start, end = snn_start, snn_end
            snn_preds = {name: p[trim:trim + end - start] for name, p in snn_preds.items()}
        else:
            print(f"  [skip] {', '.join(snn_preds)} ({label}): no overlap with the other decoders' test rows")
            snn_preds = {}
    else:
        missing += snn_names
    if missing:
        print(f"  No {label} model for: {', '.join(missing)}")

    if not preds and not snn_preds:
        return None
    if start >= end:
        raise ValueError(f"No test rows are covered by every decoder ({label}: {start}-{end})")

    n = end - start
    y_true = data['y_test_vel'][start:end]
    aligned = {name: y[start - offset:start - offset + n] for name, (y, offset) in preds.items()}
    aligned.update(snn_preds)

    metrics = {}
    for name, y_pred in aligned.items():
        m = decoder_metrics(y_true, y_pred, cfg.ci_n_splits)
        if duration_minutes is not None:
            m['train_duration_minutes'] = duration_minutes
        m['op_estimate'] = op_estimates[name]
        if name == 'speck':
            m['chip'] = chip
            energy = f"measured energy/sample {chip['energy_j'] * 1e6:.4f} uJ"
        else:
            per_sample = op_estimates[name]['per_sample']
            energy = (f"est. energy/sample {per_sample['energy_total_j_low'] * 1e6:.4f}-"
                      f"{per_sample['energy_total_j_high'] * 1e6:.4f} uJ")
        metrics[name] = m
        print(f"  {name.upper():>5s} | RMSE={m['rmse']:.4f} | CC_x={m['cc_x']:.4f} CC_y={m['cc_y']:.4f} "
              f"| R2_x={m['r2_x']:.4f} R2_y={m['r2_y']:.4f} | {energy}")
    print(f"  Scored test rows [{start}, {end}) ({n} samples)")
    return {'duration_tag': tag, 'train_duration_minutes': duration_minutes,
            'start_raw': start, 'end_raw': end, 'n_samples': n,
            'decoders': list(aligned), 'metrics': metrics,
            'arrays': {'y_true': y_true, 'y_pos': data['y_test_pos'][start:end], 'pred': aligned}}

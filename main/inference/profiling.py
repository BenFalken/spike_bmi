"""
Per-sample latency, energy and parameter count for each decoder.

Every decoder is timed the way a real-time decoder runs: one prediction per
call, one new 4 ms sample at a time, never batched (so per-call framework
overhead counts). The SNN is fed one timestep per forward call, with state
reset only at the start of each test trial.

latency_s: median over n_repeats timed passes of the per-sample time, after
    n_warmup untimed passes (Keras traces its graph on the first call).
energy_j: per-sample energy from one separate, longer block of
    n_energy_repeats passes inside an EnergyMeter (RAPL where readable,
    otherwise a CPU-utilization proxy -- see energy_meter.py), since the
    counters need a longer window than one pass of a fast decoder.
power_w: mean power over that block (its energy / its wall time).
param_count: model parameters. Not comparable across families: KF's count is
    dominated by its channel x channel noise covariance.

The 'speck' decoder is not profiled here: its latency and power are
measured on the chip during evaluation (speck.py).
"""

import time

import numpy as np
import sinabs
import torch

from decoder_eval import load_ann_decoder, load_snn_model, snn_input, snn_test_files, _load_pickle


def count_params(name, model):
    if name == 'kf':
        return int(sum(np.asarray(m).size for m in model.model))   # [A, W, H, Q]
    if name == 'wf':
        reg = model.model
        return int(np.asarray(reg.coef_).size + np.asarray(getattr(reg, 'intercept_', [])).size)
    if name in ('lstm', 'qrnn'):
        return int(model.count_params())
    return int(sum(p.numel() for p in model.parameters()))          # SNN: parameters, not buffers


def time_per_sample(run_pass, n_samples, energy_meter_cls=None, n_repeats=3, n_warmup=3,
                    n_energy_repeats=10):
    """run_pass() runs n_samples single-sample predictions. Returns
    (latency_s, energy_j, energy_method, power_w); latency and energy are
    per sample, power_w is the mean over the energy block."""
    for _ in range(n_warmup):
        run_pass()
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        run_pass()
        times.append(time.perf_counter() - start)
    latency_s = float(np.median(times)) / n_samples

    if energy_meter_cls is None:
        return latency_s, None, None, None
    with energy_meter_cls() as meter:
        for _ in range(n_energy_repeats):
            run_pass()
    if meter.latency_s < 0.01:
        print(f"    [caution] energy window was only {meter.latency_s * 1000:.2f} ms; "
              f"raise --n_energy_repeats if energy looks noisy")
    energy_j = None if meter.energy_j is None else meter.energy_j / (n_energy_repeats * n_samples)
    power_w = None if meter.energy_j is None or meter.latency_s <= 0 else meter.energy_j / meter.latency_s
    return latency_s, energy_j, meter.energy_method, power_w


def _ann_single_sample_pass(name, model, scaler, config, X_test, y_init, timing_idx):
    """A run_pass() for one ANN decoder and the rows it can predict from."""
    X = scaler.transform(X_test)
    if name == 'kf':
        rows = X[timing_idx]
        return (lambda: [model.predict(x.reshape(1, -1), y_init) for x in rows]), len(rows)

    ts = config['timesteps']
    ends = timing_idx[timing_idx >= ts - 1]
    if name == 'wf':
        windows = [X[e - ts + 1:e + 1].reshape(1, -1, order='F') for e in ends]
        return (lambda: [model.predict(w) for w in windows]), len(windows)
    windows = [X[e - ts + 1:e + 1][None, ...] for e in ends]
    return (lambda: [model.predict(w, verbose=0) for w in windows]), len(windows)


def _snn_single_timestep_pass(model, snn_dataset_path, n_timesteps):
    """A run_pass() feeding the SNN one timestep per call. forward() resets
    state on every call, so the reset is suppressed except at trial starts."""
    timesteps = []
    for path in snn_test_files(snn_dataset_path):
        spikes = snn_input(model, _load_pickle(path)['input_spikes'])   # (C, T)
        timesteps += [(torch.from_numpy(spikes[:, t].astype(np.float32))[None, None, :], t == 0)
                      for t in range(spikes.shape[1])]
        if len(timesteps) >= n_timesteps:
            break
    timesteps = timesteps[:n_timesteps]
    if getattr(model, 'use_spikingjelly', False):
        print("  WARNING: SpikingJelly SNN: state is reset every call, so this times "
              "independent single-timestep calls")

    def run_pass():
        real_reset = sinabs.utils.reset_states
        try:
            with torch.no_grad():
                for x, trial_start in timesteps:
                    sinabs.utils.reset_states = real_reset if trial_start else (lambda *a, **k: None)
                    model(x)
        finally:
            sinabs.utils.reset_states = real_reset
    return run_pass, len(timesteps)


def profile_decoders(data, cfg, decoders, snn_checkpoint_path=None, energy_meter_cls=None):
    """{name: {latency_s, energy_j, energy_method, power_w, param_count}} for the
    full-data model of every available decoder except 'speck'."""
    rng = np.random.default_rng(0)
    X_test = data['X_test']
    timing_idx = rng.choice(len(X_test), size=min(cfg.n_timing_samples, len(X_test)), replace=False)
    y_init = data['y_test_full'][:1, :]
    common = dict(energy_meter_cls=energy_meter_cls, n_energy_repeats=cfg.n_energy_repeats)

    results = {}
    for name in decoders:
        if name == 'speck':
            continue
        if name == 'snn':
            if not snn_checkpoint_path:
                continue
            print("  Profiling SNN")
            model, _, _ = load_snn_model(snn_checkpoint_path, cfg.experiment)
            run_pass, n = _snn_single_timestep_pass(model, cfg.snn_dataset_path, cfg.n_timing_samples)
            timing = time_per_sample(run_pass, n, n_repeats=2, n_warmup=2, **common)
        else:
            try:
                model, scaler, config = load_ann_decoder(cfg.model_dir, cfg.feature, name,
                                                         cfg.test_frac, X_test.shape[-1])
            except FileNotFoundError:
                continue
            print(f"  Profiling {name.upper()}")
            run_pass, n = _ann_single_sample_pass(name, model, scaler, config, X_test, y_init,
                                                  timing_idx)
            repeats = dict(n_repeats=2, n_warmup=2) if name in ('lstm', 'qrnn') else {}
            timing = time_per_sample(run_pass, n, **repeats, **common)
        latency_s, energy_j, method, power_w = timing
        results[name] = {'latency_s': latency_s, 'energy_j': energy_j, 'energy_method': method,
                         'power_w': power_w,
                         'param_count': count_params(name, model)}
        energy = f"{energy_j * 1e6:.3f} uJ ({method})" if energy_j is not None else "n/a"
        print(f"  {name.upper():>5s} | {latency_s * 1000:.4f} ms/sample | {energy} | "
              f"{results[name]['param_count']:,} params")
    return results

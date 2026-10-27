"""
Why does one checkpoint lose accuracy on Speck? Input, or network activity?

Runs one session's test trials through the same checkpoint under several
conditions, each on the host (float and quantized network, one timestep per
call, as in training) and, with --speck_devkit, on the chip:

    raw            raw spike counts (bins can hold several spikes)
    binary         input clipped to 0/1, as the binarized models were trained
    binary_wait5   binary, waiting 5 ms instead of 1 ms per timestep for the
                   chip's output (spikes arriving late spill into later steps)
    binary_thin50  binary with half the input events dropped at random:
                   less input, same weights
    binary_thr2    binary with every threshold (and min_v_mem) doubled before
                   deployment: the same input, less network activity

thin50 and thr2 change the network, so their PyTorch RMSE gets worse; what
they test is whether the chip then agrees better with the quantized network.
If lowering the network's activity (thr2) restores agreement as much as
lowering the input (thin50) does, the cause is the activity, not the input.

Per condition it reports:
  - input events per timestep, and each layer's spikes per neuron per
    timestep (host float, host quantized, and the chip with every layer
    monitored over the first --layer_trials trials);
  - per-event risk on the quantized network: the share of neuron-steps where
    the excitatory events alone reach threshold but the step's net input does
    not (training stays silent; the chip fires if the excitation arrives
    first), and where the net input reaches 2 thresholds (training fires
    several spikes at once; the chip needs a separate event for each);
  - RMSE of each version, and the chip against the quantized network: output
    spike ratio, slope and correlation of the chip's predictions on the
    quantized network's (a slope well below 1 is a compressed output), the
    correlation of the summed output spikes per step and of the EMA-smoothed
    output the decoder reads; chip power and latency.

Input statistics for the session, and for --reference_dataset if given
(e.g. an Indy session, whose chip results were close to PyTorch), come first.

Every condition scores the same trials, from step BASE_NPERSEG of each trial
(as the ANN-aligned evaluation does), against the SNN dataset's velocity.
The checkpoint is not modified: binarization is set per condition here.
Results are written to --output after every condition.

Usage (from main/inference, devkit connected):
    python speck_activity_study.py --experiment hkm \
        --checkpoint .../per_session_binary/<session>/best_model_weights.pth \
        --snn_dataset .../snn_datasets/hkm/jenkins/mua/<session> \
        --reference_dataset .../snn_datasets/bmi/indy/mua_8_group/<indy session> \
        --speck_devkit speck2fdevkit:0 --output speck_activity_<session>.json

Without --speck_devkit only the host versions run (minutes, no chip).
"""

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from decoder_eval import BASE_NPERSEG, load_snn_model, load_test_trials, unscale_velocity  # noqa: E402
import speck  # noqa: E402

# name: (binarize, fraction of input events kept, threshold scale, wait time in s)
CONDITIONS = {
    'raw': (False, 1.0, 1.0, 0.001),
    'binary': (True, 1.0, 1.0, 0.001),
    'binary_wait5': (True, 1.0, 1.0, 0.005),
    'binary_thin50': (True, 0.5, 1.0, 0.001),
    'binary_thr2': (True, 1.0, 2.0, 0.001),
}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def input_stats(trials):
    """Input statistics over trials of (C, T) spike counts."""
    x = np.concatenate([t['input_spikes'] for t in trials], axis=1)
    nonzero = x[x > 0]
    per_step = x.sum(axis=0)
    return {
        'channels': int(x.shape[0]), 'timesteps': int(x.shape[1]),
        'events_per_step_raw': float(per_step.mean()),
        'events_per_step_binary': float((x > 0).sum(axis=0).mean()),
        'events_per_step_p95_raw': float(np.percentile(per_step, 95)),
        'spikes_per_channel_per_step': float(x.mean()),
        'max_channel_spikes_per_step': float(x.mean(axis=1).max()),
        'nonzero_bins_over_1': float((nonzero > 1).mean()) if nonzero.size else 0.0,
        'max_count': float(x.max()),
    }


def condition_input(trial_spikes, binarize, keep, rng):
    x = (trial_spikes > 0).astype(np.float32) if binarize else trial_spikes.astype(np.float32)
    if keep < 1.0:
        x = x * (rng.random(x.shape) < keep)
    return x


# --------------------------------------------------------------------------- #
# Host versions
# --------------------------------------------------------------------------- #

def scaled_network(snn_seq, threshold_scale):
    """A copy of the flattened network with every neuron layer's threshold
    and lower membrane bound multiplied by threshold_scale."""
    seq = copy.deepcopy(snn_seq)
    if threshold_scale != 1.0:
        for m in seq:
            for attr in ('spike_threshold', 'min_v_mem'):
                value = getattr(m, attr, None)
                if value is None:
                    continue
                scaled = value.float() * threshold_scale if torch.is_tensor(value) else value * threshold_scale
                if attr in m._buffers:
                    m._buffers[attr] = scaled
                elif attr in m._parameters:
                    m._parameters[attr].data = scaled
                else:
                    setattr(m, attr, scaled)
    return seq


def _state(neuron, n):
    v = getattr(neuron, 'v_mem', None)
    if v is None or v.numel() != n:
        return torch.zeros(n)
    return v.detach().reshape(-1).clone()


def run_with_drive(seq, x):
    """(C, T) input through seq one timestep per call (from rest). Returns
    ([(T, n) spikes per neuron layer], [risk counts per neuron layer]); the
    risk counts are taken before each neuron update from the layer's
    weighted input split into its excitatory and inhibitory parts."""
    conv = any(isinstance(m, nn.Conv2d) for m in seq)
    xt = torch.from_numpy(x.T.astype(np.float32))
    speck.reset_states(seq)
    modules = list(seq)
    n_layers = sum(1 for m in modules if isinstance(m, (nn.Linear, nn.Conv2d)))
    spikes = [[] for _ in range(n_layers)]
    risk = [{'neuron_steps': 0, 'order_sensitive': 0, 'multi_spike': 0, 'events_in': 0.0}
            for _ in range(n_layers)]
    with torch.no_grad():
        for t in range(len(xt)):
            out = xt[t:t + 1]
            if conv:
                out = out[:, :, None, None]
            layer = -1
            for i, m in enumerate(modules):
                if isinstance(m, nn.Flatten):
                    out = m(out)
                    continue
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    layer += 1
                    w = m.weight.reshape(m.weight.shape[0], -1)
                    pre = out.reshape(-1)
                    exc = (w.clamp(min=0) @ pre)
                    inh = (w.clamp(max=0) @ pre)
                    neuron = modules[i + 1]
                    theta = float(torch.as_tensor(neuron.spike_threshold).reshape(-1)[0])
                    v = _state(neuron, w.shape[0])
                    r = risk[layer]
                    r['neuron_steps'] += w.shape[0]
                    r['order_sensitive'] += int(((v + exc >= theta) & (v + exc + inh < theta)).sum())
                    r['multi_spike'] += int((v + exc + inh >= 2 * theta).sum())
                    r['events_in'] += float(pre.sum())
                    out = m(out)
                    continue
                out = m(out)
                spikes[layer].append(out.reshape(-1).numpy().copy())
    layers = [np.stack(s) for s in spikes]
    risks = [{'order_sensitive': r['order_sensitive'] / max(r['neuron_steps'], 1),
              'multi_spike': r['multi_spike'] / max(r['neuron_steps'], 1),
              'events_in_per_step': r['events_in'] / len(xt)} for r in risk]
    return layers, risks


def decode(model, counts, velocity_scale):
    return unscale_velocity(speck.decode_spike_counts(model, counts), velocity_scale)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def rmse(pred, target):
    return float(np.sqrt(((pred - target) ** 2).mean()))


def _corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def smooth(counts, decay, stages):
    out, state = np.zeros_like(counts, dtype=float), [np.zeros(counts.shape[1]) for _ in range(stages)]
    for t, x in enumerate(counts):
        for i in range(stages):
            state[i] = decay * state[i] + x
            x = state[i]
        out[t] = x
    return out


def chip_vs_quantized(chip_counts, quant_counts, chip_pred, quant_pred, decay, stages):
    """Lists per trial (scored rows only for the predictions)."""
    cc, qc = np.concatenate(chip_counts), np.concatenate(quant_counts)
    cp, qp = np.concatenate(chip_pred), np.concatenate(quant_pred)
    sm = [(smooth(c, decay, stages), smooth(q, decay, stages)) for c, q in zip(chip_counts, quant_counts)]
    return {
        'output_spike_ratio': float(cc.sum() / max(qc.sum(), 1)),
        'slope_x': float(np.polyfit(qp[:, 0], cp[:, 0], 1)[0]),
        'slope_y': float(np.polyfit(qp[:, 1], cp[:, 1], 1)[0]),
        'pred_corr_x': _corr(qp[:, 0], cp[:, 0]), 'pred_corr_y': _corr(qp[:, 1], cp[:, 1]),
        'rmse_chip_vs_quantized': rmse(cp, qp),
        'step_corr_lag0': _corr(cc.sum(axis=1), qc.sum(axis=1)),
        'smoothed_corr': _corr(np.concatenate([s[0] for s in sm]), np.concatenate([s[1] for s in sm])),
        'feature_rate_corr': _corr(cc.mean(axis=0), qc.mean(axis=0)),
    }


# --------------------------------------------------------------------------- #
# Chip
# --------------------------------------------------------------------------- #

def open_chip(seq, n_inputs, devkit, wait_time, monitor_all):
    return speck.SpeckDevkit(seq, n_inputs, devkit, wait_time, 0.1, monitor_all)


def close_chip(device, devkit):
    try:
        device.close()
    finally:
        try:
            import sinabs.backend.dynapcnn.io as sio
            sio.close_device(devkit)          # so the next condition can open it again
        except Exception:
            pass


def run_chip(seq, inputs, n_inputs, n_outputs, devkit, wait_time, monitor_all=False):
    """Every trial from a reset chip. Returns (output counts per trial,
    per-layer counts per trial or None, {power_w, latency_s, ...})."""
    device = open_chip(seq, n_inputs, devkit, wait_time, monitor_all)
    try:
        resets = device.resets
        counts, layer_counts, loop_s, energy_j, steps, events = [], [], 0.0, 0.0, 0, 0
        for i, x in enumerate(inputs):
            device.reset()
            c, seconds, power_w, n_events = device.run(x, n_outputs)
            counts.append(c)
            if monitor_all:
                layer_counts.append(device.layer_counts)
            loop_s, energy_j, steps, events = loop_s + seconds, energy_j + power_w * seconds, steps + len(c), events + n_events
            print(f"    [chip] trial {i + 1}/{len(inputs)}: {x.shape[1]} steps, "
                  f"{seconds * 1000 / x.shape[1]:.2f} ms/step, {power_w * 1000:.2f} mW", flush=True)
    finally:
        close_chip(device, devkit)
    info = {'power_mw': 1000 * energy_j / loop_s if loop_s else None,
            'latency_ms_per_step': 1000 * loop_s / steps if steps else None,
            'input_events_per_step': events / steps if steps else None, 'reset': resets}
    return counts, (layer_counts if monitor_all else None), info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def layer_rates(per_trial_layers):
    """Mean spikes per neuron per timestep of each layer over trials."""
    n_layers = len(per_trial_layers[0])
    return [float(np.concatenate([t[k] for t in per_trial_layers]).mean()) for k in range(n_layers)]


def pick_trials(trials, n, longest):
    idx = list(range(len(trials)))
    if longest:
        idx.sort(key=lambda i: -trials[i]['input_spikes'].shape[1])
    idx = idx[:n] if n > 0 else idx
    return sorted(idx)


def run_condition(name, spec, model, snn_seq, velocity_scale, trials, args, rng_seed):
    binarize, keep, thr_scale, wait = spec
    rng = np.random.default_rng(rng_seed)
    inputs = [condition_input(t['input_spikes'], binarize, keep, rng) for t in trials]
    targets = [t['velocity'][BASE_NPERSEG:] for t in trials]
    n_in, n_out = model.layers[0].in_features, 2 * model.n_bins
    decay = float(model.temporal_decay)
    stages = model.temporal_decay_stages

    float_seq = scaled_network(snn_seq, thr_scale)
    quant_seq = speck.discretized_sequential(speck.discretize(float_seq, n_in))
    t0 = time.time()
    float_layers = [speck.run_layers(float_seq, x) for x in inputs]
    quant_runs = [run_with_drive(quant_seq, x) for x in inputs]
    quant_layers = [r[0] for r in quant_runs]
    print(f"    host runs: {time.time() - t0:.0f} s", flush=True)

    result = {'binarize': binarize, 'input_kept': keep, 'threshold_scale': thr_scale, 'wait_time_s': wait,
              'input_events_per_step': float(np.mean(np.concatenate([x.sum(axis=0) for x in inputs]))),
              'layer_rates': {'float': layer_rates(float_layers), 'quantized': layer_rates(quant_layers)}}
    n_layers = len(quant_runs[0][1])
    steps = np.array([x.shape[1] for x in inputs], float)
    result['quantized_risk'] = [
        {k: float(np.average([r[1][layer][k] for r in quant_runs], weights=steps)) for k in quant_runs[0][1][layer]}
        for layer in range(n_layers)]

    preds = {'float': [decode(model, l[-1], velocity_scale)[BASE_NPERSEG:] for l in float_layers],
             'quantized': [decode(model, l[-1], velocity_scale)[BASE_NPERSEG:] for l in quant_layers]}
    out_counts = {'float': [l[-1] for l in float_layers], 'quantized': [l[-1] for l in quant_layers]}

    if args.speck_devkit:
        chip_counts, _, info = run_chip(float_seq, inputs, n_in, n_out, args.speck_devkit, wait)
        preds['chip'] = [decode(model, c, velocity_scale)[BASE_NPERSEG:] for c in chip_counts]
        out_counts['chip'] = chip_counts
        result['chip'] = info
        result['chip_vs_quantized'] = chip_vs_quantized(
            [c[BASE_NPERSEG:] for c in chip_counts], [c[BASE_NPERSEG:] for c in out_counts['quantized']],
            preds['chip'], preds['quantized'], decay, stages)
        if args.layer_trials > 0:
            k = min(args.layer_trials, len(inputs))
            _, chip_layers, _ = run_chip(float_seq, inputs[:k], n_in, n_out, args.speck_devkit, wait,
                                         monitor_all=True)
            result['layer_rates']['chip_monitored'] = layer_rates(chip_layers)
            result['layer_rates']['quantized_same_trials'] = layer_rates(quant_layers[:k])

    target = np.concatenate(targets)
    result['rmse'] = {v: rmse(np.concatenate(p), target) for v, p in preds.items()}
    result['output_spikes_per_step'] = {v: float(np.concatenate(c).sum(axis=1).mean()) for v, c in out_counts.items()}
    return result


def print_summary(results, layer_names):
    print("\n" + "=" * 100)
    print("RMSE and output spikes per step")
    print(f"{'condition':16s} {'in ev/step':>10s} {'float':>8s} {'quant':>8s} {'chip':>8s} "
          f"{'out/step q':>10s} {'out/step chip':>13s} {'power mW':>9s}")
    for name, r in results.items():
        chip = r.get('chip', {})
        print(f"{name:16s} {r['input_events_per_step']:10.1f} {r['rmse']['float']:8.2f} "
              f"{r['rmse']['quantized']:8.2f} {r['rmse'].get('chip', float('nan')):8.2f} "
              f"{r['output_spikes_per_step']['quantized']:10.1f} "
              f"{r['output_spikes_per_step'].get('chip', float('nan')):13.1f} "
              f"{(chip.get('power_mw') or float('nan')):9.2f}")
    if any('chip_vs_quantized' in r for r in results.values()):
        print("\nChip vs. quantized network (slope < 1: compressed predictions)")
        print(f"{'condition':16s} {'spk ratio':>9s} {'slope x':>8s} {'slope y':>8s} {'corr x':>7s} "
              f"{'corr y':>7s} {'step corr':>9s} {'EMA corr':>8s} {'rmse vs q':>9s}")
        for name, r in results.items():
            c = r.get('chip_vs_quantized')
            if c:
                print(f"{name:16s} {c['output_spike_ratio']:9.2f} {c['slope_x']:8.2f} {c['slope_y']:8.2f} "
                      f"{c['pred_corr_x']:7.2f} {c['pred_corr_y']:7.2f} {c['step_corr_lag0']:9.2f} "
                      f"{c['smoothed_corr']:8.2f} {c['rmse_chip_vs_quantized']:9.2f}")
    print("\nSpikes per neuron per timestep by layer (quantized host; chip with every layer monitored)")
    for name, r in results.items():
        q = ' '.join(f"{v:6.3f}" for v in r['layer_rates']['quantized'])
        line = f"{name:16s} quant [{q}]"
        if 'chip_monitored' in r['layer_rates']:
            qs = ' '.join(f"{v:6.3f}" for v in r['layer_rates']['quantized_same_trials'])
            ch = ' '.join(f"{v:6.3f}" for v in r['layer_rates']['chip_monitored'])
            line += f"\n{'':16s} quant* [{qs}]  chip* [{ch}]  (* first --layer_trials trials)"
        print(line)
    print(f"  layers: {', '.join(layer_names)}")
    print("\nPer-event risk on the quantized network, by layer (share of neuron-steps)")
    print("  order-sensitive: excitation alone reaches threshold, net input does not")
    print("  multi-spike:     net input reaches 2 thresholds in one step")
    for name, r in results.items():
        o = ' '.join(f"{x['order_sensitive']:6.3f}" for x in r['quantized_risk'])
        m = ' '.join(f"{x['multi_spike']:6.3f}" for x in r['quantized_risk'])
        e = ' '.join(f"{x['events_in_per_step']:6.1f}" for x in r['quantized_risk'])
        print(f"{name:16s} order [{o}]  multi [{m}]  events in/step [{e}]")


def main(args):
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model, checkpoint, velocity_scale = load_snn_model(args.checkpoint, args.experiment)
    speck.check_deployable(model, checkpoint)
    snn_seq = speck.flatten_snn(model)
    snn_seq.eval()
    trained_binarized = bool(checkpoint['args'].get('binarize_input', False))
    print(f"checkpoint: {args.checkpoint}\n  binarize_input in args: {'binarize_input' in checkpoint['args']} "
          f"({trained_binarized}); reset {checkpoint['args'].get('reset_type')}, spike_fn "
          f"{checkpoint['args'].get('spike_fn')}, hidden {checkpoint['args'].get('hidden_dims')}")

    all_trials = load_test_trials(args.snn_dataset)
    report = {'checkpoint': args.checkpoint, 'snn_dataset': args.snn_dataset,
              'input_stats': {'session_all_test_trials': input_stats(all_trials)}}
    if args.reference_dataset:
        report['input_stats']['reference_all_test_trials'] = input_stats(load_test_trials(args.reference_dataset))
    print("\nInput statistics (all test trials)")
    for k, s in report['input_stats'].items():
        print(f"  {k}: " + ', '.join(f"{n}={v:.4g}" for n, v in s.items()))

    idx = pick_trials(all_trials, args.n_trials, args.longest)
    trials = [all_trials[i] for i in idx]
    report['trials'] = idx
    print(f"\n{len(trials)} of {len(all_trials)} test trials, {sum(t['input_spikes'].shape[1] for t in trials)} steps")

    first = condition_input(trials[0]['input_spikes'], True, 1.0, None)
    speck.cross_check_flattened(model, snn_seq, first)

    layer_names = [f"hidden {i + 1}" for i in range(len(model.layers) // 2 - 1)] + ['output']
    report['conditions'] = {}
    for name in args.conditions:
        print(f"\n--- {name} ---", flush=True)
        report['conditions'][name] = run_condition(name, CONDITIONS[name], model, snn_seq, velocity_scale,
                                                   trials, args, args.seed)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
    print_summary(report['conditions'], layer_names)
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True, choices=['bmi', 'hkm'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--snn_dataset', required=True, help='session directory holding test/*.pkl')
    parser.add_argument('--reference_dataset', default=None,
                        help='another session directory for the input statistics only (e.g. Indy)')
    parser.add_argument('--speck_devkit', default=None, help='e.g. speck2fdevkit:0; omit for host only')
    parser.add_argument('--conditions', nargs='+', default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument('--n_trials', type=int, default=30, help='test trials to use (0: all)')
    parser.add_argument('--longest', action='store_true', help='use the longest trials instead of the first')
    parser.add_argument('--layer_trials', type=int, default=3,
                        help='trials re-run on the chip with every layer monitored (0: skip)')
    parser.add_argument('--seed', type=int, default=0, help='for the thinned input')
    parser.add_argument('--output', default='speck_activity_study.json')
    main(parser.parse_args())

"""
Speck lab: pick an SNN training regime for hkm whose accuracy survives the chip.

Self-contained (nothing outside main/speck_lab changes or depends on it). It
trains small models on part of one session, then scores each one as

    pytorch    the trained network, one timestep per call
    quantized  the network as deployed (integer weights and thresholds), one
               timestep per call
    emulated   the quantized network as the chip runs it, one event at a time
               (emulator.py); no devkit needed
    chip       the real Speck2f (with --speck_devkit)

Two commands:

  validate  check the emulator against a real chip run before trusting it:
            emulates the trials of a speck_activity_study.py result (same
            checkpoint, binarized input) and compares layer rates, output spike
            ratio and prediction slope with what the chip produced there.

      python lab.py validate --checkpoint .../best_model_weights.pth \\
          --snn_dataset .../snn_datasets/hkm/jenkins/mua/<session> \\
          --study_json ../../docs/data/speck_activity_jenkins_20090912.json

  run       train every --regimes entry (regimes.py) on the first --n_train
            training trials (validation: the next --n_val, picking the best
            epoch), then score it on the first --n_test test trials. Models and
            results go to --out_dir/<regime>/; a model already there is reused
            (train on the cluster, then add the chip on the laptop with
            --speck_devkit). Writes results.json, a table, and two figures.

      python lab.py run --snn_dataset .../snn_datasets/hkm/jenkins/mua/<session> \\
          --out_dir lab_runs/jenkins_0912 --regimes baseline rate10 rate30 rate10_single

Scores: RMSE from step 65 of each trial (as the evaluation aligns them),
spikes per neuron per timestep by layer, the prediction slope of each chip
version on the quantized network (1 = no compression), and synaptic events
per timestep (spikes x fan-out, an energy proxy); with the chip, its power
and energy per timestep.
"""

import argparse
import json
import os
import pickle as pkl
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
for p in (MAIN, os.path.join(MAIN, 'snn_training'), os.path.join(MAIN, 'inference')):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from emulator import ChipEmulator, quantize  # noqa: E402
from regimes import REGIMES, regime_config  # noqa: E402

SCORE_FROM = 65          # first scored step of each trial (decoder_eval.BASE_NPERSEG)
VERSIONS = ('pytorch', 'quantized', 'emulated', 'chip')


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def _natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def load_trials(session_dir, split, first=0, n=None):
    d = os.path.join(session_dir, split)
    files = sorted((f for f in os.listdir(d) if f.endswith('.pkl')), key=_natural)
    files = files[first:first + n] if n else files[first:]
    trials = []
    for f in files:
        with open(os.path.join(d, f), 'rb') as fh:
            data = pkl.load(fh)
        trials.append({'x': (np.asarray(data['input_spikes']) > 0).astype(np.float32),   # binarized
                       'v': np.asarray(data['velocity'], dtype=np.float32), 'file': f})
    return trials


def velocity_scale(session_dir, train_trials, init_checkpoint=None):
    """(v_lo, v_hi, margin): from the init checkpoint (its decoder was trained
    with them), else velocity_scalers.json, else the 0.5/99.5 percentiles of
    the training trials (as compute_velocity_scalers.py)."""
    if init_checkpoint:
        a = torch.load(init_checkpoint, map_location='cpu', weights_only=False)['args']
        return a['velocity_lo'], a['velocity_hi'], a['velocity_margin'], 'init checkpoint'
    try:
        from dataset import experiment_and_subject_from_path, get_velocity_scalers
        exp, subj = experiment_and_subject_from_path(session_dir)
        return (*get_velocity_scalers(exp, subj), 'velocity_scalers.json')
    except Exception:
        v = np.concatenate([t['v'].ravel() for t in train_trials])
        lo, hi = np.percentile(v, [0.5, 99.5])
        return float(lo), float(hi), 0.05, 'training-trial percentiles'


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(cfg, n_inputs, scale):
    import sinabs
    from models.model_hkm import create_model
    return create_model(
        use_spikingjelly=False, last_layer_reset=False,
        spike_fn=sinabs.activation.MultiSpike if cfg['spike_fn'] == 'multi' else sinabs.activation.SingleSpike,
        weight_init='kaiming', min_vmem=cfg['min_vmem'], neuron_type='iaf', tau_mem=12.0,
        reset_type=cfg['reset_type'], final_layer_reset_type=None, surrogate_grad='periodic_exponential',
        use_exodus=False, use_iaf_squeeze=True, n_bins=cfg['n_bins'],
        spike_thresholds=cfg['spike_thresholds'], temporal_decay_init=0.8, learnable_temporal_decay=True,
        temporal_decay_stages=cfg['temporal_decay_stages'], num_input_channels=n_inputs,
        hidden_dims=cfg['hidden_dims'], tau_syn=None,
        velocity_lo=scale[0], velocity_hi=scale[1], velocity_margin=scale[2])


def flatten(model):
    return nn.Sequential(*[getattr(m, 'neuron', m) for m in model.layers])


def decode(model, counts, scale):
    """(T, 2 n_bins) output spike counts -> (T, 2) velocity, through the
    model's EMA cascade and population readout (as speck.decode_spike_counts)."""
    from train_snn import unscale_velocity
    decay = model.temporal_decay
    stages = [torch.zeros(1, 2 * model.n_bins) for _ in range(model.temporal_decay_stages)]
    c = torch.as_tensor(np.asarray(counts), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for t in range(len(c)):
            x = c[t:t + 1]
            for i in range(len(stages)):
                stages[i] = decay * stages[i] + x
                x = stages[i]
            out.append(model.decode_output(x))
    return unscale_velocity(torch.cat(out).numpy(), *scale)


def reset_states(seq):
    import sinabs
    for m in seq.modules():
        for name, buf in list(m._buffers.items()):
            if buf is not None and buf._base is not None:
                m._buffers[name] = buf.detach().clone()
    sinabs.reset_states(seq)


def run_steps(seq, x):
    """(C, T) input through seq (Linear or 1x1 Conv2d + IAF) one timestep per
    call from rest -> [(T, n) spikes per neuron layer]."""
    conv = any(isinstance(m, nn.Conv2d) for m in seq)
    xt = torch.from_numpy(x.T.astype(np.float32))
    if conv:
        xt = xt[:, :, None, None]
    reset_states(seq)
    steps = []
    with torch.no_grad():
        for t in range(len(xt)):
            out, spikes = xt[t:t + 1], []
            for m in seq:
                out = m(out)
                if not isinstance(m, (nn.Linear, nn.Conv2d)):
                    spikes.append(out.reshape(-1).numpy().copy())
            steps.append(spikes)
    return [np.stack(layer) for layer in zip(*steps)]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(cfg, train_trials, val_trials, scale, epochs, seed, log):
    from train_snn import spike_rate_penalty, unscale_velocity
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_in = train_trials[0]['x'].shape[0]
    model = build_model(cfg, n_in, scale)
    if cfg['init']:
        from models.model_hkm import load_model_weights
        source = torch.load(cfg['init'], map_location='cpu', weights_only=False)
        load_model_weights(model, source['model_state_dict'], neuron_type='iaf', source_description=cfg['init'])
    n_spiking = len(model.layer_widths)
    targets = cfg['spike_rate_target'] * n_spiking if len(cfg['spike_rate_target']) == 1 \
        else list(cfg['spike_rate_target'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    lo, hi, margin = scale[:3]

    def to_batch(t):
        x = torch.from_numpy(t['x'].T)[:, None, :]                                  # (T, 1, C)
        v = margin + (1 - 2 * margin) * (t['v'] - lo) / (hi - lo)
        return x, torch.from_numpy(v.astype(np.float32))[:, None, :]               # (T, 1, 2)

    best, best_state, history = float('inf'), None, []
    for epoch in range(epochs):
        model.train()
        t0, losses, rates = time.time(), [], []
        for i in rng.permutation(len(train_trials)):
            x, y = to_batch(train_trials[i])
            pred, _, _ = model(x)
            rmse = torch.sqrt(((unscale_velocity(pred, lo, hi, margin)
                                - unscale_velocity(y, lo, hi, margin)) ** 2).mean() + 1e-8)
            penalty, layer_rates = spike_rate_penalty(model, x.shape[0], targets)
            loss = rmse + cfg['spike_rate_weight'] * penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(rmse.item())
            rates.append(layer_rates)
        val = evaluate_pytorch(model, val_trials, scale)
        entry = {'epoch': epoch + 1, 'train_rmse': float(np.mean(losses)), 'val_rmse': val['rmse'],
                 'train_layer_rates': np.mean(rates, axis=0).tolist(), 'val_layer_rates': val['layer_rates']}
        msg = (f"    epoch {epoch + 1}/{epochs} ({time.time() - t0:.0f} s): train RMSE {np.mean(losses):.2f}, "
               f"val RMSE {val['rmse']:.2f}, val rates [{' '.join(f'{r:.3f}' for r in val['layer_rates'])}]")
        criterion = val['rmse']
        if cfg.get('select') == 'emulated':
            entry['val_rmse_emulated'] = criterion = evaluate_emulated(model, val_trials, scale)
            msg += f", val RMSE emulated {criterion:.2f}"
        history.append(entry)
        log(msg)
        if criterion < best:
            best, best_state = criterion, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def evaluate_pytorch(model, trials, scale):
    model.eval()
    preds, layers = [], []
    seq = flatten(model)
    for t in trials:
        spikes = run_steps(seq, t['x'])
        layers.append(spikes)
        preds.append(decode(model, spikes[-1], scale))
    target = np.concatenate([t['v'][SCORE_FROM:] for t in trials])
    pred = np.concatenate([p[SCORE_FROM:] for p in preds])
    return {'rmse': rmse(pred, target), 'layer_rates': layer_rates(layers)}


def evaluate_emulated(model, trials, scale):
    """RMSE of the model as the emulated chip runs it."""
    model.eval()
    qlayers, _ = quantize(flatten(model))
    emulator = ChipEmulator(qlayers)
    pred = np.concatenate([decode(model, emulator.run(t['x'])[-1], scale)[SCORE_FROM:] for t in trials])
    return rmse(pred, np.concatenate([t['v'][SCORE_FROM:] for t in trials]))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def rmse(pred, target):
    return float(np.sqrt(((pred - target) ** 2).mean()))


def layer_rates(per_trial):
    return [float(np.concatenate([t[k] for t in per_trial]).mean()) for k in range(len(per_trial[0]))]


def synaptic_events_per_step(x_trials, per_trial, widths):
    """Spikes arriving at each layer x that layer's width, summed, per timestep."""
    total, steps = 0.0, 0
    for x, layers in zip(x_trials, per_trial):
        sources = [x.T] + layers[:-1]
        total += sum(float(s.sum()) * w for s, w in zip(sources, widths))
        steps += x.shape[1]
    return total / steps


def slope(pred, ref):
    return float(np.mean([np.polyfit(ref[:, k], pred[:, k], 1)[0] for k in (0, 1)]))


def score(model, trials, scale, emulator_shuffle='none', devkit=None, wait_time=0.001, log=print):
    """Every version on the same trials."""
    seq = flatten(model)
    qlayers, qseq = quantize(seq)
    widths = list(model.layer_widths)
    xs = [t['x'] for t in trials]
    target = np.concatenate([t['v'][SCORE_FROM:] for t in trials])
    runs = {'pytorch': [run_steps(seq, x) for x in xs], 'quantized': [run_steps(qseq, x) for x in xs]}
    t0 = time.time()
    emulator = ChipEmulator(qlayers, shuffle=emulator_shuffle)
    runs['emulated'] = [emulator.run(x) for x in xs]
    log(f"    emulated {sum(x.shape[1] for x in xs)} steps in {time.time() - t0:.0f} s")
    chip_info = None
    if devkit:
        import speck
        device = speck.SpeckDevkit(seq, xs[0].shape[0], devkit, wait_time, 0.1, False)
        try:
            outs, loop_s, energy, steps = [], 0.0, 0.0, 0
            for x in xs:
                device.reset()
                counts, seconds, power_w, _ = device.run(x, 2 * model.n_bins)
                outs.append([counts])
                loop_s, energy, steps = loop_s + seconds, energy + power_w * seconds, steps + x.shape[1]
            runs['chip'] = outs
            chip_info = {'power_mw': 1000 * energy / loop_s, 'latency_ms': 1000 * loop_s / steps,
                         'energy_uj_per_step': 1e6 * energy / steps}
        finally:
            device.close()
            try:
                import sinabs.backend.dynapcnn.io as sio
                sio.close_device(devkit)
            except Exception:
                pass
    preds = {v: np.concatenate([decode(model, r[-1], scale)[SCORE_FROM:] for r in rs]) for v, rs in runs.items()}
    out = {'rmse': {v: rmse(p, target) for v, p in preds.items()},
           'layer_rates': {v: layer_rates(rs) for v, rs in runs.items() if len(rs[0]) == len(widths)},
           'output_rate': {v: float(np.concatenate([r[-1] for r in rs]).mean()) for v, rs in runs.items()},
           'slope_vs_quantized': {v: slope(preds[v], preds['quantized']) for v in preds if v != 'quantized'},
           'synaptic_events_per_step': {v: synaptic_events_per_step(xs, rs, widths)
                                        for v, rs in runs.items() if len(rs[0]) == len(widths)},
           'input_events_per_step': float(np.mean(np.concatenate([x.sum(axis=0) for x in xs])))}
    if chip_info:
        out['chip'] = chip_info
    return out


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_validate(args):
    """The emulator against a speck_activity_study.py chip run."""
    sys.path.insert(0, os.path.join(MAIN, 'inference'))
    from decoder_eval import load_snn_model
    with open(args.study_json) as f:
        study = json.load(f)
    chip = study['conditions']['binary']
    model, _, scale = load_snn_model(args.checkpoint, 'hkm')
    trials_idx = study['trials']
    tests = load_trials(args.snn_dataset, 'test')
    trials = [tests[i] for i in trials_idx]
    qlayers, qseq = quantize(flatten(model))
    xs = [t['x'] for t in trials]
    quant = [run_steps(qseq, x) for x in xs]
    qpred = np.concatenate([decode(model, q[-1], scale)[SCORE_FROM:] for q in quant])
    k = 3       # the study monitored every chip layer on its first 3 trials
    print(f"\nChip (study, binary input) against the emulator on the same {len(trials)} trials")
    print(f"{'':22s} {'out spk ratio':>13s} {'slope':>6s}   layer rates, first {k} trials")
    c = chip['chip_vs_quantized']
    print(f"{'chip (study)':22s} {c['output_spike_ratio']:13.2f} {np.mean([c['slope_x'], c['slope_y']]):6.2f}   "
          f"[{' '.join(f'{r:.3f}' for r in chip['layer_rates']['chip_monitored'])}]")
    print(f"{'quantized (study)':22s} {'':13s} {'':6s}   "
          f"[{' '.join(f'{r:.3f}' for r in chip['layer_rates']['quantized_same_trials'])}]")
    print(f"{'quantized (here)':22s} {'':13s} {'':6s}   [{' '.join(f'{r:.3f}' for r in layer_rates(quant[:k]))}]")
    rows = {}
    for shuffle in args.shuffle:
        t0 = time.time()
        emu = [ChipEmulator(qlayers, shuffle=shuffle).run(x) for x in xs]
        epred = np.concatenate([decode(model, e[-1], scale)[SCORE_FROM:] for e in emu])
        ratio = float(sum(e[-1].sum() for e in emu) / max(sum(q[-1].sum() for q in quant), 1))
        rows[shuffle] = {'output_spike_ratio': ratio, 'slope': slope(epred, qpred),
                         'layer_rates_first_trials': layer_rates(emu[:k]), 'seconds': time.time() - t0}
        print(f"{'emulated, shuffle=' + shuffle:22s} {ratio:13.2f} {rows[shuffle]['slope']:6.2f}   "
              f"[{' '.join(f'{r:.3f}' for r in rows[shuffle]['layer_rates_first_trials'])}]  "
              f"({time.time() - t0:.0f} s)")
    print("\nThe emulator is trustworthy when its spike ratio, slope and layer rates are close to the chip's.")
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'study': args.study_json, 'emulated': rows}, f, indent=2)


def cmd_run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'lab.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    train_trials = load_trials(args.snn_dataset, 'train', 0, args.n_train)
    val_trials = load_trials(args.snn_dataset, 'train', args.n_train, args.n_val)
    test_trials = load_trials(args.snn_dataset, 'test', 0, args.n_test)
    log(f"{args.snn_dataset}: {len(train_trials)} train, {len(val_trials)} validation, {len(test_trials)} "
        f"test trials ({sum(t['x'].shape[1] for t in test_trials)} test steps)")

    for name in args.regimes:
        cfg = regime_config(name)
        if args.init and not cfg['init']:
            cfg['init'] = args.init
        scale = velocity_scale(args.snn_dataset, train_trials, cfg['init'])
        model_path = os.path.join(args.out_dir, name, 'model.pth')
        log(f"\n=== {name}: {REGIMES[name] or 'base config'}")
        if os.path.isfile(model_path) and not args.retrain:
            saved = torch.load(model_path, map_location='cpu', weights_only=False)
            model = build_model(saved['config'], saved['n_inputs'], saved['scale'])
            from models.model_hkm import load_model_weights      # skips the neuron-state buffers
            load_model_weights(model, saved['state_dict'], neuron_type='iaf', source_description=model_path)
            scale = (*saved['scale'], 'the saved model')       # decode as it was trained
            model.eval()
            history = saved['history']
            log(f"  reusing {model_path}")
        else:
            log(f"  velocity scale from {scale[3]}: {scale[0]:.2f} .. {scale[1]:.2f}, margin {scale[2]}")
            model, history = train(cfg, train_trials, val_trials, scale[:3], cfg['epochs'] or args.epochs,
                                   args.seed, log)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save({'config': cfg, 'n_inputs': train_trials[0]['x'].shape[0], 'scale': scale[:3],
                        'state_dict': model.state_dict(), 'history': history}, model_path)
        log("  scoring")
        r = score(model, test_trials, scale[:3], args.emulator_shuffle, args.speck_devkit, args.speck_wait_time, log)
        result_path = os.path.join(args.out_dir, name, 'result.json')
        previous_chip = {}
        if os.path.isfile(result_path):
            with open(result_path) as f:
                previous_chip = json.load(f)['scores']
        if not args.speck_devkit and 'chip' in previous_chip.get('rmse', {}):
            for key in ('rmse', 'output_rate', 'slope_vs_quantized'):     # keep an earlier chip run
                r[key]['chip'] = previous_chip[key]['chip']
            r['chip'] = previous_chip.get('chip')
        log("  RMSE " + ', '.join(f"{v} {x:.1f}" for v, x in r['rmse'].items())
            + " | slope vs quantized " + ', '.join(f"{v} {x:.2f}" for v, x in r['slope_vs_quantized'].items()))
        with open(result_path, 'w') as f:      # one file per regime: array tasks may run side by side
            json.dump({'config': cfg, 'history': history, 'scores': r}, f, indent=2)

    results = collect(args.out_dir)
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print_table(results, log)
    from plots import plot_results
    for p in plot_results(results, args.out_dir):
        log(f"Saved {p}")


def collect(out_dir):
    """{regime: result} from every <out_dir>/<regime>/result.json, in regimes.py order."""
    results = {}
    for name in list(REGIMES) + sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name, 'result.json')
        if name not in results and os.path.isfile(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results


def print_table(results, log):
    log("\n" + "=" * 110)
    log(f"{'regime':16s} " + ' '.join(f"{'RMSE ' + v:>15s}" for v in VERSIONS)
        + f" {'slope emu':>9s} {'slope chip':>10s} {'out rate q':>10s} {'out rate emu':>12s} {'syn ev/step emu':>15s}")
    for name, res in results.items():
        r = res['scores']
        log(f"{name:16s} " + ' '.join(f"{r['rmse'].get(v, float('nan')):15.1f}" for v in VERSIONS)
            + f" {r['slope_vs_quantized'].get('emulated', float('nan')):9.2f}"
            + f" {r['slope_vs_quantized'].get('chip', float('nan')):10.2f}"
            + f" {r['output_rate']['quantized']:10.3f} {r['output_rate']['emulated']:12.3f}"
            + f" {r['synaptic_events_per_step'].get('emulated', float('nan')):15.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    v = sub.add_parser('validate', help='check the emulator against a speck_activity_study.py chip run')
    v.add_argument('--checkpoint', required=True, help='the checkpoint the study ran')
    v.add_argument('--snn_dataset', required=True, help='the session directory the study ran')
    v.add_argument('--study_json', required=True)
    v.add_argument('--shuffle', nargs='+', default=['none', 'input', 'all'], choices=['none', 'input', 'all'])
    v.add_argument('--output', default=None)

    r = sub.add_parser('run', help='train and compare regimes')
    r.add_argument('--snn_dataset', required=True, help='session directory with train/ and test/')
    r.add_argument('--out_dir', required=True)
    r.add_argument('--regimes', nargs='+', default=['baseline', 'rate10', 'rate30', 'rate10_single'],
                   choices=list(REGIMES))
    r.add_argument('--n_train', type=int, default=80)
    r.add_argument('--n_val', type=int, default=15)
    r.add_argument('--n_test', type=int, default=30)
    r.add_argument('--epochs', type=int, default=10)
    r.add_argument('--init', default=None, help='fine-tune every regime from this checkpoint')
    r.add_argument('--retrain', action='store_true', help='retrain models already in --out_dir')
    r.add_argument('--emulator_shuffle', default='none', choices=['none', 'input', 'all'])
    r.add_argument('--speck_devkit', default=None, help='also score on the chip, e.g. speck2fdevkit:0')
    r.add_argument('--speck_wait_time', type=float, default=0.001)
    r.add_argument('--seed', type=int, default=0)

    s = sub.add_parser('summary', help='table and figures from every regime already in --out_dir')
    s.add_argument('--out_dir', required=True)

    a = parser.parse_args()
    if a.command == 'summary':
        res = collect(a.out_dir)
        print_table(res, print)
        from plots import plot_results
        for path in plot_results(res, a.out_dir):
            print(f"Saved {path}")
    else:
        {'validate': cmd_validate, 'run': cmd_run}[a.command](a)

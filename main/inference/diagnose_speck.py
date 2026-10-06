"""
Where does the Speck decoder lose accuracy relative to the PyTorch SNN?

For every session, the SNN test trials (same dropped first window as
test_all_decoders.py) are decoded by the same checkpoint in software:

    pytorch     the network as trained, one timestep per call; this is the
                'snn' decoder of test_all_decoders.py
    quantized   the network as deployed to the chip (8-bit weights, integer
                thresholds), still one timestep per call

and compared with the chip's own predictions ('speck') from that session's
test_all_decoders.py results. pytorch -> quantized is the cost of
quantization; quantized -> speck is what the chip itself adds.

When the results also hold the chip's output spike counts, a second table
compares them with the quantized network's on the same steps: total spikes,
exactly matching steps, per-feature rates, the agreement of the EMA-smoothed
spike trains the decoder reads, the delay at which the two agree
best (spikes read after --speck_wait_time land in later steps), and the RMSE
once that delay is removed. The chip's counts are also re-decoded on the
host, which must reproduce the 'speck' RMSE. Last, a linear readout is
re-fitted to each network's output spikes (5-fold blocked cross-validation
over the scored rows): 'refit spk' estimates what calibrating the readout on the
chip could recover, next to 'refit qnt' for the quantized network.

Every column is scored on the rows and targets of the session results
(<session>_arrays.npz), after checking that the 'snn' predictions stored
there are the ones this checkpoint produces; results made with a different
checkpoint are reported and left out of the snn/speck columns. Without the
.npz, versions are scored against the SNN dataset's velocity instead.

When the results ran 'speck' with another checkpoint than 'snn'
(run_inference.sbatch with SPECK_CHECKPOINT_ROOT), pass the Speck checkpoints
as --snn_checkpoint_root: they must be the ones the results recorded for
'speck', and the rows come from the recorded offset. The 'snn' column is then
the other checkpoint's result.

Last, a figure follows one test trial through both networks layer by layer
(input at the top, output at the bottom), one stacked raster per network on
a shared time axis, coloured by spike density (spikes per neuron per
timestep, averaged over --figure_bin steps). The Speck raster comes from the
chip with every layer monitored when --speck_devkit is given, otherwise from
the network as deployed (quantized), emulated on the host.

The same densities are also animated as a GIF: per network, the layers
side by side in the direction spikes travel, each box coloured by the
current bin's density, one frame per bin.

Writes {results_dir}/speck_diagnosis.json,
{results_dir}/speck_layer_activity_{session}.png and .gif.

Usage (no devkit needed):
    python diagnose_speck.py --experiment bmi --subject indy --data_root ../../data \
        --snn_checkpoint_root ../../data/snn_checkpoints/bmi/indy/full_cohort_finetuned_medium \
        --results_dir ../../data/results/test_all_decoders_finetuned/bmi/indy

    Add --speck_devkit speck2fdevkit:0 (devkit connected) for the chip's own
    layer activity in the figure.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from decoder_eval import (BASE_NPERSEG, STEP_S, _load_pickle, checkpoint_id, load_snn_model,  # noqa: E402
                          snn_test_files, unscale_velocity)
import speck  # noqa: E402

VERSIONS = ('pytorch', 'quantized')
COLUMNS = VERSIONS + ('snn', 'speck')
MATCH_TOL = 1e-3


def decode_versions(checkpoint_path, snn_dataset_path, experiment):
    """({version: (n, 2) velocity}, (n, 2) SNN dataset velocity, {version:
    (n, 2 * n_bins) output spike counts}, decode) over the test trials, first
    window dropped; decode(counts) -> velocity, as for the chip."""
    model, checkpoint, scale = load_snn_model(checkpoint_path, experiment)
    speck.check_deployable(model, checkpoint)
    snn_seq = speck.flatten_snn(model)
    snn_seq.eval()
    quant_seq = speck.discretized_sequential(speck.discretize(snn_seq, model.layers[0].in_features))
    runners = {'pytorch': lambda x: speck.run_float(snn_seq, x),
               'quantized': lambda x: speck.run_discretized(quant_seq, x)}

    files = snn_test_files(snn_dataset_path)
    preds, counts_kept, targets = {v: [] for v in runners}, {v: [] for v in runners}, []
    for i, path in enumerate(files):
        if i == 0 and len(files) > 1:
            continue                                # dropped, as in test_all_decoders.py
        trial = _load_pickle(path)
        keep = slice(BASE_NPERSEG, None) if len(files) == 1 else slice(None)
        targets.append(trial['velocity'][keep])
        for version, run in runners.items():
            counts = run(trial['input_spikes'])
            counts_kept[version].append(counts[keep])
            preds[version].append(unscale_velocity(speck.decode_spike_counts(model, counts), scale)[keep])
    decode = lambda counts: unscale_velocity(speck.decode_spike_counts(model, counts), scale)
    decode.smooth = lambda counts: smooth(counts, model.temporal_decay, model.temporal_decay_stages)
    return ({v: np.concatenate(p) for v, p in preds.items()}, np.concatenate(targets),
            {v: np.concatenate(c) for v, c in counts_kept.items()}, decode)


def find_offset(pred, reference, tol=MATCH_TOL):
    """Offset k with pred[k:k + len(reference)] == reference (within tol), or None."""
    n, probe = len(reference), min(20, len(reference))
    for k in range(len(pred) - n + 1):
        if np.abs(pred[k:k + probe] - reference[:probe]).max() < tol and \
                np.abs(pred[k:k + n] - reference).max() < tol:
            return k
    return None


def rmse(pred, target):
    return float(np.sqrt(((pred - target) ** 2).mean()))


def smooth(counts, decay, stages):
    """The model's EMA cascade applied to spike counts (what decode_output reads)."""
    out, state = np.zeros_like(counts, dtype=float), [np.zeros(counts.shape[1]) for _ in range(stages)]
    for t, x in enumerate(counts):
        for i in range(stages):
            state[i] = decay * state[i] + x
            x = state[i]
        out[t] = x
    return out


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def refit_rmse(counts, target, smooth, warmup, folds=5, ridge=1e-3):
    """RMSE of a linear readout re-fitted to these output spike counts, by
    blocked cross-validation on the scored rows (fit on all but one contiguous
    block, score that block, for each of `folds` blocks). The features are what decode_output reads,
    the EMA-smoothed counts, as each axis's share per bin plus its log total;
    so a readout fitted on the chip's spikes can absorb a consistent
    distortion of how the chip spreads spikes over the bins."""
    acc = smooth(counts)[warmup:]
    target = target[warmup:]
    n = acc.shape[1] // 2
    features = []
    for axis in (acc[:, :n], acc[:, n:2 * n]):
        total = axis.sum(axis=1, keepdims=True)
        features += [np.divide(axis, total, out=np.zeros_like(axis), where=total > 0), np.log1p(total)]
    features = np.hstack(features)
    blocks = np.array_split(np.arange(len(features)), folds)
    errors = []
    for score in blocks:
        fit = np.setdiff1d(np.arange(len(features)), score)
        mean, std = features[fit].mean(axis=0), features[fit].std(axis=0)
        std[std < 1e-6] = np.inf                # a bin never used in the fitting half is ignored
        design = lambda rows: np.hstack([(features[rows] - mean) / std, np.ones((len(features[rows]), 1))])
        a = design(fit)
        reg = ridge * len(a) * np.eye(a.shape[1])
        reg[-1, -1] = 0                         # the intercept is not penalised
        weights = np.linalg.solve(a.T @ a + reg, a.T @ target[fit])
        errors.append(design(score) @ weights - target[score])
    return float(np.sqrt((np.concatenate(errors) ** 2).mean()))


def compare_chip_output(chip, reference, decode, target, chip_pred, max_lag=5, warmup=200):
    """The chip's output spike counts against the quantized network's on the
    same steps: spike totals, how often a step matches exactly, per-feature
    rate agreement, the delay (in steps) at which the two correlate best,
    the RMSE of the chip's spikes decoded after removing that delay, and
    the RMSE of a readout re-fitted (cross-validated) to the chip's spikes and,
    for comparison, to the quantized network's.

    The chip's spikes are also re-decoded without any shift; that RMSE must
    equal the chip's own ('speck_same_rows'), or the host-side decode
    differs. Re-decoding starts the EMA from zero, so all three RMSEs skip
    the first `warmup` steps, by which the EMA has forgotten its start."""
    total_chip, total_ref = chip.sum(axis=1), reference.sum(axis=1)
    lags = range(0, max_lag + 1)
    lag_corr = {lag: _corr(total_chip[lag:], total_ref[:len(total_ref) - lag]) for lag in lags}
    best = max(lag_corr, key=lambda lag: -1 if np.isnan(lag_corr[lag]) else lag_corr[lag])
    shifted = np.vstack([chip[best:], np.zeros((best, chip.shape[1]))])
    return {
        'spike_ratio': float(chip.sum() / max(reference.sum(), 1)),
        'steps_identical': float((chip == reference).all(axis=1).mean()),
        'feature_rate_corr': _corr(chip.mean(axis=0), reference.mean(axis=0)),
        'smoothed_corr': _corr(decode.smooth(chip).ravel(), decode.smooth(reference).ravel()),
        'step_corr_lag0': lag_corr[0],
        'best_lag': int(best), 'step_corr_best_lag': lag_corr[best],
        'step_corr_by_lag': {int(lag): c for lag, c in lag_corr.items()},
        'rmse_speck_same_rows': rmse(chip_pred[warmup:], target[warmup:]),
        'rmse_redecoded': rmse(decode(chip)[warmup:], target[warmup:]),
        'rmse_lag_corrected': rmse(decode(shifted)[warmup:], target[warmup:]),
        'rmse_refit_quantized': refit_rmse(reference, target, decode.smooth, warmup),
        'rmse_refit_speck': refit_rmse(chip, target, decode.smooth, warmup),
    }


def diagnose_session(session, checkpoint, snn_dataset_path, results_dir, experiment, max_lag=5):
    preds, target, counts, decode = decode_versions(checkpoint, snn_dataset_path, experiment)
    arrays_path = os.path.join(results_dir, 'sessions', f'{session}_arrays.npz')
    results_path = os.path.join(results_dir, 'sessions', f'{session}.json')
    full = {}
    if os.path.isfile(results_path):
        with open(results_path, 'r') as f:
            full = json.load(f).get('full') or {}
    # Results recorded with checkpoints say which one each SNN decoder ran.
    # When 'speck' ran another checkpoint than 'snn', it must be this one, and
    # the rows come from the recorded offset; otherwise this checkpoint must
    # reproduce the saved 'snn' predictions.
    recorded = full.get('checkpoints') or {}
    decoupled = 'speck' in recorded and checkpoint_id(recorded['speck']) != checkpoint_id(recorded.get('snn'))
    row, note = {}, None
    if os.path.isfile(arrays_path):
        with np.load(arrays_path) as f:
            saved = {k: f[k] for k in f.files}
        n = len(saved['y_true'])
        if decoupled:
            offset = full.get('snn_row_offset')
            if checkpoint_id(checkpoint) != checkpoint_id(recorded['speck']):
                offset, note = None, (f"speck ran {checkpoint_id(recorded['speck'])}; "
                                      "snn/speck left out, scored on the SNN dataset")
            elif offset is None or offset + n > len(preds['pytorch']):
                offset, note = None, 'no usable snn_row_offset in the results; snn/speck left out'
        else:
            offset = find_offset(preds['pytorch'], saved['pred_snn']) if 'pred_snn' in saved else None
            if offset is None:
                note = 'results made with a different checkpoint; snn/speck left out, scored on the SNN dataset'
        if offset is not None:
            target = saved['y_true']
            preds = {v: p[offset:offset + n] for v, p in preds.items()}
            counts = {v: c[offset:offset + n] for v, c in counts.items()}
            for name in ('snn', 'speck'):
                if f'pred_{name}' in saved:
                    row[name] = {'rmse': rmse(saved[f'pred_{name}'], target)}
            if decoupled and 'snn' in row:
                note = f"snn ran {checkpoint_id(recorded.get('snn'))}"
            if 'speck_counts' in saved:
                row['chip_vs_quantized'] = compare_chip_output(
                    saved['speck_counts'].astype(float), counts['quantized'], decode, target,
                    saved['pred_speck'], max_lag=max_lag)
    else:
        note = f'no {os.path.basename(arrays_path)}; scored on the SNN dataset, snn/speck left out'
    for version, p in preds.items():
        row[version] = {'rmse': rmse(p, target), 'output_spikes_per_step': float(counts[version].sum() / len(p))}
    if 'speck' in row:
        chip = full.get('metrics', {}).get('speck', {}).get('chip') or {}
        if 'output_spikes_per_step' in chip:
            row['speck']['output_spikes_per_step'] = chip['output_spikes_per_step']
    return row, note


def layer_activity(checkpoint_path, snn_dataset_path, experiment, trial, start, steps,
                   devkit=None, wait_time=0.001, raster_dt=0.1):
    """Spikes of every layer over one test trial, for the PyTorch network and
    for Speck: (input (steps, C), {network label: [(steps, n) per layer]},
    trial index used). Both networks start the trial from rest and run from
    its first step; the window [start, start + steps) is returned. Speck is
    the chip with every layer monitored when devkit is given, otherwise the
    quantized network as deployed, emulated on the host."""
    model, checkpoint, _ = load_snn_model(checkpoint_path, experiment)
    speck.check_deployable(model, checkpoint)
    snn_seq = speck.flatten_snn(model)
    snn_seq.eval()
    files = snn_test_files(snn_dataset_path)
    trial = min(trial, len(files) - 1)
    spikes = _load_pickle(files[trial])['input_spikes'][:, :start + steps]
    layers = {'PyTorch SNN': speck.run_layers(snn_seq, spikes)}
    if devkit:
        device = speck.open_speck(model, checkpoint, snn_dataset_path, devkit, wait_time, raster_dt,
                                  monitor_all=True)
        try:
            device.reset()
            device.run(spikes, 2 * model.n_bins)
            layers[f'Speck SNN (chip, {devkit})'] = device.layer_counts
        finally:
            device.close()
    else:
        quant_seq = speck.discretized_sequential(speck.discretize(snn_seq, model.layers[0].in_features))
        layers['Speck SNN (quantized as deployed, emulated on host)'] = speck.run_layers(quant_seq, spikes)
    window = slice(start, start + steps)
    return spikes.T[window], {k: [l[window] for l in v] for k, v in layers.items()}, trial


LAYER_CMAP = 'Blues'


def layer_densities(inputs, layers, bin_steps):
    """({network: (n_layers + 1, n_bins) spike density}, row labels, neuron
    counts, colour norm): each layer's spikes per neuron per timestep,
    averaged over bin_steps, input first; one norm for every network."""
    from matplotlib.colors import PowerNorm

    def density(x):
        n = len(x) // bin_steps * bin_steps
        return x[:n].reshape(-1, bin_steps, x.shape[1]).mean(axis=(1, 2))

    rasters = {name: np.vstack([density(inputs)] + [density(l) for l in layer_list])
               for name, layer_list in layers.items()}
    first = next(iter(layers.values()))
    sizes = [inputs.shape[1]] + [l.shape[1] for l in first]
    labels = ['input'] + [f'hidden {i + 1}' for i in range(len(first) - 1)] + ['output']
    vmax = max(r.max() for r in rasters.values()) or 1.0
    return rasters, labels, sizes, PowerNorm(0.5, vmin=0, vmax=vmax)


def layer_activity_figure(inputs, layers, bin_steps=5, title=''):
    """One stacked raster per network (rows: input, hidden layers, output),
    on a shared time axis; colour = spikes per neuron per timestep, averaged
    over bin_steps, on one colour scale for both networks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rasters, names, sizes, norm = layer_densities(inputs, layers, bin_steps)
    labels = [f'{name} ({size})' for name, size in zip(names, sizes)]
    n_rows = len(labels)
    duration = rasters[next(iter(rasters))].shape[1] * bin_steps * STEP_S
    fig, axes = plt.subplots(len(rasters), 1, figsize=(12, 1.2 + 0.55 * n_rows * len(rasters)),
                             sharex=True, squeeze=False)
    for ax, (name, raster) in zip(axes[:, 0], rasters.items()):
        im = ax.imshow(raster, aspect='auto', interpolation='nearest', cmap=LAYER_CMAP,
                       norm=norm, extent=(0, duration, n_rows - 0.5, -0.5))
        ax.set_yticks(np.arange(n_rows), labels, fontsize=8)
        ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
        ax.grid(which='minor', axis='y', color='white', linewidth=2)
        ax.tick_params(which='minor', length=0)
        ax.set_title(name, fontsize=10, loc='left')
        for row, layer_density in enumerate(raster.mean(axis=1)):
            ax.annotate(f'{layer_density:.3f}', (1.005, row), xycoords=('axes fraction', 'data'),
                        va='center', fontsize=7, color='0.3')
        ax.annotate('mean', (1.005, -0.5), xycoords=('axes fraction', 'data'), va='bottom',
                    fontsize=7, color='0.3')
    axes[-1, 0].set_xlabel(f'Time in trial (s; {bin_steps * STEP_S * 1000:g} ms bins)')
    bar = fig.colorbar(im, ax=axes[:, 0], fraction=0.025, pad=0.07)
    bar.set_label('Spike density (spikes / neuron / timestep)')
    fig.suptitle(title or 'Spike density by layer over time', fontsize=11)
    return fig


def save_layer_activity_gif(inputs, layers, path, bin_steps=5, title='', fps=10):
    """The data of layer_activity_figure() as an animation: per network, a
    row of layers left to right in the direction spikes travel (input,
    hidden layers, output), each a box whose height follows its neuron
    count, coloured by its spike density in the current bin (same colours
    and scale as the figure). One frame per bin, played at fps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import FancyArrowPatch, Rectangle

    rasters, names, sizes, norm = layer_densities(inputs, layers, bin_steps)
    cmap = plt.get_cmap(LAYER_CMAP)
    n_layers, n_frames = len(names), next(iter(rasters.values())).shape[1]
    heights = 0.25 + 0.75 * np.asarray(sizes) / max(sizes)     # box height ~ neuron count
    box_w, gap = 0.5, 0.6
    xs = np.arange(n_layers) * (box_w + gap)

    fig, axes = plt.subplots(len(rasters), 1, figsize=(1.6 + 1.5 * n_layers, 0.9 + 2.1 * len(rasters)),
                             squeeze=False)
    boxes, values = {}, {}
    for ax, name in zip(axes[:, 0], rasters):
        boxes[name], values[name] = [], []
        for x, h, label, size in zip(xs, heights, names, sizes):
            box = Rectangle((x, -h / 2), box_w, h, facecolor=cmap(0.0), edgecolor='black', linewidth=0.8)
            ax.add_patch(box)
            boxes[name].append(box)
            ax.text(x + box_w / 2, -0.55, f'{label}\n({size})', ha='center', va='top', fontsize=8)
            values[name].append(ax.text(x + box_w / 2, h / 2 + 0.04, '', ha='center', va='bottom', fontsize=7,
                                        color='0.3'))
        for x0, x1 in zip(xs[:-1], xs[1:]):
            ax.add_patch(FancyArrowPatch((x0 + box_w + 0.06, 0), (x1 - 0.06, 0), arrowstyle='-|>',
                                         mutation_scale=12, color='0.4', linewidth=1.2))
        ax.set_xlim(xs[0] - 0.2, xs[-1] + box_w + 0.2)
        ax.set_ylim(-0.8, 0.65)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(name, fontsize=9, loc='left')
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, 0], fraction=0.03, pad=0.03)
    bar.set_label('Spike density (spikes / neuron / timestep)', fontsize=8)
    clock = fig.suptitle('', fontsize=10)

    def draw(frame):
        t0 = frame * bin_steps * STEP_S
        clock.set_text(f"{title or 'Spike density by layer'}\n"
                       f"t = {t0:5.2f}-{t0 + bin_steps * STEP_S:.2f} s")
        for name, raster in rasters.items():
            for box, text, value in zip(boxes[name], values[name], raster[:, frame]):
                box.set_facecolor(cmap(norm(value)))
                text.set_text(f'{value:.3f}')
        return []

    FuncAnimation(fig, draw, frames=n_frames).save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def main(args):
    root = args.snn_checkpoint_root or os.path.join(
        args.data_root, 'snn_checkpoints', args.experiment, args.subject, 'per_session')
    dataset_root = os.path.join(args.data_root, 'snn_datasets', args.experiment, args.subject,
                                'mua' if args.experiment == 'hkm' else 'mua_8_group')
    results_dir = args.results_dir or os.path.join(
        args.data_root, 'results', 'test_all_decoders', args.experiment, args.subject)

    sessions = args.sessions or sorted(os.listdir(root))
    report = {'snn_checkpoint_root': root, 'results_dir': results_dir, 'sessions': {}}
    print(f"Checkpoints: {root}\nResults:     {results_dir}\n")
    print(f"RMSE\n{'session':<20s}" + ''.join(f"{c:>11s}" for c in COLUMNS))
    for session in sessions:
        checkpoint = os.path.join(root, session, args.snn_checkpoint_subdir, 'best_model_weights.pth')
        if not os.path.isfile(checkpoint):
            continue
        row, note = diagnose_session(session, checkpoint, os.path.join(dataset_root, session),
                                     results_dir, args.experiment, args.max_lag)
        report['sessions'][session] = dict(row, note=note) if note else row
        print(f"{session:<20s}" + ''.join(f"{row[c]['rmse']:11.2f}" if c in row else f"{'-':>11s}"
                                          for c in COLUMNS) + (f"   ({note})" if note else ''))

    rows = list(report['sessions'].values())
    report['mean'] = {c: {m: float(np.mean([r[c][m] for r in rows if m in r.get(c, {})]))
                          for m in ('rmse', 'output_spikes_per_step') if any(m in r.get(c, {}) for r in rows)}
                      for c in COLUMNS}
    fmt = lambda c, m: f"{report['mean'][c][m]:11.2f}" if m in report['mean'][c] else f"{'-':>11s}"
    print(f"{'mean':<20s}" + ''.join(fmt(c, 'rmse') for c in COLUMNS))
    print(f"{'out spikes/step':<20s}" + ''.join(fmt(c, 'output_spikes_per_step') for c in COLUMNS))

    chip_rows = {s: r['chip_vs_quantized'] for s, r in report['sessions'].items() if 'chip_vs_quantized' in r}
    if chip_rows:
        cols = [('spike_ratio', 'spk ratio'), ('steps_identical', 'same steps'),
                ('feature_rate_corr', 'feat corr'), ('smoothed_corr', 'EMA corr'),
                ('step_corr_lag0', 'corr lag0'),
                ('best_lag', 'best lag'), ('step_corr_best_lag', 'corr best'),
                ('rmse_speck_same_rows', 'speck'), ('rmse_redecoded', 'redecoded'),
                ('rmse_lag_corrected', 'lag-fixed'), ('rmse_refit_quantized', 'refit qnt'),
                ('rmse_refit_speck', 'refit spk')]
        print(f"\nChip output vs. quantized network, same steps; RMSEs after a 200-step warm-up "
              f"(redecoded must equal speck; refit: readout re-fitted to that network's spikes, "
              f"5-fold cross-validated)\n"
              f"{'session':<20s}" + ''.join(f"{label:>11s}" for _, label in cols))
        for session, c in chip_rows.items():
            print(f"{session:<20s}" + ''.join(f"{c[k]:11d}" if k == 'best_lag' else f"{c[k]:11.2f}"
                                              for k, _ in cols))
    elif rows:
        print("\nNo chip spike counts saved (runs before this version); re-run test_all_decoders.py "
              "with 'speck' to compare the chip's output spikes with the quantized network.")
    path = args.output or os.path.join(results_dir, 'speck_diagnosis.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved {path}")

    figure_session = args.figure_session or next(iter(report['sessions']), None)
    if args.figure_steps > 0 and figure_session:
        import matplotlib.pyplot as plt
        checkpoint = os.path.join(root, figure_session, args.snn_checkpoint_subdir, 'best_model_weights.pth')
        inputs, layers, trial = layer_activity(
            checkpoint, os.path.join(dataset_root, figure_session), args.experiment, args.figure_trial,
            args.figure_start, args.figure_steps, args.speck_devkit, args.speck_wait_time,
            args.speck_raster_dt)
        fig = layer_activity_figure(inputs, layers, args.figure_bin,
                                    f'Spike density by layer -- {figure_session}, test trial {trial}')
        path = os.path.join(results_dir, f'speck_layer_activity_{figure_session}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {path}")
        path = os.path.join(results_dir, f'speck_layer_activity_{figure_session}.gif')
        save_layer_activity_gif(inputs, layers, path, args.figure_bin,
                                f'Spike density by layer -- {figure_session}, test trial {trial}',
                                fps=args.figure_fps)
        print(f"Saved {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', required=True, choices=['bmi', 'hkm'])
    parser.add_argument('--subject', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--snn_checkpoint_root', default=None,
                        help='Default: {data_root}/snn_checkpoints/{experiment}/{subject}/per_session')
    parser.add_argument('--snn_checkpoint_subdir', default='')
    parser.add_argument('--results_dir', default=None,
                        help='Session results made with these checkpoints (default: results/test_all_decoders/...)')
    parser.add_argument('--sessions', nargs='*', default=None)
    parser.add_argument('--max_lag', type=int, default=50,
                        help='Largest chip output delay (in steps) searched for the best-lag correlation')
    parser.add_argument('--output', default=None)
    figure = parser.add_argument_group('layer activity figure')
    figure.add_argument('--figure_session', default=None, help='Default: the first session diagnosed')
    figure.add_argument('--figure_trial', type=int, default=1,
                        help='Test trial index (default 1, the first one scored)')
    figure.add_argument('--figure_start', type=int, default=0, help='First timestep shown')
    figure.add_argument('--figure_steps', type=int, default=750,
                        help='Timesteps shown (4 ms each); 0 skips the figure')
    figure.add_argument('--figure_bin', type=int, default=5,
                        help='Timesteps averaged per colour cell (and per GIF frame)')
    figure.add_argument('--figure_fps', type=int, default=10,
                        help='GIF frames per second (default 10: 200 ms of trial per second at --figure_bin 5)')
    figure.add_argument('--speck_devkit', default=None,
                        help="Record the Speck raster on this devkit (e.g. speck2fdevkit:0), every "
                             "layer monitored; default: emulate the quantized network on the host")
    figure.add_argument('--speck_wait_time', type=float, default=0.001)
    figure.add_argument('--speck_raster_dt', type=float, default=0.1)
    main(parser.parse_args())

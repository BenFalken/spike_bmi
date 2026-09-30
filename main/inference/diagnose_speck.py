"""
Where does the Speck decoder lose accuracy relative to the PyTorch SNN?

For every session, the SNN test trials (same dropped first window as
test_all_decoders.py) are decoded by the same checkpoint in software:

    pytorch     the network as trained, one timestep per call; this is the
                'snn' decoder of test_all_decoders.py
    quantized   the network as deployed to the chip (8-bit weights, integer
                thresholds), still one timestep per call
    specksim    the quantized network in samna's event-driven Speck simulator,
                where each input event updates the membranes on its own

and compared with the chip's own predictions ('speck') from that session's
test_all_decoders.py results. pytorch -> quantized is the cost of
quantization; quantized -> speck is what the chip itself adds.

When the results also hold the chip's output spike counts, a second table
compares them with the quantized network's on the same steps: total spikes,
exactly matching steps, per-feature rates, the delay at which the two agree
best (spikes read after --speck_wait_time land in later steps), and the RMSE
once that delay is removed. The chip's counts are also re-decoded on the
host, which must reproduce the 'speck' RMSE.

Every column is scored on the rows and targets of the session results
(<session>_arrays.npz), after checking that the 'snn' predictions stored
there are the ones this checkpoint produces; results made with a different
checkpoint are reported and left out of the snn/speck columns. Without the
.npz, versions are scored against the SNN dataset's velocity instead.

Writes {results_dir}/speck_diagnosis.json.

Usage (no devkit needed; specksim needs samna):
    python diagnose_speck.py --experiment bmi --subject indy --data_root ../../data \
        --snn_checkpoint_root ../../data/snn_checkpoints/bmi/indy/full_cohort_finetuned_medium \
        --results_dir ../../data/results/test_all_decoders_finetuned/bmi/indy
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from decoder_eval import (BASE_NPERSEG, _load_pickle, load_snn_model, snn_test_files,  # noqa: E402
                          unscale_velocity)
import speck  # noqa: E402

VERSIONS = ('pytorch', 'quantized', 'specksim')
COLUMNS = VERSIONS + ('snn', 'speck')
MATCH_TOL = 1e-3


def decode_versions(checkpoint_path, snn_dataset_path, experiment, use_specksim=True):
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
    if use_specksim:
        runners['specksim'] = lambda x: speck.run_specksim(quant_seq, x, 2 * model.n_bins)

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


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def compare_chip_output(chip, reference, decode, target, chip_pred, max_lag=5, warmup=200):
    """The chip's output spike counts against the quantized network's on the
    same steps: spike totals, how often a step matches exactly, per-feature
    rate agreement, the delay (in steps) at which the two correlate best,
    and the RMSE of the chip's spikes decoded after removing that delay.

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
        'step_corr_lag0': lag_corr[0],
        'best_lag': int(best), 'step_corr_best_lag': lag_corr[best],
        'rmse_speck_same_rows': rmse(chip_pred[warmup:], target[warmup:]),
        'rmse_redecoded': rmse(decode(chip)[warmup:], target[warmup:]),
        'rmse_lag_corrected': rmse(decode(shifted)[warmup:], target[warmup:]),
    }


def diagnose_session(session, checkpoint, snn_dataset_path, results_dir, experiment, use_specksim):
    preds, target, counts, decode = decode_versions(checkpoint, snn_dataset_path, experiment, use_specksim)
    arrays_path = os.path.join(results_dir, 'sessions', f'{session}_arrays.npz')
    row, note = {}, None
    if os.path.isfile(arrays_path):
        with np.load(arrays_path) as f:
            saved = {k: f[k] for k in f.files}
        offset = find_offset(preds['pytorch'], saved['pred_snn']) if 'pred_snn' in saved else None
        if offset is None:
            note = 'results made with a different checkpoint; snn/speck left out, scored on the SNN dataset'
        else:
            n = len(saved['y_true'])
            target = saved['y_true']
            preds = {v: p[offset:offset + n] for v, p in preds.items()}
            counts = {v: c[offset:offset + n] for v, c in counts.items()}
            for name in ('snn', 'speck'):
                if f'pred_{name}' in saved:
                    row[name] = {'rmse': rmse(saved[f'pred_{name}'], target)}
            if 'speck_counts' in saved:
                row['chip_vs_quantized'] = compare_chip_output(
                    saved['speck_counts'].astype(float), counts['quantized'], decode, target,
                    saved['pred_speck'])
    else:
        note = f'no {os.path.basename(arrays_path)}; scored on the SNN dataset, snn/speck left out'
    for version, p in preds.items():
        row[version] = {'rmse': rmse(p, target), 'output_spikes_per_step': float(counts[version].sum() / len(p))}
    if 'speck' in row:
        results_path = os.path.join(results_dir, 'sessions', f'{session}.json')
        with open(results_path, 'r') as f:
            chip = json.load(f)['full']['metrics'].get('speck', {}).get('chip') or {}
        if 'output_spikes_per_step' in chip:
            row['speck']['output_spikes_per_step'] = chip['output_spikes_per_step']
    return row, note


def main(args):
    root = args.snn_checkpoint_root or os.path.join(
        args.data_root, 'snn_checkpoints', args.experiment, args.subject, 'per_session')
    dataset_root = os.path.join(args.data_root, 'snn_datasets', args.experiment, args.subject,
                                'mua' if args.experiment == 'hkm' else 'mua_8_group')
    results_dir = args.results_dir or os.path.join(
        args.data_root, 'results', 'test_all_decoders', args.experiment, args.subject)
    try:
        import samna  # noqa: F401
        use_specksim = not args.skip_specksim
    except ImportError:
        print("samna is not installed: skipping specksim")
        use_specksim = False

    sessions = args.sessions or sorted(os.listdir(root))
    report = {'snn_checkpoint_root': root, 'results_dir': results_dir, 'sessions': {}}
    print(f"Checkpoints: {root}\nResults:     {results_dir}\n")
    print(f"RMSE\n{'session':<20s}" + ''.join(f"{c:>11s}" for c in COLUMNS))
    for session in sessions:
        checkpoint = os.path.join(root, session, args.snn_checkpoint_subdir, 'best_model_weights.pth')
        if not os.path.isfile(checkpoint):
            continue
        row, note = diagnose_session(session, checkpoint, os.path.join(dataset_root, session),
                                     results_dir, args.experiment, use_specksim)
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
                ('feature_rate_corr', 'feat corr'), ('step_corr_lag0', 'corr lag0'),
                ('best_lag', 'best lag'), ('step_corr_best_lag', 'corr best'),
                ('rmse_speck_same_rows', 'speck'), ('rmse_redecoded', 'redecoded'),
                ('rmse_lag_corrected', 'lag-fixed')]
        print(f"\nChip output vs. quantized network, same steps; RMSEs after a 200-step warm-up "
              f"(redecoded must equal speck)\n"
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
    parser.add_argument('--skip_specksim', action='store_true')
    parser.add_argument('--output', default=None)
    main(parser.parse_args())

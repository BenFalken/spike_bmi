"""
Where does the Speck decoder lose accuracy relative to the PyTorch SNN?

For every session, the SNN test trials (same dropped first window as
test_all_decoders.py) are decoded by four versions of the same network:

    float       the flattened PyTorch network, one timestep per call (= 'snn')
    discretized the network as quantized for the chip (8-bit weights,
                integer thresholds), still one timestep per call
    specksim    the quantized network in samna's event-driven Speck simulator:
                each input event updates the membranes on its own, as on chip
    speck       the chip itself, from the session's test_all_decoders.py results

float -> discretized is the cost of quantization; discretized -> specksim
is the cost of event-driven processing (within a timestep, a neuron can
spike and reset before later, possibly inhibitory, events arrive);
specksim -> speck is what the physical chip adds (timing, wait time,
power-on state). RMSE here is against the SNN dataset's own velocity,
so float/discretized/specksim differ slightly from the aligned values in
the session files; the 'speck' and 'snn' columns are copied from those.

Writes {results_dir}/speck_diagnosis.json.

Usage (no devkit needed; specksim needs samna):
    python diagnose_speck.py --experiment bmi --subject indy --data_root ../../data \
        [--snn_checkpoint_root .../snn_checkpoints/bmi/indy/full_cohort_finetuned_medium]
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

VERSIONS = ('float', 'discretized', 'specksim')


def diagnose_session(checkpoint_path, snn_dataset_path, experiment, use_specksim=True):
    model, checkpoint, scale = load_snn_model(checkpoint_path, experiment)
    speck.check_deployable(model, checkpoint)
    snn_seq = speck.flatten_snn(model)
    snn_seq.eval()
    disc_seq = speck.discretized_sequential(speck.discretize(snn_seq, model.layers[0].in_features))
    n_outputs = 2 * model.n_bins
    runners = {'float': lambda x: speck.run_float(snn_seq, x),
               'discretized': lambda x: speck.run_discretized(disc_seq, x)}
    if use_specksim:
        runners['specksim'] = lambda x: speck.run_specksim(disc_seq, x, n_outputs)

    files = snn_test_files(snn_dataset_path)
    preds, spikes, targets = {v: [] for v in runners}, {v: 0.0 for v in runners}, []
    for i, path in enumerate(files):
        if i == 0 and len(files) > 1:
            continue                                # dropped, as in test_all_decoders.py
        trial = _load_pickle(path)
        keep = slice(BASE_NPERSEG, None) if len(files) == 1 else slice(None)
        targets.append(trial['velocity'][keep])
        for version, run in runners.items():
            counts = run(trial['input_spikes'])
            spikes[version] += counts[keep].sum()
            preds[version].append(unscale_velocity(speck.decode_spike_counts(model, counts), scale)[keep])
    y = np.concatenate(targets)
    out = {}
    for version in runners:
        p = np.concatenate(preds[version])
        out[version] = {'rmse': float(np.sqrt(((p - y) ** 2).mean())),
                        'output_spikes_per_step': float(spikes[version] / len(y))}
    return out


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
    report = {'snn_checkpoint_root': root, 'sessions': {}}
    header = f"{'session':<20s}" + ''.join(f"{v:>13s}" for v in VERSIONS + ('snn*', 'speck*'))
    print(f"RMSE per version (* = from the session results)\n{header}")
    for session in sessions:
        checkpoint = os.path.join(root, session, args.snn_checkpoint_subdir, 'best_model_weights.pth')
        if not os.path.isfile(checkpoint):
            continue
        row = diagnose_session(checkpoint, os.path.join(dataset_root, session), args.experiment, use_specksim)
        results_path = os.path.join(results_dir, 'sessions', f'{session}.json')
        if os.path.isfile(results_path):
            with open(results_path, 'r') as f:
                metrics = json.load(f).get('full', {}).get('metrics', {})
            for name in ('snn', 'speck'):
                if name in metrics:
                    row[name + '_results'] = {'rmse': metrics[name]['rmse']}
            chip = metrics.get('speck', {}).get('chip') or {}
            if 'output_spikes_per_step' in chip:
                row['speck_results']['output_spikes_per_step'] = chip['output_spikes_per_step']
        report['sessions'][session] = row
        cells = [row.get(k, {}).get('rmse') for k in VERSIONS + ('snn_results', 'speck_results')]
        print(f"{session:<20s}" + ''.join(f"{c:13.2f}" if c is not None else f"{'-':>13s}" for c in cells))

    keys = VERSIONS + ('snn_results', 'speck_results')
    rows = report['sessions'].values()
    report['mean'] = {k: {m: float(np.mean([r[k][m] for r in rows if m in r.get(k, {})]))
                          for m in ('rmse', 'output_spikes_per_step')
                          if any(m in r.get(k, {}) for r in rows)} for k in keys}
    print(f"{'mean':<20s}" + ''.join(f"{report['mean'][k].get('rmse', np.nan):13.2f}" for k in keys))
    print(f"{'out spikes/step':<20s}" + ''.join(
        f"{report['mean'][k].get('output_spikes_per_step', np.nan):13.2f}" for k in keys))
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
                        help='Where the session results are (default: results/test_all_decoders/...)')
    parser.add_argument('--sessions', nargs='*', default=None)
    parser.add_argument('--skip_specksim', action='store_true')
    parser.add_argument('--output', default=None)
    main(parser.parse_args())

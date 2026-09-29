"""
Combine one subject's per-session results (test_all_decoders.py output) and
draw the report figures.

Reads   {results_dir}/sessions/*.json
        {results_dir}/aggregated_summary.json   (optional; the Speck laptop
                                                 run, aggregate_speck_results.py)
Writes  {results_dir}/combined_metrics.json            {session: {decoder: metrics}}
        {results_dir}/combined_metrics_durations.json  {session: {"Nmin": {decoder: metrics}}}
        {results_dir}/efficiency_summary.json          per-decoder means across sessions
        {results_dir}/decoder_efficiency.png, decoder_energy.png, decoder_comparison_4x2.png

The Speck results ('speck' and 'snn_pytorch') are added to the figures only;
the combined JSON files hold the cluster evaluation alone.

Usage:
    python make_report.py --results_dir $BMI_DATA_ROOT/results/test_all_decoders/bmi/indy
"""

import argparse
import copy
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from report_figures import (DECODER_ORDER, LAPTOP_DECODERS, comparison_4x2_figure, energy_figure,
                            efficiency_figure, mean_ci)


def load_sessions(sessions_dir):
    sessions = {}
    for path in sorted(glob.glob(os.path.join(sessions_dir, '*.json'))):
        with open(path, 'r') as f:
            content = json.load(f)
        if 'full' not in content:
            print(f"[skip] {os.path.basename(path)}: no full-data results yet")
            continue
        sessions[content['session']] = content
    if not sessions:
        raise FileNotFoundError(f"No completed session results in {sessions_dir}")
    subjects = {s.split('_')[0] for s in sessions}
    if len(subjects) > 1:
        print(f"WARNING: sessions from several subjects are pooled: {sorted(subjects)}")
    return sessions


def load_speck_summary(path):
    """{name: {session: {...}}, ...} for 'speck' (on chip) and 'snn_pytorch'
    (the laptop's PyTorch run) from an aggregate_speck_results.py summary."""
    if not path or not os.path.isfile(path):
        print(f"[speck] no Speck summary at {path}; figures show the cluster decoders only")
        return {}
    with open(path, 'r') as f:
        summary = json.load(f)
    merged = {}
    for impl_key, name in (('torch', 'snn_pytorch'), ('speck', 'speck')):
        impl = summary.get('impls', {}).get(impl_key)
        if impl is None:
            continue
        n = len(impl['sessions'])
        power = impl.get('power_mw', {}).get('values')
        latency_ms = impl['latency_per_timestep_ms']['values']
        energy = ([(p / 1000) * (ms / 1000) for p, ms in zip(power, latency_ms)]
                  if power is not None and len(power) == len(latency_ms) else [None] * n)
        merged[name] = {
            'accuracy': {s: {'rmse': r, 'cc_x': cx, 'cc_y': cy} for s, r, cx, cy in zip(
                impl['sessions'], impl['rmse_vs_gt']['values'], impl['cc_x_vs_gt']['values'],
                impl['cc_y_vs_gt']['values'])},
            'latency_s': [ms / 1000 for ms in latency_ms],
            'energy_j': energy,
            'param_count': summary.get('dynapcnn_param_count'),
            'energy_method': impl.get('energy_method'),
        }
        print(f"[speck] merged {n} session(s) of '{impl_key}' as {name}")
    return merged


def efficiency_records(sessions, speck):
    """One record per decoder: means (and CIs) across sessions of RMSE,
    latency, parameter count and energy."""
    samples = {}
    for content in sessions.values():
        for name, m in content['full']['metrics'].items():
            samples.setdefault(name, {'rmse': [], 'latency_s': [], 'param_count': [], 'energy_j': [],
                                      'methods': {}})['rmse'].append(m['rmse'])
        profile = content.get('profile') or {}
        for name, p in profile.get('decoders', {}).items():
            entry = samples.setdefault(name, {'rmse': [], 'latency_s': [], 'param_count': [],
                                              'energy_j': [], 'methods': {}})
            entry['latency_s'].append(p['latency_s'])
            entry['param_count'].append(p['param_count'])
            entry['energy_j'].append(p['energy_j'])
            method = profile.get('energy_method')
            if method and p['energy_j'] is not None:
                entry['methods'][method] = entry['methods'].get(method, 0) + 1
    for name, s in speck.items():
        if s['param_count'] is None:
            print(f"[speck] {name}: summary has no dynapcnn_param_count; left out of efficiency figures")
            continue
        samples[name] = {'rmse': [a['rmse'] for a in s['accuracy'].values()], 'latency_s': s['latency_s'],
                         'param_count': [s['param_count']] * len(s['latency_s']), 'energy_j': s['energy_j'],
                         'methods': {s['energy_method']: len(s['latency_s'])} if s['energy_method'] else {}}

    records = []
    for name in (d for d in DECODER_ORDER if d in samples):
        s = samples[name]
        if not s['rmse'] or not s['latency_s']:
            print(f"[efficiency] {name}: missing {'RMSE' if not s['rmse'] else 'profiling'} data, left out")
            continue
        if len(set(s['param_count'])) > 1:
            print(f"WARNING: {name.upper()} parameter count differs across sessions: {sorted(set(s['param_count']))}")
        if len(s['methods']) > 1:
            print(f"WARNING: {name.upper()} energy mixes measurement methods across sessions: {s['methods']}")
        rmse, rmse_lo, rmse_hi = mean_ci(s['rmse'])
        latency, latency_lo, latency_hi = mean_ci(s['latency_s'])
        energies = [e for e in s['energy_j'] if e is not None]
        energy = mean_ci(energies) if energies else (None, None, None)
        records.append({
            'name': name, 'rmse': rmse, 'rmse_lo': rmse_lo, 'rmse_hi': rmse_hi,
            'latency_s': latency, 'latency_lo': latency_lo, 'latency_hi': latency_hi,
            'param_count': float(np.mean(s['param_count'])),
            'energy_j': energy[0], 'energy_lo': energy[1], 'energy_hi': energy[2],
            'energy_method': max(s['methods'], key=s['methods'].get) if energies and s['methods'] else None,
            'n_sessions': len(s['latency_s']), 'n_rmse_sessions': len(s['rmse']),
            'n_energy_sessions': len(energies),
            'latency_cohort': 'laptop' if name in LAPTOP_DECODERS else 'cluster',
        })
    return records


def _write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


def _save_figure(fig, path):
    fig.savefig(path, dpi=200 if '4x2' in path else 150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def main(args):
    results_dir = args.results_dir
    sessions = load_sessions(os.path.join(results_dir, 'sessions'))
    print(f"Loaded {len(sessions)} session(s) from {results_dir}/sessions")

    combined = {s: c['full']['metrics'] for s, c in sessions.items()}
    durations = {s: {tag: r['metrics'] for tag, r in c.get('durations', {}).items() if r}
                 for s, c in sessions.items()}
    durations = {s: d for s, d in durations.items() if d}
    _write_json(combined, os.path.join(results_dir, 'combined_metrics.json'))
    _write_json(durations, os.path.join(results_dir, 'combined_metrics_durations.json'))

    speck = load_speck_summary(args.speck_summary_path or os.path.join(results_dir, 'aggregated_summary.json'))
    records = efficiency_records(sessions, speck)
    _write_json({'decoders': {r['name']: {k: v for k, v in r.items() if k != 'name'} for r in records}},
                os.path.join(results_dir, 'efficiency_summary.json'))
    for r in records:
        energy = (f"{r['energy_j'] * 1e6:.3f} uJ ({r['energy_method']})" if r['energy_j'] is not None
                  else 'n/a')
        print(f"  {r['name']:>11s} | RMSE {r['rmse']:.2f} | {r['latency_s'] * 1000:.4f} ms/sample | "
              f"{r['param_count']:,.0f} params | energy {energy} | {r['n_sessions']} sessions")

    if records:
        _save_figure(efficiency_figure(records, error_bars=args.error_bars),
                     os.path.join(results_dir, 'decoder_efficiency.png'))
        fig = energy_figure(records)
        if fig is not None:
            _save_figure(fig, os.path.join(results_dir, 'decoder_energy.png'))

    combined_for_figure = copy.deepcopy(combined)
    for name, s in speck.items():
        for session, metrics in s['accuracy'].items():
            if session in combined_for_figure:
                combined_for_figure[session][name] = metrics
    _save_figure(comparison_4x2_figure(combined_for_figure, durations),
                 os.path.join(results_dir, 'decoder_comparison_4x2.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', required=True,
                        help='Subject results directory containing sessions/')
    parser.add_argument('--speck_summary_path', default=None,
                        help='Default: {results_dir}/aggregated_summary.json')
    parser.add_argument('--error_bars', action='store_true',
                        help='Draw 95%% CI error bars on the efficiency figure')
    main(parser.parse_args())

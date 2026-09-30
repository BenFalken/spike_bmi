"""
Combine one subject's per-session results (test_all_decoders.py output) and
draw the report figures.

Reads   {results_dir}/sessions/*.json
Writes  {results_dir}/combined_metrics.json            {session: {decoder: metrics}}
        {results_dir}/combined_metrics_durations.json  {session: {"Nmin": {decoder: metrics}}}
        {results_dir}/efficiency_summary.json          {machine: {decoder: means across sessions}}
        {results_dir}/decoder_efficiency.png, decoder_energy.png, decoder_comparison_4x2.png

Usage:
    python make_report.py --results_dir $BMI_DATA_ROOT/results/test_all_decoders/bmi/indy
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from report_figures import DECODER_ORDER, comparison_4x2_figure, energy_figure, efficiency_figure, mean_ci


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


def _profiles(content):
    """{machine: {decoder: profile}}, also for files with the older single
    'profile' section (cluster only, one energy_method for all decoders)."""
    if 'profiles' in content:
        return {m: p.get('decoders', {}) for m, p in content['profiles'].items()}
    old = content.get('profile') or {}
    return {'cluster': {name: {**p, 'energy_method': p.get('energy_method', old.get('energy_method'))}
                        for name, p in old.get('decoders', {}).items()}} if old else {}


def efficiency_records(sessions):
    """One record per (machine, decoder): means (and CIs) across sessions of
    RMSE, latency, parameter count and energy."""
    samples = {}   # (machine, decoder) -> lists
    for content in sessions.values():
        metrics = content['full']['metrics']
        for machine, profile in _profiles(content).items():
            for name, p in profile.items():
                if name not in metrics:
                    continue
                s = samples.setdefault((machine, name), {'rmse': [], 'latency_s': [], 'param_count': [],
                                                         'energy_j': [], 'methods': {}})
                s['rmse'].append(metrics[name]['rmse'])
                s['latency_s'].append(p['latency_s'])
                s['param_count'].append(p['param_count'])
                s['energy_j'].append(p['energy_j'])
                if p.get('energy_method') and p['energy_j'] is not None:
                    s['methods'][p['energy_method']] = s['methods'].get(p['energy_method'], 0) + 1

    records = []
    order = {d: i for i, d in enumerate(DECODER_ORDER)}
    for machine, name in sorted(samples, key=lambda k: (k[0], order.get(k[1], len(order)))):
        s = samples[(machine, name)]
        label = f"{name.upper()} ({machine})"
        if len(set(s['param_count'])) > 1:
            print(f"WARNING: {label} parameter count differs across sessions: {sorted(set(s['param_count']))}")
        if len(s['methods']) > 1:
            print(f"WARNING: {label} energy mixes measurement methods across sessions: {s['methods']}")
        rmse, rmse_lo, rmse_hi = mean_ci(s['rmse'])
        latency, latency_lo, latency_hi = mean_ci(s['latency_s'])
        energies = [e for e in s['energy_j'] if e is not None]
        energy = mean_ci(energies) if energies else (None, None, None)
        records.append({
            'name': name, 'machine': machine, 'rmse': rmse, 'rmse_lo': rmse_lo, 'rmse_hi': rmse_hi,
            'latency_s': latency, 'latency_lo': latency_lo, 'latency_hi': latency_hi,
            'param_count': float(np.mean(s['param_count'])),
            'energy_j': energy[0], 'energy_lo': energy[1], 'energy_hi': energy[2],
            'energy_method': max(s['methods'], key=s['methods'].get) if energies and s['methods'] else None,
            'n_sessions': len(s['latency_s']), 'n_energy_sessions': len(energies),
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

    records = efficiency_records(sessions)
    summary = {}
    for r in records:
        summary.setdefault(r['machine'], {})[r['name']] = {k: v for k, v in r.items()
                                                          if k not in ('name', 'machine')}
    _write_json(summary, os.path.join(results_dir, 'efficiency_summary.json'))
    for r in records:
        energy = (f"{r['energy_j'] * 1e6:.3f} uJ ({r['energy_method']})" if r['energy_j'] is not None
                  else 'n/a')
        print(f"  {r['machine']:>10s} {r['name']:>5s} | RMSE {r['rmse']:.2f} | "
              f"{r['latency_s'] * 1000:.4f} ms/sample | {r['param_count']:,.0f} params | "
              f"energy {energy} | {r['n_sessions']} sessions")

    if records:
        _save_figure(efficiency_figure(records, error_bars=args.error_bars),
                     os.path.join(results_dir, 'decoder_efficiency.png'))
        fig = energy_figure(records)
        if fig is not None:
            _save_figure(fig, os.path.join(results_dir, 'decoder_energy.png'))

    _save_figure(comparison_4x2_figure(combined, durations),
                 os.path.join(results_dir, 'decoder_comparison_4x2.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', required=True,
                        help='Subject results directory containing sessions/')
    parser.add_argument('--error_bars', action='store_true',
                        help='Draw 95%% CI error bars on the efficiency figure')
    main(parser.parse_args())

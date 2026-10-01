"""
Compare the Speck diagnosis of several SNN training runs (e.g. per-session,
fine-tuned with hard reset, fine-tuned with soft reset) in one figure, from
the speck_diagnosis.json each diagnose_speck.py run writes.

    a  mean RMSE as the network goes from PyTorch to quantized to the chip
    b  per session, the RMSE the chip adds to the quantized network
    c  per session, PyTorch RMSE against Speck RMSE

Only sessions with chip results (snn/speck scored on the same checkpoint)
are used.

Usage (no devkit needed):
    python compare_speck_runs.py \
        --run "per-session=../../data/results/per_session/bmi/indy/speck_diagnosis.json" \
        --run "fine-tuned, hard=../../data/results/full_cohort_finetuned_medium/bmi/indy/speck_diagnosis.json" \
        --run "fine-tuned, soft=../../data/results/full_cohort_finetuned_medium_soft/bmi/indy/speck_diagnosis.json" \
        --output ../../data/results/speck_run_comparison.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ['#2a78d6', '#eb6834', '#1baf7a']   # blue, orange, aqua: distinguishable with colour-vision deficiency
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
STAGES = [('pytorch', 'PyTorch'), ('quantized', 'Quantized'), ('speck', 'Speck')]


def load_run(path):
    """{measure: per-session array} over the sessions with chip results."""
    with open(path) as f:
        sessions = json.load(f)['sessions']
    rows = [r for r in sessions.values() if 'speck' in r]
    out = {name: np.array([r[name]['rmse'] for r in rows]) for name, _ in STAGES}
    chip = [r['chip_vs_quantized'] for r in rows if 'chip_vs_quantized' in r]
    out['spike_ratio'] = np.array([c['spike_ratio'] for c in chip])
    out['gap'] = out['speck'] - out['quantized']
    return out


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _sem(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _spread(values, gap):
    """Label positions near `values`, at least `gap` apart, in the same order."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        pos[b] = max(pos[b], pos[a] + gap)
    return pos


def stage_panel(ax, runs):
    x = np.arange(len(STAGES))
    offsets = np.linspace(-0.06, 0.06, len(runs)) if len(runs) > 1 else [0.0]
    finals = []
    for (label, run), color, off in zip(runs.items(), COLORS, offsets):
        means = [run[s].mean() for s, _ in STAGES]
        ax.errorbar(x + off, means, yerr=[_sem(run[s]) for s, _ in STAGES], color=color, linewidth=2,
                    marker='o', markersize=8, markeredgecolor='white', markeredgewidth=1.5, capsize=0,
                    elinewidth=1)
        finals.append(means[-1])
    span = np.ptp(ax.get_ylim())
    for (label, _), y, y_label in zip(runs.items(), finals, _spread(finals, 0.045 * span)):
        ax.annotate(f"{label}  {y:.1f}", (x[-1] + 0.1, y), xytext=(x[-1] + 0.16, y_label),
                    va='center', color=INK, fontsize=9)
    ax.set_xticks(x, [name for _, name in STAGES])
    ax.set_xlim(-0.3, len(STAGES) - 0.15)
    _style(ax, 'a  Mean RMSE from training to chip (± s.e.m.)', ylabel='RMSE')


def gap_panel(ax, runs, rng):
    for i, ((label, run), color) in enumerate(zip(runs.items(), COLORS)):
        jitter = rng.uniform(-0.18, 0.18, len(run['gap']))
        ax.scatter(i + jitter, run['gap'], s=22, color=color, alpha=0.75, edgecolors='white', linewidths=0.5)
        mean = run['gap'].mean()
        ax.hlines(mean, i - 0.3, i + 0.3, color=INK, linewidth=2)
        ax.annotate(f"{mean:.1f}", (i + 0.3, mean), xytext=(4, 0), textcoords='offset points',
                    va='center', color=INK, fontsize=9)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(len(runs)), list(runs))
    _style(ax, 'b  RMSE the chip adds, per session (bar: mean)', ylabel='Speck − quantized RMSE')


def scatter_panel(ax, runs):
    lo = min(min(r['pytorch'].min(), r['speck'].min()) for r in runs.values()) - 2
    hi = max(max(r['pytorch'].max(), r['speck'].max()) for r in runs.values()) + 2
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle='--')
    ax.annotate('Speck = PyTorch', (hi, hi), xytext=(-4, -12), textcoords='offset points', ha='right',
                color=MUTED, fontsize=8)
    for (label, run), color in zip(runs.items(), COLORS):
        ax.scatter(run['pytorch'], run['speck'], s=30, color=color, alpha=0.8, edgecolors='white',
                   linewidths=0.6, label=label)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK)
    _style(ax, 'c  Per session: PyTorch vs. Speck', xlabel='PyTorch RMSE', ylabel='Speck RMSE')


def main(args):
    runs = {}
    for spec in args.run:
        label, path = spec.split('=', 1)
        runs[label] = load_run(path)
    if len(runs) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} runs per figure")

    print(f"{'run':<20s}{'sessions':>9s}" + ''.join(f"{name:>11s}" for _, name in STAGES)
          + f"{'chip adds':>11s}{'spk ratio':>11s}")
    for label, run in runs.items():
        fmt = lambda k: f"{run[k].mean():11.2f}" if len(run[k]) else f"{'-':>11s}"
        print(f"{label:<20s}{len(run['speck']):9d}" + ''.join(fmt(s) for s, _ in STAGES)
              + fmt('gap') + fmt('spike_ratio'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    stage_panel(axes[0], runs)
    gap_panel(axes[1], runs, np.random.default_rng(0))
    scatter_panel(axes[2], runs)
    fig.tight_layout(w_pad=3)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor='white')
    print(f"Saved {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=PATH',
                        help='A run label and its speck_diagnosis.json; repeat per run (up to 3)')
    parser.add_argument('--output', default='speck_run_comparison.png')
    parser.add_argument('--dpi', type=int, default=150)
    main(parser.parse_args())

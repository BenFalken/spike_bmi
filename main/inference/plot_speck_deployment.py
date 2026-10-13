"""
The Jenkins (hkm) Speck deployment in one figure: how far the chip's
decoding error came down, and which factors caused the gap to PyTorch.

    speck_deployment_jenkins.png
      a  chip RMSE along the fixes, first deployment to the deployed setting
      b  hidden layer 1 threshold factor (1 ms wait): PyTorch, emulator, chip
      c  chip wait per timestep (h1 x2), with the 4 ms real-time budget
      d  RMSE change per factor tested: causes, ruled out, alternatives
      e  chip power, f  energy per timestep, for the settings measured

Numbers come from docs/data/speck_deployment_jenkins_20090912.json (the
chip and emulator runs of speck_lab/ and speck_activity_study.py). The
deployed 3 ms setting's power was not recorded there: it is drawn as an
estimate (hatched) unless --wait3_json gives that run's probe JSON
(lab_runs/probe_wait_0.003.json), whose measured chip numbers then replace it.

Usage:
    python plot_speck_deployment.py --data ../../docs/data/speck_deployment_jenkins_20090912.json \
        --out_dir ../../docs [--wait3_json lab_runs/probe_wait_0.003.json]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Reference palette (dataviz skill), light mode; slots 1-3 validated all-pairs.
SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
PYTORCH, CHIP, EMULATED = '#2a78d6', '#eb6834', '#1baf7a'


def style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=10.5, color=INK, fontweight='bold')


def label_line(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', fontsize=8.5,
                color=INK, va='center', ha='left')


def panel_path(ax, d):
    path = d['path']
    xs = range(len(path))
    colors = [CHIP if p.get('deployed') else '#f2b08f' for p in path]
    ax.bar(xs, [p['rmse'] for p in path], width=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    for x, p in zip(xs, path):
        ax.text(x, p['rmse'] + 2, f"{p['rmse']:.1f}", ha='center', va='bottom', fontsize=9, color=INK,
                fontweight='bold' if p.get('deployed') else 'normal')
    ref = d['reference']['pytorch']
    ax.axhline(ref, color=PYTORCH, linewidth=2, linestyle='--')
    ax.text(-0.45, ref + 2, f"PyTorch {ref:.1f}", color=INK, fontsize=8.5, ha='left', va='bottom')
    ax.set_xticks(list(xs))
    ax.set_xticklabels([p['label'] for p in path], fontsize=7.6)
    ax.set_ylim(0, 135)
    ax.set_ylabel('chip RMSE (velocity)', color=INK_2, fontsize=9)
    style(ax, 'a  Chip error, first deployment to deployed')


def panel_threshold(ax, d):
    s = d['threshold_sweep']
    for key, color, marker, name in (('pytorch', PYTORCH, 'o', 'PyTorch'), ('emulated', EMULATED, 's', 'emulator'),
                                     ('chip', CHIP, '^', 'chip (1 ms wait)')):
        ax.plot(s['factor'], s[key], color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
        label_line(ax, s['factor'][-1], s[key][-1], name)
    ax.axvline(2.0, color=MUTED, linewidth=1, linestyle=':')
    ax.text(2.03, 42, 'deployed x2', fontsize=8, color=INK_2, va='bottom')
    ax.set_xlim(0.9, 3.75)
    ax.set_ylim(40, 130)
    ax.set_xlabel('hidden layer 1 threshold factor (no retraining)', color=INK_2, fontsize=9)
    ax.set_ylabel('RMSE', color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    style(ax, 'b  Threshold: fixes the excess firing')


def panel_wait(ax, d):
    s = d['wait_sweep']
    budget = 4.0 - s['host_overhead_ms']
    ax.axvspan(budget, 5.4, color=GRID, alpha=0.7, linewidth=0)
    ax.text(budget + 0.08, 76, 'step > 4 ms:\nslower than\nreal time', fontsize=8, color=INK_2, va='top')
    ax.plot(s['wait_ms'], s['chip'], color=CHIP, linewidth=2, marker='^', markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label='chip (h1 x2)')
    for w, r in zip(s['wait_ms'], s['chip']):
        if w in (1.0, 3.0, 5.0):
            ax.annotate(f"{r:.1f}", (w, r), xytext=(0, 7), textcoords='offset points', ha='center',
                        fontsize=8.5, color=INK, fontweight='bold' if w == 3.0 else 'normal')
    for value, color, name, dy in ((d['reference']['emulator_x2'], EMULATED, 'emulator (no time limit)', 5),
                                   (d['reference']['pytorch'], PYTORCH, 'PyTorch', -5)):
        ax.axhline(value, color=color, linewidth=2, linestyle='--', label=name)
        ax.text(1.9, value + dy * 0.12, name, fontsize=8, color=INK, va='bottom' if dy > 0 else 'top')
    ax.set_xlim(0.8, 5.4)
    ax.set_ylim(50, 90)
    ax.set_xlabel('chip wait per timestep (ms)', color=INK_2, fontsize=9)
    ax.set_ylabel('RMSE', color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0.2, 1.0))
    style(ax, 'c  Wait: lets the chip finish each step')


def panel_factors(ax, d):
    factors = d['factors']
    rows, labels, y = [], [], 0.8
    group_rows = {}
    for f in factors:
        if f['group'] not in group_rows:
            if group_rows:
                y += 0.6
            group_rows[f['group']] = y
            y += 0.9
        rows.append(y)
        labels.append(f['label'])
        y += 1
    shades = {'Causes (fixed)': CHIP, 'Ruled out': MUTED, 'Alternatives not adopted': '#c3c2b7'}
    for f, r in zip(factors, rows):
        ax.barh(r, f['delta'], height=0.62, color=shades[f['group']], edgecolor=SURFACE, linewidth=1.5)
        sign = '+' if f['delta'] > 0 else ('' if f['delta'] < 0 else '±')
        ax.text(f['delta'] + (1 if f['delta'] >= 0 else -1), r, f"{sign}{f['delta']:.1f}",
                va='center', ha='left' if f['delta'] >= 0 else 'right', fontsize=8.5, color=INK)
    for group, r in group_rows.items():
        ax.text(-58, r, group, fontsize=9, fontweight='bold', color=INK, va='center')
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.set_ylim(rows[-1] + 0.7, -0.4)     # top to bottom
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlim(-58, 32)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('change in RMSE (negative = better)', color=INK_2, fontsize=9)
    style(ax, 'd  What each factor did')


def panel_energy(ax, d, key, unit, title):
    entries = d['energy']
    xs = range(len(entries))
    for x, e in zip(xs, entries):
        measured = e.get(key) is not None
        value = e[key] if measured else e['estimate'][key]
        deployed = '(deployed)' in e['label']
        ax.bar(x, value, width=0.62, color=CHIP if deployed else '#f2b08f', edgecolor=SURFACE if measured else INK_2,
               linewidth=2 if measured else 1, hatch=None if measured else '///', alpha=1 if measured else 0.55)
        ax.text(x, value * 1.02 + 0.1, f"{value:.1f}" + ('' if measured else '\n(est.)'), ha='center', va='bottom',
                fontsize=8.5, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([e['label'] for e in entries], fontsize=7.8)
    top = max((e[key] if e.get(key) is not None else e['estimate'][key]) for e in entries)
    ax.set_ylim(0, top * 1.3)
    ax.set_ylabel(unit, color=INK_2, fontsize=9)
    style(ax, title)


def main(args):
    with open(args.data) as f:
        d = json.load(f)
    if args.wait3_json:
        with open(args.wait3_json) as f:
            chip = json.load(f)['x2 on layer 1']['scores']['chip']
        for e in d['energy']:
            if e['label'].startswith('h1 x2, 3 ms'):
                e.update(power_mw=chip['power_mw'], latency_ms=chip['latency_ms'],
                         energy_uj=chip['energy_uj_per_step'])
    fig = plt.figure(figsize=(16, 10.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], hspace=0.42, wspace=0.3)
    panel_path(fig.add_subplot(grid[0, 0]), d)
    panel_threshold(fig.add_subplot(grid[0, 1]), d)
    panel_wait(fig.add_subplot(grid[0, 2]), d)
    ax_f = fig.add_subplot(grid[1, 0])
    panel_factors(ax_f, d)
    pos = ax_f.get_position()
    ax_f.set_position([pos.x0 + 0.11, pos.y0, pos.width + 0.12, pos.height])
    energy = grid[1, 2].subgridspec(2, 1, hspace=0.75)
    panel_energy(fig.add_subplot(energy[0]), d, 'power_mw', 'mW', 'e  Chip power')
    panel_energy(fig.add_subplot(energy[1]), d, 'energy_uj', 'µJ per timestep', 'f  Chip energy per timestep')
    fig.suptitle(f"Speck deployment of the Jenkins SNN ({d['session'].split('_')[1]}): "
                 f"{d['path'][0]['rmse']:.1f} → {d['path'][2]['rmse']:.1f} RMSE on the chip "
                 f"(PyTorch {d['reference']['pytorch']:.1f})",
                 x=0.06, ha='left', fontsize=13, color=INK, fontweight='bold')
    fig.text(0.06, 0.935, f"{d['test']}. Two causes: hidden layer 1 fires on early excitatory events "
             "(fixed by doubling its threshold at deployment, no retraining), and output read before the chip "
             "finishes a step (fixed by a 3 ms wait).", fontsize=9, color=INK_2)
    fig.legend(handles=[Patch(color=CHIP, label='deployed setting / cause fixed'),
                        Patch(color='#f2b08f', label='other chip setting'),
                        Patch(color=MUTED, label='ruled out'), Patch(color='#c3c2b7', label='alternative not adopted'),
                        Patch(facecolor=SURFACE, edgecolor=INK_2, hatch='///', label='estimate (not measured)')],
               loc='lower center', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'speck_deployment_jenkins.png')
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print(f"Saved {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data', default=os.path.join(here, '../../docs/data/speck_deployment_jenkins_20090912.json'))
    parser.add_argument('--out_dir', default=os.path.join(here, '../../docs'))
    parser.add_argument('--wait3_json', default=None,
                        help='probe JSON of the 3 ms run (lab_runs/probe_wait_0.003.json): measured power')
    main(parser.parse_args())

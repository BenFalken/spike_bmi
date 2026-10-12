"""
Figures from speck_activity_study.py: firing rates, chip power and energy,
Indy (bmi) against Jenkins (hkm), PyTorch against quantized against Speck.

    speck_activity_rates.png       spikes per neuron per timestep through the
                                   layers (input to output), one panel per subject
    speck_activity_conditions.png  Jenkins study conditions: the deployed
                                   network's output rate against how well the chip
                                   follows the quantized network (prediction slope,
                                   added RMSE)
    speck_activity_energy.png      input events, chip power, latency and energy
                                   per timestep, Indy against every Jenkins condition

Jenkins comes from a study JSON (--jenkins_json). Indy comes from another
study JSON if given (--indy_json, the 'binary' condition), else from the
README's numbers for the per-session hard-reset Indy model: layer densities of
one 3 s test trial (PyTorch and Speck; no quantized run), and chip power,
latency and energy as means over 36 sessions. Speck layer rates are those of
the chip with every layer monitored (study: first --layer_trials trials);
PyTorch and quantized rates cover all the study's trials.

Usage:
    python plot_speck_activity.py --jenkins_json ../../docs/data/speck_activity_jenkins_20090912.json \
        --out_dir ../../docs
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill), light mode; slots 1-3 validated as a set.
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'PyTorch': ('#2a78d6', 'o'), 'quantized': ('#eb6834', 's'), 'Speck': ('#1baf7a', '^')}
REFERENCE_GRAY = '#8a8984'
LAYERS = ['input', 'hidden 1', 'hidden 2', 'hidden 3', 'output']

# README (per-session hard-reset Indy model): layer spike density of one test
# trial, and chip power / latency / energy means over sessions.
INDY_README = {
    'label': 'Indy (bmi)',
    'note': 'README: one 3 s trial; no quantized run',
    'rates': {'PyTorch': [0.054, 0.035, 0.056, 0.087, 0.232],
              'Speck': [0.054, 0.041, 0.078, 0.146, 0.459]},
    'power_mw': 2.15, 'latency_ms': 1.27, 'energy_uj': 2.73,
    'input_events_per_step': 6.11,
}

CONDITION_LABELS = {
    'raw': 'raw counts', 'binary': 'binary (as trained)', 'binary_wait5': 'binary, 5 ms wait',
    'binary_thin50': 'half the input', 'binary_thr2': 'thresholds x2',
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def study_rates(study, condition='binary'):
    c = study['conditions'][condition]
    n_in = study['input_stats']['session_all_test_trials']['channels']
    x_in = c['input_events_per_step'] / n_in
    lr = c['layer_rates']
    rates = {'PyTorch': [x_in] + lr['float'], 'quantized': [x_in] + lr['quantized']}
    if 'chip_monitored' in lr:
        rates['Speck'] = [x_in] + lr['chip_monitored']
    return rates


def subject_from_study(study, label, condition='binary'):
    c = study['conditions'][condition]
    chip = c.get('chip', {})
    return {'label': label, 'note': f"study, '{condition}' condition",
            'rates': study_rates(study, condition),
            'power_mw': chip.get('power_mw'), 'latency_ms': chip.get('latency_ms_per_step'),
            'energy_uj': chip['power_mw'] * chip['latency_ms_per_step'] if chip else None,
            'input_events_per_step': c['input_events_per_step']}


def fig_rates(subjects, path):
    fig, axes = plt.subplots(1, len(subjects), figsize=(5.2 * len(subjects), 4.2), sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    x = np.arange(len(LAYERS))
    for ax, subj in zip(axes, subjects):
        style(ax)
        labelled = []                      # (output value, label) already placed
        for name, (color, marker) in SERIES.items():
            if name not in subj['rates']:
                continue
            y = subj['rates'][name]
            ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3)
            # Series ending within 5% of one another share one end label.
            same = [i for i, (v, _) in enumerate(labelled) if abs(v - y[-1]) <= 0.05 * v]
            if same:
                v, text = labelled[same[0]]
                labelled[same[0]] = (v, text.replace(f" {v:.2f}", f" = {name} {v:.2f}"))
            else:
                labelled.append((y[-1], f"{name} {y[-1]:.2f}"))
        for v, text in labelled:
            ax.annotate(text, (x[-1], v), xytext=(8, 0), textcoords='offset points',
                        va='center', fontsize=9, color=INK)
        if 'Speck' in subj['rates']:
            ratio = subj['rates']['Speck'][-1] / subj['rates']['PyTorch'][-1]
            ax.text(0.02, 0.97, f"output: Speck / PyTorch = {ratio:.1f}x", transform=ax.transAxes,
                    va='top', fontsize=9, color=INK_2)
        ax.set_yscale('log')
        ax.set_xticks(x, LAYERS)
        ax.set_xlim(-0.3, len(LAYERS) - 1 + 2.2)
        ax.set_title(f"{subj['label']}\n{subj['note']}", fontsize=10, color=INK, loc='left', pad=8)
    axes[0].set_ylabel('spikes per neuron per timestep (log)', color=INK_2, fontsize=10)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.06, 1.07), labelcolor=INK)
    fig.suptitle('Firing rate through the network: PyTorch, quantized and on Speck',
                 y=1.15, x=0.07, ha='left', fontsize=12, color=INK)
    fig.text(0.01, -0.04, 'Input: spikes per input channel per timestep. Jenkins Speck rates: chip with every '
             'layer monitored over the first study trials.', fontsize=8, color=INK_2)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def fig_conditions(study, path):
    conds = list(study['conditions'])
    out_rate = [study['conditions'][c]['layer_rates']['quantized'][-1] for c in conds]
    slope = [np.mean([study['conditions'][c]['chip_vs_quantized'][k] for k in ('slope_x', 'slope_y')])
             for c in conds]
    added = [study['conditions'][c]['rmse']['chip'] - study['conditions'][c]['rmse']['quantized']
             for c in conds]
    color, marker = SERIES['Speck']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURFACE)
    panels = [(axes[0], slope, 'prediction slope, chip on quantized\n(1 = no compression)', 1.0),
              (axes[1], added, 'RMSE added by the chip\n(chip - quantized)', 0.0)]
    for ax, y, ylabel, ref in panels:
        span = max(max(y), ref) - min(min(y), ref) or 1.0
        style(ax)
        ax.axhline(ref, color=INK_2, linewidth=1, linestyle='--', zorder=1)
        ax.scatter(out_rate, y, s=70, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.2,
                   zorder=3)
        placed = []                        # conditions at nearly the same point share a label
        for c, xv, yv in zip(conds, out_rate, y):
            near = [g for g in placed if abs(np.log(g['x'] / xv)) < 0.25 and abs(g['y'] - yv) < 0.06 * span]
            if near:
                near[0]['names'].append(CONDITION_LABELS.get(c, c))
            else:
                placed.append({'x': xv, 'y': yv, 'names': [CONDITION_LABELS.get(c, c)]})
        x_mid = np.sqrt(min(out_rate) * max(out_rate))       # log-scale middle
        y_mid = (max(y) + min(y)) / 2
        for g in placed:
            left = g['x'] > x_mid                              # label toward the panel's middle
            above = g['y'] >= y_mid                            # and into the empty half
            ax.annotate('\n'.join(g['names']), (g['x'], g['y']),
                        xytext=(-8 if left else 8, 8 if above else -8), textcoords='offset points',
                        fontsize=8.5, color=INK, ha='right' if left else 'left',
                        va='bottom' if above else 'top')
        ax.set_xscale('log')
        ax.set_xlabel('quantized network output rate (spikes per neuron per timestep, log)',
                      color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[1].set_ylim(0, max(added) * 1.2)
    fig.suptitle('Jenkins: the chip follows the trained network only when the network fires sparsely',
                 fontsize=11.5, color=INK, x=0.01, ha='left')
    fig.text(0.01, -0.05, 'One session (20090912), 30 test trials. Thinning the input lowers activity '
             'only partly; doubling thresholds (same input) lowers it 30x.', fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def fig_energy(indy, study, path):
    rows = [(indy['label'], indy['input_events_per_step'], indy['power_mw'], indy['latency_ms'],
             indy['energy_uj'], True)]
    for c, v in study['conditions'].items():
        chip = v['chip']
        rows.append((f"Jenkins: {CONDITION_LABELS.get(c, c)}", v['input_events_per_step'], chip['power_mw'],
                     chip['latency_ms_per_step'], chip['power_mw'] * chip['latency_ms_per_step'], False))
    labels = [r[0] for r in rows]
    metrics = [('input events per timestep', 1, '{:.1f}'), ('chip power (mW)', 2, '{:.2f}'),
               ('latency (ms per timestep)', 3, '{:.2f}'), ('energy per timestep (uJ)', 4, '{:.1f}')]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    y = np.arange(len(rows))[::-1]
    for ax, (title, k, fmt) in zip(axes, metrics):
        style(ax)
        ax.grid(axis='y', visible=False)
        vals = [r[k] for r in rows]
        colors = [REFERENCE_GRAY if r[5] else SERIES['PyTorch'][0] for r in rows]
        ax.barh(y, vals, height=0.62, color=colors, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v, yi, ' ' + fmt.format(v), va='center', fontsize=8.5, color=INK)
        ax.set_xlim(0, max(vals) * 1.3)
        ax.set_title(title, fontsize=9.5, color=INK, loc='left')
    axes[0].set_yticks(y, labels, fontsize=9, color=INK)
    fig.suptitle('Speck input load, power and energy: Indy (gray) against Jenkins (blue)',
                 fontsize=11.5, color=INK, x=0.01, ha='left')
    fig.text(0.01, -0.06, 'Energy per timestep = mean chip power x loop time per timestep. Power alone depends '
             'on the loop rate (the 5 ms wait lowers mW but raises energy). Indy: README means over sessions.',
             fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def main(args):
    with open(args.jenkins_json) as f:
        jenkins = json.load(f)
    if args.indy_json:
        with open(args.indy_json) as f:
            indy = subject_from_study(json.load(f), 'Indy (bmi)')
    else:
        indy = INDY_README
    jen = subject_from_study(jenkins, 'Jenkins (hkm)')
    jen['note'] = "study, 'binary' condition (as trained)"
    os.makedirs(args.out_dir, exist_ok=True)
    outputs = [os.path.join(args.out_dir, f'speck_activity_{n}.png') for n in ('rates', 'conditions', 'energy')]
    fig_rates([indy, jen], outputs[0])
    fig_conditions(jenkins, outputs[1])
    fig_energy(indy, jenkins, outputs[2])
    for p in outputs:
        print(f"Saved {p}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jenkins_json', required=True, help='speck_activity_study.py output for Jenkins')
    parser.add_argument('--indy_json', default=None,
                        help="speck_activity_study.py output for an Indy session (default: README numbers)")
    parser.add_argument('--out_dir', default='.')
    main(parser.parse_args())

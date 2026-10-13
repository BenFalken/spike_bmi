"""
Figures for lab.py run (also: python plots.py <out_dir> to redraw from results.json).

    lab_rmse.png    RMSE of every regime as PyTorch, quantized, emulated chip (with
                    the model's readout and with one calibrated on the emulated chip)
                    and, when run, the real chip; the prediction slope of the chip
                    versions on the quantized network under each group
    lab_rates.png   spikes per neuron per timestep by layer, one panel per regime
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill), light mode; the six slots validated as a set.
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'pytorch': ('PyTorch', '#2a78d6', 'o'), 'quantized': ('quantized', '#eb6834', 's'),
          'emulated': ('emulated chip', '#1baf7a', '^'),
          'emulated_cal': ('emulated chip, calibrated readout', '#eda100', 'v'),
          'chip': ('Speck chip', '#e87ba4', 'D'), 'chip_cal': ('Speck chip, calibrated readout', '#008300', 'P')}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def plot_rmse(results, path):
    names = list(results)
    versions = [v for v in SERIES if any(v in results[n]['scores']['rmse'] for n in names)]
    width = 0.8 / len(versions)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.9 * len(names) + 2), 4.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis='x', visible=False)
    x = np.arange(len(names))
    top = max(results[n]['scores']['rmse'].get(v, 0) for n in names for v in versions)
    for i, v in enumerate(versions):
        label, color, _ = SERIES[v]
        vals = [results[n]['scores']['rmse'].get(v, np.nan) for n in names]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width * 0.92, color=color, label=label, zorder=3)
        for p, val in zip(pos, vals):
            if np.isfinite(val):
                ax.text(p, val + top * 0.01, f"{val:.0f}", ha='center', va='bottom', fontsize=7.5, color=INK)
    slopes = []
    for n in names:
        s = results[n]['scores']['slope_vs_quantized']
        parts = [f"emu {s['emulated']:.2f}"] + ([f"chip {s['chip']:.2f}"] if 'chip' in s else [])
        slopes.append(f"{n}\nslope: {', '.join(parts)}")
    ax.set_xticks(x, slopes, fontsize=8.5, color=INK)
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel('RMSE (velocity units)', color=INK_2, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=min(len(versions), 3), loc='upper left', bbox_to_anchor=(0, 1.18),
              labelcolor=INK)
    fig.suptitle('Decoding accuracy by training regime: PyTorch against the chip', fontsize=11.5, color=INK,
                 x=0.01, ha='left', y=1.12)
    fig.text(0.01, -0.06, 'slope: chip predictions regressed on the quantized network\'s (1 = no compression).',
             fontsize=8, color=INK_2)
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def plot_rates(results, path):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.8), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, names):
        style(ax)
        rates = results[n]['scores']['layer_rates']
        n_layers = len(next(iter(rates.values())))
        x = np.arange(n_layers)
        for v, (label, color, marker) in SERIES.items():
            if v in rates:
                ax.plot(x, rates[v], color=color, marker=marker, linewidth=2, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
        ax.set_yscale('log')
        ax.set_xticks(x, [f'h{i + 1}' for i in range(n_layers - 1)] + ['out'])
        ax.set_title(n, fontsize=10, color=INK, loc='left')
        out = rates.get('emulated', [np.nan])[-1] / max(rates['quantized'][-1], 1e-9)
        ax.text(0.03, 0.95, f"output emu / quant {out:.1f}x", transform=ax.transAxes, va='top',
                fontsize=8, color=INK_2)
    axes[0].set_ylabel('spikes per neuron per timestep (log)', color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=len(labels), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.01, 1.08), labelcolor=INK)
    fig.suptitle('Firing rate by layer for each regime', fontsize=11.5, color=INK, x=0.01, ha='left', y=1.15)
    fig.savefig(path, dpi=170, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


def plot_results(results, out_dir):
    paths = [os.path.join(out_dir, 'lab_rmse.png'), os.path.join(out_dir, 'lab_rates.png')]
    plot_rmse(results, paths[0])
    plot_rates(results, paths[1])
    return paths


if __name__ == '__main__':
    with open(os.path.join(sys.argv[1], 'results.json')) as f:
        for p in plot_results(json.load(f), sys.argv[1]):
            print(f"Saved {p}")

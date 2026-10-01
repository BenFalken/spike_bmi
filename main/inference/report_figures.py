"""
Report figures across the sessions of one subject (called by make_report.py).

    efficiency figure   RMSE vs. per-sample latency, marker area ~ parameter
                        count; one panel per profiled machine (e.g. the cluster
                        and the Speck-connected laptop)
    energy figure       energy per sample, with proxy estimates and real
                        (RAPL / chip) measurements in separate panels
    4x2 figure          a/b RMSE and CC boxplots over sessions (box = quartiles,
                        whiskers = 1.5 IQR, white dot = mean) with Wilcoxon
                        significance against the best decoder, c/d pairwise
                        matrices: % of sessions the row decoder wins, median
                        paired difference and significance,
                        e/f accuracy vs. training duration, g/h accuracy vs.
                        days since the first session. Wilcoxon p-values are
                        Holm-corrected within each panel.

'speck' (the SNN run on the chip) shares the SNN's colour and is told apart
by hatching (dashed in line plots).
"""

import re
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import t as t_dist, wilcoxon

DECODER_ORDER = ['kf', 'wf', 'lstm', 'qrnn', 'snn', 'speck']
REAL_ENERGY_METHODS = {'rapl', 'chip_power_monitor'}

EFFICIENCY_COLORS = {'lstm': 'darkorange', 'qrnn': 'seagreen', 'kf': 'purple', 'wf': 'goldenrod',
                     'snn': 'royalblue', 'speck': 'royalblue'}
COMPARISON_COLORS = {'kf': '#e8271c', 'wf': '#8fdc3c', 'snn': '#00bcd4', 'lstm': '#6a1fc9',
                     'qrnn': '#e91e8c', 'speck': '#00bcd4'}
HATCHES = {'speck': '///'}


def display_label(name):
    return 'SNN (Speck)' if name == 'speck' else name.upper()


def mean_ci(values, confidence=0.95):
    """(mean, low, high) with a t-distribution confidence interval."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    sem = values.std(ddof=1) / np.sqrt(len(values))
    margin = float(t_dist.ppf(1 - (1 - confidence) / 2, df=len(values) - 1) * sem)
    return mean, mean - margin, mean + margin


def average_cc(entry):
    return (entry['cc_x'] + entry['cc_y']) / 2 if entry.get('cc_y') is not None else entry['cc_x']


# --------------------------------------------------------------------------- #
# Efficiency and energy
# --------------------------------------------------------------------------- #

def efficiency_panel(ax, records, colors=EFFICIENCY_COLORS, error_bars=False):
    """RMSE vs. latency (log x); records have name, rmse, latency_s,
    param_count, and (for error_bars) rmse_lo/hi, latency_lo/hi."""
    params = np.array([r['param_count'] for r in records], dtype=float)
    max_params = params.max() if len(params) else 1.0
    sizes = 60 + 900 * np.sqrt(params / max_params)     # marker area ~ sqrt(params)
    for rec, size in zip(records, sizes):
        x, y = rec['latency_s'] * 1000, rec['rmse']
        ax.scatter(x, y, s=size, color=colors.get(rec['name'], 'gray'), edgecolor='black',
                   linewidth=0.8, alpha=0.85, zorder=3, hatch=HATCHES.get(rec['name']))
        ax.annotate(display_label(rec['name']), (x, y), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=9, fontweight='bold')
        if error_bars:
            xerr = [[max(0, x - rec['latency_lo'] * 1000)], [max(0, rec['latency_hi'] * 1000 - x)]]
            yerr = [[max(0, y - rec['rmse_lo'])], [max(0, rec['rmse_hi'] - y)]]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='none', ecolor=colors.get(rec['name'], 'gray'),
                        elinewidth=1.2, capsize=3, zorder=2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Per-sample inference latency (ms, log scale)')
    ax.set_ylabel('Mean RMSE (across sessions)')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.margins(y=0.18)
    handles = [ax.scatter([], [], s=60 + 900 * np.sqrt(f), color='gray', edgecolor='black', alpha=0.6,
                          label=f'{int(f * max_params):,} params') for f in (0.1, 0.5, 1.0)]
    ax.legend(handles=handles, title='Parameter count (rough scale)', loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, title_fontsize=8, frameon=False)


def efficiency_figure(records, error_bars=False):
    """One panel per profiled machine: latency is only comparable within a
    panel."""
    machines = list(dict.fromkeys(r['machine'] for r in records))
    cohorts = [(m, [r for r in records if r['machine'] == m]) for m in machines]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(7 * len(cohorts), 6 if len(cohorts) > 1 else 5.5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (title, recs) in zip(axes[0], cohorts):
        efficiency_panel(ax, recs, error_bars=error_bars)
        ax.set_title(f'Decoding accuracy vs. computational cost -- {title}\n'
                     f'(mean across sessions; marker size = parameter count)', fontsize=9)
    if len(cohorts) > 1:
        fig.suptitle('Latency is not comparable across panels (different machines)', fontsize=11, y=1.03)
    fig.tight_layout()
    return fig


def _energy_panel(ax, records, title):
    energies = [r['energy_j'] * 1e6 for r in records]      # uJ
    for i, (rec, e) in enumerate(zip(records, energies)):
        ax.bar(i, e, color=EFFICIENCY_COLORS.get(rec['name'], 'gray'), edgecolor='black',
               linewidth=0.8, hatch=HATCHES.get(rec['name']), zorder=3)
        if rec['n_energy_sessions'] > 1:
            ax.errorbar(i, e, yerr=[[max(0, e - rec['energy_lo'] * 1e6)], [max(0, rec['energy_hi'] * 1e6 - e)]],
                        fmt='none', ecolor='black', elinewidth=1.2, capsize=3, zorder=4)
        method = {'rapl': 'RAPL', 'chip_power_monitor': 'chip',
                  'proxy_psutil': 'proxy'}.get(rec['energy_method'], rec['energy_method'] or 'unknown')
        ax.annotate(f"{display_label(rec['name'])}\n{rec['machine']}, {method}", (i, e), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(np.arange(len(records)), [])
    ax.set_yscale('log')
    ax.set_title(title, fontsize=10)
    ax.grid(True, which='both', axis='y', linestyle=':', alpha=0.4)
    ax.margins(y=0.3)


def energy_figure(records):
    """Energy per decoded sample. Proxy estimates and real measurements are
    different quantities, so they go in separate panels. None if no decoder
    has energy data."""
    usable = [r for r in records if r.get('energy_j') is not None]
    groups = [(title, recs) for title, recs in [
        ('Estimated (proxy, not measured)', [r for r in usable if r['energy_method'] not in REAL_ENERGY_METHODS]),
        ('Measured (real)', [r for r in usable if r['energy_method'] in REAL_ENERGY_METHODS]),
    ] if recs]
    if not groups:
        return None
    widths = [max(5.5 if i == 0 else 3.5, 1.3 * len(recs)) for i, (_, recs) in enumerate(groups)]
    fig, axes = plt.subplots(1, len(groups), figsize=(sum(widths), 5), sharey=True, squeeze=False)
    for ax, (title, recs) in zip(axes[0], groups):
        _energy_panel(ax, recs, 'Energy per decoded sample\n' + title if len(groups) == 1 else title)
    axes[0][0].set_ylabel('Mean energy per sample (uJ, log scale)')
    if len(groups) > 1:
        fig.suptitle('Energy per decoded sample -- proxy estimates and real measurements are not '
                     'directly comparable', fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4x2 accuracy comparison
# --------------------------------------------------------------------------- #

def session_date(session_id):
    """First 8-digit YYYYMMDD group in the session ID (bmi and NWB naming)."""
    match = re.search(r'(\d{8})', session_id)
    return datetime.strptime(match.group(1), '%Y%m%d') if match else None


def _stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def _wilcoxon_p(a, b):
    try:
        return wilcoxon(a, b)[1]
    except ValueError:          # e.g. all differences zero
        return 1.0


def _boxplot(ax, values, decoders, sig, ylabel):
    box = ax.boxplot([values[d] for d in decoders], tick_labels=[display_label(d) for d in decoders],
                     patch_artist=True, showmeans=True, widths=0.6,
                     meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black',
                                    markersize=6, linewidth=1.2))
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    for patch, d in zip(box['boxes'], decoders):
        patch.set_facecolor(COMPARISON_COLORS[d])
        patch.set_edgecolor('black')
        if d in HATCHES:
            patch.set_hatch(HATCHES[d])
    for artist in box['whiskers'] + box['caps']:
        artist.set_color('black')
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    all_vals = [v for d in decoders for v in values[d]]
    pad = (max(all_vals) - min(all_vals)) * 0.05
    for i, d in enumerate(decoders, start=1):
        if sig[d]:
            ax.text(i, max(values[d]) + pad, sig[d], ha='center', va='bottom', fontsize=11)
    ax.set_ylabel(ylabel)


def _holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def _pairwise_matrix(ax, values, decoders, higher_is_better):
    """Cell (row, col): how often and by how much the row decoder beats the
    column decoder over the paired sessions. Colour = percentage of sessions
    in which the row is better (ties count half; 50% = no consistent winner);
    text = median paired difference row - col in the metric's units, with
    stars from the two-sided Wilcoxon signed-rank test, Holm-corrected over
    all pairs. The two triangles mirror each other (100 - % and -diff)."""
    n = len(decoders)
    sign = 1 if higher_is_better else -1
    win = np.full((n, n), np.nan)
    diff = np.full((n, n), np.nan)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    p_adj = _holm([_wilcoxon_p(values[decoders[i]], values[decoders[j]]) for i, j in pairs])
    pmat = np.full((n, n), np.nan)
    for (i, j), p in zip(pairs, p_adj):
        pmat[i, j] = pmat[j, i] = p
    for i, a in enumerate(decoders):
        for j, b in enumerate(decoders):
            if i != j:
                d = np.asarray(values[a], dtype=float) - np.asarray(values[b], dtype=float)
                diff[i, j] = np.median(d)
                win[i, j] = 100 * (np.mean(sign * d > 0) + 0.5 * np.mean(d == 0))
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color='0.85')
    im = ax.imshow(np.ma.masked_invalid(win), cmap=cmap, vmin=0, vmax=100, aspect='equal')
    labels = [display_label(d) for d in decoders]
    ax.set_xticks(np.arange(n), labels, rotation=45, ha='left')
    ax.set_yticks(np.arange(n), labels)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='both', length=0)
    for i in range(n):
        for j in range(n):
            if i != j:
                text = f"{diff[i, j]:+.2g}" + (f"\n{_stars(pmat[i, j])}" if _stars(pmat[i, j]) else '\nn.s.')
                ax.text(j, i, text, ha='center', va='center', fontsize=7.5, linespacing=1.1,
                        color='white' if abs(win[i, j] - 50) > 35 else 'black')
    cax = make_axes_locatable(ax).append_axes('bottom', size='6%', pad=0.55)
    bar = ax.figure.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 25, 50, 75, 100])
    bar.set_label('Sessions where the row decoder is better (%)')


def _over_x(ax, x_values, series, decoders, ylabel, xlabel, fmt, clip_decoder=None, clip_margin=1.15):
    """series[d] = (mean, low, high) arrays parallel to x_values; points are
    evenly spaced in sorted order with the actual values as tick labels.
    clip_decoder's points above the other decoders' range are drawn as
    labelled arrows at the top instead of stretching the axis."""
    order = np.argsort(x_values)
    x_pos = np.arange(len(order))
    for d in decoders:
        y, lo, hi = (np.asarray(a, dtype=float)[order] for a in series[d])
        ax.errorbar(x_pos, y, yerr=np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)]),
                    marker='o', linestyle='--' if d in HATCHES else '-', color=COMPARISON_COLORS[d],
                    linewidth=1.5, markersize=5, capsize=2, elinewidth=1, label=display_label(d))
    ax.set_xticks(x_pos, [fmt(v) for v in np.asarray(x_values)[order]], rotation=90, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.02)
    if clip_decoder in decoders:
        others = [np.nanmax(np.asarray(series[d][2], dtype=float)) for d in decoders if d != clip_decoder]
        others = [h for h in others if np.isfinite(h)]
        if others:
            cap = max(others) * clip_margin
            ax.set_ylim(0, cap)
            color = COMPARISON_COLORS[clip_decoder]
            for xi, yi in zip(x_pos, np.asarray(series[clip_decoder][0], dtype=float)[order]):
                if np.isfinite(yi) and yi > cap:
                    ax.annotate(f"{clip_decoder.upper()}\n{yi:.0f}", xy=(xi, cap), xytext=(xi, cap * 0.90),
                                ha='center', va='top', fontsize=7, color=color, fontweight='bold',
                                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2))


def comparison_4x2_figure(combined, durations=None):
    """combined: {session: {decoder: metrics}}; durations: {session: {"Nmin":
    {decoder: metrics}}} or None. Sessions missing any decoder are left out
    of rows a-d and g-h so the paired tests compare the same sessions."""
    present = {d for entry in combined.values() for d in entry}
    decoders = [d for d in DECODER_ORDER if d in present]
    sessions = []
    for s in sorted(combined):
        missing = [d for d in decoders if d not in combined[s]]
        if missing:
            print(f"  [4x2] {s}: missing {missing}, left out")
        elif session_date(s) is None:
            print(f"  [4x2] {s}: no YYYYMMDD date in the session ID, left out")
        else:
            sessions.append(s)
    if not sessions:
        raise RuntimeError("No session has results for every decoder")

    rmse = {d: [combined[s][d]['rmse'] for s in sessions] for d in decoders}
    cc = {d: [average_cc(combined[s][d]) for s in sessions] for d in decoders}
    reference = min(decoders, key=lambda d: np.mean(rmse[d]))
    print(f"  [4x2] {len(sessions)} sessions; significance vs. {reference.upper()}")

    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    for col, (values, ylabel) in enumerate([(rmse, 'Average RMSE'), (cc, 'Average CC')]):
        others = [d for d in decoders if d != reference]
        p_adj = _holm([_wilcoxon_p(values[d], values[reference]) for d in others])
        sig = {reference: '', **{d: _stars(p) for d, p in zip(others, p_adj)}}
        _boxplot(axes[0, col], values, decoders, sig, ylabel)
        _pairwise_matrix(axes[1, col], values, decoders, higher_is_better=(col == 1))

    # e/f: accuracy vs. training duration, mean and 95% CI across sessions
    samples = {}   # (decoder, minutes) -> ([rmse], [cc])
    for per_tag in (durations or {}).values():
        for tag, entry in per_tag.items():
            for d, m in entry.items():
                r, c = samples.setdefault((d, float(tag[:-3])), ([], []))
                r.append(m['rmse'])
                c.append(average_cc(m))
    duration_decoders = [d for d in DECODER_ORDER if any(k[0] == d for k in samples)]
    minutes = sorted({k[1] for k in samples})
    for col, (idx, ylabel, clip) in enumerate([(0, 'RMSE', 'wf'), (1, 'Correlation', None)]):
        ax = axes[2, col]
        if not duration_decoders:
            ax.text(0.5, 0.5, 'No training-duration results', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            continue
        series = {}
        for d in duration_decoders:
            stats = [mean_ci(samples[(d, m)][idx]) if (d, m) in samples else (np.nan,) * 3 for m in minutes]
            series[d] = tuple(zip(*stats))
        # WF's unregularized short-duration fits can explode; clip them so
        # the other decoders stay readable (RMSE only; CC is bounded).
        _over_x(ax, minutes, series, duration_decoders, ylabel, 'Training Duration (min)',
                lambda x: f"{x:g}", clip_decoder=clip)

    # g/h: accuracy vs. days since the first session
    first = min(session_date(s) for s in sessions)
    days = [(session_date(s) - first).days for s in sessions]
    for col, (key, ylabel) in enumerate([('rmse', 'RMSE'), ('cc', 'Correlation')]):
        series = {}
        for d in decoders:
            entries = [combined[s][d] for s in sessions]
            if key == 'rmse':
                series[d] = ([e['rmse'] for e in entries], [e.get('rmse_ci_low', e['rmse']) for e in entries],
                             [e.get('rmse_ci_high', e['rmse']) for e in entries])
            else:
                mean = [average_cc(e) for e in entries]
                series[d] = (mean, [e.get('cc_ci_low', m) for e, m in zip(entries, mean)],
                             [e.get('cc_ci_high', m) for e, m in zip(entries, mean)])
        _over_x(axes[3, col], days, series, decoders, ylabel, 'Days Since Implantation',
                lambda x: str(int(round(x))))

    handles, labels = axes[3, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(decoders), bbox_to_anchor=(0.5, 1.01),
               frameon=False)
    for ax, letter in zip(axes.flat, 'abcdefgh'):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

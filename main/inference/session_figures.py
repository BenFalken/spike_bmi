"""
Per-session diagnostic figures (test_all_decoders.py --figures_dir).

Saved as {figures_dir}/{session}_{name}.png:
    comparison        predicted vs. true velocity over the test set
    trajectory_grid   integrated 2D hand path per ~1 s segment, each anchored
                      at the true start position
    velocity_grid     the same segments in velocity space (no integration,
                      so small errors are not smoothed away)
    rolling_rmse, cumulative_loss, rmse_bar, error_boxplot, scatter_r2,
    error_vs_speed
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'lstm': 'darkorange', 'qrnn': 'seagreen', 'kf': 'purple', 'wf': 'goldenrod',
          'snn': 'royalblue'}
STYLES = {'lstm': '--', 'qrnn': '--', 'kf': '-.', 'wf': '-.', 'snn': ':'}
STEP_S = 0.004


def reconstruct_path(anchor, velocities, step_time=STEP_S):
    """anchor (2,) followed by the cumulative integral of velocities (m, 2)."""
    return np.vstack([anchor, anchor + np.cumsum(np.asarray(velocities) * step_time, axis=0)])


def segment_grid_figure(y_common, pred_common, session, segment_samples=260, n_segments=16,
                        reconstruct=True):
    """Grid of the first n_segments segment_samples-long segments.

    reconstruct=True: y_common is position and predictions (velocity) are
    integrated from each segment's true start. False: y_common and the
    predictions are both velocity and are plotted directly."""
    n_segments = min(n_segments, len(y_common) // segment_samples)
    if n_segments == 0:
        print(f"  [skip] segment grid: fewer than {segment_samples} scored samples")
        return None
    space = 'position' if reconstruct else 'velocity'
    side = int(np.ceil(np.sqrt(n_segments)))
    fig, axes = plt.subplots(side, side, figsize=(3.2 * side, 3.2 * side))
    axes = np.atleast_1d(axes).flatten()
    for k, ax in enumerate(axes[:n_segments]):
        i0, i1 = k * segment_samples, (k + 1) * segment_samples - 1
        true = y_common[i0:i1 + 1]
        ax.plot(true[:, 0], true[:, 1], color='black', linewidth=1.6, label='true', zorder=5)
        for name, y_pred in pred_common.items():
            path = reconstruct_path(true[0], y_pred[i0:i1]) if reconstruct else y_pred[i0:i1 + 1]
            ax.plot(path[:, 0], path[:, 1], color=COLORS.get(name, 'gray'), linewidth=1.0,
                    alpha=0.85, label=name.upper())
        ax.scatter(*true[0], marker='s', s=28, color='black', zorder=6, label='segment start')
        ax.scatter(*true[-1], marker='o', s=40, facecolor='none', edgecolor='black',
                   linewidth=1.3, zorder=6, label=f'true {space} end')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"segment {k} ({i1 - i0 + 1} samples)", fontsize=8)
    for ax in axes[n_segments:]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{session}: true vs. decoded hand {space} per segment", fontsize=13, y=1.05)
    fig.legend(handles, labels, loc='upper center', ncol=min(len(labels), 8),
               bbox_to_anchor=(0.5, 1.00), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _save(fig, figures_dir, session, name):
    if fig is None:
        return
    path = os.path.join(figures_dir, f"{session}_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_session_figures(result, session, figures_dir, roll_window=20, speed_bins=8,
                         segment_samples=260, n_segments=16):
    """result: decoder_eval.evaluate_decoders() output for the full-data run."""
    os.makedirs(figures_dir, exist_ok=True)
    y_true, y_pos, pred = result['arrays']['y_true'], result['arrays']['y_pos'], result['arrays']['pred']
    metrics, names, n = result['metrics'], list(pred), len(y_true)
    sq_err = {name: (y - y_true) ** 2 for name, y in pred.items()}
    per_sample_rmse = {name: np.sqrt(e.mean(axis=1)) for name, e in sq_err.items()}
    xlabel = f"Test sample (from test row {result['start_raw']})"
    style = lambda name: dict(color=COLORS.get(name, 'gray'), linestyle=STYLES.get(name, '-'))

    _save(segment_grid_figure(y_pos, pred, session, segment_samples, n_segments), figures_dir,
          session, 'trajectory_grid')
    _save(segment_grid_figure(y_true, pred, session, segment_samples, n_segments, reconstruct=False),
          figures_dir, session, 'velocity_grid')

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for a, col, axis in zip(ax, [0, 1], 'xy'):
        a.plot(y_true[:, col], label='true', color='black', linewidth=1.2)
        for name, y in pred.items():
            a.plot(y[:, col], label=f"{name.upper()} pred", linewidth=0.9, alpha=0.85, **style(name))
        a.set_ylabel(f'velocity ({axis})')
        a.legend(loc='upper right', fontsize=7, ncol=2)
    ax[-1].set_xlabel(xlabel)
    ax[0].set_title(f'{session}: test-set predictions vs. true velocity')
    fig.tight_layout()
    _save(fig, figures_dir, session, 'comparison')

    fig, ax = plt.subplots(figsize=(12, 4))
    kernel = np.ones(roll_window) / roll_window
    for name, e in sq_err.items():
        rolling = np.full(n, np.nan)
        if n >= roll_window:
            rolling[roll_window - 1:] = np.sqrt(np.convolve(e.mean(axis=1), kernel, mode='valid'))
        ax.plot(rolling, label=name.upper(), **style(name))
    ax.set(xlabel=xlabel, ylabel=f'Rolling RMSE (window={roll_window})',
           title=f'{session}: decoder error over time')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    _save(fig, figures_dir, session, 'rolling_rmse')

    fig, ax = plt.subplots(figsize=(12, 4))
    for name, e in sq_err.items():
        ax.plot(np.sqrt(np.cumsum(e.mean(axis=1)) / np.arange(1, n + 1)), label=name.upper(),
                **style(name))
    ax.set(xlabel=xlabel, ylabel='Cumulative RMSE', title=f'{session}: cumulative loss')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    _save(fig, figures_dir, session, 'cumulative_loss')

    x_pos, width = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 4))
    ax.bar(x_pos - width / 2, [metrics[m]['rmse_x'] for m in names], width, label='RMSE (x)',
           color='steelblue')
    ax.bar(x_pos + width / 2, [metrics[m]['rmse_y'] for m in names], width, label='RMSE (y)',
           color='indianred')
    ax.set_xticks(x_pos, [m.upper() for m in names])
    ax.set(ylabel='RMSE', title=f'{session}: RMSE by decoder')
    ax.legend()
    fig.tight_layout()
    _save(fig, figures_dir, session, 'rmse_bar')

    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 4))
    ax.boxplot([per_sample_rmse[m] for m in names], tick_labels=[m.upper() for m in names],
               showfliers=False)
    ax.set(ylabel='Per-sample RMSE (x, y combined)', title=f'{session}: error distribution')
    fig.tight_layout()
    _save(fig, figures_dir, session, 'error_boxplot')

    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.4), squeeze=False)
    for j, name in enumerate(names):
        for row, axis in enumerate('xy'):
            a, t, p = axes[row][j], y_true[:, row], pred[name][:, row]
            a.scatter(t, p, s=4, alpha=0.3, color=COLORS.get(name, 'gray'))
            lims = [min(t.min(), p.min()), max(t.max(), p.max())]
            a.plot(lims, lims, color='black', linewidth=0.8, linestyle='--')
            a.set_title(f"{name.upper()} ({axis}), R²={metrics[name][f'r2_{axis}']:.3f}", fontsize=9)
            if row == 1:
                a.set_xlabel('true velocity')
            if j == 0:
                a.set_ylabel('predicted velocity')
    fig.suptitle(f'{session}: predicted vs. true velocity')
    fig.tight_layout()
    _save(fig, figures_dir, session, 'scatter_r2')

    speed = np.linalg.norm(y_true, axis=1)
    n_bins = min(speed_bins, max(3, n // 20))
    edges = np.quantile(speed, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.digitize(speed, edges[1:-1])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in names:
        binned = [per_sample_rmse[name][bin_idx == b].mean() if np.any(bin_idx == b) else np.nan
                  for b in range(n_bins)]
        ax.plot(0.5 * (edges[:-1] + edges[1:]), binned, marker='o', label=name.upper(), **style(name))
    ax.set(xlabel='True speed |velocity| (quantile bins)', ylabel='Mean per-sample RMSE',
           title=f'{session}: error vs. movement speed')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    _save(fig, figures_dir, session, 'error_vs_speed')

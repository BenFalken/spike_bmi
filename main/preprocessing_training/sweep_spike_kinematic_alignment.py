"""
Sweeps every session's real, already-processed {session}_binning.h5 file
under {root}/{experiment}/{subject}/mua/*_binning.h5 (experiment in
{bmi, hkm}), computes the same spike-vs-movement alignment diagnostics as
the original single-session version, and produces one aggregate figure:
a grid with one COLUMN per experiment and one ROW per subject-slot (2x2
for this project's current 2-subjects-per-experiment reality; grows
automatically if that changes) -- e.g.

    +---------- bmi ----------+---------- hkm ----------+
    |          indy           |         jenkins          |
    +--------------------------+--------------------------+
    |          loco            |        nitschke          |
    +--------------------------+--------------------------+

Each subplot shows that subject's lag-vs-correlation relationship as
overlaid, semi-transparent vertical bars -- one bar-set PER SESSION, all
drawn at alpha=0.4 in C0, so regions where multiple sessions agree
appear MORE saturated (a visual density/consensus cue) rather than as a
tangle of separate line curves, which reads as sparse and messy for
subjects with only 4-6 sessions. A dotted, fully-opaque mean-fit line
(C0, alpha=1.0) is overlaid on top of the bar cloud as the summary trend.

The underlying lag search itself still runs at the full --step-ms
resolution (e.g. 4ms) -- --bar-lag-step-ms only controls how coarsely
that fine-grained curve gets REBINNED for the bar plot's own display,
since raw 4ms-wide bars across a +/-1000ms window would be far too thin
to read.

Per-session numeric results are still saved to alignment_summary.csv.

Usage:
    python sweep_spike_kinematic_alignment.py \
        --root /users/bfalkenb/scratch/bfalkenb/data/dataset \
        --output-dir ./alignment_check
"""
import argparse
import csv
import glob
import os
import re

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NHP_NAMES = {
    0: {0: 'NHP-I', 1: 'NHP-L'},
    1: {0: 'NHP-J', 1: 'NHP-N'}
}

def smooth(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


def lag_correlation_search(a, b, max_lag_bins, step_ms):
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            aa, bb = a[:lag], b[-lag:]
        elif lag > 0:
            aa, bb = a[lag:], b[:-lag]
        else:
            aa, bb = a, b
        if len(aa) < 10:
            corrs.append(np.nan)
            continue
        corrs.append(np.corrcoef(aa, bb)[0, 1])
    corrs = np.array(corrs)
    best_idx = np.nanargmax(np.abs(corrs))
    return lags, corrs, lags[best_idx] * step_ms, corrs[best_idx]


def discover_sessions(root):
    """Returns {experiment: {subject: [(session_label, h5_path), ...]}}
    for every {root}/{experiment}/{subject}/mua/*_binning.h5 found."""
    out = {}
    for h5_path in sorted(glob.glob(os.path.join(root, "*", "*", "mua", "*_binning.h5"))):
        parts = h5_path.split(os.sep)
        experiment, subject = parts[-4], parts[-3]
        session_label = os.path.basename(h5_path)[:-len("_binning.h5")]
        out.setdefault(experiment, {}).setdefault(subject, []).append((session_label, h5_path))
    return out


def short_label(session_label):
    """Shortens a session label for display -- pulls out just the date
    if the label matches this project's own naming conventions
    (indy_20160407_02, sub-Nitschke_ses-20090910_...), else truncates."""
    m = re.search(r"(\d{8})", session_label)
    if m:
        return m.group(1)
    return session_label[:12]


def compute_session_alignment(h5_path, step_ms, smooth_ms, max_lag_ms):
    """Pure computation, no printing/plotting -- returns a dict of
    results for one session, reusable both per-session and for the
    sweep's own aggregation."""
    with h5py.File(h5_path, "r") as f:
        X_mua = f["X_mua"][()]
        y_task = f["y_task"][()]

    n_rows, n_channels = X_mua.shape
    total_spike_rate = X_mua.sum(axis=1)
    vel_x, vel_y = y_task[:, 2], y_task[:, 3]
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)

    smooth_window = max(1, int(round(smooth_ms / step_ms)))
    spike_smooth = smooth(total_spike_rate, smooth_window)
    speed_smooth = smooth(speed, smooth_window)
    zero_lag_corr = np.corrcoef(spike_smooth, speed_smooth)[0, 1]

    max_lag_bins = int(round(max_lag_ms / step_ms))
    lags, lag_corrs, best_lag_ms, best_lag_corr = lag_correlation_search(
        spike_smooth, speed_smooth, max_lag_bins, step_ms)

    silent_frac = float(np.mean(X_mua.sum(axis=0) == 0))
    still_frac = float(np.mean(speed < 1e-6))

    lag_is_far_from_zero = abs(best_lag_ms) > step_ms * 2
    lag_much_stronger = abs(best_lag_corr) > 2 * abs(zero_lag_corr) and abs(best_lag_corr) > 0.15
    if lag_much_stronger and lag_is_far_from_zero:
        diagnosis = "LIKELY_TIMING_OFFSET"
    elif abs(best_lag_corr) < 0.1:
        diagnosis = "LOW_EVERYWHERE"
    else:
        diagnosis = "OK"

    return {
        "h5_path": h5_path, "n_rows": n_rows, "n_channels": n_channels,
        "duration_min": n_rows * step_ms / 1000 / 60,
        "silent_channel_frac": silent_frac, "still_bin_frac": still_frac,
        "zero_lag_corr": float(zero_lag_corr),
        "best_lag_ms": float(best_lag_ms), "best_lag_corr": float(best_lag_corr),
        "diagnosis": diagnosis, "lags_ms": lags * step_ms, "lag_corrs": lag_corrs,
    }


def main(args):
    sessions_by_exp = discover_sessions(args.root)
    if not sessions_by_exp:
        raise SystemExit(f"No *_binning.h5 files found under {args.root}/*/*/mua/")

    os.makedirs(args.output_dir, exist_ok=True)
    results = []  # flat list of dicts, one per session, each tagged with experiment/subject/label

    for experiment, by_subject in sessions_by_exp.items():
        for subject, sessions in by_subject.items():
            print(f"=== {experiment}/{subject}: {len(sessions)} session(s) ===")
            for session_label, h5_path in sessions:
                try:
                    r = compute_session_alignment(h5_path, args.step_ms, args.smooth_ms, args.max_lag_ms)
                except Exception as e:
                    print(f"  ERROR on {session_label}: {e} -- skipping")
                    continue
                r["experiment"], r["subject"], r["session_label"] = experiment, subject, session_label
                results.append(r)
                print(f"  {session_label}: zero_lag={r['zero_lag_corr']:.3f}, "
                      f"best_lag={r['best_lag_ms']:.0f}ms (r={r['best_lag_corr']:.3f}), "
                      f"diagnosis={r['diagnosis']}")

    if not results:
        raise SystemExit("Every discovered session errored out -- nothing to plot.")

    # --- Save per-session numeric results, so this doesn't only live in one figure ---
    csv_path = os.path.join(args.output_dir, "alignment_summary.csv")
    csv_fields = ["experiment", "subject", "session_label", "n_rows", "n_channels", "duration_min",
                  "silent_channel_frac", "still_bin_frac", "zero_lag_corr", "best_lag_ms",
                  "best_lag_corr", "diagnosis"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_fields})
    print(f"\nSaved per-session results to {csv_path}")

    # --- Aggregate figure: one column per experiment, one row per
    # subject-slot, each subplot a semi-transparent bar overlay + dotted
    # mean-fit line ---
    experiments = sorted({r["experiment"] for r in results})
    subjects_by_exp = {
        exp: sorted({r["subject"] for r in results if r["experiment"] == exp})
        for exp in experiments
    }
    n_rows = max(len(v) for v in subjects_by_exp.values())
    n_cols = len(experiments)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.25 * 3 * n_cols, 1.25 * 2 * n_rows), squeeze=False)

    bar_step = args.bar_lag_step_ms
    for col, experiment in enumerate(experiments):
        for row, subject in enumerate(subjects_by_exp[experiment]):
            ax = axes[row][col]
            subj_results = [r for r in results if r["experiment"] == experiment and r["subject"] == subject]

            fine_lags_ms = subj_results[0]["lags_ms"]
            bin_edges = np.arange(fine_lags_ms.min(), fine_lags_ms.max() + bar_step, bar_step)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            all_binned = []

            alph = float(1.0/len(subj_results))

            for r in subj_results:
                binned = np.full(len(bin_centers), np.nan)
                for i in range(len(bin_centers)):
                    mask = (fine_lags_ms >= bin_edges[i]) & (fine_lags_ms < bin_edges[i + 1])
                    if mask.any():
                        binned[i] = np.nanmean(r["lag_corrs"][mask])
                all_binned.append(binned)
                # Each session's own bar set, alpha=0.4 -- overlapping
                # sessions at the same lag bin visually darken where
                # they agree, rather than needing a separate legend
                # entry per session.
                ax.bar(bin_centers, binned, width=bar_step * 0.9, color="C0", alpha=alph, linewidth=0)

            mean_curve = np.nanmean(np.vstack(all_binned), axis=0)
            ax.plot(bin_centers, mean_curve, color="C0", alpha=1.0, linestyle=":", linewidth=2.2, label=f"{len(subj_results)} sessions")
            ax.legend(loc='upper right', fontsize='x-small')
            ax.axvline(0, color="gray", linestyle="--", linewidth=1)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            if row == 0:
                ax.set_title(f"Dataset {col+1}")
            ax.set_xlabel("Lag (ms)")
            sub = NHP_NAMES[col][row]
            ax.set_ylabel(f"{sub}\nCorrelation")

        # Blank any unused cells if this experiment has fewer subjects
        # than the tallest column
        for row in range(len(subjects_by_exp[experiment]), n_rows):
            axes[row][col].axis("off")

    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, "alignment_aggregate.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved aggregate figure to {fig_path}")

    n_offset = sum(1 for r in results if r["diagnosis"] == "LIKELY_TIMING_OFFSET")
    n_low = sum(1 for r in results if r["diagnosis"] == "LOW_EVERYWHERE")
    print(f"\n{len(results)} session(s) total: {n_offset} likely timing offset, "
          f"{n_low} low correlation everywhere (not a simple offset), "
          f"{len(results) - n_offset - n_low} look OK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/users/bfalkenb/scratch/bfalkenb/data/dataset",
                         help="Root containing {experiment}/{subject}/mua/*_binning.h5")
    parser.add_argument("--output-dir", type=str, default="./alignment_check")
    parser.add_argument("--step-ms", type=float, default=4.0)
    parser.add_argument("--smooth-ms", type=float, default=200.0)
    parser.add_argument("--max-lag-ms", type=float, default=1000.0)
    parser.add_argument("--bar-lag-step-ms", type=float, default=40.0,
                         help="Bin width (ms) for rebinning the lag axis specifically for the "
                              "bar plot's own display (default 40ms) -- the underlying lag search "
                              "still runs at --step-ms resolution regardless.")
    args = parser.parse_args()
    main(args)

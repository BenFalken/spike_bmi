"""
Visualizes and quantifies how well a session's spike activity actually
lines up with its own hand-movement data -- built to investigate why
some HKM sessions produce good decoder RMSE (~50) while others produce
very poor RMSE (500+) with near-zero CC (~0.06-0.09), which is the
signature of a decoder that has learned essentially nothing.

Deliberately does TWO different things, not just one, because they point
to different root causes:
  1. Zero-lag correlation between spike activity and movement speed --
     low here just means "not much relationship," which could be a
     genuine alignment bug OR just a session with weak encoding.
  2. A LAG SEARCH across a real time window -- if correlation is much
     STRONGER at some nonzero lag than at zero lag, that's a direct,
     specific signature of a timing OFFSET (spikes and kinematics
     genuinely correspond, just shifted), not just noisy/weak data.
     Finding no strong correlation at ANY lag points away from a simple
     offset and toward a different problem (data quality, a corrupted
     segment, wrong channels, etc).

Reads directly from the real, already-processed {session}_binning.h5
file -- the exact same file eval_kf_decoder.py/eval_wf_decoder.py
actually train on, so this is diagnosing the SAME data the poor RMSE
came from, not a re-derived proxy.

Usage:
    python inspect_spike_kinematic_alignment.py \
        --h5-path /users/bfalkenb/scratch/bfalkenb/data/dataset/hkm/nitschke/sub-Nitschke_ses-20100923_behavior+ecephys_binning.h5 \
        --output-dir ./alignment_check --session-label 20100923

    # Compare directly against a known-good session:
    python inspect_spike_kinematic_alignment.py \
        --h5-path /path/to/some_good_session_binning.h5 \
        --output-dir ./alignment_check --session-label some_good_session
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main(args):
    with h5py.File(args.h5_path, "r") as f:
        X_mua = f["X_mua"][()]    # (n_rows, n_channels)
        y_task = f["y_task"][()]  # (n_rows, 6): pos_x, pos_y, vel_x, vel_y, acc_x, acc_y

    n_rows, n_channels = X_mua.shape
    step_s = args.step_ms / 1000.0
    time_axis = np.arange(n_rows) * step_s

    total_spike_rate = X_mua.sum(axis=1)
    vel_x, vel_y = y_task[:, 2], y_task[:, 3]
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)

    smooth_window = max(1, int(round(args.smooth_ms / args.step_ms)))
    spike_smooth = smooth(total_spike_rate, smooth_window)
    speed_smooth = smooth(speed, smooth_window)
    zero_lag_corr = np.corrcoef(spike_smooth, speed_smooth)[0, 1]

    silent_frac = np.mean(X_mua.sum(axis=0) == 0)
    still_frac = np.mean(speed < 1e-6)

    print(f"=== {args.h5_path} ===")
    print(f"  n_rows={n_rows}, n_channels={n_channels}, duration={n_rows * step_s / 60:.1f} min")
    print(f"  Total spike rate/bin: mean={total_spike_rate.mean():.3f}, std={total_spike_rate.std():.3f}, "
          f"max={total_spike_rate.max():.1f}")
    print(f"  {silent_frac * 100:.1f}% of channels are ENTIRELY SILENT (zero spikes, whole session)")
    print(f"  Movement speed: mean={speed.mean():.3f}, std={speed.std():.3f}, "
          f"{still_frac * 100:.1f}% of bins have ~zero speed")
    print(f"  Zero-lag spike-rate vs. speed correlation (smoothed {args.smooth_ms:.0f}ms): {zero_lag_corr:.4f}")

    max_lag_bins = int(round(args.max_lag_ms / args.step_ms))
    lags, lag_corrs, best_lag_ms, best_lag_corr = lag_correlation_search(
        spike_smooth, speed_smooth, max_lag_bins, args.step_ms)
    print(f"  Best correlation over +/-{args.max_lag_ms:.0f}ms lag search: {best_lag_corr:.4f} "
          f"at lag={best_lag_ms:.0f}ms (positive = spikes shifted LATER than movement)")

    print()
    lag_is_far_from_zero = abs(best_lag_ms) > args.step_ms * 2
    lag_much_stronger = abs(best_lag_corr) > 2 * abs(zero_lag_corr) and abs(best_lag_corr) > 0.15
    if lag_much_stronger and lag_is_far_from_zero:
        print(f"  DIAGNOSIS: correlation is much stronger at a nonzero lag ({best_lag_ms:.0f}ms, "
              f"r={best_lag_corr:.3f}) than at zero lag (r={zero_lag_corr:.3f}) -- this is a direct, "
              f"specific signature of a TIME-ALIGNMENT OFFSET between the spike stream and the "
              f"kinematic stream, not just noisy data.")
    elif abs(best_lag_corr) < 0.1:
        print(f"  DIAGNOSIS: correlation stays low across the ENTIRE lag search, not just at zero lag -- "
              f"this points AWAY FROM a simple timing offset and toward something else: a corrupted/"
              f"misassigned data segment, wrong channel mapping, or this session genuinely having very "
              f"weak movement-related modulation. A timing shift alone would not explain this.")
    else:
        print(f"  DIAGNOSIS: zero-lag correlation is reasonably strong already ({zero_lag_corr:.3f}) -- "
              f"no clear sign of a major alignment problem in this specific check.")

    os.makedirs(args.output_dir, exist_ok=True)
    label = args.session_label or os.path.basename(args.h5_path)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(time_axis / 60, total_spike_rate, color="steelblue", alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel("Total spike count/bin", color="steelblue")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_title(f"{label}: spike activity vs. movement speed, full session")
    ax0b = axes[0].twinx()
    ax0b.plot(time_axis / 60, speed, color="darkorange", alpha=0.7, linewidth=0.5)
    ax0b.set_ylabel("Movement speed", color="darkorange")

    zoom_bins = min(int(round(args.zoom_seconds * 1000 / args.step_ms)), n_rows)
    axes[1].plot(time_axis[:zoom_bins], total_spike_rate[:zoom_bins], color="steelblue")
    axes[1].set_ylabel("Total spike count/bin", color="steelblue")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"First {args.zoom_seconds:.0f}s, zoomed")
    ax1b = axes[1].twinx()
    ax1b.plot(time_axis[:zoom_bins], speed[:zoom_bins], color="darkorange")
    ax1b.set_ylabel("Movement speed", color="darkorange")

    axes[2].plot(lags * args.step_ms, lag_corrs, color="seagreen")
    axes[2].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[2].axvline(best_lag_ms, color="crimson", linestyle=":", linewidth=1.5,
                     label=f"best: {best_lag_corr:.3f} @ {best_lag_ms:.0f}ms")
    axes[2].set_xlabel("Lag (ms) -- positive = spikes shifted later relative to movement")
    axes[2].set_ylabel("Correlation")
    axes[2].set_title("Cross-correlation vs. lag")
    axes[2].legend()

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, f"{label}_alignment.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./alignment_check")
    parser.add_argument("--session-label", type=str, default=None)
    parser.add_argument("--step-ms", type=float, default=4.0)
    parser.add_argument("--smooth-ms", type=float, default=200.0,
                         help="Smoothing window before correlating (default 200ms) -- raw "
                              "4ms-bin spike counts are too noisy to correlate meaningfully bin-by-bin")
    parser.add_argument("--max-lag-ms", type=float, default=1000.0)
    parser.add_argument("--zoom-seconds", type=float, default=30.0)
    args = parser.parse_args()
    main(args)

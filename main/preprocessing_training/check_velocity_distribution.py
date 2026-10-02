"""
Distribution diagnostic for velocity targets, per-session and combined
across sessions -- used to choose a correct scaling formula for the SNN's
[0,1]-bounded population-vector readout (see dataset.py's CustomDataset,
currently 0.5*(v/150+1), which is known to be wrong: real velocities for
at least one session range roughly -430 to +569, far outside the assumed
+/-150, and the formula's single `scale` constant can't represent an
asymmetric range anyway).

Reads the RAW (pre-scaling) velocity directly from each session's exported
SNN .pkl trial files (data['velocity'], i.e. y_trace -- the full per-bin
trace within each trial, not just one value per trial), flattening every
timestep of every trial into one big per-session sample pool. This is
exactly the data CustomDataset.__getitem__ scales -- using it directly
means these stats describe precisely what the scaling formula needs to
handle, not a re-derived proxy.

Usage:
    # Explicit session list:
    python check_velocity_distribution.py \
        --sessions-root ./datasets/bmi \
        --sessions indy_20160407_02 indy_20160627_01

    # Auto-discover every session present:
    python check_velocity_distribution.py --sessions-root ./datasets/bmi --all
"""

import argparse
import glob
import os
import pickle as pkl

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_session_velocities(session_dir, splits=("train", "test")):
    """Flatten every timestep's velocity, from every trial, in the given
    splits of one session, into (vx, vy) arrays."""
    vx_all, vy_all = [], []
    for split in splits:
        split_dir = os.path.join(session_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fpath in sorted(glob.glob(os.path.join(split_dir, "*.pkl"))):
            with open(fpath, "rb") as f:
                data = pkl.load(f)
            v = np.asarray(data["velocity"])   # (nperseg, 2)
            vx_all.append(v[:, 0])
            vy_all.append(v[:, 1])
    if not vx_all:
        return np.array([]), np.array([])
    return np.concatenate(vx_all), np.concatenate(vy_all)


def summarize(arr, label):
    """Print mean/median/std/percentile summary, return the same as a dict."""
    stats = {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p0_5": float(np.percentile(arr, 0.5)),
        "p99_5": float(np.percentile(arr, 99.5)),
    }
    print(f"  {label}: n={stats['n']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
          f"std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}, "
          f"[0.5,99.5]pct=[{stats['p0_5']:.2f}, {stats['p99_5']:.2f}]")
    return stats


def compute_scaling_bounds(v_pooled, lo_pct=0.5, hi_pct=99.5, margin=0.05):
    """Robust percentile-based scaling bounds, with headroom built in.

    Returns (v_lo, v_hi, margin) such that
        scaled = margin + (1 - 2*margin) * (v - v_lo) / (v_hi - v_lo)
    maps the [lo_pct, hi_pct] percentile range of v_pooled to
    [margin, 1-margin] (NOT the literal [0,1] extremes) -- see module
    docstring for why the readout architecture makes the literal
    boundary values effectively unreachable, so leaving headroom rather
    than targeting them is deliberate, not conservative-for-no-reason.
    """
    v_lo, v_hi = np.percentile(v_pooled, [lo_pct, hi_pct])
    return float(v_lo), float(v_hi), margin


def apply_scaling(v, v_lo, v_hi, margin):
    return margin + (1 - 2 * margin) * (np.asarray(v) - v_lo) / (v_hi - v_lo)


def main(args):
    if args.all:
        sessions = sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(args.sessions_root, "*"))
            if os.path.isdir(p) and (os.path.isdir(os.path.join(p, "train"))
                                      or os.path.isdir(os.path.join(p, "test")))
        )
    else:
        sessions = args.sessions

    if not sessions:
        raise SystemExit(f"No sessions found under {args.sessions_root} "
                          f"(pass --sessions explicitly or use --all)")

    print(f"Loading {len(sessions)} session(s)...")
    per_session = {}
    for session in sessions:
        session_dir = os.path.join(args.sessions_root, session)
        vx, vy = load_session_velocities(session_dir)
        if len(vx) == 0:
            print(f"  [skip] {session}: no .pkl files found under {session_dir}")
            continue
        per_session[session] = (vx, vy)

    if not per_session:
        raise SystemExit("No velocity data loaded for any session -- check --sessions-root/--sessions")

    # --- Per-session stats ---
    print("\n=== Per-session stats ===")
    all_stats = {}
    for session, (vx, vy) in per_session.items():
        print(f"\n{session}:")
        all_stats[session] = {
            "vx": summarize(vx, "vx"),
            "vy": summarize(vy, "vy"),
        }

    # --- Combined (pooled across all sessions) stats ---
    print("\n=== Combined (all sessions pooled) ===")
    vx_combined = np.concatenate([v[0] for v in per_session.values()])
    vy_combined = np.concatenate([v[1] for v in per_session.values()])
    combined_stats = {
        "vx": summarize(vx_combined, "vx (combined)"),
        "vy": summarize(vy_combined, "vy (combined)"),
    }

    # --- How much do sessions actually differ? Directly informs the
    # global-vs-per-session bounds decision (see accompanying discussion) ---
    print("\n=== Per-session range spread (helps decide global vs. per-session bounds) ===")
    session_ranges = {s: (st["vx"]["p0_5"], st["vx"]["p99_5"]) for s, st in all_stats.items()}
    lo_spread = np.ptp([r[0] for r in session_ranges.values()])
    hi_spread = np.ptp([r[1] for r in session_ranges.values()])
    print(f"  vx [0.5,99.5]pct lower bound varies by {lo_spread:.1f} across sessions")
    print(f"  vx [0.5,99.5]pct upper bound varies by {hi_spread:.1f} across sessions")
    print("  (large spread here -> per-session bounds would use each session's dynamic "
          "range more fully; small spread -> global bounds cost little and add consistency)")

    # --- Recommended scaling bounds (combined, both axes pooled together
    # so vx/vy share one scale -- change if the plot shows they shouldn't) ---
    print(f"\n=== Recommended scaling bounds (global, {args.lo_pct}/{args.hi_pct} percentile, "
          f"margin={args.margin}) ===")
    v_both_combined = np.concatenate([vx_combined, vy_combined])
    v_lo, v_hi, margin = compute_scaling_bounds(v_both_combined, args.lo_pct, args.hi_pct, args.margin)
    print(f"  v_lo={v_lo:.2f}, v_hi={v_hi:.2f}, margin={margin}")
    print(f"  Formula: scaled = {margin} + {1 - 2*margin:.2f} * (v - {v_lo:.2f}) / ({v_hi:.2f} - {v_lo:.2f})")
    print(f"  Inverse (for eval-side denormalization): "
          f"v = {v_lo:.2f} + ({v_hi:.2f} - {v_lo:.2f}) * (scaled - {margin}) / {1 - 2*margin:.2f}")
    frac_clipped = np.mean((v_both_combined < v_lo) | (v_both_combined > v_hi))
    print(f"  {frac_clipped:.2%} of samples fall outside [v_lo, v_hi] and will clip at the boundary")

    # --- Plots ---
    n_sessions = len(per_session)
    fig, axes = plt.subplots(n_sessions + 1, 2, figsize=(12, 3 * (n_sessions + 1)), squeeze=False)

    for i, (session, (vx, vy)) in enumerate(per_session.items()):
        axes[i][0].hist(vx, bins=100, color="steelblue", alpha=0.8)
        axes[i][0].set_title(f"{session}: vx (mean={all_stats[session]['vx']['mean']:.1f}, "
                              f"std={all_stats[session]['vx']['std']:.1f})")
        axes[i][1].hist(vy, bins=100, color="darkorange", alpha=0.8)
        axes[i][1].set_title(f"{session}: vy (mean={all_stats[session]['vy']['mean']:.1f}, "
                              f"std={all_stats[session]['vy']['std']:.1f})")

    axes[n_sessions][0].hist(vx_combined, bins=100, color="steelblue", alpha=0.8)
    axes[n_sessions][0].set_title(f"COMBINED: vx (mean={combined_stats['vx']['mean']:.1f}, "
                                   f"std={combined_stats['vx']['std']:.1f})")
    axes[n_sessions][1].hist(vy_combined, bins=100, color="darkorange", alpha=0.8)
    axes[n_sessions][1].set_title(f"COMBINED: vy (mean={combined_stats['vy']['mean']:.1f}, "
                                   f"std={combined_stats['vy']['std']:.1f})")

    for row in axes:
        for a in row:
            a.axvline(v_lo, color="red", linestyle="--", linewidth=1, alpha=0.6)
            a.axvline(v_hi, color="red", linestyle="--", linewidth=1, alpha=0.6)

    fig.tight_layout()
    out_path = args.output or "velocity_distributions.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-root", type=str, required=True,
                         help="Directory containing one subfolder per session, "
                              "each with train/ and test/ .pkl files, e.g. ./datasets/bmi")
    parser.add_argument("--sessions", type=str, nargs="+", default=[],
                         help="Explicit list of session names (subfolder names under --sessions-root)")
    parser.add_argument("--all", action="store_true",
                         help="Auto-discover every session under --sessions-root")
    parser.add_argument("--lo-pct", type=float, default=0.5,
                         help="Lower percentile for the robust scaling bound (default 0.5)")
    parser.add_argument("--hi-pct", type=float, default=99.5,
                         help="Upper percentile for the robust scaling bound (default 99.5)")
    parser.add_argument("--margin", type=float, default=0.05,
                         help="Map the percentile range to [margin, 1-margin] rather than "
                              "the literal [0,1] -- see module docstring for why (default 0.05)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)

"""
Computes velocity scaling bounds (v_lo, v_hi, margin) for every
experiment/subject pair found under --snn-datasets-root, and writes them
out as one JSON mapping.

Deliberately reuses check_velocity_distribution.py's own functions
directly (load_session_velocities, compute_scaling_bounds) rather than
reimplementing the same percentile logic a second time -- this
guarantees every subject's bounds are derived by the EXACT SAME
methodology that originally produced Indy's existing constants
(0.5/99.5 percentile, vx and vy pooled together into one shared bound,
margin=0.05), not a similar-but-subtly-different one. If
check_velocity_distribution.py's own method ever changes, this script
picks that change up automatically rather than silently drifting out of
sync with it.

Per-subject, not per-experiment or global: each experiment/subject pair
gets its own (v_lo, v_hi) pair, pooling every session THAT SUBJECT has --
see the accompanying discussion for why (different subjects' actual
velocity ranges differ, and nothing in this project pools SNN training
data across subjects, so there's no structural reason to force a shared
scale). "bmi" and "hkm" are naturally separate too, being different
apparatus/tasks entirely.

Directory layout assumed: {snn_datasets_root}/{experiment}/{subject}/...
with individual sessions (each holding train/ and/or test/ *.pkl files)
somewhere beneath that -- discovered recursively at whatever depth they
actually sit at (e.g. directly under subject, or nested one level deeper
under a group-size directory like mua_8_group/), rather than a single,
fixed, assumed depth.

Usage:
    python compute_velocity_scalers.py \
        --snn-datasets-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets \
        --output velocity_scalers.json

    # Narrow to specific experiments/subjects rather than discovering all:
    python compute_velocity_scalers.py \
        --snn-datasets-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets \
        --experiments hkm --subjects jenkins nitschke \
        --output ../snn_training/velocity_scalers_hkm.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_velocity_distribution import load_session_velocities, compute_scaling_bounds, summarize


def find_session_dirs(subject_dir):
    """Recursively finds every directory under subject_dir that directly
    holds a train/ or test/ subfolder with at least one .pkl file in it
    -- i.e. every actual session directory, regardless of how many
    intermediate levels (e.g. a group-size directory) sit between the
    subject root and the session itself."""
    session_dirs = []
    for root, dirnames, _ in os.walk(subject_dir):
        for split in ("train", "test"):
            split_dir = os.path.join(root, split)
            if os.path.isdir(split_dir) and any(f.endswith(".pkl") for f in os.listdir(split_dir)):
                session_dirs.append(root)
                break
    return sorted(set(session_dirs))


def compute_subject_bounds(subject_dir, lo_pct, hi_pct, margin, verbose=True):
    """Pools every session under subject_dir together (vx and vy from
    every session, every timestep, every trial, all combined) and
    returns (v_lo, v_hi, margin, diagnostics) -- diagnostics includes
    per-session spread, matching check_velocity_distribution.py's own
    "how much do sessions actually differ" reporting, so the per-subject
    choice stays informed rather than opaque."""
    session_dirs = find_session_dirs(subject_dir)
    if not session_dirs:
        return None

    per_session_ranges = {}
    vx_all, vy_all = [], []
    for session_dir in session_dirs:
        session_id = os.path.relpath(session_dir, subject_dir)
        vx, vy = load_session_velocities(session_dir)
        if len(vx) == 0:
            continue
        vx_all.append(vx)
        vy_all.append(vy)
        per_session_ranges[session_id] = (
            float(np.percentile(vx, lo_pct)), float(np.percentile(vx, hi_pct)))

    if not vx_all:
        return None

    vx_combined = np.concatenate(vx_all)
    vy_combined = np.concatenate(vy_all)
    v_both_combined = np.concatenate([vx_combined, vy_combined])

    v_lo, v_hi, margin = compute_scaling_bounds(v_both_combined, lo_pct, hi_pct, margin)
    frac_clipped = float(np.mean((v_both_combined < v_lo) | (v_both_combined > v_hi)))

    if len(per_session_ranges) > 1:
        lo_spread = float(np.ptp([r[0] for r in per_session_ranges.values()]))
        hi_spread = float(np.ptp([r[1] for r in per_session_ranges.values()]))
    else:
        lo_spread = hi_spread = 0.0

    if verbose:
        print(f"    {len(session_dirs)} session(s), {len(v_both_combined)} pooled samples "
              f"(vx+vy combined)")
        print(f"    v_lo={v_lo:.2f}, v_hi={v_hi:.2f}, margin={margin}, "
              f"{frac_clipped:.2%} of samples clip")
        print(f"    per-session [{lo_pct},{hi_pct}]pct spread: lower varies by {lo_spread:.1f}, "
              f"upper varies by {hi_spread:.1f}")

    return {
        "v_lo": v_lo,
        "v_hi": v_hi,
        "margin": margin,
        "n_sessions": len(session_dirs),
        "n_samples": int(len(v_both_combined)),
        "frac_clipped": frac_clipped,
        "per_session_spread": {"lo": lo_spread, "hi": hi_spread},
    }


def main(args):
    root = args.snn_datasets_root
    if not os.path.isdir(root):
        raise SystemExit(f"--snn-datasets-root does not exist: {root}")

    experiments = args.experiments or sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    mapping = {}
    for experiment in experiments:
        experiment_dir = os.path.join(root, experiment)
        if not os.path.isdir(experiment_dir):
            print(f"[skip] experiment={experiment}: {experiment_dir} does not exist")
            continue

        subjects = args.subjects or sorted(
            d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d)))

        mapping[experiment] = {}
        for subject in subjects:
            subject_dir = os.path.join(experiment_dir, subject)
            if not os.path.isdir(subject_dir):
                print(f"[skip] {experiment}/{subject}: {subject_dir} does not exist")
                continue

            print(f"\n{experiment}/{subject}:")
            result = compute_subject_bounds(subject_dir, args.lo_pct, args.hi_pct, args.margin)
            if result is None:
                print(f"    [skip] no session data found under {subject_dir}")
                continue
            mapping[experiment][subject] = result

    print("\n=== Summary ===")
    for experiment, subjects in mapping.items():
        for subject, bounds in subjects.items():
            print(f"  {experiment}/{subject}: v_lo={bounds['v_lo']:.2f}, "
                  f"v_hi={bounds['v_hi']:.2f}, margin={bounds['margin']} "
                  f"({bounds['n_sessions']} sessions, {bounds['frac_clipped']:.2%} clipped)")

    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snn-datasets-root", type=str, required=True,
                         help="Root containing {experiment}/{subject}/... "
                              "e.g. /users/bfalkenb/scratch/bfalkenb/data/snn_datasets")
    parser.add_argument("--experiments", type=str, nargs="+", default=[],
                         help="Explicit experiment list (default: auto-discover every "
                              "subdirectory of --snn-datasets-root)")
    parser.add_argument("--subjects", type=str, nargs="+", default=[],
                         help="Explicit subject list per experiment (default: auto-discover "
                              "every subdirectory of each experiment)")
    parser.add_argument("--lo-pct", type=float, default=0.5)
    parser.add_argument("--hi-pct", type=float, default=99.5)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="velocity_scalers.json")
    args = parser.parse_args()
    main(args)

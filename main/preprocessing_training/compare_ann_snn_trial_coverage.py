"""
Visualizes, per trial and aggregated per subject, how much of each raw
NWB trial's own data actually survives into the ANN dataset (4ms
stepping, no discarding) versus the SNN dataset (256ms non-overlapping
windows, whole-window-or-nothing -- a trial shorter than one window
contributes ZERO windows at all, and any trailing samples that don't
fill a full window are dropped).

Built specifically to inform whether the 256ms-window SNN approach is
worth keeping for HKM's own short, independent-trial structure, or
whether enough raw data is being lost that a variable-length-trial SNN
approach (sacrificing uniform batch sizes for full data coverage,
matching the ANN pipeline) is the better direction.

Reads three CONFIRMED, real schemas directly (not guessed):
  - Stage 1 raw per-trial (convert_nwb_trials_to_raw_h5.py's own output):
    {output_root}/raw_trials/{session}/{session}_trial{id:04d}.h5,
    'task_time' -- the actual raw kinematic sample timestamps, i.e. the
    ground-truth total duration for that trial, independent of either
    downstream pipeline's own processing.
  - Stage 2b ANN per-trial windowed (run_dense_windowing_for_all_trials.sh's
    own output): {output_root}/ann_windowed_per_trial/{session}/
    {session}_trial{id:04d}_{method}.h5, 'y_task' -- len(y_task) is this
    trial's own row count in the ANN pipeline, confirmed (via
    combine_trial_windows_to_ann_h5.py's own code) to include EVERY row
    from every trial, no discarding.
  - Stage 2a SNN per-trial windowed (run_windowing_for_all_trials.sh's own
    output): {output_root}/snn_windowed_per_trial/{session}/
    {session}_trial{id:04d}_snn.h5, 'X_raster' -- X_raster.shape[0] is
    this trial's own window count; each window covers T_BASE=65 raw
    samples (256ms at 4ms/step, confirmed elsewhere in this project).

Usage:
    python compare_ann_snn_trial_coverage.py \
        --output-root /users/bfalkenb/scratch/bfalkenb/data/hkm_purgatory \
        --sessions sub-Jenkins_ses-20090912_behavior+ecephys sub-Jenkins_ses-20090916_behavior+ecephys \
        --label jenkins --plot-dir ./coverage_check
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_BASE = 65  # raw samples per SNN window (256ms at 4ms/step) -- confirmed
# elsewhere in this project (check_datasets.py's own real output).

TRIAL_ID_PATTERN = re.compile(r"_trial(\d+)")
# Matches combine_trial_windows_to_ann_h5.py's own discovery pattern
# EXACTLY (r"_trial(\d+)_\w+\.h5$") -- that script globs *.h5 broadly and
# matches by trial ID via regex, NEVER assuming a specific method-tag
# string. An earlier version of this script instead GUESSED the ANN
# filename directly ({session}_trial{id:04d}_{method}.h5, method
# defaulting to "binning") -- when that guess didn't match the real
# filename for most trials, os.path.isfile() silently returned False and
# n_ann defaulted to 0, producing a real, misleading "ANN retains only
# 16%" result that had nothing to do with the actual data. Fixed by
# discovering the ANN file the same way that script's own, proven logic
# does: glob broadly, match by trial ID, don't assume the tag.
ANN_TRIAL_PATTERN = re.compile(r"_trial(\d+)_\w+\.h5$")
SNN_TRIAL_PATTERN = re.compile(r"_trial(\d+)_snn\.h5$")  # matches
# combine_trial_windows_to_session.py's own confirmed regex exactly


def find_trial_file(directory, trial_id, pattern):
    """Globs *.h5 in directory and returns the path whose trial ID
    (extracted via `pattern`) matches trial_id, or None if not found.
    Raises if MORE than one file matches (an ambiguous directory,
    better to fail loudly than silently pick one)."""
    if not os.path.isdir(directory):
        return None
    matches = []
    for path in glob.glob(os.path.join(directory, "*.h5")):
        m = pattern.search(os.path.basename(path))
        if m and int(m.group(1)) == trial_id:
            matches.append(path)
    if len(matches) > 1:
        raise RuntimeError(f"Trial {trial_id} matched {len(matches)} files in {directory}: "
                            f"{matches} -- ambiguous, refusing to silently pick one.")
    return matches[0] if matches else None


def discover_trial_ids(raw_dir):
    ids = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.h5"))):
        m = TRIAL_ID_PATTERN.search(os.path.basename(path))
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def per_trial_coverage(output_root, session_id, method="binning", step_ms=4.0):
    raw_dir = os.path.join(output_root, "raw_trials", session_id)
    ann_dir = os.path.join(output_root, "ann_windowed_per_trial", session_id)
    snn_dir = os.path.join(output_root, "snn_windowed_per_trial", session_id)

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"No raw_trials dir found for {session_id}: {raw_dir}")

    trial_ids = discover_trial_ids(raw_dir)
    if not trial_ids:
        raise FileNotFoundError(f"No trial files found under {raw_dir}")

    records = []
    n_ann_missing = 0
    n_snn_missing = 0
    for trial_id in trial_ids:
        raw_path = os.path.join(raw_dir, f"{session_id}_trial{trial_id:04d}.h5")
        ann_path = find_trial_file(ann_dir, trial_id, ANN_TRIAL_PATTERN)
        snn_path = find_trial_file(snn_dir, trial_id, SNN_TRIAL_PATTERN)

        if not os.path.isfile(raw_path):
            continue
        with h5py.File(raw_path, "r") as f:
            n_raw = len(f["task_time"][()])
        if n_raw == 0:
            continue

        n_ann = 0
        if ann_path is not None:
            with h5py.File(ann_path, "r") as f:
                n_ann = len(f["y_task"][()])
        else:
            n_ann_missing += 1

        n_snn = 0
        if snn_path is not None:
            with h5py.File(snn_path, "r") as f:
                n_snn = f["X_raster"].shape[0] * T_BASE
        else:
            n_snn_missing += 1

        records.append({
            "session_id": session_id, "trial_id": trial_id,
            "n_raw": n_raw, "n_ann": n_ann, "n_snn": n_snn,
            "duration_ms": n_raw * step_ms,
            "ann_coverage_frac": n_ann / n_raw if n_raw else 0.0,
            "snn_coverage_frac": n_snn / n_raw if n_raw else 0.0,
        })

    # Loud, not silent: a genuinely missing file (trial dropped during
    # windowing, e.g. too short even for the ANN side's own minimum
    # prior-history window) is real and fine to report as 0 coverage --
    # but if MOST trials are missing, that's very likely a path/pattern
    # problem again, not real data, and silently proceeding would
    # reproduce exactly the misleading result this rewrite was meant to
    # fix.
    if records:
        if n_ann_missing / len(records) > 0.5:
            print(f"  WARNING: {session_id}: {n_ann_missing}/{len(records)} trials have NO "
                  f"matching ANN file at all in {ann_dir} -- before trusting this session's "
                  f"ANN coverage numbers, confirm that directory actually contains this "
                  f"session's windowed files.")
        if n_snn_missing / len(records) > 0.5:
            print(f"  WARNING: {session_id}: {n_snn_missing}/{len(records)} trials have NO "
                  f"matching SNN file at all in {snn_dir} -- before trusting this session's "
                  f"SNN coverage numbers, confirm that directory actually contains this "
                  f"session's windowed files.")
    return records


def main(args):
    all_records = []
    for session_id in args.sessions:
        try:
            recs = per_trial_coverage(args.output_root, session_id, args.method, args.step_ms)
        except FileNotFoundError as e:
            print(f"  [skip] {session_id}: {e}")
            continue
        all_records.extend(recs)
        print(f"  {session_id}: {len(recs)} trial(s)")

    if not all_records:
        raise SystemExit("No trials found across any of the given sessions -- nothing to plot.")

    durations_ms = np.array([r["duration_ms"] for r in all_records])
    ann_frac = np.array([r["ann_coverage_frac"] for r in all_records])
    snn_frac = np.array([r["snn_coverage_frac"] for r in all_records])
    n_raw_total = sum(r["n_raw"] for r in all_records)
    n_ann_total = sum(r["n_ann"] for r in all_records)
    n_snn_total = sum(r["n_snn"] for r in all_records)
    n_zero_snn = sum(1 for r in all_records if r["n_snn"] == 0)

    print(f"\n{'='*72}\n{args.label} -- {len(all_records)} trial(s) across {len(args.sessions)} session(s)\n{'='*72}")
    print(f"  Total raw samples available:  {n_raw_total}")
    print(f"  ANN dataset retains: {n_ann_total} ({100*n_ann_total/n_raw_total:.1f}%)")
    print(f"  SNN dataset retains: {n_snn_total} ({100*n_snn_total/n_raw_total:.1f}%)")
    print(f"  Trials contributing ZERO SNN windows (shorter than one {T_BASE * args.step_ms:.0f}ms "
          f"window): {n_zero_snn} / {len(all_records)} ({100*n_zero_snn/len(all_records):.1f}%)")

    os.makedirs(args.plot_dir, exist_ok=True)

    # --- Figure 1: coverage fraction distribution, ANN vs SNN ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 21)
    ax.hist(ann_frac, bins=bins, alpha=0.6, label=f"ANN (mean={ann_frac.mean():.2f})", color="royalblue")
    ax.hist(snn_frac, bins=bins, alpha=0.6, label=f"SNN (mean={snn_frac.mean():.2f})", color="crimson")
    ax.set_xlabel("Fraction of trial's own raw data retained")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"{args.label}: per-trial data coverage, ANN vs. SNN ({len(all_records)} trials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "coverage_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(args.plot_dir, 'coverage_distribution.png')}")

    # --- Figure 2: coverage fraction vs. trial duration -- shows WHERE
    # the SNN's coverage loss concentrates (short trials specifically) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(durations_ms)
    ax.scatter(durations_ms[order], ann_frac[order], s=12, alpha=0.6, label="ANN", color="royalblue")
    ax.scatter(durations_ms[order], snn_frac[order], s=12, alpha=0.6, label="SNN", color="crimson")
    ax.axvline(T_BASE * args.step_ms, color="gray", linestyle="--", linewidth=1,
               label=f"one SNN window ({T_BASE * args.step_ms:.0f}ms)")
    ax.set_xlabel("Trial duration (ms)")
    ax.set_ylabel("Fraction of trial's own raw data retained")
    ax.set_title(f"{args.label}: coverage vs. trial duration ({len(all_records)} trials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "coverage_vs_duration.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(args.plot_dir, 'coverage_vs_duration.png')}")

    # --- Figure 3: trial duration distribution itself, with the
    # one-window threshold marked -- shows what fraction of trials are
    # simply too short to ever produce a full SNN window at all ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(durations_ms, bins=30, color="gray", alpha=0.7)
    ax.axvline(T_BASE * args.step_ms, color="crimson", linestyle="--", linewidth=1.5,
               label=f"one SNN window ({T_BASE * args.step_ms:.0f}ms)")
    ax.set_xlabel("Trial duration (ms)")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"{args.label}: trial duration distribution ({len(all_records)} trials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.plot_dir, "trial_duration_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {os.path.join(args.plot_dir, 'trial_duration_distribution.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, required=True,
                         help="run_nwb_pipeline.sh's own --output-root (contains "
                              "raw_trials/, ann_windowed_per_trial/, snn_windowed_per_trial/)")
    parser.add_argument("--sessions", type=str, nargs="+", required=True,
                         help="One or more session_ids (matching raw_trials/{session_id}/)")
    parser.add_argument("--method", type=str, default="binning",
                         help="ANN windowing method tag in the filename (default 'binning')")
    parser.add_argument("--step-ms", type=float, default=4.0)
    parser.add_argument("--label", type=str, default="subject")
    parser.add_argument("--plot-dir", type=str, default="./coverage_check")
    args = parser.parse_args()
    main(args)

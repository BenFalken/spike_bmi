"""
ANN-side counterpart to combine_trial_windows_to_session.py. Reads every
per-trial dense-windowed ANN output for one session (make_dataset.py's
own output, run unmodified once per trial file via
run_dense_windowing_for_all_trials.sh), concatenates them in
trial-chronological order into ONE combined h5 file matching the
existing X_sua/X_mua/y_task schema test_all_decoders.py already reads
via --input_filepath -- plus one new thing: an 'n_train' attr holding a
TRIAL-AWARE split boundary, which test_all_decoders.py now reads back
automatically (see its own run_session(), modified alongside this
script) instead of deriving one internally via compute_aligned_split().

WHY THIS NEEDS ITS OWN BOUNDARY, NOT compute_aligned_split(): that
function assumes ONE continuous, densely-windowed stream, reconstructing
the raw (unwindowed) session length as N + base_nperseg -- true for a
real single session, but not what a trial-concatenated file represents.
Splitting by naive row count here could put some of a trial's own
near-duplicate, densely-overlapping windows in train and the REST of
that same trial in test -- real information leakage between train and
test, not just an approximation error. This script's own split instead
finds the nearest whole-TRIAL boundary to the requested test_frac,
identical reasoning to combine_trial_windows_to_session.py's SNN-side
version, applied to row counts instead of window counts (ANN dense
windowing produces vastly more rows per trial than the SNN's
non-overlapping windowing, but the trial-boundary-safety principle is
unchanged).

HOW TRIAL IDENTITY SURVIVES WINDOWING: same situation as the SNN side --
make_dataset.py doesn't read or propagate any custom attrs from its
input file, so trial order is recovered from the FILENAME. Expects files
named "{session_id}_trial{trial_id:04d}_{method}.h5", matching
run_dense_windowing_for_all_trials.sh's own naming convention.

CLI usage:
    python combine_trial_windows_to_ann_h5.py \
        --windowed-dir ann_windowed_per_trial/nitschke_20100923 \
        --output-path datasets/bmi/nitschke_20100923_binning.h5 \
        --test_frac 0.1
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np

TRIAL_ID_PATTERN = re.compile(r"_trial(\d+)_\w+\.h5$")


def load_trial_windows(path):
    with h5py.File(path, "r") as f:
        return {
            "X_sua": f["X_sua"][()],
            "X_mua": f["X_mua"][()],
            "y_task": f["y_task"][()],
        }


def main(args):
    paths = sorted(glob.glob(os.path.join(args.windowed_dir, "*.h5")))
    trials = []
    for path in paths:
        m = TRIAL_ID_PATTERN.search(os.path.basename(path))
        if not m:
            print(f"  [skip] {path}: filename doesn't match "
                  f"'..._trial####_<method>.h5' -- can't recover trial order from it")
            continue
        trial_id = int(m.group(1))
        trials.append((trial_id, path))

    trials.sort(key=lambda t: t[0])  # chronological, since trial_id reflects NWB trials table order
    if not trials:
        raise FileNotFoundError(f"No '..._trial####_<method>.h5' files found under {args.windowed_dir}")

    print(f"Found {len(trials)} trial windowed file(s), trial_id range "
          f"[{trials[0][0]}, {trials[-1][0]}]")

    per_trial_data = []
    n_rows_per_trial = []
    for trial_id, path in trials:
        data = load_trial_windows(path)
        n_rows = len(data["y_task"])
        if n_rows == 0:
            print(f"  [skip] trial {trial_id}: zero rows")
            continue
        per_trial_data.append((trial_id, data))
        n_rows_per_trial.append(n_rows)

    total_rows = sum(n_rows_per_trial)
    print(f"Total rows across {len(per_trial_data)} non-empty trial(s): {total_rows}")

    # Trial-level split -- identical logic to combine_trial_windows_to_
    # session.py's own, applied to row counts (see module docstring).
    cumulative = np.cumsum(n_rows_per_trial)
    train_row_target = (1 - args.test_frac) * total_rows
    split_trial_idx = int(np.searchsorted(cumulative, train_row_target, side="left"))
    split_trial_idx = min(max(split_trial_idx, 0), len(per_trial_data) - 1)

    train_trials = per_trial_data[:split_trial_idx + 1]
    test_trials = per_trial_data[split_trial_idx + 1:]
    if not test_trials:
        test_trials = [train_trials[-1]]
        train_trials = train_trials[:-1]
        print("  NOTE: test_frac too small to naturally cross a trial boundary -- "
              "holding out the single last trial instead of producing an empty test set.")

    n_train = sum(len(d["y_task"]) for _, d in train_trials)
    n_test = sum(len(d["y_task"]) for _, d in test_trials)
    print(f"Split: {len(train_trials)} train trial(s) ({n_train} rows), "
          f"{len(test_trials)} test trial(s) ({n_test} rows)")
    print(f"  train trial_ids: {[t for t, _ in train_trials]}")
    print(f"  test trial_ids:  {[t for t, _ in test_trials]}")

    ordered_trials = train_trials + test_trials  # train block first, then test -- n_train indexes this directly
    X_sua = np.concatenate([d["X_sua"] for _, d in ordered_trials], axis=0)
    X_mua = np.concatenate([d["X_mua"] for _, d in ordered_trials], axis=0)
    y_task = np.concatenate([d["y_task"] for _, d in ordered_trials], axis=0)
    assert len(X_sua) == len(X_mua) == len(y_task) == total_rows

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with h5py.File(args.output_path, "w") as f:
        f["X_sua"] = X_sua
        f["X_mua"] = X_mua
        f["y_task"] = y_task
        f.attrs["n_train"] = n_train  # read automatically by test_all_decoders.py's run_session()

    print(f"\nSaved combined dataset to {args.output_path} "
          f"(X_sua={X_sua.shape}, X_mua={X_mua.shape}, y_task={y_task.shape}, n_train attr={n_train})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowed-dir", type=str, required=True,
                         help="Directory of per-trial dense-windowed ANN h5 files "
                              "(make_dataset.py output, one per trial)")
    parser.add_argument("--output-path", type=str, required=True,
                         help="Path to write the combined h5 file, e.g. "
                              "datasets/bmi/{session}_binning.h5 -- matching test_all_decoders.py's "
                              "own --input_filepath naming convention")
    parser.add_argument("--test_frac", type=float, default=0.1,
                         help="Fraction of rows (chronologically last, by whole trial) "
                              "held out as test")
    args = parser.parse_args()
    main(args)

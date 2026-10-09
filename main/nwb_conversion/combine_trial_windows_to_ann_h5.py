"""
Combine one HKM session's per-trial ANN datasets into the session's ANN
dataset, the {session}_binning.h5 that the KF, WF, LSTM and QRNN train on
and test_all_decoders.py evaluates.

Inputs: the per-trial raw files (convert_nwb_trials_to_raw_h5.py, --raw-dir)
and their dense-windowed datasets (make_dataset.py run once per trial by
run_dense_windowing_for_all_trials.sh, --windowed-dir, named
{session}_trial####_{method}.h5). A trial of T samples gives T - 65 rows;
row j scores raw sample j + 65 of its trial. Trials of 65 samples or fewer
give none (the SNN dataset still uses them).

The split is trial_split.py's, so the test trials are exactly those of the
SNN dataset (make_snn_dataset_whole_trial.py). Rows are written train trials
first, then test trials, each in trial order.

Output (--output-path):
    X_sua, X_mua, y_task   make_dataset.py's arrays, concatenated
    trial_id  (n_rows,)    the trial each row belongs to
    attrs: n_train         first test row (read as the train/test boundary by
                           single_subject_pipeline.py and test_all_decoders.py)
           test_frac, nperseg

Usage:
    python combine_trial_windows_to_ann_h5.py --raw-dir raw_trials/SESSION \
        --windowed-dir ann_windowed_per_trial/SESSION --output-path .../SESSION_binning.h5
"""

import argparse
import os

import h5py
import numpy as np

from trial_split import split_raw_trials

NPERSEG = 65   # 256 ms ANN window in 4 ms samples (run_dense_windowing_for_all_trials.sh)


def load_trial_rows(windowed_dir, raw_path, n_samples, method):
    """The trial's dense-windowed arrays, or None if it is too short for a
    window. Checks it has the expected n_samples - NPERSEG rows."""
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    path = os.path.join(windowed_dir, f"{stem}_{method}.h5")
    n_expected = max(0, n_samples - NPERSEG)
    if not os.path.exists(path):
        if n_expected:
            raise FileNotFoundError(f"{path} is missing, but its {n_samples}-sample trial should "
                                    f"give {n_expected} rows")
        return None
    with h5py.File(path, "r") as f:
        data = {key: f[key][()] for key in ("X_sua", "X_mua", "y_task")}
    if len(data["y_task"]) != n_expected:
        raise ValueError(f"{path}: {len(data['y_task'])} rows, expected {n_expected} for a "
                         f"{n_samples}-sample trial with {NPERSEG}-sample windows at 4 ms steps")
    return data


def main(args):
    train, test = split_raw_trials(args.raw_dir, args.test_frac)
    blocks, n_train = [], None
    for split, trials in (("train", train), ("test", test)):
        if split == "test":
            n_train = sum(len(b["y_task"]) for b in blocks)
        n_short = 0
        for trial_id, n_samples, raw_path in trials:
            data = load_trial_rows(args.windowed_dir, raw_path, n_samples, args.method)
            if data is None:
                n_short += 1
                continue
            data["trial_id"] = np.full(len(data["y_task"]), trial_id, dtype=np.int32)
            blocks.append(data)
        print(f"  {split}: {len(trials) - n_short} trial(s) with rows, {n_short} of <= {NPERSEG} "
              f"samples without")
    combined = {key: np.concatenate([b[key] for b in blocks], axis=0)
                for key in ("X_sua", "X_mua", "y_task", "trial_id")}
    if n_train == len(combined["y_task"]):
        raise ValueError("No test trial is long enough for an ANN row; raise --test_frac")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with h5py.File(args.output_path, "w") as f:
        for key, value in combined.items():
            f[key] = value
        f.attrs["n_train"] = n_train
        f.attrs["test_frac"] = args.test_frac
        f.attrs["nperseg"] = NPERSEG
    print(f"Saved {args.output_path}: X_mua {combined['X_mua'].shape}, {n_train} train rows, "
          f"{len(combined['y_task']) - n_train} test rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Per-trial raw files (convert_nwb_trials_to_raw_h5.py output)")
    parser.add_argument("--windowed-dir", type=str, required=True,
                        help="Per-trial dense-windowed datasets (make_dataset.py output)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Session dataset to write, .../dataset/hkm/<subject>/mua/<session>_<method>.h5")
    parser.add_argument("--method", type=str, default="binning")
    parser.add_argument("--test_frac", type=float, default=0.1,
                        help="Fraction of the session's samples (whole trials, the last ones) held out")
    main(parser.parse_args())

"""
The train/test split of a trial-structured (HKM) session, shared by
combine_trial_windows_to_ann_h5.py and make_snn_dataset_whole_trial.py so
that the ANN and SNN datasets hold out exactly the same trials.

Both read the per-trial raw files of convert_nwb_trials_to_raw_h5.py. Trials
are ordered by trial ID (chronological) and split whole, never mid-trial:
the chronologically first trials covering (1 - test_frac) of the session's
samples are the training set, the rest the test set.
"""

import glob
import os
import re

import h5py
import numpy as np

RAW_TRIAL_PATTERN = re.compile(r"_trial(\d+)\.h5$")


def read_raw_trials(raw_dir):
    """[(trial_id, n_samples, path)] of every {session}_trial####.h5 in
    raw_dir, sorted by trial ID."""
    trials = []
    for path in glob.glob(os.path.join(raw_dir, "*.h5")):
        match = RAW_TRIAL_PATTERN.search(os.path.basename(path))
        if not match:
            continue
        with h5py.File(path, "r") as f:
            n_samples = len(f["task_time"])
        trials.append((int(match.group(1)), n_samples, path))
    if not trials:
        raise FileNotFoundError(f"No '..._trial####.h5' files in {raw_dir}")
    return sorted(trials)


def n_train_trials(n_samples, test_frac):
    """Number of leading trials in the training set: the trial in which the
    cumulative sample count reaches (1 - test_frac) of the total goes to
    training. At least one trial is held out."""
    cumulative = np.cumsum(n_samples)
    n_train = int(np.searchsorted(cumulative, (1 - test_frac) * cumulative[-1], side="left")) + 1
    return min(max(n_train, 1), len(n_samples) - 1)


def split_raw_trials(raw_dir, test_frac):
    """(train_trials, test_trials), each a read_raw_trials() list."""
    trials = read_raw_trials(raw_dir)
    if len(trials) < 2:
        raise ValueError(f"{raw_dir} holds {len(trials)} trial(s); a train/test split needs two")
    n_train = n_train_trials([n for _, n, _ in trials], test_frac)
    train, test = trials[:n_train], trials[n_train:]
    n_test_samples = sum(n for _, n, _ in test)
    print(f"Trial split (test_frac={test_frac}): {len(train)} train trial(s), {len(test)} test trial(s) "
          f"({n_test_samples / sum(n for _, n, _ in trials):.1%} of samples); first test trial ID "
          f"{test[0][0]}")
    return train, test

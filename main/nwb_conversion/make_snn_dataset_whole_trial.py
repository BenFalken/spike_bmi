"""
Build one HKM session's SNN dataset: one .pkl per trial, covering the whole
trial at 4 ms per timestep (no windowing, no grouping).

HKM trials are short, separate reaches, and many are shorter than a 256 ms
window, so windowing them would discard data. Each trial is therefore one
variable-length training example; train_snn.py needs --batch-size 1 for
these datasets, and its model state is reset at the start of every trial.

Reads the per-trial raw files of convert_nwb_trials_to_raw_h5.py and splits
them with trial_split.py, so the test trials are exactly those of the ANN
dataset (combine_trial_windows_to_ann_h5.py). Spikes are binned against the
trial's own 4 ms sample times as in make_snn_dataset.py; velocity is the raw
file's task_data[:, 2:4].

Output: {dest_root}/{experiment}/{subject}/{feature}/{session}/{train,test}/{i}.pkl,
numbered in trial order, each
    input_spikes (n_units, T) float32, velocity (T, 2) float32, trial_id int

As a last check, the build fails if any velocity sample is faster than
--max-speed: the despiking in convert_nwb_trials_to_raw_h5.py should leave
none (see hkm_despike.py).

Usage:
    python make_snn_dataset_whole_trial.py --raw-dir raw_trials/SESSION --session-id SESSION \
        --dest-root .../snn_datasets --experiment hkm --subject jenkins
"""

import argparse
import os
import pickle as pkl

import h5py
import numpy as np

from hkm_despike import DEFAULT_MAX_SPEED
from trial_split import split_raw_trials


def bin_whole_trial(task_time, task_data, spike_trains):
    """(X_raster (n_units, T), velocity (T, 2)) for one trial: spike counts
    in 4 ms bins centred on the trial's own sample times."""
    dt = np.diff(task_time).mean()
    bin_edges = np.concatenate((task_time - dt / 2, [task_time[-1] + dt / 2]))
    X_raster = np.zeros((len(spike_trains), len(task_time)), dtype=np.float32)
    for u, spikes in enumerate(spike_trains):
        X_raster[u] = np.histogram(np.asarray(spikes), bin_edges)[0]
    return X_raster, task_data[:, 2:4].astype(np.float32)


def write_split(trials, split_dir, feature, max_speed):
    """Bin and save every trial; returns [(trial_id, max speed)] of trials
    faster than max_speed."""
    os.makedirs(split_dir, exist_ok=True)
    for name in os.listdir(split_dir):
        if name.endswith(".pkl"):
            os.remove(os.path.join(split_dir, name))
    too_fast = []
    for i, (trial_id, _, path) in enumerate(trials):
        with h5py.File(path, "r") as f:
            X_raster, velocity = bin_whole_trial(f["task_time"][()], f["task_data"][()],
                                                 f[f"{feature}_trains"][()])
        speed = float(np.linalg.norm(velocity, axis=1).max())
        if max_speed and speed > max_speed:
            too_fast.append((trial_id, speed))
        with open(os.path.join(split_dir, f"{i}.pkl"), "wb") as f:
            pkl.dump({"input_spikes": X_raster, "velocity": velocity, "trial_id": trial_id}, f)
    print(f"Wrote {len(trials)} trial(s) to {split_dir}")
    return too_fast


def main(args):
    train, test = split_raw_trials(args.raw_dir, args.test_frac)
    dest_dir = os.path.join(args.dest_root, args.experiment, args.subject, args.feature, args.session_id)
    too_fast = []
    for split, trials in (("train", train), ("test", test)):
        too_fast += write_split(trials, os.path.join(dest_dir, split), args.feature, args.max_speed)

    lengths = [n for _, n, _ in train + test]
    print(f"Trial length (timesteps): min {min(lengths)}, median {int(np.median(lengths))}, "
          f"max {max(lengths)}; train with --batch-size 1")
    if too_fast:
        worst = max(too_fast, key=lambda t: t[1])
        raise SystemExit(f"{len(too_fast)} trial(s) have velocity above --max-speed {args.max_speed:g} "
                         f"(fastest: trial {worst[0]}, {worst[1]:.0f} units/s); tracking glitches "
                         f"survived the despiking in convert_nwb_trials_to_raw_h5.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Per-trial raw files (convert_nwb_trials_to_raw_h5.py output)")
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--dest-root", type=str, required=True, help=".../snn_datasets")
    parser.add_argument("--experiment", type=str, default="hkm")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--feature", type=str, default="mua", choices=["sua", "mua"])
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                        help="Fail if any velocity sample is faster (units/s); 0 disables the check")
    main(parser.parse_args())

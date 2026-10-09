"""
Builds whole-trial, variable-length SNN training data for HKM: one .pkl
per raw NWB trial, covering that trial's ENTIRE duration, no 256ms
windowing and no group_size chunking at all -- the data-processing
choice settled on directly because HKM's own trials are short,
independent reaches (confirmed via compare_ann_snn_trial_coverage.py: a
large fraction of trials are shorter than even one 256ms window, so
windowing was silently discarding real data rather than a neutral
choice). Training on these requires --batch-size 1 (see dataset.py's own
collate_fn(), which torch.stacks a batch -- with batch_size=1 there's
never more than one tensor to stack, so variable trial length is already
handled by the EXISTING dataloader with zero code changes there).

Reads Stage 1's raw per-trial output directly (convert_nwb_trials_to_raw_h5.py,
{output_root}/raw_trials/{session_id}/{session_id}_trial####.h5) --
trial boundaries already exist at that stage (one file per NWB trial),
so unlike make_snn_dataset.py (which slices a single whole-SESSION file
into many small windows itself) there is no windowing/slicing to do at
all: bin each trial's own COMPLETE task_time range into native 4ms bins,
in one pass, and that trial's own X_raster/velocity IS the training
example.

BINNING LOGIC: adapted directly from make_snn_dataset.py's own per-window
np.histogram-based approach -- same bin_edges construction, same spike-
counting method, same velocity extraction (task_data[:, 2:4], already
precomputed by convert_nwb_trials_to_raw_h5.py, not re-derived here) --
just applied ONCE across nperseg = the trial's own FULL length, not a
fixed small window repeated many times. Deliberately NOT importing
make_snn_dataset.py directly (its own main() is built around single-
whole-session --input_filepath/--output_filepath and internal window-
stepping this script has no use for) -- same reasoning
snn_inference_utils.py's own module docstring gives for copying rather
than importing: the two scripts solve genuinely different shaped
problems even though the core bin-the-spikes-against-bin_edges step is
identical, and forcing a shared import here would mean threading dead
windowing parameters through a function that doesn't need them.

TRAIN/TEST SPLIT: same chronological, whole-trial-only principle as
combine_trial_windows_to_grouped_session.py (never split mid-trial) --
adapted to split by cumulative RAW SAMPLE count instead of cumulative
WINDOW count, since a "window" is no longer a meaningful unit here at
all (every trial contributes exactly one example, of whatever length it
naturally has).

OUTPUT SCHEMA: {"input_spikes": (C, T_trial) float32, "velocity":
(T_trial, 2) float32} for BOTH train and test -- matching CustomDataset's
own read exactly (data['input_spikes'].T, data['velocity']), confirmed
directly against dataset.py's own __getitem__(). No window_start_time/
y_end fields (unlike the old windowed test/ convention) -- those existed
for window-level ANN alignment calibration, which doesn't apply here:
every file is now just "one whole trial," train or test alike.

Writes DIRECTLY to the real, final destination
(snn_datasets/{experiment}/{subject}/mua/{session_id}/{train,test}/*.pkl)
-- no intermediate windowed_per_trial/mua_1_group/mua_8_group staging at
all, since there's no windowing or grouping step left to stage between
raw and final.

CLI usage:
    python make_snn_dataset_whole_trial.py \
        --output-root /users/bfalkenb/scratch/bfalkenb/data/hkm_purgatory \
        --session-id sub-Jenkins_ses-20090912_behavior+ecephys \
        --dest-root /users/bfalkenb/scratch/bfalkenb/data/snn_datasets \
        --experiment hkm --subject jenkins \
        --feature mua --test_frac 0.1
"""

import argparse
import glob
import os
import re
import pickle as pkl

import h5py
import numpy as np
from tqdm import tqdm

TRIAL_ID_PATTERN = re.compile(r"_trial(\d+)\.h5$")
DELTA_TIME = 0.004  # native sampling interval -- MUST match make_snn_dataset.py's
# hardcoded value and convert_nwb_trials_to_raw_h5.py's own task_time sampling rate


def discover_raw_trials(raw_dir):
    """Returns [(trial_id, path), ...] sorted by trial_id -- same
    discovery convention as combine_trial_windows_to_grouped_session.py's
    own TRIAL_ID_PATTERN, applied to Stage 1's raw filenames instead of
    Stage 2a's windowed ones."""
    trials = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.h5"))):
        m = TRIAL_ID_PATTERN.search(os.path.basename(path))
        if not m:
            print(f"  [skip] {path}: filename doesn't match '..._trial####.h5'")
            continue
        trials.append((int(m.group(1)), path))
    trials.sort(key=lambda t: t[0])
    return trials


def bin_whole_trial(task_time, task_data, spike_trains):
    """Bins ONE trial's ENTIRE task_time range into native 4ms bins --
    the same np.histogram(spk, bin_edges) approach make_snn_dataset.py
    uses per-window, here applied once with nperseg = len(task_time) (the
    WHOLE trial) instead of a fixed small window repeated many times.
    Returns (X_raster, velocity): X_raster (num_units, T) float32,
    velocity (T, 2) float32 -- T = len(task_time), the trial's own
    natural length, whatever that happens to be.
    """
    nperseg = len(task_time)
    num_units = len(spike_trains)

    # Same bin_edges construction as make_snn_dataset.py's own per-window
    # version: nperseg+1 edges around the ACTUAL sample times (not an
    # assumed-uniform grid), so real small timing jitter in task_time
    # doesn't silently misalign spikes to the wrong bin.
    dt = np.diff(task_time).mean()
    bin_edges = np.concatenate((task_time - dt / 2, [task_time[-1] + dt / 2]))

    X_raster = np.zeros((num_units, nperseg), dtype=np.float32)
    for u, spk in enumerate(spike_trains):
        spk = np.asarray(spk)
        X_raster[u, :] = np.histogram(spk, bin_edges)[0]

    # velocity columns, same convention as make_snn_dataset.py/evaluate_ann.py:
    # task_data is N x 6 (pos_x, pos_y, vel_x, vel_y, acc_x, acc_y) --
    # already precomputed upstream, not re-derived here.
    velocity = task_data[:, 2:4].astype(np.float32)

    return X_raster, velocity


def main(args):
    raw_dir = os.path.join(args.output_root, "raw_trials", args.session_id)
    trials = discover_raw_trials(raw_dir)
    if not trials:
        raise FileNotFoundError(f"No '..._trial####.h5' files found under {raw_dir}")
    print(f"Found {len(trials)} raw trial file(s), trial_id range "
          f"[{trials[0][0]}, {trials[-1][0]}]")

    per_trial_data = []
    n_raw_samples_per_trial = []
    for trial_id, path in tqdm(trials, desc="Binning whole trials"):
        with h5py.File(path, "r") as f:
            task_data = f["task_data"][()]
            task_time = f["task_time"][()]
            spike_trains = f[f"{args.feature}_trains"][()]

        if len(task_time) == 0:
            print(f"  [skip] trial {trial_id}: zero raw samples")
            continue

        X_raster, velocity = bin_whole_trial(task_time, task_data, spike_trains)
        per_trial_data.append((trial_id, X_raster, velocity))
        n_raw_samples_per_trial.append(len(task_time))

    if not per_trial_data:
        raise RuntimeError(f"Every trial under {raw_dir} had zero raw samples -- nothing to write.")

    total_raw_samples = sum(n_raw_samples_per_trial)
    print(f"Total raw samples across {len(per_trial_data)} non-empty trial(s): {total_raw_samples} "
          f"(~{total_raw_samples * DELTA_TIME:.1f}s)")

    # Same chronological, whole-trial-only split principle as
    # combine_trial_windows_to_grouped_session.py -- adapted to split by
    # cumulative RAW SAMPLE count (there is no "window count" here at
    # all: every trial contributes exactly one example).
    cumulative = np.cumsum(n_raw_samples_per_trial)
    train_sample_target = (1 - args.test_frac) * total_raw_samples
    split_trial_idx = int(np.searchsorted(cumulative, train_sample_target, side="left"))
    split_trial_idx = min(max(split_trial_idx, 0), len(per_trial_data) - 1)

    train_trials = per_trial_data[:split_trial_idx + 1]
    test_trials = per_trial_data[split_trial_idx + 1:]
    if not test_trials:
        test_trials = [train_trials[-1]]
        train_trials = train_trials[:-1]
        print("  NOTE: test_frac too small to naturally cross a trial boundary -- "
              "holding out the single last trial instead of producing an empty test set.")

    print(f"Split: {len(train_trials)} train trial(s), {len(test_trials)} test trial(s)")
    print(f"  train trial_ids: {[t for t, _, _ in train_trials]}")
    print(f"  test trial_ids:  {[t for t, _, _ in test_trials]}")

    dest_dir = os.path.join(args.dest_root, args.experiment, args.subject, "mua", args.session_id)
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for d in (train_dir, test_dir):
        for fname in os.listdir(d):
            if fname.endswith(".pkl"):
                os.remove(os.path.join(d, fname))

    for idx, (trial_id, X_raster, velocity) in enumerate(train_trials):
        with open(os.path.join(train_dir, f"{idx}.pkl"), "wb") as fh:
            pkl.dump({"input_spikes": X_raster, "velocity": velocity}, fh)
    print(f"Wrote {len(train_trials)} whole-trial train file(s) to {train_dir}")

    for idx, (trial_id, X_raster, velocity) in enumerate(test_trials):
        with open(os.path.join(test_dir, f"{idx}.pkl"), "wb") as fh:
            pkl.dump({"input_spikes": X_raster, "velocity": velocity}, fh)
    print(f"Wrote {len(test_trials)} whole-trial test file(s) to {test_dir}")

    trial_lengths = [X.shape[1] for _, X, _ in per_trial_data]
    print(f"\nTrial length (timesteps): min={min(trial_lengths)}, max={max(trial_lengths)}, "
          f"mean={np.mean(trial_lengths):.0f} -- REMINDER: variable-length trials require "
          f"--batch-size 1 for training (see module docstring).")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, required=True,
                         help="run_nwb_pipeline.sh's own --output-root (contains raw_trials/)")
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--dest-root", type=str, required=True,
                         help="e.g. /users/bfalkenb/scratch/bfalkenb/data/snn_datasets -- "
                              "output lands under {dest-root}/{experiment}/{subject}/mua/{session-id}/")
    parser.add_argument("--experiment", type=str, required=True, choices=["bmi", "hkm"])
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--feature", type=str, default="mua", choices=["sua", "mua"])
    parser.add_argument("--test_frac", type=float, default=0.1)
    args = parser.parse_args()
    main(args)

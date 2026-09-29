"""
Build a long-trial SNN dataset for one session by concatenating runs of
--group-size consecutive windows from make_snn_dataset.py's output.

make_snn_dataset.py's back-to-back windows are first rejoined into the
continuous recording, which is then split chronologically at the same
boundary export_snn_pkl.py and the ANN decoders use:
  - train: consecutive trials of group_size * nperseg timesteps each. The
    shorter trailing trial is kept unless --discard-train-remainder is given
    (equal-length trials are needed for batch_size > 1).
  - test: one single trial covering the whole test region.

Output layout: {output_dir}/{session_id}/{train,test}/{i}.pkl, each a dict with
    input_spikes  (n_units, T)  float32
    velocity      (T, 2)        float32

Usage:
    python combine_snn_dataset.py --h5-path .../indy_20160407_02_snn.h5 \
        --session-id indy_20160407_02 --group-size 8 --test-frac 0.1 \
        --output-dir datasets/bmi/indy/mua_8_group --discard-train-remainder
"""

import argparse
import os
import pickle as pkl
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.preprocessing import aligned_train_boundary

STEP_TIME_S = 0.004


def load_continuous_recording(h5_path):
    """Rejoin back-to-back SNN windows into the continuous recording.

    Returns (input_spikes (C, T), velocity (T, 2), nperseg, total_raw_samples).
    T is n_windows * nperseg, a few samples shorter than the raw session,
    whose length total_raw_samples is used for the train/test boundary."""
    with h5py.File(h5_path, "r") as f:
        window_start_time = f["window_start_time"][:]
        X_raster = f["X_raster"][:]   # (n_windows, C, nperseg)
        y_trace = f["y_trace"][:]     # (n_windows, nperseg, 2)
        total_raw_samples = int(f.attrs["total_raw_samples"])

    n_windows, n_channels, nperseg = X_raster.shape
    stride = np.diff(window_start_time)
    if not np.allclose(stride, stride[0]):
        raise ValueError(f"{h5_path}: windows are not evenly spaced, so they cannot be "
                         f"concatenated into a continuous recording.")
    print(f"{n_windows} windows, stride={stride[0]}s, "
          f"total duration={n_windows * stride[0] / 60:.2f} minutes")

    input_spikes = np.transpose(X_raster, (1, 0, 2)).reshape(n_channels, -1)
    velocity = y_trace.reshape(-1, 2)
    return input_spikes, velocity, nperseg, total_raw_samples


def _save_trial(split_dir, idx, input_spikes, velocity):
    path = os.path.join(split_dir, f"{idx}.pkl")
    with open(path, "wb") as f:
        pkl.dump({"input_spikes": input_spikes.astype(np.float32),
                  "velocity": velocity.astype(np.float32)}, f)
    print(f"  {path}: input_spikes shape={input_spikes.shape}, velocity shape={velocity.shape} "
          f"({velocity.shape[0] * STEP_TIME_S:.1f}s)")


def save_chunked(split_dir, input_spikes, velocity, chunk_len, discard_remainder):
    """Save consecutive chunk_len-timestep trials; returns the number written."""
    os.makedirs(split_dir, exist_ok=True)
    total_len = input_spikes.shape[1]
    starts = list(range(0, total_len, chunk_len))
    if discard_remainder and total_len % chunk_len:
        starts = starts[:-1]
    print(f"\n{split_dir}: {total_len} timesteps -> {len(starts)} trial(s) of up to "
          f"{chunk_len} timesteps")
    for idx, start in enumerate(starts):
        _save_trial(split_dir, idx, input_spikes[:, start:start + chunk_len],
                    velocity[start:start + chunk_len])
    return len(starts)


def main(args):
    input_spikes, velocity, nperseg, total_raw_samples = load_continuous_recording(args.h5_path)
    total_len = input_spikes.shape[1]
    # Same boundary as export_snn_pkl.py and the ANN decoders.
    n_train = aligned_train_boundary(total_raw_samples, args.test_frac, base_nperseg=nperseg)
    print(f"Train/test boundary at sample {n_train} of {total_raw_samples} "
          f"(test_frac={args.test_frac})")

    output_dir = os.path.join(args.output_dir, args.session_id)
    n_train_trials = save_chunked(os.path.join(output_dir, "train"),
                                  input_spikes[:, :n_train], velocity[:n_train],
                                  args.group_size * nperseg, args.discard_train_remainder)

    # One whole test trial: evaluation needs no backward pass, so memory does
    # not require chunking, and per-trial metric averages are not skewed by a
    # short remainder trial.
    n_test_trials = 0
    if n_train < total_len:
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        _save_trial(test_dir, 0, input_spikes[:, n_train:], velocity[n_train:])
        n_test_trials = 1
    print(f"\nDone: {n_train_trials} train trial(s), {n_test_trials} test trial(s) under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5-path", type=str, required=True,
                        help="SNN dataset (make_snn_dataset.py output, --ol_time 0)")
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=256,
                        help="Windows per train trial (256 x 65 samples = ~66.6 s)")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="datasets/bmi/mua_huge")
    parser.add_argument("--discard-train-remainder", action="store_true",
                        help="Drop the shorter trailing train trial so all train trials have "
                             "equal length (required for batch_size > 1)")
    main(parser.parse_args())

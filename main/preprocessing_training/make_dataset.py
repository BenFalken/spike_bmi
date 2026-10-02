"""
Build the ANN decoding dataset for one session: one spike-count feature per
unit per sliding window, paired with the kinematics at the window's end.

Input  (--input_filepath):  process_data.py output.
Output (--output_filepath): .h5 with
    X_sua   (n_windows, n_sorted_units)
    X_mua   (n_windows, n_channels)
    y_task  (n_windows, 6)   task_data at the sample just after each window

The defaults (--wdw_time 0.256, --ol_time 0.252) give 65-sample windows
advanced by one native 4 ms sample, so row i starts at raw sample i.

A session too short for a single window (possible for individual NWB trials)
writes nothing and exits with status 0, so per-trial batch loops continue.
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.features import extract

DELTA_TIME = 0.004  # native sampling interval (s)


def window_params(wdw_time, ol_time, delta_time=DELTA_TIME):
    """(nperseg, noverlap) in samples. A window of wdw_time spans nperseg =
    wdw_time/dt + 1 samples; ol_time=0 gives noverlap=0 (no shared samples)."""
    nperseg = int(wdw_time / delta_time) + 1
    noverlap = int(round(ol_time / delta_time))
    if ol_time > 0:
        noverlap += 1
    return nperseg, noverlap


def main(args):
    run_start = time.time()
    print(f"Loading spike and kinematic data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_data = f['task_data'][()]
        task_time = f['task_time'][()]
        sua_trains = f['sua_trains'][()]
        mua_trains = f['mua_trains'][()]

    nperseg, noverlap = window_params(args.wdw_time, args.ol_time)
    if args.method == 'gaussian':
        method_kwargs = {'window': 'gaussian', 'std': int(args.std_time / DELTA_TIME)}
    elif args.method == 'baks':
        method_kwargs = {'a': args.alpha}
    else:
        method_kwargs = {}

    def rates(train):
        return extract(train, task_time, nperseg, noverlap, method=args.method, **method_kwargs)

    # The target depends only on the window positions, not on the spikes.
    _, y_task = extract(mua_trains[0], task_time, nperseg, noverlap, task=task_data,
                        method=args.method, **method_kwargs)
    if y_task.shape[0] == 0:
        print(f"[skip] {args.input_filepath}: {len(task_time)} samples is shorter than one "
              f"{nperseg}-sample window; no output written.")
        return

    if len(sua_trains):
        X_sua = np.asarray([rates(train) for train in sua_trains]).T
    else:
        X_sua = np.zeros((len(y_task), 0))

    X_mua, run_times = [], []
    for train in mua_trains:
        start = time.time()
        X_mua.append(rates(train))
        run_times.append((time.time() - start) / X_mua[-1].shape[0])
    X_mua = np.asarray(X_mua).T
    run_times = np.asarray(run_times)
    print(f"Average (std) run time for spike rate estimation: "
          f"{run_times.mean() * 1e6} ({run_times.std() * 1e6}) µs")

    print(f"Storing dataset into file : {args.output_filepath}")
    with h5py.File(args.output_filepath, 'w') as f:
        f['X_sua'] = X_sua
        f['X_mua'] = X_mua
        f['y_task'] = y_task
    print(f"Finished whole processes within {(time.time() - run_start) / 60:.2f} minutes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_filepath', type=str, required=True,
                        help='Spike and kinematic data (process_data.py output)')
    parser.add_argument('--output_filepath', type=str, required=True, help='Output dataset file')
    parser.add_argument('--method', type=str, default='binning',
                        choices=['binning', 'gaussian', 'baks'], help='Spike rate estimation method')
    parser.add_argument('--wdw_time', type=float, default=0.256, help='Window width (s)')
    parser.add_argument('--ol_time', type=float, default=0.252,
                        help='Overlap between consecutive windows (s); step = wdw_time - ol_time')
    parser.add_argument('--std_time', type=float, default=0.060, help='Gaussian kernel width (s)')
    parser.add_argument('--alpha', type=float, default=4., help='BAKS shape parameter')
    main(parser.parse_args())

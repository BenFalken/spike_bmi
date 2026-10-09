"""
Create dataset containing the input data (spike rates) and the output data (kinematics)

Default --wdw_time/--ol_time (0.256/0.252) give a dense, 4ms-step (step=1
native sample) window -- the ANN pipeline's current convention project-
wide, matching this project's SNN pipeline's shared WDW_TIME base unit
(65 native samples = 256ms) while intentionally NOT matching the SNN's own
--ol_time (the SNN uses non-overlapping windows; see make_snn_dataset.py's
module docstring for why ANN and SNN windowing are now deliberately
different, not "matched"). If you call this script directly rather than
through single_subject_pipeline.py (which always passes these explicitly
regardless of the defaults here), these are the values you'll get.
"""

# import packages
import argparse
import numpy as np
import h5py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.features import extract
import time as timer

def main(args):
    run_start = timer.time()
    print(f"Loading spike and kinematic data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_data = f['task_data'][()]  # kinematic data
        task_time = f['task_time'][()]  # time associated with the kinematic data
        sua_train = f['sua_trains'][()] # sorted spike times (single unit activity)
        mua_train = f['mua_trains'][()] # unsorted spike times (threshold crossing/multi unit activity)

    num_sua = len(sua_train)
    num_mua = len(mua_train)

    delta_time = 0.004 # sampling interval in second
    nperseg = int(args.wdw_time / delta_time) + 1
    # FIX: previously `noverlap = int(args.ol_time / delta_time) + 1` applied
    # its "+1" unconditionally, so --ol_time 0.0 still produced noverlap=1 --
    # one shared raw sample between consecutive windows, not true
    # non-overlap. The +1 is only correct when there's a real, nonzero
    # overlap to be inclusive about (matches the identical fix already
    # applied to make_snn_dataset.py). Not currently exercised by this
    # script's typical usage (ol_time is normally nonzero here), but fixed
    # proactively for consistency and in case --ol_time 0.0 is ever used on
    # the ANN side.
    noverlap = int(round(args.ol_time / delta_time))
    if args.ol_time > 0:
        noverlap += 1

    X_sua = []
    X_mua = []
    for i in range(num_sua):
        #print(f"Extracting SUA/sorted spike features from unit no: {i}")
        if args.method == 'binning':
            sua_rate, y_task = extract(sua_train[i], task_time, nperseg, noverlap, task=task_data, method=args.method)
        elif args.method == 'gaussian':
            std_time = args.std_time
            std = int(std_time/delta_time)
            sua_rate, y_task = extract(sua_train[i], task_time, nperseg, noverlap, task=task_data, method=args.method, window=args.method, std=std)
        elif args.method == 'baks':
            sua_rate, y_task = extract(sua_train[i], task_time, nperseg, noverlap, task=task_data, method=args.method, a=args.alpha)
        X_sua.append(sua_rate)

    run_times = []
    for i in range(num_mua):
        #print(f"Extracting MUA/threshold crossing features from channel no: {i}")
        extract_start = timer.time()
        if args.method == 'binning':
            mua_rate = extract(mua_train[i], task_time, nperseg, noverlap, task=None, method=args.method)
        elif args.method == 'gaussian':
            std_time = args.std_time
            std = int(std_time/delta_time)
            mua_rate = extract(mua_train[i], task_time, nperseg, noverlap, task=None, method=args.method, window=args.method, std=std)
        elif args.method == 'baks':
            mua_rate = extract(mua_train[i], task_time, nperseg, noverlap, task=None, method=args.method, a=args.alpha)
        extract_end = timer.time()

        # BUG FIX: this trial's own task_time can be shorter than nperseg
        # (this file's dense-window "1 window minimum" requirement) even
        # though it's genuinely long enough for the SNN's own whole-trial
        # path (no minimum window length there at all -- see
        # make_snn_dataset_whole_trial.py's own module docstring) -- e.g.
        # convert_nwb_trials_to_raw_h5.py's --min-samples default was
        # deliberately lowered from 130 (2 SNN windows) to 5 (just past the
        # velocity/acceleration computation's own real crash floor), for
        # the SNN side specifically, with the KNOWN, accepted consequence
        # that some trials now pass Stage 1 without being long enough for
        # even ONE ANN window. extract() then returns a genuinely empty,
        # zero-row mua_rate for EVERY channel (not just this one -- all
        # channels share the same task_time/nperseg/noverlap, so this is
        # checked on the FIRST channel and trusted for the rest, rather
        # than re-checking num_mua times for a result that can't differ).
        # Previously: `run_time = (extract_end - extract_start) /
        # mua_rate.shape[0]` divided by this zero unconditionally --
        # ZeroDivisionError, uncaught, which (given
        # run_dense_windowing_for_all_trials.sh's own set -eo pipefail)
        # took the ENTIRE REST OF THE SESSION's per-trial loop down with
        # it, not just this one trial -- confirmed directly: this is
        # exactly what produced a missing {session}_binning.h5 for an
        # otherwise-successful session. Detected here, BEFORE the
        # division, and exits cleanly (code 0, no output file written --
        # this trial genuinely has zero ANN windows to contribute, so
        # writing a degenerate, zero-row output would be wrong, not just
        # avoiding a crash) so the wrapper's own per-trial loop continues
        # to the next trial instead of aborting the whole session.
        if mua_rate.shape[0] == 0:
            print(f"[skip] {args.input_filepath}: this trial has {len(task_time)} raw "
                  f"samples, fewer than the {nperseg} needed for even one {args.wdw_time*1000:.0f}ms "
                  f"dense window -- zero ANN windows possible for this trial. No output written; "
                  f"exiting cleanly (not an error) so the per-trial loop continues.")
            return
        run_time = (extract_end - extract_start) / mua_rate.shape[0]
        run_times.append(run_time)
        X_mua.append(mua_rate)
    run_times = np.asarray(run_times)
    print(f"Average (std) run time for spike rate estimation: {run_times.mean()*1e6} ({run_times.std()*1e6}) µs")
    # convert to array
    X_sua = np.asarray(X_sua).T
    X_mua = np.asarray(X_mua).T

    print(f"Storing dataset into file : {args.output_filepath}")
    with h5py.File(args.output_filepath, 'w') as f:
        f['X_sua'] = X_sua
        f['X_mua'] = X_mua
        f['y_task'] = y_task

    run_end = timer.time()
    print(f"Finished whole processes within {(run_end-run_start)/60:.2f} minutes")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath',   type=str,   help='Path to the spike dan kinematic data')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the created dataset')
    parser.add_argument('--method',           type=str,   default='binning',  help='Spike rate estimation method')
    parser.add_argument('--wdw_time',         type=float, default=0.256,      help='Segment window size (s) -- see module docstring for why 0.256, not the older 0.240')
    parser.add_argument('--ol_time',          type=float, default=0.252,      help='Overlap window size (s) -- 0.252 -> dense 4ms-step windows; see module docstring')
    parser.add_argument('--std_time',         type=float, default=0.060,      help='Bandwidth of Gaussian window (s)')
    parser.add_argument('--alpha',            type=float, default=4.,         help='Shape parameter of BAKS')
    args = parser.parse_args()
    main(args)

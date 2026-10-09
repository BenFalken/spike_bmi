"""
Evaluating spike-based BMI decoding using Wiener filter (WF)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# import packages
import argparse
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom, transform_data, chronological_holdout_split
from bmi.decoders import WienerDecoder
from sklearn.metrics import root_mean_squared_error
from bmi.metrics import pearson_corrcoef
import time as timer
import threading
from tqdm import tqdm

import os
import json
import pickle as pkl  # add to imports


class Heartbeat:
    """Prints a periodic 'still running' message from a background thread
    for the duration of a `with Heartbeat(...):` block -- for calls
    (transform_data, model.fit) whose internal implementation isn't
    visible/instrumentable here (no access to bmi/preprocessing.py's or
    bmi/decoders.py's own source), so a real, fractional progress bar
    isn't possible: if WienerDecoder.fit() is a single closed-form
    matrix solve rather than an iterative loop, there is no "50% done"
    to report. This answers a narrower, more honest question instead --
    is the process still alive and how long has this specific step been
    running -- without claiming to know how close to finished it is."""
    def __init__(self, label, interval=30):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._start = None

    def __enter__(self):
        self._start = timer.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = timer.time() - self._start
            print(f"    ...{self.label} still running ({elapsed:.0f}s elapsed)", flush=True)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1)
        return False


def minutes_to_samples(minutes, step_ms=120.0):
    """Convert a duration in minutes to a number of (overlapping) trial
    samples, given the step size between consecutive trials in ms.

    E.g. 240 ms trials stepped every 120 ms -> 500 samples/minute, so
    1 minute = 500 samples, 2 minutes = 1000 samples, etc.
    """
    return int(round(minutes * 60000.0 / step_ms))


def add_suffix_to_path(path, suffix):
    """Insert a suffix before a file's extension, e.g.
    add_suffix_to_path('results.h5', '5min') -> 'results_5min.h5'
    """
    base, ext = os.path.splitext(path)
    return f"{base}_{suffix}{ext}"


def resolve_gap_samples(args):
    """Same purpose as the KF script's version of this function, with one
    addition: WF stacks --timesteps consecutive rows of X into a single
    training example (via transform_data), so a training example's actual
    reach into the past is (window_width_rows + timesteps - 1), not just
    the window width. Using the KF-style "2x window width" formula here
    would under-purge -- a training example built from rows ending near
    the split boundary would still reach `timesteps - 1` rows further back
    than that formula accounts for, but that's fine (it's on the training
    side, looking backward, away from the gap). The real risk is the FIRST
    test example after the boundary: it also stacks `timesteps` rows, so
    it reaches (timesteps - 1) rows into what should be purged territory
    unless the gap accounts for that reach too. Hence
    2 * (window_width_rows + timesteps - 1) instead of KF's plain
    2 * window_width_rows.
    """
    if args.gap_samples is not None:
        return args.gap_samples
    window_width_rows = args.wdw_time * 1000.0 / args.step_ms
    gap_samples = int(np.ceil(2 * (window_width_rows + args.timesteps - 1)))
    print(f"--gap_samples not set; using computed default {gap_samples} "
          f"(2 x [window_width_rows={window_width_rows:.1f} + (timesteps-1)={args.timesteps - 1}], "
          f"from --wdw_time={args.wdw_time}s, --step_ms={args.step_ms}, --timesteps={args.timesteps})")
    return gap_samples


def main(args):
    run_start = timer.time()

    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]
    y = y[:, 2:4]

    config = {'reg_type'    : args.reg_type,
              'reg_alpha'   : args.reg_alpha,
              'timesteps'   : args.timesteps}
    print(f"Hyperparameter configuration: {config}")

    gap_samples = resolve_gap_samples(args)

    # Parse the list of training durations (in minutes) to sweep over.
    # If none were given, fall back to the original behavior: a single
    # pass using the full training set available in each fold.
    if args.train_durations:
        duration_minutes_list = [float(x) for x in args.train_durations.split(',') if x.strip() != '']
    else:
        duration_minutes_list = [None]

    samples_per_minute = 60000.0 / args.step_ms
    print(f"Step size: {args.step_ms} ms -> {samples_per_minute:.1f} samples/minute")

    for duration_minutes in duration_minutes_list:

        if duration_minutes is not None:
            n_train_samples = minutes_to_samples(duration_minutes, step_ms=args.step_ms)
            duration_tag = f"{duration_minutes:g}min"
            print(f"\n=== Training with {duration_minutes:g} minute(s) of data "
                  f"(~{n_train_samples} samples) ===")
        else:
            n_train_samples = None
            duration_tag = "full"
            print("\n=== Training with full available training data (no duration cap) ===")

        rmse_test_folds = []
        cc_test_folds = []

        # keep references to the last fold's test data/predictions for the
        # per-duration output file (mirrors original script's behavior)
        y_test_last = None
        y_test_pred_last = None

        tscv = TimeSeriesSplitCustom(n_splits=args.n_folds,
                                      test_size=int(args.test_size * len(y)),
                                      min_train_size=int(args.min_train_size * len(y)),
                                      gap=gap_samples)
        print(f"  Starting {args.n_folds}-fold CV loop...", flush=True)
        for fold_i, (train_idx, test_idx) in enumerate(
                tqdm(tscv.split(X, y), total=args.n_folds, desc="  CV folds", unit="fold")):
            fold_start = timer.time()
            print(f"  Fold-{fold_i + 1}/{args.n_folds}: {len(train_idx)} train rows, "
                  f"{len(test_idx)} test rows", flush=True)

            # specify training set
            X_train_full = X[train_idx, :]
            y_train_full = y[train_idx, :]

            # specify test set
            X_test = X[test_idx, :]
            y_test = y[test_idx, :]

            if n_train_samples is not None:
                if n_train_samples > len(y_train_full):
                    print(f"  Fold-{fold_i + 1}: requested {n_train_samples} samples "
                          f"({duration_minutes:g} min) but only {len(y_train_full)} "
                          f"available in this fold's training set; skipping fold.")
                    rmse_test_folds.append(np.nan)
                    cc_test_folds.append(np.nan)
                    continue
                # take the most recent n_train_samples samples, i.e. those
                # immediately preceding the test window, since these are
                # most representative of the data regime nearest to test time
                X_train = X_train_full[-n_train_samples:, :]
                y_train = y_train_full[-n_train_samples:, :]
            else:
                X_train = X_train_full
                y_train = y_train_full

            # standardize input data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test_fold = scaler.transform(X_test)
            print(f"  Fold-{fold_i + 1}: scaler fit ({timer.time() - fold_start:.1f}s elapsed)",
                  flush=True)

            # transform data into sequence data
            with Heartbeat(f"Fold-{fold_i + 1} tap-delay transform (train)"):
                X_train, y_train = transform_data(X_train, y_train, timesteps=args.timesteps)
            with Heartbeat(f"Fold-{fold_i + 1} tap-delay transform (test)"):
                X_test_fold, y_test_fold = transform_data(X_test_fold, y_test, timesteps=args.timesteps)
            print(f"  Fold-{fold_i + 1}: tap-delay transform done -> X_train shape={X_train.shape} "
                  f"({timer.time() - fold_start:.1f}s elapsed)", flush=True)

            # reshape data
            X_train = X_train.reshape(X_train.shape[0], (X_train.shape[1] * X_train.shape[2]), order='F')
            X_test_fold = X_test_fold.reshape(X_test_fold.shape[0], (X_test_fold.shape[1] * X_test_fold.shape[2]), order='F')

            # instantiate model
            model = WienerDecoder(args.reg_type, args.reg_alpha)
            # fit model
            print(f"  Fold-{fold_i + 1}: starting fit on X_train shape={X_train.shape} "
                  f"({timer.time() - fold_start:.1f}s elapsed)", flush=True)
            train_start = timer.time()
            with Heartbeat(f"Fold-{fold_i + 1} WienerDecoder.fit()"):
                model.fit(X_train, y_train)
            train_end = timer.time()
            train_time = (train_end - train_start) / 60
            print(f"  Fold-{fold_i + 1}: fit done in {train_time:.2f} min "
                  f"({timer.time() - fold_start:.1f}s elapsed)", flush=True)

            # predict using the trained model
            y_test_pred = model.predict(X_test_fold)

            # evaluate performance
            rmse_test = root_mean_squared_error(y_test_fold, y_test_pred)
            cc_test = pearson_corrcoef(y_test_fold, y_test_pred)
            print(f"  Fold-{fold_i + 1}: RMSE={rmse_test:.2f}, CC={np.mean(cc_test) if hasattr(cc_test, '__len__') else cc_test:.2f} "
                  f"-- fold total {timer.time() - fold_start:.1f}s", flush=True)

            rmse_test_folds.append(rmse_test)
            cc_test_folds.append(cc_test)

            y_test_last = y_test_fold
            y_test_pred_last = y_test_pred

        for i in range(args.n_folds):
            cc_display = np.mean(cc_test_folds[i]) if hasattr(cc_test_folds[i], '__len__') else cc_test_folds[i]
            print(f"Fold-{i + 1} | RMSE test = {rmse_test_folds[i]:.2f}, CC test = {cc_display:.2f}")

        # Same fix as train_test_split_eval_kf_decoder.py -- only tag the
        # filename when a specific duration was actually requested,
        # matching train_test_split_eval_dl_decoders.py's convention.
        if duration_minutes is not None:
            output_filepath_d = add_suffix_to_path(args.output_filepath, duration_tag)
        else:
            output_filepath_d = args.output_filepath
        print(f"Storing results into file: {output_filepath_d}")
        with h5py.File(output_filepath_d, 'w') as f:
            if y_test_last is not None:
                f['y_test'] = y_test_last
                f['y_test_pred'] = y_test_pred_last
            f['rmse_test_folds'] = np.asarray(rmse_test_folds)
            f['cc_test_folds'] = np.asarray(cc_test_folds)
            if duration_minutes is not None:
                f.attrs['train_duration_minutes'] = duration_minutes
                f.attrs['train_duration_samples'] = n_train_samples
            f.attrs['split_gap_samples'] = gap_samples

        if args.model_dir:
            feature_dir = args.model_dir  # os.path.join(args.model_dir, args.feature)
            os.makedirs(feature_dir, exist_ok=True)

            # Chronological split, matching what a downstream comparison script
            # will reproduce -- NOT the CV split above, which exists only for
            # evaluating generalization and doesn't correspond to a fixed test set.
            # Gap-purged via the same shared helper KF's eval script uses --
            # previously this was plain index slicing with no gap at all.
            X_train_final_full, y_train_final_full, _, _ = chronological_holdout_split(
                X, y, args.test_frac, gap=gap_samples, n_train_override=args.n_train_override)

            if n_train_samples is not None and n_train_samples > len(y_train_final_full):
                print(f"  Skipping final model save for duration {duration_minutes:g} min: "
                      f"only {len(y_train_final_full)} chronological training samples available.")
            else:
                if n_train_samples is not None:
                    X_train_final = X_train_final_full[-n_train_samples:, :]
                    y_train_final = y_train_final_full[-n_train_samples:, :]
                else:
                    X_train_final = X_train_final_full
                    y_train_final = y_train_final_full

                scaler_final = StandardScaler()
                X_train_final = scaler_final.fit_transform(X_train_final)

                with Heartbeat("Final model tap-delay transform"):
                    X_train_final, y_train_final = transform_data(X_train_final, y_train_final, timesteps=args.timesteps)
                X_train_final = X_train_final.reshape(
                    X_train_final.shape[0], (X_train_final.shape[1] * X_train_final.shape[2]), order='F')
                print(f"  Final model: starting fit on X_train_final shape={X_train_final.shape}", flush=True)

                final_fit_start = timer.time()
                final_model = WienerDecoder(args.reg_type, args.reg_alpha)
                with Heartbeat("Final model WienerDecoder.fit()"):
                    final_model.fit(X_train_final, y_train_final)
                print(f"  Final model: fit done in {(timer.time() - final_fit_start) / 60:.2f} min",
                      flush=True)

                model_tag = f"_{duration_tag}" if duration_minutes is not None else ""
                model_path  = os.path.join(feature_dir, f"wf{model_tag}_model.pkl")
                scaler_path = os.path.join(feature_dir, f"wf{model_tag}_scaler.pkl")
                config_path = os.path.join(feature_dir, f"wf{model_tag}_config.json")

                with open(model_path, 'wb') as f:
                    pkl.dump(final_model, f)
                with open(scaler_path, 'wb') as f:
                    pkl.dump(scaler_final, f)
                with open(config_path, 'w') as f:
                    json.dump({**config, 'test_frac': args.test_frac, 'feature': args.feature,
                               'input_dim': X.shape[-1],           # raw channel count, for a sua/mua sanity check
                               'flattened_dim': X_train_final.shape[-1],  # post-reshape dim (= input_dim * timesteps)
                               'decoder': 'wf', 'train_duration_minutes': duration_minutes,
                               'train_duration_samples': n_train_samples,
                               'split_gap_samples': gap_samples}, f, indent=2)

                print(f"Saved final WF model bundle to {feature_dir} (duration={duration_tag})")

    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print(f"Whole processes took {run_time:.2f} minutes")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument('--input_filepath',   type=str,   help='File path to the dataset')
    parser.add_argument('--output_filepath',  type=str,   help='File path to the stored result')
    parser.add_argument('--feature',          type=str,   default='mua',  help='Type of spiking activity (sua or mua)')
    parser.add_argument('--timesteps',        type=int,   default=4,      help='Number of timesteps')
    parser.add_argument('--reg_type',         type=str,   default='',     help='Regularization type')
    parser.add_argument('--reg_alpha',        type=float, default=0,      help='Regularization constant')
    parser.add_argument('--n_folds',          type=int,   default=2,      help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,    help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,    help='Testing data size')

    parser.add_argument('--model_dir', type=str, default='',
                     help='If set, fit and save one final model + scaler for reuse downstream')
    parser.add_argument('--test_frac', type=float, default=0.1,
                     help='Chronological holdout fraction for the saved final model')

    parser.add_argument('--train_durations', type=str, default='',
                     help='Comma-separated list of training durations in minutes, e.g. '
                          '"1,2,3,4,5,6,7,8,9,10". For each duration, the training data '
                          'in every CV fold (and the final chronological-split model, if '
                          '--model_dir is set) is truncated to the most recent N minutes '
                          'worth of samples, where minutes are converted to sample counts '
                          'using --step_ms. Each duration gets its own output results file '
                          '(and model bundle). If omitted, the script runs once using the '
                          'full training set (original behavior).')
    parser.add_argument('--step_ms', type=float, default=120.0,
                     help='Step size between consecutive rows of X, in ms (i.e. the --ol_time '
                          'used to build the input dataset: step = wdw_time - ol_time). Used to '
                          'convert --train_durations minutes into a number of samples, and to '
                          'compute the default purge gap (see --wdw_time/--gap_samples). '
                          'Default 120 ms matches 240 ms trials with 50%% overlap; use 4 ms for '
                          'densely-strided (paper-matching) windowing.')
    parser.add_argument('--wdw_time', type=float, default=0.256,
                     help='Window width (s) used to build the input dataset -- NOT used for '
                          'windowing here, only to compute a generous default purge gap between '
                          'train/test splits (see --gap_samples). Must match whatever was '
                          'actually passed to make_dataset.py for --input_filepath.')
    parser.add_argument('--gap_samples', type=int, default=None,
                     help='Purge gap, in rows of X, excluded between train and test in both '
                          'the CV folds and the final chronological holdout. If not set, '
                          'computed as 2 * (wdw_time*1000/step_ms + timesteps - 1) -- see '
                          '--wdw_time, --step_ms, --timesteps. The (timesteps - 1) term matters '
                          'here specifically because WF stacks --timesteps consecutive rows per '
                          'prediction (transform_data), so a training/test example reaches '
                          'timesteps-1 rows further back than the window width alone would '
                          'suggest. Set to 0 to reproduce the original (pre-fix) ungapped '
                          'behavior.')
    parser.add_argument('--n_train_override', type=int, default=None,
                     help='Explicit train-row count for the FINAL (chronological-split, saved) '
                          'model, bypassing the base_nperseg-aligned split computation entirely. '
                          'For normal, single-continuous-session data, leave unset -- only needed '
                          'for trial-structured data where that computation does not apply (see '
                          'combine_trial_windows_to_ann_h5.py, which writes a matching n_train '
                          'attr onto its own output file -- read that value and pass it here). '
                          'Does NOT affect the CV folds above, which still use the ungapped-by-'
                          'this-fix TimeSeriesSplitCustom split.')

    args = parser.parse_args()
    main(args)

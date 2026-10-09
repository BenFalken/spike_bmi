"""
Evaluating spike-based BMI decoding using Kalman filter (KF)
"""

# import packages
import argparse
import h5py
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom, chronological_holdout_split
from bmi.decoders import KalmanDecoder
from sklearn.metrics import root_mean_squared_error
from bmi.metrics import pearson_corrcoef
import time as timer

import pickle as pkl  # add to imports
import json


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
    """A purge gap between train/test splits is needed whenever consecutive
    rows of X are built from overlapping raw windows -- without it, the
    last training row and first test row can share nearly all their raw
    samples, leaking test information into training. This matters little
    under coarse, weakly-overlapping windowing, but becomes a real risk
    under densely-strided (e.g. 4 ms step) windowing, where adjacent rows
    can be built from all but one native sample of shared data.

    If --gap_samples is set explicitly, use it as-is. Otherwise, compute a
    generous default from the window-construction parameters: 2x the
    window width, expressed in rows of X (window_width_ms / step_ms).
    --wdw_time must match whatever was passed to make_dataset.py for this
    input file, or this default won't mean anything -- this script has no
    way to verify that from the .h5 file alone.
    """
    if args.gap_samples is not None:
        return args.gap_samples
    window_width_rows = args.wdw_time * 1000.0 / args.step_ms
    gap_samples = int(np.ceil(2 * window_width_rows))
    print(f"--gap_samples not set; using computed default {gap_samples} "
          f"(2 x window_width_rows={window_width_rows:.1f}, from "
          f"--wdw_time={args.wdw_time}s, --step_ms={args.step_ms})")
    return gap_samples


def main(args):
    run_start = timer.time()
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]

    # define model configuration
    config = {'reg_type'    : args.reg_type,
              'reg_alpha'   : args.reg_alpha}
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
        else:
            n_train_samples = None
            duration_tag = "full"

        # Per-duration resume check -- WITHOUT this, a timeout partway
        # through a --train_durations sweep loses ALL progress on resume:
        # single_subject_pipeline.py's own cache-check only looks at the
        # LAST duration's output to decide whether to invoke this script
        # AT ALL, but once invoked, this loop used to redo every
        # requested duration from scratch, including ones that had
        # already completed and been written to disk before the timeout.
        # Uses the exact same output_filepath_d / model_path construction
        # as the real write further down, so this check can never drift
        # out of sync with what actually gets produced.
        expected_output_path = (add_suffix_to_path(args.output_filepath, duration_tag)
                                 if duration_minutes is not None else args.output_filepath)
        expected_model_tag = f"_{duration_tag}" if duration_minutes is not None else ""
        expected_model_path = (os.path.join(args.model_dir, f"kf{expected_model_tag}_model.pkl")
                                if args.model_dir else None)
        already_done = os.path.exists(expected_output_path) and (
            expected_model_path is None or os.path.exists(expected_model_path))
        if already_done:
            print(f"\n[skip] duration {duration_tag}: {expected_output_path} "
                  f"{'(+ its cached model) ' if expected_model_path else ''}already exists -- "
                  f"delete it first if you actually want to redo this duration.")
            continue

        if duration_minutes is not None:
            print(f"\n=== Training with {duration_minutes:g} minute(s) of data "
                  f"(~{n_train_samples} samples) ===")
        else:
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
        for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
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

            # instantiate model
            model = KalmanDecoder(args.reg_type, args.reg_alpha)
            # fit model
            train_start = timer.time()
            model.fit(X_train, y_train)  # train model
            train_end = timer.time()
            train_time = (train_end - train_start) / 60
            #print(f"Training the model took {train_time:.2f} minutes")

            # predict using the trained model
            y_test_pred = model.predict(X_test_fold, y_test[:1, :])
            # select the x-y velocity components
            y_test_vel = y_test[:, 2:4]        # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)
            y_test_pred_vel = y_test_pred[:, 2:4]  # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)

            # evaluate performance
            rmse_test = root_mean_squared_error(y_test_vel, y_test_pred_vel)
            cc_test = pearson_corrcoef(y_test_vel, y_test_pred_vel)

            rmse_test_folds.append(rmse_test)
            cc_test_folds.append(cc_test)

            y_test_last = y_test_vel
            y_test_pred_last = y_test_pred_vel

        for i in range(args.n_folds):
            cc_display = np.mean(cc_test_folds[i]) if hasattr(cc_test_folds[i], '__len__') else cc_test_folds[i]
            print(f"Fold-{i + 1} | RMSE test = {rmse_test_folds[i]:.2f}, CC test = {cc_display:.2f}")

        # Matches train_test_split_eval_dl_decoders.py's convention: only
        # tag the filename when a specific duration was actually requested.
        # Previously this always applied a "_full" suffix (even when
        # duration_minutes is None / no duration cap requested), which
        # didn't match what single_subject_pipeline.py's cache-check and
        # load_full_results() look for -- so the full-data eval could
        # never be detected as cached (always re-ran) and its results were
        # invisible to the final comparison plot.
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
            # Namespace by feature so a mismatched --feature flag downstream
            # produces a FileNotFoundError instead of a silently wrong-shape load
            # (this bit us with the DL decoder cache -- see eval_dl_decoder.py).
            feature_dir = args.model_dir  # os.path.join(args.model_dir, args.feature)
            os.makedirs(feature_dir, exist_ok=True)

            # Chronological split, matching what a downstream comparison script
            # will reproduce -- NOT the CV split above, which exists only for
            # evaluating generalization and doesn't correspond to a fixed test set.
            # Gap-purged the same as the CV folds above, via the same shared
            # helper WF/DL's eval scripts use -- previously this was plain
            # index slicing with no gap at all, which is the split every
            # downstream figure/comparison actually depends on, so leaving it
            # ungapped would have meant "fixing" only the metric you look at
            # least, not the one that matters most.
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

                final_model = KalmanDecoder(args.reg_type, args.reg_alpha)
                final_model.fit(X_train_final, y_train_final)

                # Same fix as the results file above: bare bundle name
                # when untagged, matching the DL scripts' convention
                # (final_model_bundle_paths()'s duration_minutes is not
                # None check) -- was previously always "kf_full_*" for
                # the no-duration-cap case.
                model_tag = f"_{duration_tag}" if duration_minutes is not None else ""
                model_path  = os.path.join(feature_dir, f"kf{model_tag}_model.pkl")
                scaler_path = os.path.join(feature_dir, f"kf{model_tag}_scaler.pkl")
                config_path = os.path.join(feature_dir, f"kf{model_tag}_config.json")

                with open(model_path, 'wb') as f:
                    pkl.dump(final_model, f)
                with open(scaler_path, 'wb') as f:
                    pkl.dump(scaler_final, f)
                with open(config_path, 'w') as f:
                    json.dump({**config,
                               'test_frac': args.test_frac,
                               'feature': args.feature,
                               'input_dim': X.shape[-1],
                               'decoder': 'kf',
                               'train_duration_minutes': duration_minutes,
                               'train_duration_samples': n_train_samples,
                               'split_gap_samples': gap_samples}, f, indent=2)

                print(f"Saved final KF model bundle to {feature_dir} (duration={duration_tag})")

    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print(f"Whole processes took {run_time:.2f} minutes")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument('--input_filepath',   type=str,   help='File path to the dataset')
    parser.add_argument('--output_filepath',  type=str,   help='File path to the stored result')
    parser.add_argument('--feature',          type=str,   default='mua',  help='Type of spiking activity (sua or mua)')
    parser.add_argument('--reg_type',         type=str,   default='',     help='Regularization type')
    parser.add_argument('--reg_alpha',        type=float, default=0,      help='Regularization constant')
    parser.add_argument('--n_folds',          type=int,   default=2,      help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,    help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,    help='Testing data size')

    parser.add_argument('--model_dir', type=str, default='',
                     help='If set, fit and save one final model + scaler for reuse downstream '
                          '(one bundle per training duration, if --train_durations is set)')
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
                          'convert --train_durations minutes into a number of samples '
                          '(samples_per_minute = 60000 / step_ms), and to compute the default '
                          'purge gap (see --wdw_time/--gap_samples). Default 120 ms matches '
                          '240 ms trials with 50%% overlap (120 ms steps); use 4 ms for '
                          'densely-strided (paper-matching) windowing.')
    parser.add_argument('--wdw_time', type=float, default=0.256,
                     help='Window width (s) used to build the input dataset -- NOT used for '
                          'windowing here (that already happened upstream in make_dataset.py), '
                          'only to compute a generous default purge gap between train/test '
                          'splits (see --gap_samples). Must match whatever was actually passed '
                          'to make_dataset.py for --input_filepath, or the computed default '
                          "won't mean anything -- this script can't verify that from the .h5 "
                          'file alone.')
    parser.add_argument('--gap_samples', type=int, default=None,
                     help='Purge gap, in rows of X, excluded between train and test in both '
                          'the CV folds and the final chronological holdout -- prevents the '
                          'last training row and first test row from being built from nearly '
                          'identical overlapping raw windows. If not set, computed as '
                          '2 * wdw_time*1000/step_ms (see --wdw_time, --step_ms). Set to 0 to '
                          'reproduce the original (pre-fix) ungapped behavior.')
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

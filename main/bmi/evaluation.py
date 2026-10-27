"""
Shared evaluation loop for the ANN-side decoder scripts
(preprocessing_training/eval_{kf,wf}_decoder.py, eval_dl_decoders.py).

For each requested training duration (or the full training set), a decoder
script gets up to two outputs:

1. Cross-validation results (only with --n_folds > 0; off by default):
   chronological K-fold CV (TimeSeriesSplitCustom) with a purge gap between
   train and test. Per-fold RMSE/CC and the last fold's predictions go to an
   .h5 file. These are quick-look numbers; the reported results come from
   inference/test_all_decoders.py.
2. A final model bundle (with --model_dir): one model fit on the
   chronological training split (chronological_holdout_split). Downstream
   inference scripts load these bundles and evaluate them on the held-out
   test split. The purge gap also separates this split, except where the
   boundary is a trial boundary (--n_train_override, trial-structured NWB
   data): different trials share no samples, so no gap is needed there.

With a duration cap, each training set is cut down to its most recent
`minutes` of rows, the ones just before the test data. Outputs for a
duration get a `_{minutes}min` suffix; full-data outputs get no suffix.
Outputs that already exist are skipped unless --overwrite is given, so an
interrupted sweep picks up where it stopped.

Each decoder script provides two functions:
    cv_fold(X_train, y_train, X_test, y_test) -> (y_true, y_pred)
        Fit on one fold's standardized training data and predict its test
        data. Returns velocity targets and predictions, both (n, 2).
    fit_final(X_train, y_train, scaler, suffix, meta)
        Fit on the standardized chronological training split and save the
        bundle. `meta` holds split/duration fields for the bundle's config.
"""

import os

import h5py
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from bmi.metrics import pearson_corrcoef
from bmi.preprocessing import TimeSeriesSplitCustom, chronological_holdout_split


def add_eval_args(parser):
    """CLI arguments shared by every decoder evaluation script."""
    parser.add_argument('--input_filepath', type=str, required=True,
                        help='ANN dataset (make_dataset.py output)')
    parser.add_argument('--output_filepath', type=str, required=True,
                        help='CV results file; per-duration results get a _{N}min suffix')
    parser.add_argument('--feature', type=str, default='mua', choices=['sua', 'mua'],
                        help='Which input matrix to use: X_sua or X_mua')
    parser.add_argument('--n_folds', type=int, default=0,
                        help='Number of CV folds (default 0: no cross-validation, only the final model)')
    parser.add_argument('--min_train_size', type=float, default=0.5,
                        help='Minimum training size of the first CV fold (fraction of rows)')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Test size of each CV fold (fraction of rows)')
    parser.add_argument('--model_dir', type=str, default='',
                        help='If set, fit and save a final model bundle per duration here')
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Held-out fraction for the final model\'s chronological split')
    parser.add_argument('--train_durations', type=str, default='',
                        help='Comma-separated training durations in minutes (e.g. "1,2,5"). '
                             'If omitted, train once on all available training data.')
    parser.add_argument('--step_ms', type=float, default=4.0,
                        help='Time between consecutive dataset rows (wdw_time - ol_time used '
                             'by make_dataset.py), in ms')
    parser.add_argument('--wdw_time', type=float, default=0.256,
                        help='Window width used by make_dataset.py, in s; only used to size '
                             'the default purge gap')
    parser.add_argument('--gap_samples', type=int, default=None,
                        help='Rows purged between train and test. Default: see default_gap_samples()')
    parser.add_argument('--n_train_override', type=int, default=None,
                        help='Train/test boundary (in rows) for the final model. Required for '
                             'trial-structured NWB datasets (combine_trial_windows_to_ann_h5.py '
                             'stores it as the n_train attribute).')
    parser.add_argument('--overwrite', '--force_retrain', dest='overwrite', action='store_true',
                        help='Recompute results and model bundles even if they already exist')
    return parser


def minutes_to_samples(minutes, step_ms):
    """Number of dataset rows spanning `minutes`, given the row spacing."""
    return int(round(minutes * 60000.0 / step_ms))


def parse_train_durations(train_durations):
    """'1,2,5' -> [1.0, 2.0, 5.0]; '' -> [None] (a single full-data run)."""
    durations = [float(x) for x in train_durations.split(',') if x.strip()]
    return durations or [None]


def duration_suffix(duration_minutes):
    """Filename suffix for a duration: '' for full data, else e.g. '_5min'."""
    return "" if duration_minutes is None else f"_{duration_minutes:g}min"


def add_suffix_to_path(path, suffix):
    """add_suffix_to_path('results.h5', '_5min') -> 'results_5min.h5'"""
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def default_gap_samples(wdw_time, step_ms, timesteps=1):
    """Purge gap, in rows, between training and test data.

    With 4 ms steps, consecutive rows share almost all of their raw window,
    so the training rows right before the boundary carry information about
    the first test rows. A decoder that stacks `timesteps` rows per example
    (WF, LSTM, QRNN) reaches `timesteps - 1` rows further back. The gap is
    twice that reach: 2 * (window_rows + timesteps - 1).
    """
    window_width_rows = wdw_time * 1000.0 / step_ms
    return int(np.ceil(2 * (window_width_rows + timesteps - 1)))


def _outputs_exist(paths, overwrite, label):
    if not overwrite and all(os.path.exists(p) for p in paths):
        print(f"[skip] {label}: {paths[0]} already exists")
        return True
    return False


def cross_validate(args, X, y, gap_samples, n_train_samples, cv_fold):
    """Run chronological K-fold CV.

    Returns (rmse_folds, cc_folds, y_true_last, y_pred_last). A fold whose
    training set is shorter than `n_train_samples` scores NaN.
    """
    tscv = TimeSeriesSplitCustom(n_splits=args.n_folds,
                                 test_size=int(args.test_size * len(y)),
                                 min_train_size=int(args.min_train_size * len(y)),
                                 gap=gap_samples)
    rmse_folds, cc_folds = [], []
    y_true_last = y_pred_last = None
    for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
        fold = f"Fold-{fold_i + 1}/{args.n_folds}"
        if n_train_samples is not None and n_train_samples > len(train_idx):
            print(f"  {fold}: needs {n_train_samples} training rows, only {len(train_idx)} "
                  f"available; scoring NaN")
            rmse_folds.append(np.nan)
            cc_folds.append(np.nan)
            continue
        if n_train_samples is not None:
            train_idx = train_idx[-n_train_samples:]
        print(f"  {fold}: {len(train_idx)} train rows, {len(test_idx)} test rows", flush=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_true, y_pred = cv_fold(X_train, y[train_idx], X_test, y[test_idx])

        rmse_folds.append(root_mean_squared_error(y_true, y_pred))
        cc_folds.append(pearson_corrcoef(y_true, y_pred))
        print(f"  {fold}: RMSE={rmse_folds[-1]:.2f}, CC={cc_folds[-1]:.2f}", flush=True)
        y_true_last, y_pred_last = y_true, y_pred
    return rmse_folds, cc_folds, y_true_last, y_pred_last


def write_cv_results(path, rmse_folds, cc_folds, y_test, y_test_pred,
                     duration_minutes, n_train_samples, gap_samples):
    print(f"Storing results into file: {path}")
    with h5py.File(path, 'w') as f:
        if y_test is not None:
            f['y_test'] = y_test
            f['y_test_pred'] = y_test_pred
        f['rmse_test_folds'] = np.asarray(rmse_folds)
        f['cc_test_folds'] = np.asarray(cc_folds)
        if duration_minutes is not None:
            f.attrs['train_duration_minutes'] = duration_minutes
            f.attrs['train_duration_samples'] = n_train_samples
        f.attrs['split_gap_samples'] = gap_samples


def run_evaluation(args, X, y, gap_samples, cv_fold, fit_final=None, bundle_paths=None,
                   on_duration_start=None):
    """CV and (optionally) a final model fit for every requested duration.

    bundle_paths(suffix) lists the files of a saved bundle; if all exist the
    final fit is skipped. on_duration_start(), if given, runs before each
    duration's work (the DL script uses it to reseed).
    """
    final_gap = 0 if args.n_train_override is not None else gap_samples
    print(f"Purge gap: {gap_samples} rows between CV folds, {final_gap} before the final model's "
          f"test split; {60000.0 / args.step_ms:.1f} rows/minute")
    for duration_minutes in parse_train_durations(args.train_durations):
        suffix = duration_suffix(duration_minutes)
        if duration_minutes is None:
            n_train_samples = None
            print("\n=== Training on all available training data ===")
        else:
            n_train_samples = minutes_to_samples(duration_minutes, args.step_ms)
            print(f"\n=== Training on {duration_minutes:g} minute(s) "
                  f"({n_train_samples} rows) ===")
        if on_duration_start is not None:
            on_duration_start()

        results_path = add_suffix_to_path(args.output_filepath, suffix)
        if args.n_folds > 0 and not _outputs_exist([results_path], args.overwrite, "CV results"):
            rmse, cc, y_true, y_pred = cross_validate(args, X, y, gap_samples,
                                                      n_train_samples, cv_fold)
            write_cv_results(results_path, rmse, cc, y_true, y_pred,
                             duration_minutes, n_train_samples, gap_samples)

        if not args.model_dir or fit_final is None:
            continue
        os.makedirs(args.model_dir, exist_ok=True)
        if _outputs_exist(bundle_paths(suffix), args.overwrite, "final model bundle"):
            continue

        X_train, y_train, _, _ = chronological_holdout_split(
            X, y, args.test_frac, gap=final_gap, n_train_override=args.n_train_override)
        if n_train_samples is not None:
            if n_train_samples > len(y_train):
                print(f"  Not saving a final model: only {len(y_train)} training rows available")
                continue
            X_train, y_train = X_train[-n_train_samples:], y_train[-n_train_samples:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        meta = {'test_frac': args.test_frac,
                'feature': args.feature,
                'train_duration_minutes': duration_minutes,
                'train_duration_samples': n_train_samples,
                'split_gap_samples': final_gap}
        fit_final(X_train, y_train, scaler, suffix, meta)
        print(f"Saved final model bundle to {args.model_dir} "
              f"(duration={duration_minutes if duration_minutes is not None else 'full'})")

"""
Evaluating deep learning based BMI decoders implemented with TensorFlow
"""

# import packages

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from bmi.preprocessing import TimeSeriesSplitCustom, transform_data, chronological_holdout_split
from bmi.utils import seed_tensorflow, count_params
from bmi.decoders import QRNNDecoder, LSTMDecoder, MLPDecoder
from sklearn.metrics import root_mean_squared_error
from bmi.metrics import pearson_corrcoef
import time as timer
import pickle as pkl  # add to imports


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


def final_model_bundle_paths(model_dir, decoder, duration_minutes, duration_tag):
    """Return the (weights_path, scaler_path, config_path) a final model
    bundle for this decoder/duration would be saved to. Centralized here
    so the "does this already exist" check and the "save it" code can't
    drift apart.
    """
    model_tag = f"_{duration_tag}" if duration_minutes is not None else ""
    weights_path = os.path.join(model_dir, f"{decoder}{model_tag}.weights.h5")
    scaler_path  = os.path.join(model_dir, f"{decoder}{model_tag}_scaler.pkl")
    config_path  = os.path.join(model_dir, f"{decoder}{model_tag}_config.json")
    return weights_path, scaler_path, config_path


def bundle_exists(paths):
    """True only if every file in the bundle is present (a partial bundle
    from an interrupted run should NOT be treated as cached)."""
    return all(os.path.exists(p) for p in paths)


def resolve_gap_samples(args, decoder, timesteps):
    """Same purpose as the KF/WF scripts' versions of this function. The
    (timesteps - 1) tap-stacking term only applies to LSTM/QRNN -- MLP
    never calls transform_data, so its examples reach no further into the
    past than the window itself, same as KF. `timesteps` here is whatever
    ended up in `config` (a CLI default OR a loaded --config_filepath
    value), NOT necessarily args.timesteps -- see the call site in main().
    """
    if args.gap_samples is not None:
        return args.gap_samples
    window_width_rows = args.wdw_time * 1000.0 / args.step_ms
    effective_timesteps = timesteps if decoder in ('qrnn', 'lstm') else 1
    gap_samples = int(np.ceil(2 * (window_width_rows + effective_timesteps - 1)))
    print(f"--gap_samples not set; using computed default {gap_samples} "
          f"(2 x [window_width_rows={window_width_rows:.1f} + "
          f"(effective_timesteps-1)={effective_timesteps - 1}], decoder={decoder}, "
          f"from --wdw_time={args.wdw_time}s, --step_ms={args.step_ms})")
    return gap_samples


def main(args):
    run_start = timer.time()
    
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]   
    # select the x-y velocity components
    y = y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)

    print("Hyperparameter configuration setting")
    if args.config_filepath:
        # open JSON hyperparameter configuration file
        print(f"Using hyperparameter configuration from a file: {args.config_filepath}")
        with open(args.config_filepath, 'r') as f:
            config = json.load(f)
        
    else:
        # define model configuration
        config = {'timesteps'    : args.timesteps,
                  'n_layers'     : args.n_layers,
                  'units'        : args.units,
                  'batch_size'   : args.batch_size,
                  'learning_rate': args.learning_rate,
                  'dropout'      : args.dropout,
                  'optimizer'    : args.optimizer,
                  'epochs'       : args.epochs}
    config['input_dim'] = X.shape[-1]
    config['output_dim'] = y.shape[-1]
    config['window_size'] = args.window_size
    config['loss'] = args.loss
    config['metric'] = args.metric
    print(f"Hyperparameter configuration: {config}")

    # set seed for reproducibility
    seed_tensorflow(args.seed)

    # NOTE: gap resolution uses config['timesteps'], not args.timesteps --
    # when --config_filepath is set, config['timesteps'] may come from that
    # file rather than the CLI default, and that's the value that actually
    # governs how far transform_data() reaches into the past for LSTM/QRNN.
    gap_samples = resolve_gap_samples(args, args.decoder, config['timesteps'])

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

        # ------------------------------------------------------------------
        # CACHE CHECK #1: CV evaluation results for this duration.
        # ------------------------------------------------------------------
        if duration_minutes is not None:
            output_filepath_d = add_suffix_to_path(args.output_filepath, duration_tag)
        else:
            output_filepath_d = args.output_filepath

        cv_already_done = os.path.exists(output_filepath_d) and not args.force_retrain
        if cv_already_done:
            print(f"  [cache hit] {output_filepath_d} already exists -- skipping CV "
                  f"evaluation for duration={duration_tag} (use --force_retrain to redo)")
        else:
            rmse_test_folds = []
            cc_test_folds = []

            y_test_fold_last = None
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
                    X_train = X_train_full[-n_train_samples:, :]
                    y_train = y_train_full[-n_train_samples:, :]
                else:
                    X_train = X_train_full
                    y_train = y_train_full

                # standardize input data
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test_fold = scaler.transform(X_test)

                if (args.decoder == 'qrnn') or (args.decoder == 'lstm'):
                    # transform data into sequence data
                    X_train, y_train = transform_data(X_train, y_train, timesteps=config['timesteps'])
                    X_test_fold, y_test_fold = transform_data(X_test_fold, y_test, timesteps=config['timesteps'])
                else:
                    y_test_fold = y_test

                # Create and compile model
                print("Compiling and training a model")
                if args.decoder == 'qrnn':
                    model = QRNNDecoder(config)
                elif args.decoder == 'lstm':
                    model = LSTMDecoder(config)
                elif args.decoder == 'mlp':
                    model = MLPDecoder(config)
                total_count, _, _ = count_params(model)
                # fit model
                train_start = timer.time()
                history = model.fit(X_train, y_train, validation_data=None, epochs=config['epochs'],
                                     verbose=args.verbose, callbacks=None)
                train_end = timer.time()
                train_time = (train_end - train_start) / 60
                print(f"Training the model took {train_time:.2f} minutes")

                # predict using the trained model
                y_test_pred = model.predict(X_test_fold, batch_size=config['batch_size'], verbose=args.verbose)

                # evaluate performance
                print("Evaluating the model performance")
                rmse_test = root_mean_squared_error(y_test_fold, y_test_pred)
                cc_test = pearson_corrcoef(y_test_fold, y_test_pred)

                rmse_test_folds.append(rmse_test)
                cc_test_folds.append(cc_test)

                y_test_fold_last = y_test_fold
                y_test_pred_last = y_test_pred

            for i in range(args.n_folds):
                cc_display = np.mean(cc_test_folds[i]) if hasattr(cc_test_folds[i], '__len__') else cc_test_folds[i]
                print(f"Fold-{i + 1} | RMSE test = {rmse_test_folds[i]:.2f}, CC test = {cc_display:.2f}")

            print(f"Storing results into file: {output_filepath_d}")
            with h5py.File(output_filepath_d, 'w') as f:
                if y_test_fold_last is not None:
                    f['y_test'] = y_test_fold_last
                    f['y_test_pred'] = y_test_pred_last
                f['rmse_test_folds'] = np.asarray(rmse_test_folds)
                f['cc_test_folds'] = np.asarray(cc_test_folds)
                if duration_minutes is not None:
                    f.attrs['train_duration_minutes'] = duration_minutes
                    f.attrs['train_duration_samples'] = n_train_samples
                f.attrs['split_gap_samples'] = gap_samples

        # ------------------------------------------------------------------
        # CACHE CHECK #2: final chronological-split model bundle for this
        # decoder/duration.
        # ------------------------------------------------------------------
        if args.model_dir:
            os.makedirs(args.model_dir, exist_ok=True)

            weights_path, scaler_path, config_path = final_model_bundle_paths(
                args.model_dir, args.decoder, duration_minutes, duration_tag)

            if bundle_exists([weights_path, scaler_path, config_path]) and not args.force_retrain:
                print(f"  [cache hit] final model bundle for decoder={args.decoder}, "
                      f"duration={duration_tag} already exists at {args.model_dir} "
                      f"-- skipping fit (use --force_retrain to redo)")
            else:
                # Chronological split, matching what a downstream comparison script
                # will reproduce -- NOT the CV split above, which exists only for
                # evaluating generalization and doesn't correspond to a fixed test set.
                # Gap-purged via the same shared helper KF/WF's eval scripts use --
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

                    if args.decoder in ('qrnn', 'lstm'):
                        X_train_final, y_train_final = transform_data(
                            X_train_final, y_train_final, timesteps=config['timesteps'])

                    if args.decoder == 'qrnn':
                        final_model = QRNNDecoder(config)
                    elif args.decoder == 'lstm':
                        final_model = LSTMDecoder(config)
                    else:
                        final_model = MLPDecoder(config)

                    seed_tensorflow(args.seed)
                    final_model.fit(X_train_final, y_train_final, validation_data=None,
                                     epochs=config['epochs'], verbose=args.verbose, callbacks=None)

                    final_model.save_weights(weights_path)
                    with open(scaler_path, 'wb') as f:
                        pkl.dump(scaler_final, f)
                    with open(config_path, 'w') as f:
                        json.dump({**config, 'test_frac': args.test_frac, 'feature': args.feature,
                                   'decoder': args.decoder, 'train_duration_minutes': duration_minutes,
                                   'train_duration_samples': n_train_samples,
                                   'split_gap_samples': gap_samples}, f, indent=2)

                    print(f"Saved final {args.decoder} model bundle to {args.model_dir} "
                          f"(duration={duration_tag})")

    run_end = timer.time()
    run_time = (run_end - run_start) / 60
    print(f"Whole processes took {run_time:.2f} minutes")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--input_filepath',   type=str,   help='Path to the dataset file')
    parser.add_argument('--output_filepath',  type=str,   help='Path to the result file')
    parser.add_argument('--seed',             type=float, default=42,      help='Seed for reproducibility')
    parser.add_argument('--feature',          type=str,   default='mua',   help='Type of spiking activity (sua or mua)')
    parser.add_argument('--decoder',          type=str,   default='qrnn',  help='Deep learning based decoding algorithm')
    parser.add_argument('--n_folds',          type=int,   default=2,       help='Number of cross validation folds')
    parser.add_argument('--min_train_size',   type=float, default=0.5,     help='Minimum (fraction) of training data size')
    parser.add_argument('--test_size',        type=float, default=0.1,     help='Testing data size')
    parser.add_argument('--config_filepath',  type=str,   default='',      help='JSON hyperparameter configuration file')
    parser.add_argument('--timesteps',        type=int,   default=5,       help='Number of timesteps')
    parser.add_argument('--n_layers',         type=int,   default=1,       help='Number of layers')
    parser.add_argument('--units',            type=int,   default=600,     help='Number of units (hidden state size)')
    parser.add_argument('--window_size',      type=int,   default=2,       help='Window size')
    parser.add_argument('--dropout',          type=float, default=0.1,     help='Dropout rate')
    parser.add_argument('--optimizer',        type=str,   default='Adam',  help='Optimizer')
    parser.add_argument('--epochs',           type=int,   default=50,      help='Number of epochs')
    parser.add_argument('--batch_size',       type=int,   default=32,      help='Batch size')
    parser.add_argument('--learning_rate',    type=float, default=0.001,   help='Learning rate')
    parser.add_argument('--loss',             type=str,   default='mse',   help='Loss function')
    parser.add_argument('--metric',           type=str,   default='mse',   help='Predictive performance metric')
    parser.add_argument('--verbose',          type=int,   default=0,       help='Wether or not to print the output')

    parser.add_argument('--model_dir', type=str, default='',
                     help='If set, fit and save one final model + scaler for reuse downstream '
                          '(one bundle per training duration, if --train_durations is set)')
    parser.add_argument('--test_frac', type=float, default=0.1,
                     help='Chronological holdout fraction for the saved final model. '
                          'MUST match the downstream script\'s --test_frac.')

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
                          'used to build the input dataset). Used to convert --train_durations '
                          'minutes into a number of samples, and to compute the default purge '
                          'gap (see --wdw_time/--gap_samples).')
    parser.add_argument('--wdw_time', type=float, default=0.256,
                     help='Window width (s) used to build the input dataset -- NOT used for '
                          'windowing here, only to compute a generous default purge gap between '
                          'train/test splits (see --gap_samples). Must match whatever was '
                          'actually passed to make_dataset.py for --input_filepath.')
    parser.add_argument('--gap_samples', type=int, default=None,
                     help='Purge gap, in rows of X, excluded between train and test in both '
                          'the CV folds and the final chronological holdout. If not set, '
                          'computed as 2 * (wdw_time*1000/step_ms + effective_timesteps - 1), '
                          'where effective_timesteps is config["timesteps"] for LSTM/QRNN '
                          '(which stack that many rows per prediction via transform_data) or 1 '
                          'for MLP (which does not). Set to 0 to reproduce the original '
                          '(pre-fix) ungapped behavior.')
    parser.add_argument('--n_train_override', type=int, default=None,
                     help='Explicit train-row count for the FINAL (chronological-split, saved) '
                          'model, bypassing the base_nperseg-aligned split computation entirely. '
                          'For normal, single-continuous-session data, leave unset -- only needed '
                          'for trial-structured data where that computation does not apply (see '
                          'combine_trial_windows_to_ann_h5.py, which writes a matching n_train '
                          'attr onto its own output file -- read that value and pass it here). '
                          'Does NOT affect the CV folds above, which still use the ungapped-by-'
                          'this-fix TimeSeriesSplitCustom split.')

    parser.add_argument('--force_retrain', action='store_true',
                     help='If set, ignore any cached CV-results files and final model '
                          'bundles for each duration and redo everything from scratch. '
                          'By default (flag omitted), a duration is skipped whenever its '
                          'results file already exists, and a final model bundle is '
                          'skipped whenever its weights/scaler/config files all already '
                          'exist -- this lets a long sweep be safely re-run/resumed after '
                          'an interruption.')

    args = parser.parse_args()
    main(args)

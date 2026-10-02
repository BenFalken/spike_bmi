"""
Evaluate a deep-learning velocity decoder (LSTM, QRNN or MLP; TensorFlow)
on one session's ANN dataset.

LSTM and QRNN see --timesteps consecutive rows per example; MLP sees one.
Hyperparameters come from --config_filepath if given (a JSON with timesteps,
n_layers, units, batch_size, learning_rate, dropout, optimizer, epochs),
otherwise from the CLI. See bmi/evaluation.py for the CV, duration-sweep and
model-bundle logic.

Bundle files in --model_dir: {decoder}{suffix}.weights.h5,
{decoder}{suffix}_scaler.pkl, {decoder}{suffix}_config.json.
"""

import argparse
import json
import os
import pickle as pkl
import sys
import time

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.decoders import LSTMDecoder, MLPDecoder, QRNNDecoder
from bmi.evaluation import add_eval_args, default_gap_samples, run_evaluation
from bmi.preprocessing import transform_data
from bmi.utils import count_params, seed_tensorflow

BUILDERS = {'lstm': LSTMDecoder, 'qrnn': QRNNDecoder, 'mlp': MLPDecoder}
SEQUENCE_DECODERS = ('lstm', 'qrnn')


def bundle_paths(model_dir, decoder, suffix):
    return [os.path.join(model_dir, f"{decoder}{suffix}.weights.h5"),
            os.path.join(model_dir, f"{decoder}{suffix}_scaler.pkl"),
            os.path.join(model_dir, f"{decoder}{suffix}_config.json")]


def load_config(args, input_dim, output_dim):
    if args.config_filepath:
        print(f"Using hyperparameter configuration from a file: {args.config_filepath}")
        with open(args.config_filepath, 'r') as f:
            config = json.load(f)
    else:
        config = {'timesteps': args.timesteps, 'n_layers': args.n_layers, 'units': args.units,
                  'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                  'dropout': args.dropout, 'optimizer': args.optimizer, 'epochs': args.epochs}
    config.update(input_dim=input_dim, output_dim=output_dim, window_size=args.window_size,
                  loss=args.loss, metric=args.metric)
    return config


def main(args):
    run_start = time.time()
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()][:, 2:4]  # velocity

    config = load_config(args, X.shape[-1], y.shape[-1])
    print(f"Hyperparameter configuration: {config}")
    is_sequence = args.decoder in SEQUENCE_DECODERS
    build = BUILDERS[args.decoder]

    if args.gap_samples is not None:
        gap_samples = args.gap_samples
    else:
        gap_samples = default_gap_samples(args.wdw_time, args.step_ms,
                                          config['timesteps'] if is_sequence else 1)

    def to_sequences(X_, y_):
        return transform_data(X_, y_, timesteps=config['timesteps']) if is_sequence else (X_, y_)

    def cv_fold(X_train, y_train, X_test, y_test):
        X_train, y_train = to_sequences(X_train, y_train)
        X_test, y_test = to_sequences(X_test, y_test)
        model = build(config)
        count_params(model)
        fit_start = time.time()
        model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                  verbose=args.verbose)
        print(f"  Training took {(time.time() - fit_start) / 60:.2f} minutes")
        return y_test, model.predict(X_test, batch_size=config['batch_size'], verbose=args.verbose)

    def fit_final(X_train, y_train, scaler, suffix, meta):
        X_train, y_train = to_sequences(X_train, y_train)
        model = build(config)
        seed_tensorflow(args.seed)
        model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                  verbose=args.verbose)
        weights_path, scaler_path, config_path = bundle_paths(args.model_dir, args.decoder, suffix)
        model.save_weights(weights_path)
        with open(scaler_path, 'wb') as f:
            pkl.dump(scaler, f)
        with open(config_path, 'w') as f:
            json.dump({**config, **meta, 'decoder': args.decoder}, f, indent=2)

    # Reseeding per duration makes each duration's result independent of
    # which other durations ran (or were skipped as cached) in this process.
    run_evaluation(args, X, y, gap_samples, cv_fold, fit_final,
                   bundle_paths=lambda suffix: bundle_paths(args.model_dir, args.decoder, suffix),
                   on_duration_start=lambda: seed_tensorflow(args.seed))
    print(f"Whole processes took {(time.time() - run_start) / 60:.2f} minutes")


if __name__ == '__main__':
    parser = add_eval_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--decoder', type=str, default='qrnn', choices=list(BUILDERS))
    parser.add_argument('--config_filepath', type=str, default='',
                        help='JSON hyperparameter file; overrides the hyperparameter flags below')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--timesteps', type=int, default=5, help='Rows per example (LSTM/QRNN)')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of recurrent/dense layers')
    parser.add_argument('--units', type=int, default=600, help='Hidden units per layer')
    parser.add_argument('--window_size', type=int, default=2, help='QRNN convolution window')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help="'Adam' or 'RMSprop'")
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training and prediction batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')
    parser.add_argument('--metric', type=str, default='mse', help='Training metric')
    parser.add_argument('--verbose', type=int, default=0, help='Keras verbosity')
    main(parser.parse_args())

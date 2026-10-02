"""
Evaluate a Wiener filter (WF) velocity decoder on one session's ANN dataset.

Each example stacks --timesteps consecutive rows of X (a tapped delay line)
and is flattened into one feature vector for a linear regression onto
velocity. See bmi/evaluation.py for the CV, duration-sweep and model-bundle
logic.

Bundle files in --model_dir: wf{suffix}_model.pkl, wf{suffix}_scaler.pkl,
wf{suffix}_config.json.
"""

import argparse
import json
import os
import pickle as pkl
import sys
import threading
import time

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.decoders import WienerDecoder
from bmi.evaluation import add_eval_args, default_gap_samples, run_evaluation
from bmi.preprocessing import transform_data


class Heartbeat:
    """Print an elapsed-time message every `interval` seconds while the
    `with` block runs. Used around the tap-delay transform and the fit,
    which can take many minutes on large (e.g. 192-channel) sessions and
    otherwise print nothing."""

    def __init__(self, label, interval=30):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()

    def __enter__(self):
        self._start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            print(f"    ...{self.label} still running ({time.time() - self._start:.0f}s elapsed)",
                  flush=True)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1)
        return False


def tap_delay(X, y, timesteps):
    """(n, channels) -> (n - timesteps, channels * timesteps), channel-major."""
    X_seq, y_seq = transform_data(X, y, timesteps=timesteps)
    return X_seq.reshape(X_seq.shape[0], X_seq.shape[1] * X_seq.shape[2], order='F'), y_seq


def bundle_paths(model_dir, suffix):
    return [os.path.join(model_dir, f"wf{suffix}_{name}")
            for name in ("model.pkl", "scaler.pkl", "config.json")]


def main(args):
    run_start = time.time()
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()][:, 2:4]  # velocity

    config = {'reg_type': args.reg_type, 'reg_alpha': args.reg_alpha, 'timesteps': args.timesteps}
    print(f"Hyperparameter configuration: {config}")
    gap_samples = (args.gap_samples if args.gap_samples is not None
                   else default_gap_samples(args.wdw_time, args.step_ms, args.timesteps))

    def fit(X_train, y_train):
        with Heartbeat("tap-delay transform"):
            X_train, y_train = tap_delay(X_train, y_train, args.timesteps)
        print(f"  Fitting on X_train shape={X_train.shape}", flush=True)
        model = WienerDecoder(args.reg_type, args.reg_alpha)
        with Heartbeat("WienerDecoder.fit()"):
            model.fit(X_train, y_train)
        return model, X_train.shape[-1]

    def cv_fold(X_train, y_train, X_test, y_test):
        model, _ = fit(X_train, y_train)
        X_test, y_test = tap_delay(X_test, y_test, args.timesteps)
        return y_test, model.predict(X_test)

    def fit_final(X_train, y_train, scaler, suffix, meta):
        model, flattened_dim = fit(X_train, y_train)
        model_path, scaler_path, config_path = bundle_paths(args.model_dir, suffix)
        with open(model_path, 'wb') as f:
            pkl.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pkl.dump(scaler, f)
        with open(config_path, 'w') as f:
            json.dump({**config, **meta,
                       'input_dim': X.shape[-1],         # channels
                       'flattened_dim': flattened_dim,   # channels * timesteps
                       'decoder': 'wf'}, f, indent=2)

    run_evaluation(args, X, y, gap_samples, cv_fold, fit_final,
                   bundle_paths=lambda suffix: bundle_paths(args.model_dir, suffix))
    print(f"Whole processes took {(time.time() - run_start) / 60:.2f} minutes")


if __name__ == '__main__':
    parser = add_eval_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--timesteps', type=int, default=4, help='Taps (rows) per example')
    parser.add_argument('--reg_type', type=str, default='',
                        help="Regression type: '', 'l1', 'l2' or 'l12'")
    parser.add_argument('--reg_alpha', type=float, default=0, help='Regularization strength')
    main(parser.parse_args())

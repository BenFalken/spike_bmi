"""
Evaluate a Kalman filter (KF) velocity decoder on one session's ANN dataset.

The KF state is the full 6-D kinematics vector (position, velocity,
acceleration); predictions are seeded with the first test sample's state and
scored on the velocity columns only. See bmi/evaluation.py for the CV,
duration-sweep and model-bundle logic.

Bundle files in --model_dir: kf{suffix}_model.pkl, kf{suffix}_scaler.pkl,
kf{suffix}_config.json.
"""

import argparse
import json
import os
import pickle as pkl
import sys
import time

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.decoders import KalmanDecoder
from bmi.evaluation import add_eval_args, default_gap_samples, run_evaluation

VELOCITY = slice(2, 4)  # y_task columns: pos_x, pos_y, vel_x, vel_y, acc_x, acc_y


def bundle_paths(model_dir, suffix):
    return [os.path.join(model_dir, f"kf{suffix}_{name}")
            for name in ("model.pkl", "scaler.pkl", "config.json")]


def main(args):
    run_start = time.time()
    print(f"Reading dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y = f['y_task'][()]

    config = {'reg_type': args.reg_type, 'reg_alpha': args.reg_alpha}
    print(f"Hyperparameter configuration: {config}")
    gap_samples = (args.gap_samples if args.gap_samples is not None
                   else default_gap_samples(args.wdw_time, args.step_ms))

    def cv_fold(X_train, y_train, X_test, y_test):
        model = KalmanDecoder(args.reg_type, args.reg_alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test, y_test[:1, :])
        return y_test[:, VELOCITY], y_pred[:, VELOCITY]

    def fit_final(X_train, y_train, scaler, suffix, meta):
        model = KalmanDecoder(args.reg_type, args.reg_alpha)
        model.fit(X_train, y_train)
        model_path, scaler_path, config_path = bundle_paths(args.model_dir, suffix)
        with open(model_path, 'wb') as f:
            pkl.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pkl.dump(scaler, f)
        with open(config_path, 'w') as f:
            json.dump({**config, **meta, 'input_dim': X.shape[-1], 'decoder': 'kf'}, f, indent=2)

    run_evaluation(args, X, y, gap_samples, cv_fold, fit_final,
                   bundle_paths=lambda suffix: bundle_paths(args.model_dir, suffix))
    print(f"Whole processes took {(time.time() - run_start) / 60:.2f} minutes")


if __name__ == '__main__':
    parser = add_eval_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--reg_type', type=str, default='',
                        help="Regression used to fit the KF matrices: '', 'l1', 'l2' or 'l12'")
    parser.add_argument('--reg_alpha', type=float, default=0, help='Regularization strength')
    main(parser.parse_args())

"""
Split one session's SNN dataset (make_snn_dataset.py output) chronologically
into train/test and write one .pkl per window, the format read by
snn_training/dataset.py.

Output layout: {output_path}/train/{i}.pkl and {output_path}/test/{i}.pkl,
each a dict with
    input_spikes       (n_units, nperseg)  float32
    velocity           (nperseg, 2)        float32
    window_start_time  float
    y_end              (2,)                float32

The boundary comes from bmi.preprocessing.aligned_train_boundary() applied to
the raw session length, the same computation the ANN decoders use, so both
hold out the same stretch of recording. Existing .pkl files in train/ and
test/ are deleted first.
"""

import argparse
import os
import pickle as pkl
import sys

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.preprocessing import aligned_train_boundary


def main(args):
    print(f"Loading SNN dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X_raster = f['X_raster'][()]
        y_trace = f['y_trace'][()]
        y_end = f['y_end'][()]
        window_start_time = f['window_start_time'][()]
        total_raw_samples = f.attrs['total_raw_samples']
    print(f"Velocity stats: mean={y_trace.mean():.2f}, std={y_trace.std():.2f}, "
          f"max={y_trace.max():.2f}, min={y_trace.min():.2f}")

    n_windows, _, nperseg = X_raster.shape
    n_train = aligned_train_boundary(total_raw_samples, args.test_frac, base_nperseg=nperseg) // nperseg
    n_test = n_windows - n_train
    if n_train <= 0 or n_test <= 0:
        raise ValueError(f"test_frac={args.test_frac} gives n_train={n_train}, n_test={n_test} "
                         f"for {n_windows} windows.")
    print(f"Total windows: {n_windows} -> train: {n_train}, test: {n_test} "
          f"(total_raw_samples={total_raw_samples}, test_frac={args.test_frac})")

    split_dirs = {name: os.path.join(args.output_path, name) for name in ('train', 'test')}
    for d in split_dirs.values():
        os.makedirs(d, exist_ok=True)
        for fname in os.listdir(d):
            if fname.endswith('.pkl'):
                os.remove(os.path.join(d, fname))

    for i in tqdm(range(n_windows), desc="Writing .pkl trials"):
        sample = {
            'input_spikes': X_raster[i].astype(np.float32),
            'velocity': y_trace[i].astype(np.float32),
            'window_start_time': window_start_time[i],
            'y_end': y_end[i].astype(np.float32),
        }
        if i < n_train:
            out_path = os.path.join(split_dirs['train'], f'{i}.pkl')
        else:
            out_path = os.path.join(split_dirs['test'], f'{i - n_train}.pkl')
        with open(out_path, 'wb') as fh:
            pkl.dump(sample, fh)

    print(f"Wrote {n_train} train files to {split_dirs['train']}")
    print(f"Wrote {n_test} test files to {split_dirs['test']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_filepath', type=str, required=True,
                        help='SNN dataset (make_snn_dataset.py output)')
    parser.add_argument('--output_path', type=str, default='datasets/bmi/',
                        help="Directory to write train/ and test/ into")
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Held-out fraction; must match the ANN decoders\' --test_frac')
    main(parser.parse_args())

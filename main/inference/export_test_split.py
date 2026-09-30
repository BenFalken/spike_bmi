"""
Write test-split-only copies of the ANN datasets, for running inference on a
machine with little disk space (e.g. the Speck-connected laptop).

For every {input_dir}/*_binning.h5 this keeps the rows test_all_decoders.py
evaluates (the same boundary it computes) of X_{feature} and y_task, and sets
the n_train attribute to 0, so test_all_decoders.py treats the whole copy as
the test split. The original boundary is kept as the source_n_train attribute.

Usage:
    python export_test_split.py \
        --input_dir  $BMI_DATA_ROOT/dataset/bmi/indy/mua \
        --output_dir /path/to/laptop_copy/dataset/bmi/indy/mua
"""

import argparse
import glob
import os
import sys

import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decoder_eval import test_split_start  # noqa: E402


def export(path, output_path, feature, test_frac):
    with h5py.File(path, 'r') as f:
        X = f[f'X_{feature}']
        n_train = test_split_start(len(X), test_frac, f.attrs.get('n_train'))
        with h5py.File(output_path, 'w') as out:
            out.create_dataset(f'X_{feature}', data=X[n_train:], compression='gzip')
            out.create_dataset('y_task', data=f['y_task'][n_train:], compression='gzip')
            out.attrs.update({k: v for k, v in f.attrs.items() if k != 'n_train'})
            out.attrs.update(n_train=0, source_n_train=n_train, source_test_frac=test_frac)
        return len(X) - n_train, len(X)


def main(args):
    paths = sorted(glob.glob(os.path.join(args.input_dir, '*_binning.h5')))
    if not paths:
        raise FileNotFoundError(f"No *_binning.h5 in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    for path in paths:
        output_path = os.path.join(args.output_dir, os.path.basename(path))
        if os.path.abspath(output_path) == os.path.abspath(path):
            raise ValueError("--output_dir must differ from --input_dir")
        n_test, n_rows = export(path, output_path, args.feature, args.test_frac)
        print(f"{os.path.basename(path)}: kept {n_test} of {n_rows} rows -> {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', required=True, help='Directory of full *_binning.h5 datasets')
    parser.add_argument('--output_dir', required=True, help='Where to write the test-only copies')
    parser.add_argument('--feature', default='mua', choices=['sua', 'mua'])
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Must match the value the models were trained with')
    main(parser.parse_args())

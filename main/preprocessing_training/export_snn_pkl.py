"""
Convert the sliding-window SNN dataset (make_snn_dataset.py output, a single .h5
file) into the per-window .pkl format expected by CustomDataset / create_dataloaders
(bmi_dataset.py) -- so the existing training code needs zero changes.

Shape compatibility (no reshaping needed -- verified against CustomDataset.__getitem__):
  'input_spikes' : (num_units, nperseg)  -- CustomDataset applies .T itself to get
                    (time_bins, units); our X_raster[i] is already in this pre-transpose shape.
  'velocity'      : (nperseg, 2)          -- CustomDataset uses this directly, no transpose;
                    our y_trace[i] is already in this shape.


Train/test split: chronological holdout on window index, boundary aligned
to a multiple of nperseg (256ms at this project's native 4ms/sample).
This replaces both the original random per-trial shuffle (which risked
test windows sitting immediately adjacent in time to training windows)
AND a later, still-imperfect fix that computed n_test as a naive
round(test_frac * N) -- that rounds against N (this file's OWN trial
count, non-overlapping windowing), a completely different granularity
than the ANN dataset's own row count (dense, step=1 windowing) uses for
ITS train/test boundary (bmi.preprocessing.chronological_holdout_split()).
Even with the identical nominal --test_frac, two independent roundings
against very different granularities don't generally land on the same
real moment -- confirmed in practice on a real session (found to be the
exact same recording on both sides, but offset by 332 raw samples, purely
from this). compute_aligned_train_boundary() below computes the boundary
from total_raw_samples (an attr saved directly by the CURRENT
make_snn_dataset.py) using the IDENTICAL formula
chronological_holdout_split() uses on the ANN side -- MUST be kept in
sync with that function if either ever changes; they are two independent
copies of one formula living in two separate, currently-unmerged project
directories, not an import relationship.
"""

import argparse
import os
import h5py
import numpy as np
import pickle as pkl
from tqdm import tqdm


def compute_aligned_train_boundary(total_raw_samples, test_frac, base_nperseg):
    """MUST exactly match bmi.preprocessing.chronological_holdout_split()'s
    internal boundary computation -- see module docstring. Returns
    n_train_raw, an exact multiple of base_nperseg."""
    naive_n_test_raw = round(test_frac * total_raw_samples)
    naive_n_train_raw = total_raw_samples - naive_n_test_raw
    return (naive_n_train_raw // base_nperseg) * base_nperseg


def main(args):
    print(f"Loading SNN dataset from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        X_raster = f['X_raster'][()]
        y_trace = f['y_trace'][()]
        y_end = f['y_end'][()]
        window_start_time = f['window_start_time'][()]
        total_raw_samples = f.attrs.get('total_raw_samples')

    if total_raw_samples is None:
        raise KeyError(
            f"{args.input_filepath} has no 'total_raw_samples' attribute -- this file was "
            f"built by an older make_snn_dataset.py that didn't save it. Re-run the CURRENT "
            f"make_snn_dataset.py to regenerate {args.input_filepath} before running this "
            f"script; the aligned train/test split below depends on this value being exact, "
            f"not reconstructed (see module docstring for why reconstruction isn't reliable "
            f"for this file's non-overlapping windowing).")

    print(f"Velocity stats: mean={y_trace.mean():.2f}, std={y_trace.std():.2f}, max={y_trace.max():.2f}, min={y_trace.min():.2f}")

    N = X_raster.shape[0]
    nperseg = X_raster.shape[-1]
    if nperseg != 65:
        print(f"WARNING: nperseg={nperseg}, not the project's usual 65 (256ms @ 4ms/sample). "
              f"eval_all_decoders.py's build_snn_ann_alignment()/compute_aligned_split() "
              f"hardcode base_nperseg=65 -- if this session's windowing genuinely differs, "
              f"those need updating to match, or the alignment fix here won't agree with them.")

    n_train_raw = compute_aligned_train_boundary(total_raw_samples, args.test_frac, base_nperseg=nperseg)
    assert n_train_raw % nperseg == 0
    n_train = n_train_raw // nperseg
    n_test = N - n_train
    print(f"Total windows: {N} -> train: {n_train}, test: {n_test} "
          f"(chronological holdout, aligned to {nperseg}-sample boundary from "
          f"total_raw_samples={total_raw_samples}, test_frac={args.test_frac})")
    if n_test <= 0 or n_train <= 0:
        raise ValueError(
            f"Aligned split gives n_train={n_train}, n_test={n_test} -- test_frac="
            f"{args.test_frac} is too extreme for {N} trials at nperseg={nperseg}.")

    train_dir = os.path.join(args.output_path, 'train')
    test_dir = os.path.join(args.output_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # clear any pre-existing .pkl files so counters stay contiguous and consistent with this run
    for d in (train_dir, test_dir):
        for fname in os.listdir(d):
            if fname.endswith('.pkl'):
                os.remove(os.path.join(d, fname))

    # tqdm here specifically because this loop previously gave ZERO
    # progress feedback between "Total windows: N -> ..." and "Wrote N
    # train files" -- for large N (thousands of individual small .pkl
    # files, genuinely slow on a shared/network filesystem), that silence
    # is indistinguishable in a SLURM log from a real hang. See
    # single_subject_pipeline.py's run_cmd() for the matching stdout-
    # buffering fix on the calling side.
    for i in tqdm(range(N), desc="Writing .pkl trials"):
        sample = {
            'input_spikes': X_raster[i].astype(np.float32),  # (num_units, nperseg)
            'velocity': y_trace[i].astype(np.float32),         # (nperseg, 2)
            'window_start_time': window_start_time[i],
            'y_end': y_end[i].astype(np.float32),
        }
        if i < n_train:
            out_path = os.path.join(train_dir, f'{i}.pkl')
        else:
            out_path = os.path.join(test_dir, f'{i - n_train}.pkl')
        with open(out_path, 'wb') as fh:
            pkl.dump(sample, fh)

    print(f"Wrote {n_train} train files to {train_dir}")
    print(f"Wrote {n_test} test files to {test_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, required=True,
                         help='Path to the SNN dataset .h5 file (make_snn_dataset.py output)')
    parser.add_argument('--output_path', type=str, default='datasets/bmi/',
                         help="Base directory to write 'train/' and 'test/' subfolders into "
                              "(matches CustomDataset's default data_path)")
    parser.add_argument('--test_frac', type=float, default=0.1,
                         help='Fraction of windows (chronologically last) held out as test -- '
                              'MUST match the ANN side\'s --test_frac (see module docstring)')
    args = parser.parse_args()
    main(args)
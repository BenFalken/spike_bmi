"""
Sanity-check one session's ANN and SNN datasets; exit with status 1 on failure.

The two datasets use different windowing (dense 4 ms steps vs. back-to-back
windows), so each is checked on its own:
  - required keys are present and all arrays have the same number of rows
  - summary statistics, and channels whose values never vary (dead channels)
  - SNN only: if the file claims non_overlapping, consecutive windows start
    exactly nperseg * delta_time apart

Usage:
    python check_datasets.py --dataset_dirname DIR --raw_stem indy_20160627_01 --methods binning
"""

import argparse
import os
import sys

import h5py
import numpy as np


def _report_array_stats(name, arr):
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}, std={arr.std():.4f}")


def _check_zero_variance_channels(name, X):
    """X is (n_windows, n_units[, ...]); warn about units that never vary."""
    if X.ndim < 2:
        return
    per_unit_std = X.reshape(X.shape[0], X.shape[1], -1).std(axis=(0, 2))
    dead = np.where(per_unit_std < 1e-12)[0]
    if len(dead):
        print(f"  WARNING: {name} has {len(dead)} near-zero-variance channel(s) "
              f"(indices: {dead.tolist()[:20]}{'...' if len(dead) > 20 else ''})")
    else:
        print(f"  OK: no near-zero-variance channels in {name}")


def _load(path, keys):
    """Returns ({key: array}, attrs) or None if the file or a key is missing."""
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    with h5py.File(path, 'r') as f:
        missing = [k for k in keys if k not in f]
        if missing:
            print(f"  MISSING KEYS: {missing}")
            return None
        return {k: f[k][()] for k in keys}, dict(f.attrs)


def _rows_consistent(arrays):
    counts = {k: len(v) for k, v in arrays.items()}
    if len(set(counts.values())) != 1:
        print(f"  ROW COUNT MISMATCH: {counts}")
        return False
    print(f"  Row count: {next(iter(counts.values()))} (consistent across {', '.join(counts)})")
    return True


def check_ann_dataset(dataset_dirname, raw_stem, method):
    path = os.path.join(dataset_dirname, f"{raw_stem}_{method}.h5")
    print(f"\n=== ANN dataset ({method}): {path} ===")
    loaded = _load(path, ('X_sua', 'X_mua', 'y_task'))
    if loaded is None:
        return False
    d, _ = loaded
    ok = _rows_consistent(d)
    for key in ('X_sua', 'X_mua'):
        _report_array_stats(key, d[key])
        _check_zero_variance_channels(key, d[key])
    _report_array_stats('y_task', d['y_task'])
    _report_array_stats('y_task velocity (cols 2:4)', d['y_task'][:, 2:4])
    return ok


def check_snn_dataset(dataset_dirname, raw_stem):
    path = os.path.join(dataset_dirname, f"{raw_stem}_snn.h5")
    print(f"\n=== SNN dataset: {path} ===")
    loaded = _load(path, ('X_raster', 'y_trace', 'y_end', 'window_start_time'))
    if loaded is None:
        return False
    d, attrs = loaded
    wdw_time, delta_time = attrs.get('wdw_time'), attrs.get('delta_time', 0.004)
    non_overlapping = bool(attrs.get('non_overlapping', False))
    print(f"  feature={attrs.get('feature')}, wdw_time={wdw_time}, ol_time={attrs.get('ol_time')}, "
          f"non_overlapping={non_overlapping}")

    ok = _rows_consistent(d)
    _report_array_stats('X_raster (per-bin spikes)', d['X_raster'])
    _check_zero_variance_channels('X_raster', d['X_raster'])
    _report_array_stats('y_trace velocity', d['y_trace'])
    _report_array_stats('y_end velocity (lagged target)', d['y_end'])

    starts = d['window_start_time']
    if non_overlapping and len(starts) > 1:
        expected_step = (round(wdw_time / delta_time) + 1) * delta_time
        observed_step = np.diff(starts)
        bad = np.where(~np.isclose(observed_step, expected_step, atol=1e-9))[0]
        if len(bad):
            print(f"  FAIL: {len(bad)} consecutive window gap(s) differ from "
                  f"{expected_step * 1000:.3f} ms (first at index {bad[0]}: "
                  f"{observed_step[bad[0]] * 1000:.3f} ms)")
            ok = False
        else:
            print(f"  OK: all {len(observed_step)} window gaps are {expected_step * 1000:.3f} ms "
                  f"(back-to-back, no shared samples)")
    return ok


def main(args):
    all_ok = True
    for method in args.methods.split(','):
        all_ok &= check_ann_dataset(args.dataset_dirname, args.raw_stem, method.strip())
    all_ok &= check_snn_dataset(args.dataset_dirname, args.raw_stem)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'ONE OR MORE CHECKS FAILED'}")
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset_dirname', type=str, required=True,
                        help='Directory containing {raw_stem}_{method}.h5 and {raw_stem}_snn.h5')
    parser.add_argument('--raw_stem', type=str, required=True, help='Session identifier')
    parser.add_argument('--methods', type=str, default='binning',
                        help='Comma-separated ANN dataset methods to check (e.g. binning,gaussian)')
    main(parser.parse_args())

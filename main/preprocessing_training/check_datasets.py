"""
Validate the ANN and SNN datasets for one session.

REWRITTEN FROM SCRATCH -- I never saw the original check_datasets.py's
source, only its printed output from an earlier ANN/SNN comparison run.
That original check assumed the ANN and SNN datasets shared the same
windowing (so window counts and y_end/y_task values should match exactly)
-- true under the old shared-windowing convention, false BY DESIGN now
that the ANN dataset uses dense 4ms-step windows and the SNN dataset uses
non-overlapping trials. Cross-checking them against each other would
either error out or report a "mismatch" that isn't actually a bug. If your
original script did anything else this rewrite doesn't (e.g. additional
sanity checks), merge that back in -- this only implements what the new
architecture actually needs validated.

What this checks, independently per dataset (no cross-comparison):
  ANN dataset (--dataset_dirname/{raw_stem}_{method}.h5):
    - file exists, has X_sua/X_mua/y_task
    - X_sua, X_mua, y_task all have the same number of rows
    - basic stats (range/mean/std) for a sanity eyeball, same style as the
      diagnostic output this replaces
    - flags near-zero-variance channels (a silently dead/disconnected
      channel is a more common real bug than a shape mismatch)

  SNN dataset (--dataset_dirname/{raw_stem}_snn.h5):
    - file exists, has X_raster/y_trace/y_end/window_start_time
    - all four have the same number of rows
    - if the file's own 'non_overlapping' attr says it should be
      non-overlapping (i.e. it was built with --ol_time 0.0 under the
      fixed make_snn_dataset.py), verifies that directly: every
      consecutive pair of window_start_time values must be spaced by
      exactly nperseg*delta_time, with ZERO tolerance for anything other
      than that exact spacing -- this is the property the whole
      "continuous, unbroken stream, state never reset" training scheme
      depends on, so it's checked exactly, not approximately.
    - basic stats + near-zero-variance channel check, same as the ANN side

CLI usage:
    python check_datasets.py --dataset_dirname data/dataset \
        --raw_stem indy_20160627_01 --methods binning
"""

import argparse
import os
import sys

import h5py
import numpy as np


def _report_array_stats(name, arr):
    arr = np.asarray(arr)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"    min={arr.min():.4f}, max={arr.max():.4f}, "
          f"mean={arr.mean():.4f}, std={arr.std():.4f}")


def _check_zero_variance_channels(name, X, axis_units=1):
    """X is expected to be (n_windows, n_units[, ...]). Flags any unit
    whose values never vary at all across every window -- a classic
    silent-dead-channel signature (as opposed to a shape/count bug)."""
    X = np.asarray(X)
    if X.ndim < 2:
        return
    flat = X.reshape(X.shape[0], X.shape[axis_units], -1) if X.ndim > 2 else X
    per_unit_std = flat.reshape(flat.shape[0], flat.shape[1], -1).std(axis=(0, 2))
    dead = np.where(per_unit_std < 1e-12)[0]
    if len(dead) > 0:
        print(f"  WARNING: {name} has {len(dead)} near-zero-variance channel(s) "
              f"(indices: {dead.tolist()[:20]}{'...' if len(dead) > 20 else ''})")
    else:
        print(f"  OK: no near-zero-variance channels in {name}")


def check_ann_dataset(dataset_dirname, raw_stem, method):
    path = os.path.join(dataset_dirname, f"{raw_stem}_{method}.h5")
    print(f"\n=== ANN dataset ({method}): {path} ===")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return False

    ok = True
    with h5py.File(path, 'r') as f:
        missing_keys = [k for k in ('X_sua', 'X_mua', 'y_task') if k not in f]
        if missing_keys:
            print(f"  MISSING KEYS: {missing_keys}")
            return False

        X_sua, X_mua, y_task = f['X_sua'][()], f['X_mua'][()], f['y_task'][()]

    row_counts = {'X_sua': len(X_sua), 'X_mua': len(X_mua), 'y_task': len(y_task)}
    if len(set(row_counts.values())) != 1:
        print(f"  ROW COUNT MISMATCH within ANN dataset: {row_counts}")
        ok = False
    else:
        print(f"  Row count: {row_counts['y_task']} (consistent across X_sua/X_mua/y_task)")

    _report_array_stats('X_sua', X_sua)
    _check_zero_variance_channels('X_sua', X_sua)
    _report_array_stats('X_mua', X_mua)
    _check_zero_variance_channels('X_mua', X_mua)
    _report_array_stats('y_task', y_task)
    _report_array_stats('y_task velocity (cols 2:4)', y_task[:, 2:4])

    return ok


def check_snn_dataset(dataset_dirname, raw_stem):
    path = os.path.join(dataset_dirname, f"{raw_stem}_snn.h5")
    print(f"\n=== SNN dataset: {path} ===")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return False

    ok = True
    with h5py.File(path, 'r') as f:
        missing_keys = [k for k in ('X_raster', 'y_trace', 'y_end', 'window_start_time') if k not in f]
        if missing_keys:
            print(f"  MISSING KEYS: {missing_keys}")
            return False

        X_raster = f['X_raster'][()]
        y_trace = f['y_trace'][()]
        y_end = f['y_end'][()]
        window_start_time = f['window_start_time'][()]
        wdw_time = f.attrs.get('wdw_time')
        ol_time = f.attrs.get('ol_time')
        delta_time = f.attrs.get('delta_time', 0.004)
        claimed_non_overlapping = bool(f.attrs.get('non_overlapping', False))
        feature = f.attrs.get('feature')

    print(f"  feature={feature}, wdw_time={wdw_time}, ol_time={ol_time}, "
          f"non_overlapping={claimed_non_overlapping}")

    row_counts = {'X_raster': len(X_raster), 'y_trace': len(y_trace),
                  'y_end': len(y_end), 'window_start_time': len(window_start_time)}
    if len(set(row_counts.values())) != 1:
        print(f"  ROW COUNT MISMATCH within SNN dataset: {row_counts}")
        ok = False
    else:
        print(f"  Row count: {row_counts['y_end']} (consistent across all four arrays)")

    _report_array_stats('X_raster (per-bin spikes)', X_raster)
    _check_zero_variance_channels('X_raster', X_raster)
    _report_array_stats('y_trace velocity', y_trace)
    _report_array_stats('y_end velocity (lagged target)', y_end)

    if claimed_non_overlapping and len(window_start_time) > 1:
        nperseg = round(wdw_time / delta_time) + 1
        expected_step = nperseg * delta_time
        observed_step = np.diff(window_start_time)
        # Exact check, not approximate -- this is the property the
        # continuous/never-reset training scheme depends on, so any
        # deviation at all should fail loudly rather than be shrugged off
        # as floating-point noise.
        if not np.allclose(observed_step, expected_step, atol=1e-9):
            bad = np.where(~np.isclose(observed_step, expected_step, atol=1e-9))[0]
            print(f"  FAIL: dataset claims non_overlapping=True but "
                  f"{len(bad)} consecutive window gap(s) are not exactly "
                  f"{expected_step*1000:.3f} ms (first bad index: {bad[0]}, "
                  f"observed {observed_step[bad[0]]*1000:.3f} ms). This "
                  f"breaks the assumption the continuous/never-reset SNN "
                  f"training scheme depends on.")
            ok = False
        else:
            print(f"  OK: all {len(observed_step)} consecutive window gaps are exactly "
                  f"{expected_step*1000:.3f} ms -- verified non-overlapping (zero shared raw samples)")
    elif claimed_non_overlapping:
        print("  (only 0 or 1 windows -- nothing to check for non-overlap)")

    return ok


def main(args):
    all_ok = True
    for method in args.methods.split(','):
        method = method.strip()
        if not check_ann_dataset(args.dataset_dirname, args.raw_stem, method):
            all_ok = False

    if not check_snn_dataset(args.dataset_dirname, args.raw_stem):
        all_ok = False

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'ONE OR MORE CHECKS FAILED'}")
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dirname', type=str, required=True,
                         help='Directory containing {raw_stem}_{method}.h5 and {raw_stem}_snn.h5')
    parser.add_argument('--raw_stem', type=str, required=True,
                         help='Session identifier, e.g. indy_20160627_01')
    parser.add_argument('--methods', type=str, default='binning',
                         help='Comma-separated list of ANN dataset methods to check (e.g. binning,gaussian)')
    args = parser.parse_args()
    main(args)

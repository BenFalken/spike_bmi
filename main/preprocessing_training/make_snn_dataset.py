"""
Build the SNN dataset for one session: per-window spike rasters at native
4 ms resolution, with the matching velocity trace.

Input  (--input_filepath):  process_data.py output (the same file make_dataset.py reads).
Output (--output_filepath): .h5 with
    X_raster           (n_windows, n_units, nperseg)  spike counts per 4 ms bin
    y_trace            (n_windows, nperseg, 2)        velocity at each bin (zero lag)
    y_end              (n_windows, 2)                 velocity at the sample after the
                                                      window (the ANN's y_task convention)
    window_start_time  (n_windows,)
  attrs: wdw_time, ol_time, delta_time, feature, non_overlapping, total_raw_samples

By default (--ol_time 0) windows are back to back with no shared samples, so
concatenating them in order reconstructs the continuous recording.
total_raw_samples (the unwindowed session length) is what export_snn_pkl.py
uses to place the train/test boundary at the same moment as the ANN split.
"""

import argparse

import h5py
import numpy as np

from make_dataset import DELTA_TIME, window_params


def main(args):
    print(f"Loading spike and kinematic data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_data = f['task_data'][()]
        task_time = f['task_time'][()]
        spike_trains = f[f'{args.feature}_trains'][()]
    velocity = task_data[:, 2:4]
    num_units = len(spike_trains)
    print(f"Number of units ({args.feature}): {num_units}")

    nperseg, noverlap = window_params(args.wdw_time, args.ol_time)
    step = nperseg - noverlap
    assert step > 0, "ol_time must be smaller than wdw_time"
    print(f"Window: {nperseg} samples (~{nperseg * DELTA_TIME * 1000:.0f} ms), "
          f"step: {step} samples (~{step * DELTA_TIME * 1000:.1f} ms)")

    # Same window placement as bmi.features.extract(): the sample at end_idx
    # (used for y_end) must exist.
    n_windows = max(0, (len(task_time) - 1 - nperseg) // step + 1)
    X_raster = np.zeros((n_windows, num_units, nperseg), dtype=np.float32)
    y_trace = np.zeros((n_windows, nperseg, 2), dtype=np.float32)
    y_end = np.zeros((n_windows, 2), dtype=np.float32)
    window_start_time = np.zeros(n_windows, dtype=np.float64)

    for w in range(n_windows):
        start_idx = w * step
        end_idx = start_idx + nperseg
        t_seg = task_time[start_idx:end_idx]
        dt = np.diff(t_seg).mean()
        bin_edges = np.concatenate((t_seg - dt / 2, [t_seg[-1] + dt / 2]))  # one bin per sample
        for u, spikes in enumerate(spike_trains):
            X_raster[w, u, :] = np.histogram(np.asarray(spikes), bin_edges)[0]
        y_trace[w] = velocity[start_idx:end_idx]
        y_end[w] = velocity[end_idx]
        window_start_time[w] = t_seg[0]
    print(f"Built {n_windows} windows")

    if noverlap == 0 and n_windows > 1:
        observed_step = np.diff(window_start_time)
        expected_step = nperseg * DELTA_TIME
        if not np.allclose(observed_step, expected_step, atol=DELTA_TIME / 2):
            raise RuntimeError(
                f"Windows are not evenly spaced by {expected_step * 1000:.1f} ms (observed "
                f"{observed_step.min() * 1000:.2f}-{observed_step.max() * 1000:.2f} ms); "
                f"task_time is probably not uniformly sampled.")

    print(f"Storing dataset into file: {args.output_filepath}")
    with h5py.File(args.output_filepath, 'w') as f:
        f['X_raster'] = X_raster
        f['y_trace'] = y_trace
        f['y_end'] = y_end
        f['window_start_time'] = window_start_time
        f.attrs['wdw_time'] = args.wdw_time
        f.attrs['ol_time'] = args.ol_time
        f.attrs['delta_time'] = DELTA_TIME
        f.attrs['feature'] = args.feature
        f.attrs['non_overlapping'] = bool(noverlap == 0)
        f.attrs['total_raw_samples'] = len(task_time)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_filepath', type=str, required=True,
                        help='Spike and kinematic data (process_data.py output)')
    parser.add_argument('--output_filepath', type=str, required=True, help='Output dataset file')
    parser.add_argument('--feature', type=str, default='mua', choices=['sua', 'mua'],
                        help='Spike trains to rasterize')
    parser.add_argument('--wdw_time', type=float, default=0.256, help='Window width (s)')
    parser.add_argument('--ol_time', type=float, default=0.0,
                        help='Overlap between consecutive windows (s); 0 = back-to-back windows')
    main(parser.parse_args())

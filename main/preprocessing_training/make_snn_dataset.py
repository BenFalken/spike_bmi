"""
Build a sliding-window spike raster dataset for the SNN.

THIS REVISION corrects the module docstring's framing (previously
inaccurate, not just stale): ANN and SNN windowing are now DELIBERATELY
DIFFERENT, not "numerically identical whenever --ol_time > 0" as an
earlier version of this file claimed. Current project-wide convention
(see single_subject_pipeline.py):
  - ANN (make_dataset.py): dense, near-total-overlap 4ms-step windows
    (--wdw_time 0.256 --ol_time 0.252) -- matches the reference paper's
    own windowing convention.
  - SNN (this file): non-overlapping WDW_TIME-wide trials
    (--wdw_time 0.256 --ol_time 0.0) -- one trial per WDW_TIME-wide
    stretch of the recording, no shared raw samples between consecutive
    trials. This is the DEFAULT here now, not just an option.
Both share the same WDW_TIME (0.256s = 65 native samples at 4ms/sample),
which is what lets eval_all_decoders.py's build_snn_ann_alignment() line
up one SNN trial's full output against 65 consecutive ANN dense-window
predictions covering the identical raw sample range -- but the STEP
between windows is intentionally NOT shared between the two pipelines.

Same input file as make_dataset.py (the output of process_raw_data.py) ->
identical task_time/task_data arrays, so windows can be indexed against
the same raw sample positions on both sides even though window COUNT
differs enormously between the two (dense vs. non-overlapping).

Same unit selection as the ANN's --feature choice (sua_trains or
mua_trains from the processed file), not an independently recomputed rate
threshold.

Two target conventions provided, matching the ANN's two extraction
functions:
    y_end   -> task[end_idx], the single lagged point extract() uses (MUA/
               binning default pipeline). Use this to compare against the
               ANN's default --feature mua run.
    y_trace -> task[start_idx:end_idx], the full zero-lag per-bin trace
               extract2() uses (SUA pipeline). Use this if comparing
               against --feature sua.
Pick whichever matches the --feature you're evaluating the ANN with --
this axis (feature type) is independent of the windowing-regime point
above and is unaffected by it.

Sub-bin resolution inside each window is left at native 4ms (nperseg bins
per window, e.g. 65 bins for wdw_time=0.256) -- this is the one place the
SNN is *expected* to differ from the ANN's single collapsed feature per
window, since exploiting that fine-grained structure is the point of
using an SNN at all.
"""

import argparse
import h5py
import numpy as np


def main(args):
    print(f"Loading spike and kinematic data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_data = f['task_data'][()]                    # N x 6: pos_x,pos_y,vel_x,vel_y,acc_x,acc_y
        task_time = f['task_time'][()]                     # N, sampled at 250 Hz (dt = 0.004 s)
        spike_trains = f[f'{args.feature}_trains'][()]      # ragged array of spike times, one per unit

    num_units = len(spike_trains)
    print(f"Number of units ({args.feature}): {num_units}")

    # velocity columns, matching evaluate_ann.py's y = y[:, 2:4]
    velocity = task_data[:, 2:4]

    delta_time = 0.004  # native sampling interval -- MUST match create_dataset.py's hardcoded value
    nperseg = int(args.wdw_time / delta_time) + 1     # identical formula to extract()/extract2()
    # See FIX note in the module docstring: the +1 only applies when there's
    # a real overlap to be inclusive about. ol_time=0.0 must give noverlap=0
    # exactly (-> step == nperseg -> zero shared raw samples), not 1.
    noverlap = int(round(args.ol_time / delta_time))
    if args.ol_time > 0:
        noverlap += 1
    step = nperseg - noverlap
    assert step > 0, "ol_time must be smaller than wdw_time"

    print(f"Window: {nperseg} samples (~{nperseg*delta_time*1000:.0f} ms), "
          f"step: {step} samples (~{step*delta_time*1000:.1f} ms)"
          + (" [non-overlapping: step == window width]" if noverlap == 0 else ""))

    n_windows_max = (len(task_time) - nperseg) // step + 1
    X_raster = np.zeros((n_windows_max, num_units, nperseg), dtype=np.float32)
    y_trace = np.zeros((n_windows_max, nperseg, 2), dtype=np.float32)   # zero-lag full trace (extract2-style)
    y_end = np.zeros((n_windows_max, 2), dtype=np.float32)               # lagged single point (extract-style)
    window_start_time = np.zeros(n_windows_max, dtype=np.float64)

    win_idx = 0
    for i in range(len(task_time)):
        start_idx = i * step
        end_idx = start_idx + nperseg
        if end_idx > len(task_time) - 1:
            break  # identical break condition to extract()/extract2()

        t_seg = task_time[start_idx:end_idx]       # length nperseg
        dt = np.diff(t_seg).mean()
        bin_edges = np.concatenate((t_seg - dt / 2, [t_seg[-1] + dt / 2]))  # nperseg+1 edges -> nperseg bins

        for u, spk in enumerate(spike_trains):
            spk = np.asarray(spk)
            X_raster[win_idx, u, :] = np.histogram(spk, bin_edges)[0]

        y_trace[win_idx] = velocity[start_idx:end_idx]   # same nperseg samples as X_raster, zero lag
        y_end[win_idx] = velocity[end_idx]                 # matches extract()'s task[end_idx] lag convention
        window_start_time[win_idx] = t_seg[0]
        win_idx += 1

    print(f"Built {win_idx} windows")

    if noverlap == 0 and win_idx > 1:
        # Cheap self-check: with true non-overlapping windows, consecutive
        # window_start_time values should differ by exactly wdw_time (i.e.
        # nperseg native samples). If this ever fires, something upstream
        # broke the non-overlap guarantee -- fail loudly rather than write
        # a dataset that silently doesn't have the property it claims to.
        starts = window_start_time[:win_idx]
        observed_step = np.diff(starts)
        expected_step = nperseg * delta_time
        if not np.allclose(observed_step, expected_step, atol=delta_time / 2):
            raise RuntimeError(
                f"Non-overlapping windows requested (ol_time=0) but consecutive "
                f"window_start_time values are not uniformly spaced by "
                f"{expected_step*1000:.1f} ms as expected (observed range: "
                f"{observed_step.min()*1000:.2f}-{observed_step.max()*1000:.2f} ms). "
                f"This should not happen -- please report.")
        print(f"OK: verified {win_idx - 1} consecutive window gaps are all "
              f"exactly {expected_step*1000:.1f} ms (non-overlapping, as requested)")

    print(f"Storing dataset into file: {args.output_filepath}")
    with h5py.File(args.output_filepath, 'w') as f:
        f['X_raster'] = X_raster[:win_idx]
        f['y_trace'] = y_trace[:win_idx]
        f['y_end'] = y_end[:win_idx]
        f['window_start_time'] = window_start_time[:win_idx]
        f.attrs['wdw_time'] = args.wdw_time
        f.attrs['ol_time'] = args.ol_time
        f.attrs['delta_time'] = delta_time
        f.attrs['feature'] = args.feature
        f.attrs['non_overlapping'] = bool(noverlap == 0)
        # The RAW (unwindowed) session length -- saved directly rather
        # than left for a downstream script to reconstruct, because it
        # CAN'T be reconstructed exactly from this file's own window
        # count for non-overlapping windowing (unlike the ANN's dense,
        # step=1 windowing, where total_raw_samples = N + nperseg holds
        # exactly). export_snn_pkl.py needs this exact value to compute a
        # train/test boundary that lands on the same real moment as the
        # ANN's own boundary -- see that script and
        # bmi.preprocessing.chronological_holdout_split(), which compute
        # the ANN-side boundary from the identical quantity.
        f.attrs['total_raw_samples'] = len(task_time)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, required=True,
                         help='Path to processed spike/kinematic file (output of process_raw_data.py) -- '
                              'MUST be the same file passed to create_dataset.py for the ANN')
    parser.add_argument('--output_filepath', type=str, required=True,
                         help='Path to output SNN dataset file')
    parser.add_argument('--feature', type=str, default='mua', choices=['sua', 'mua'],
                         help='Spike train type -- MUST match --feature used for the ANN dataset you are comparing against')
    parser.add_argument('--wdw_time', type=float, default=0.256,
                         help='Window size (s) -- shared base unit with the ANN dataset '
                              '(see module docstring); should match ANN --wdw_time')
    parser.add_argument('--ol_time', type=float, default=0.0,
                         help='Overlap (s); step = wdw_time - ol_time. Default 0.0 = '
                              'non-overlapping trials, the current project convention -- '
                              'deliberately NOT matched to the ANN dataset\'s own --ol_time '
                              '(see module docstring for why ANN and SNN windowing now '
                              'intentionally differ). Set nonzero only if you specifically '
                              'want SNN trials with real overlap, e.g. for reproducing an '
                              'older-style comparison against the ANN\'s own windowing.')
    args = parser.parse_args()
    main(args)

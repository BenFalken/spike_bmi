"""
Builds TWO genuinely continuous streams -- train and test -- from the
SNN's pre-windowing h5 source (e.g. .../dataset/mua/{session}_snn.h5),
saved directly under train/test subdirectories matching
create_dataloaders()'s own expected structure. Extends the original,
test-only version of this script (which extracted only the test region
into a single flat file) once it became clear the train split was also
needed -- for training a model on one genuinely continuous stream via
truncated BPTT, not just evaluating one. Not a concatenation of
already-selected, non-adjacent windowed trials (see the conversation
this was built from: concatenating already-windowed trials that aren't
consecutive would NOT give genuine continuity, since they can be evenly
spaced across the dataset with real gaps between them).

Confirmed directly against the real file (inspect_snn_continuous_source.py's
actual output on indy_20160407_02_snn.h5, not assumed):
  - window_start_time has a CONSTANT 0.26s stride everywhere (= 65 samples
    x 4ms, this project's base_nperseg) -- windows are genuinely
    non-overlapping and back-to-back, so concatenating X_raster/y_trace in
    window order reconstructs the true continuous recording.
  - 3145 windows -> 13.63 minutes total, consistent with (and close to)
    this same session's 785 mua_large trials x 4 base windows/trial = 3140
    -- independent cross-check, not just this file's own claim.

OPEN QUESTION, not fully resolved: y_end != y_trace[:, -1, :] for this
session's data (confirmed False on the first 5 windows) -- y_end is NOT
simply y_trace's final timestep. This script uses y_trace, not y_end, for
velocity -- justified below by checking y_trace's value range against
this project's own known physical velocity bounds (velocity_lo=-280.56,
velocity_hi=316.54), which is direct evidence about what each array
actually represents, not just an assumption from matching shapes. If
that check doesn't clearly support y_trace, this prints a loud warning
and does NOT proceed with saving -- read that output before trusting
the result.

Train/test boundary reuses test_all_decoders.py's own
compute_aligned_split() (imported directly, not reimplemented) --
applied to this h5 file's own total sample count. This will not
necessarily land EXACTLY on the same raw sample as mua_large's own
historical train/test boundary (this file has 3145 windows vs. the
5-window-smaller 3140 implied by 785 mua_large trials -- a ~0.16%
discrepancy, likely edge-trimming during mua_large's own construction)
-- for exploratory continuous-decoding analysis this is negligible (a
fraction of a second out of a 13+ minute recording), but it's a real,
documented approximation, not a guarantee of bit-for-bit alignment with
the historical split.

CLI usage:
    python build_continuous_trials.py \
        --h5-path /users/bfalkenb/data/bfalkenb/data/dataset/mua/indy_20160407_02_snn.h5 \
        --session-id indy_20160407_02 \
        --test-frac 0.1 \
        --output-dir datasets/bmi/mua_combined
"""

import argparse
import os
import pickle as pkl

import h5py
import numpy as np

from test_all_decoders import compute_aligned_split

BASE_NPERSEG = 65        # this project's established base-window size (256ms at 4ms/sample)
STEP_TIME_S = 0.004
V_LO_FALLBACK = -280.56  # this project's established physical velocity bounds --
V_HI_FALLBACK = 316.54   # used here only as a SANITY-CHECK reference range, not applied
                          # as a transform (input_spikes/velocity are saved in the SAME
                          # physical units the .pkl pipeline already expects -- forward
                          # scaling happens later, inside the dataloader, same as every
                          # other .pkl file in this project).


def _save_split(output_dir, split_name, input_spikes, velocity):
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    output_path = os.path.join(split_dir, "0.pkl")
    with open(output_path, "wb") as f:
        pkl.dump({
            "input_spikes": input_spikes.astype(np.float32),
            "velocity": velocity.astype(np.float32),
        }, f)
    duration_s = velocity.shape[0] * STEP_TIME_S
    print(f"Saved {split_name}: {output_path} -- input_spikes shape={input_spikes.shape}, "
          f"velocity shape={velocity.shape} ({duration_s:.1f}s = {duration_s/60:.2f} min)")


def load_and_verify_continuous_source(h5_path):
    """Loads the SNN's pre-windowing h5 source, verifies it's genuinely
    safe to concatenate (constant stride, no gaps -- see
    inspect_snn_continuous_source.py), disambiguates y_trace vs y_end,
    and reconstructs the full continuous recording. Shared between
    build_continuous_trials.py and make_huge_dataset.py specifically so
    this safety-critical logic exists in exactly one place -- extracted
    here rather than duplicated, once a second script needed it.

    Returns (X_continuous_full (C, total_samples), y_continuous_full
    (total_samples, 2), n_windows) on success, or (None, None, None) if
    y_trace's value range doesn't look like physical velocity (caller
    should treat this as "do not proceed", not silently substitute a
    default).
    """
    with h5py.File(h5_path, "r") as f:
        window_start_time = f["window_start_time"][:]
        X_raster = f["X_raster"][:]      # (n_windows, C, window_len)
        y_trace = f["y_trace"][:]        # (n_windows, window_len, 2)
        y_end = f["y_end"][:]            # (n_windows, 2)

    n_windows, n_channels, window_len = X_raster.shape
    assert window_len == BASE_NPERSEG, (
        f"Expected window_len={BASE_NPERSEG}, got {window_len} -- this script's "
        f"boundary-conversion math assumes this project's established base window size.")

    # --- Confirm stride is still what inspect_snn_continuous_source.py verified --
    # cheap, and this script should never silently trust a stale assumption about
    # a file it's about to concatenate wholesale.
    diffs = np.diff(window_start_time)
    if not np.allclose(diffs, diffs[0]):
        raise ValueError(
            f"window_start_time is not uniformly spaced in this file -- re-run "
            f"inspect_snn_continuous_source.py and resolve that before proceeding. "
            f"Concatenating in order would NOT be safe here.")
    print(f"Confirmed: {n_windows} windows, constant stride={diffs[0]}s, "
          f"total duration={n_windows * diffs[0] / 60:.2f} minutes")

    # --- y_trace vs y_end: direct evidence, not a shape-based guess ---
    y_trace_flat = y_trace.reshape(-1, 2)
    print(f"\ny_trace value range: vx=[{y_trace_flat[:,0].min():.2f}, {y_trace_flat[:,0].max():.2f}], "
          f"vy=[{y_trace_flat[:,1].min():.2f}, {y_trace_flat[:,1].max():.2f}]")
    print(f"y_end value range:   vx=[{y_end[:,0].min():.2f}, {y_end[:,0].max():.2f}], "
          f"vy=[{y_end[:,1].min():.2f}, {y_end[:,1].max():.2f}]")
    print(f"This project's known physical velocity bounds: "
          f"[{V_LO_FALLBACK:.2f}, {V_HI_FALLBACK:.2f}]")

    def _range_overlap_score(arr):
        # Compares the array's own SPAN (max-min) against the expected
        # velocity span, not just whether its values happen to fall inside
        # a wide outer bound -- a narrow [0,1]-normalized quantity would
        # trivially satisfy a containment-only check despite being a
        # completely different scale. Confirmed this distinction actually
        # matters: caught this exact failure mode directly, testing
        # against a synthetic y_end deliberately built with a [0,1] range
        # -- the earlier containment-only version wrongly called it
        # "plausible" too.
        span = arr.max() - arr.min()
        expected_span = V_HI_FALLBACK - V_LO_FALLBACK
        ratio = span / expected_span
        return 0.3 < ratio < 3.0, ratio

    y_trace_plausible, y_trace_ratio = _range_overlap_score(y_trace_flat)
    y_end_plausible, y_end_ratio = _range_overlap_score(y_end)
    print(f"\ny_trace span / expected velocity span = {y_trace_ratio:.3f} "
          f"-- plausible as physical velocity? {y_trace_plausible}")
    print(f"y_end span / expected velocity span = {y_end_ratio:.3f} "
          f"-- plausible as physical velocity?   {y_end_plausible}")

    if not y_trace_plausible:
        print("\nWARNING: y_trace's value range does NOT look like this project's known "
              "physical velocity range -- do NOT trust this script's output without "
              "understanding why before proceeding further.")
        return None, None, None

    # --- Reconstruct the full continuous recording (verified reshape logic --
    # see the hand-traceable test this was checked against before writing this) ---
    X_continuous_full = np.transpose(X_raster, (1, 0, 2)).reshape(n_channels, -1)
    y_continuous_full = y_trace.reshape(-1, 2)
    total_raw_samples_full = n_windows * BASE_NPERSEG
    assert X_continuous_full.shape == (n_channels, total_raw_samples_full)
    assert y_continuous_full.shape == (total_raw_samples_full, 2)

    return X_continuous_full, y_continuous_full, n_windows


def main(args):
    X_continuous_full, y_continuous_full, n_windows = load_and_verify_continuous_source(args.h5_path)
    if X_continuous_full is None:
        return
    n_channels = X_continuous_full.shape[0]
    total_raw_samples_full = n_windows * BASE_NPERSEG


    # --- Train/test boundary, reusing test_all_decoders.py's own formula --
    # not reimplemented here ---
    N = total_raw_samples_full - BASE_NPERSEG
    n_train, n_test = compute_aligned_split(N, args.test_frac, base_nperseg=BASE_NPERSEG)
    assert n_train % BASE_NPERSEG == 0, "compute_aligned_split()'s own guarantee -- should never fail"
    train_window_boundary = n_train // BASE_NPERSEG
    print(f"\ncompute_aligned_split(N={N}, test_frac={args.test_frac}) -> "
          f"n_train={n_train} samples ({train_window_boundary} windows), n_test={n_test} samples")
    print(f"Train region: windows [0:{train_window_boundary}), raw samples [0:{n_train})")
    print(f"Test region:  windows [{train_window_boundary}:{n_windows}), "
          f"raw samples [{n_train}:{total_raw_samples_full})")

    train_input_spikes = X_continuous_full[:, :n_train]
    train_velocity = y_continuous_full[:n_train, :]
    test_input_spikes = X_continuous_full[:, n_train:]
    test_velocity = y_continuous_full[n_train:, :]

    # --- Save, matching the existing SNN .pkl format exactly (physical-unit
    # velocity, (C, T) input_spikes -- same convention as every other .pkl
    # file this project's create_dataloaders() already reads), under
    # train/test subdirectories so create_dataloaders() works directly
    # against this output without any manual file-moving. ---
    output_dir = os.path.join(args.output_dir, args.session_id)
    print()
    _save_split(output_dir, "train", train_input_spikes, train_velocity)
    _save_split(output_dir, "test", test_input_spikes, test_velocity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="datasets/bmi/mua_combined")
    args = parser.parse_args()
    main(args)

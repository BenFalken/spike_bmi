"""
Directly inspects an already-converted h5 file's own recorded shapes and
(if present) timing metadata, to settle questions about actual sampling
rate / window size empirically rather than by inference from CLI flags
used to produce it.

Usage:
    # SNN-side h5 (from make_snn_dataset.py / combine_trial_windows_to_session.py):
    python inspect_h5_timing.py --h5-path .../sub-Jenkins_ses-20090912_snn.h5

    # ANN-side binning h5:
    python inspect_h5_timing.py --h5-path .../sub-Jenkins_ses-20090912_binning.h5
"""
import argparse
import h5py
import numpy as np


def main(args):
    with h5py.File(args.h5_path, "r") as f:
        print(f"Keys: {list(f.keys())}")
        print(f"Top-level attrs: {dict(f.attrs)}")
        print()

        for key in f.keys():
            ds = f[key]
            if isinstance(ds, h5py.Dataset):
                print(f"{key}: shape={ds.shape}, dtype={ds.dtype}")
                if dict(ds.attrs):
                    print(f"  attrs: {dict(ds.attrs)}")

        # If a task_time-style array is present, compute the ACTUAL
        # native sampling interval directly from consecutive timestamps
        # -- this is the ground truth, not an assumption from any CLI flag.
        for time_key in ("task_time", "t", "time", "timestamps"):
            if time_key in f:
                t = f[time_key][()].squeeze()
                dt = np.diff(t)
                print(f"\n'{time_key}' found -- ACTUAL native sampling interval:")
                print(f"  mean dt = {dt.mean()*1000:.4f} ms ({1/dt.mean():.2f} Hz)")
                print(f"  std dt  = {dt.std()*1000:.6f} ms")
                break
        else:
            print("\nNo raw timestamp array found in this file (expected for an "
                  "already-windowed h5) -- infer window size from X_raster's own "
                  "last-axis shape instead, if present:")
            for raster_key in ("X_raster", "X_mua", "X_sua"):
                if raster_key in f:
                    shape = f[raster_key].shape
                    print(f"  {raster_key} shape={shape} -- if this is "
                          f"(n_windows, channels, nperseg), nperseg={shape[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", type=str, required=True)
    args = parser.parse_args()
    main(args)

"""
Converts one NWB session file (Jenkins/Nitschke-style: Position/Hand +
units with a 1:1 electrode mapping + a real trials table) into one raw h5
file PER TRIAL, matching process_data.py's own output schema exactly --
so make_dataset.py / make_snn_dataset.py / export_snn_pkl.py need ZERO
modification to consume them.

WHY PER TRIAL, NOT ONE FILE PER SESSION: confirmed directly (not assumed)
that ~78% of this data's session duration is inter-trial rest, not
sensor dropout -- concatenating a whole session into one continuous
task_time array would silently create ~3000 fake seams (windows that
splice together moments that were actually 1-300+ seconds apart in real
time), reproducing the exact spurious-velocity-spike bug visible in the
very first plot that started this conversion effort. Splitting at real
trial boundaries means no window the existing pipeline ever builds can
straddle a seam, without touching any downstream code at all. Multiple
trial files belonging to the same session get pooled back together
LATER, at the WINDOWED-output stage (after make_snn_dataset.py has
already run on each one independently) -- see combine_trial_windows_to_
session.py. Do NOT try to merge trial files at the raw stage this script
produces; that reintroduces the same seam problem one level up.

SCHEMA (matches process_data.py's --output_filepath exactly):
  task_time  : (N,) float64, uniformly spaced at delta_time=0.004s exactly
  task_data  : (N, 6) float32 -- pos_x, pos_y, vel_x, vel_y, acc_x, acc_y,
               computed via diff/dt on THIS trial's own resampled grid
  target_pos : (N, 2) float32 -- placeholder zeros; confirmed neither
               make_dataset.py nor make_snn_dataset.py actually reads
               this key, only process_data.py writes it
  sua_trains : ragged (h5py vlen float64), one entry per unit
  mua_trains : IDENTICAL to sua_trains here -- confirmed directly (not
               assumed) this dataset has an exact 1:1 unit:electrode
               mapping (192 units, 192 electrodes, min=max=mean=1.00
               units/electrode), so "merge all units on a channel" is a
               no-op: there is never more than one unit to merge.

RESAMPLING: native NWB position sampling is ~676Hz, not the pipeline's
required 250Hz (confirmed directly: 9,827,142 samples / 14,534s session
duration). Each trial's hand position is linearly interpolated onto a
NEW, uniform 250Hz grid spanning that trial's own [start_time, stop_time]
-- chosen over exclusion specifically BECAUSE this is now per-trial, not
whole-session: within a single short trial, any internal gap is
tracking noise, not inter-trial rest, so interpolating across it doesn't
fabricate a semantically different kind of moment (see module docstring
history in the conversation this script came from for the whole-session
case, where exclusion was the right call instead). Trials with an
internal gap exceeding --max-gap-ms are skipped outright rather than
interpolated through, matching the original reference decoder's own
per-trial gap check (SingleSessionSingleTrialDataset).

CLI usage:
    python convert_nwb_trials_to_raw_h5.py \
        --nwb-path /Users/benjaminfalkenburg/Dropbox/SNN_Main/data/HKM/sub-Nitschke_ses-20090812_behavior+ecephys.nwb \
        --output-dir raw_converted/nitschke_20100923 \
        --max-gap-ms 20 --min-samples 5
"""

import argparse
import os

import glob
import json

import h5py
import numpy as np

from hkm_despike import despike_position, DEFAULT_MAX_SPEED

DELTA_TIME = 0.004  # 250 Hz -- MUST match make_dataset.py/make_snn_dataset.py's hardcoded value


def build_uniform_grid(trial_start, trial_stop, delta_time=DELTA_TIME):
    """Uniform grid spanning [trial_start, trial_stop] at EXACTLY
    delta_time spacing. np.arange with a float step can drift by the
    last sample due to floating point accumulation over many steps --
    built via integer sample COUNT instead, so the final spacing is
    exact, not merely close."""
    n_samples = int(np.floor((trial_stop - trial_start) / delta_time)) + 1
    return trial_start + np.arange(n_samples) * delta_time


def convert_one_trial(trial_id, trial_start, trial_stop, hand_timestamps, hand_xy,
                       unit_spike_times, max_gap_s, min_samples, max_speed=DEFAULT_MAX_SPEED,
                       despike_log=None):
    """Returns (task_time, task_data, target_pos, sua_trains, mua_trains) or
    None if this trial should be skipped (too few samples, internal gap
    too large, or too short after resampling).

    max_speed: native position intervals faster than this (units/s) are tracking glitches and are
    removed before resampling -- see hkm_despike.py. 0/None disables. Per-trial glitch counts are
    appended to despike_log (a list) when given."""
    mask = (hand_timestamps >= trial_start) & (hand_timestamps <= trial_stop)
    t_trial = hand_timestamps[mask]
    xy_trial = hand_xy[mask]

    if len(t_trial) < 2:
        return None, f"trial {trial_id}: fewer than 2 hand samples in [{trial_start:.3f}, {trial_stop:.3f}]"

    # Remove tracking glitches BEFORE the gap check/resampling/diff: a position jump of a few hundred
    # units between two native samples becomes a ~70,000 units/s velocity spike after diff/dt.
    xy_trial, glitch = despike_position(t_trial, xy_trial, max_speed)
    if despike_log is not None and glitch["n_glitch_intervals"]:
        despike_log.append({"trial_id": int(trial_id), **glitch})

    max_gap = np.max(np.diff(t_trial))
    if max_gap > max_gap_s:
        return None, f"trial {trial_id}: internal hand-tracking gap of {max_gap*1000:.1f}ms exceeds max_gap_ms"

    task_time = build_uniform_grid(t_trial[0], t_trial[-1])
    if len(task_time) < min_samples:
        return None, f"trial {trial_id}: only {len(task_time)} samples after resampling, need >= {min_samples}"

    # Linear interpolation onto the new uniform grid -- np.interp requires
    # the x-coordinates (t_trial) to be strictly increasing, which real
    # NWB timestamps should already be; not re-sorting defensively here
    # since a non-monotonic timestamp array would indicate a much more
    # fundamental problem worth surfacing, not silently working around.
    pos_x = np.interp(task_time, t_trial, xy_trial[:, 0])
    pos_y = np.interp(task_time, t_trial, xy_trial[:, 1])
    task_pos = np.stack([pos_x, pos_y], axis=1)

    # Velocity/acceleration -- IDENTICAL formula and padding convention to
    # process_data.py, computed on THIS trial's own resampled dt=0.004s grid.
    task_vel = np.diff(task_pos, axis=0) / DELTA_TIME
    task_acc = np.diff(task_vel, axis=0) / DELTA_TIME
    task_vel = np.concatenate((task_vel, task_vel[-1:, :]), axis=0)
    task_acc = np.concatenate((task_acc, task_acc[-2:, :]), axis=0)
    task_data = np.concatenate((task_pos, task_vel, task_acc), axis=1).astype(np.float32)

    target_pos = np.zeros((len(task_time), 2), dtype=np.float32)  # placeholder; unused downstream

    sua_trains = []
    for spk in unit_spike_times:
        spk = np.asarray(spk)
        idx = np.where((spk >= task_time[0]) & (spk <= task_time[-1]))[0]
        sua_trains.append(spk[idx].astype(np.float64))
    mua_trains = [s.copy() for s in sua_trains]  # identical given 1:1 unit:electrode mapping

    return (task_time.astype(np.float64), task_data, target_pos, sua_trains, mua_trains), None


def main(args):
    from pynwb import NWBHDF5IO  # imported here so hkm_despike/convert_one_trial work without pynwb
    print(f"Reading NWB file: {args.nwb_path}")
    io = NWBHDF5IO(args.nwb_path, "r")
    nwbfile = io.read()

    hand_series = nwbfile.processing["behavior"].data_interfaces["Position"]["Hand"]
    hand_xy = np.asarray(hand_series.data)[:, :2]  # first 2 columns only, matching the reference decoder
    hand_timestamps = np.asarray(hand_series.timestamps)

    units_df = nwbfile.units.to_dataframe()
    unit_spike_times = [np.asarray(units_df.iloc[i]["spike_times"]) for i in range(len(units_df))]
    print(f"Units: {len(unit_spike_times)}")

    trials_df = nwbfile.trials.to_dataframe()
    n_trials = len(trials_df)
    print(f"Trials: {n_trials}")

    session_id = os.path.splitext(os.path.basename(args.nwb_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    stale = glob.glob(os.path.join(args.output_dir, f"{session_id}_trial*.h5"))
    if stale:
        if not args.overwrite:
            raise SystemExit(f"{len(stale)} existing trial file(s) for {session_id} in {args.output_dir} -- "
                             f"stale files from an earlier (uncleaned) run would silently mix with the new "
                             f"ones. Re-run with --overwrite to delete them first.")
        for fpath in stale:
            os.remove(fpath)
        print(f"Removed {len(stale)} stale trial file(s) from {args.output_dir}")
    despike_log = []

    max_gap_s = args.max_gap_ms / 1000.0
    n_written, n_skipped = 0, 0
    for trial_id in range(n_trials):
        trial_start = float(trials_df.iloc[trial_id]["start_time"])
        trial_stop = float(trials_df.iloc[trial_id]["stop_time"])

        result, skip_reason = convert_one_trial(
            trial_id, trial_start, trial_stop, hand_timestamps, hand_xy,
            unit_spike_times, max_gap_s, args.min_samples,
            max_speed=args.max_speed, despike_log=despike_log)

        if result is None:
            print(f"  [skip] {skip_reason}")
            n_skipped += 1
            continue

        task_time, task_data, target_pos, sua_trains, mua_trains = result
        out_path = os.path.join(args.output_dir, f"{session_id}_trial{trial_id:04d}.h5")
        with h5py.File(out_path, "w") as f:
            f["task_time"] = task_time
            f["task_data"] = task_data
            f["target_pos"] = target_pos
            dt = h5py.special_dtype(vlen=np.dtype("f8"))
            f.create_dataset("sua_trains", data=np.asarray(sua_trains, dtype=dt))
            f.create_dataset("mua_trains", data=np.asarray(mua_trains, dtype=dt))
            f.attrs["trial_id"] = trial_id
            f.attrs["trial_start_time"] = trial_start
            f.attrs["trial_stop_time"] = trial_stop
        n_written += 1

    io.close()
    print(f"\n{n_written} trial file(s) written to {args.output_dir}, {n_skipped} skipped "
          f"(of {n_trials} total)")

    n_glitches = sum(g["n_glitch_intervals"] for g in despike_log)
    print(f"Despike (max_speed={args.max_speed}): {n_glitches} glitch interval(s) removed across "
          f"{len(despike_log)} trial(s)")
    if despike_log:
        worst = max((s for g in despike_log for s in g["glitch_speeds"]), default=0.0)
        print(f"  fastest removed interval: {worst:.0f} units/s")
    report_path = os.path.join(args.output_dir, f"{session_id}_despike_report.json")
    with open(report_path, "w") as f:
        json.dump({"session_id": session_id, "max_speed": args.max_speed, "n_trials_total": n_trials,
                   "n_trials_written": n_written, "n_trials_with_glitches": len(despike_log),
                   "n_glitch_intervals": n_glitches, "trials": despike_log}, f, indent=1)
    print(f"  report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-gap-ms", type=float, default=20.0,
                         help="Trials with an internal hand-tracking gap larger than this are "
                              "skipped entirely rather than interpolated through. Default matches "
                              "the original reference decoder's own bin_size=0.02s check.")
    parser.add_argument("--min-samples", type=int, default=5,
                         help="Minimum resampled samples (at 250Hz) required to keep a trial. "
                              "Default 5 (20ms) -- a genuine safety floor, not an arbitrary round "
                              "number: convert_one_trial()'s own task_vel/task_acc computation "
                              "(np.diff twice, then padded) crashes outright below 4 samples "
                              "(confirmed directly: n=3 raises a concatenate shape-mismatch error, "
                              "n=4 is the real minimum that works), so 5 leaves a small margin "
                              "above that hard floor rather than sitting exactly on it. Previously "
                              "defaulted to 130 ('enough for 2 non-overlapping 65-sample SNN "
                              "windows') -- that rationale no longer applies now that the SNN path "
                              "processes each trial's own full, variable length directly (see "
                              "make_snn_dataset_whole_trial.py) rather than windowing it, and was "
                              "silently discarding real, short-but-otherwise-valid trials for both "
                              "the SNN AND ANN paths (this filter runs upstream of both) even "
                              "though ANN's own dense windowing only ever needed one 65-sample "
                              "window to produce a single row. Confirmed via "
                              "compare_ann_snn_trial_coverage.py that a meaningful fraction of "
                              "real HKM trials fall under the old 130-sample threshold.")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                         help="Native hand-position intervals faster than this (position units/s) are "
                              "tracking glitches and are removed before resampling (see hkm_despike.py). "
                              "Default 3000 is ~3.5x the real 99.5th-percentile speed (~865) and far "
                              "below the ~70,000 glitch spikes. 0 disables cleaning.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Delete existing {session}_trial*.h5 files in --output-dir first.")
    args = parser.parse_args()
    main(args)

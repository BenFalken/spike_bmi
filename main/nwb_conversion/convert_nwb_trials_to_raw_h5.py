"""
Convert one NWB session (Jenkins/Nitschke: Position/Hand, units, and a trials
table) into one raw .h5 file per trial, in process_data.py's output schema,
so make_dataset.py runs on each trial unchanged.

Why per trial: about 78% of a session is rest between reaches. Concatenating
the session into one continuous recording would splice moments seconds or
minutes apart into ~3000 artificial seams, each a fake velocity spike. Each
trial is therefore its own recording, and the datasets are pooled from the
per-trial outputs later (combine_trial_windows_to_ann_h5.py,
make_snn_dataset_whole_trial.py).

Sessions recorded in several runs (some Nitschke files concatenate runs,
each with its own clock): every trial is first assigned to the run (hand
"piece") holding its reach, and its samples and spikes are taken from that
piece only; see nwb_pieces.py. Trials that cannot be assigned are skipped.
A single-run session (Jenkins, Nitschke 20090922) is converted as before.

Per trial:
  - hand-tracking glitches are removed from the native position samples
    (hkm_despike.py, --max-speed; a safety net: in multi-run sessions the
    spikes came from mixing runs, which the piece assignment prevents);
  - trials with an internal tracking gap over --max-gap-ms, or fewer than
    --min-samples samples after resampling, are skipped;
  - position is linearly interpolated from the native ~676 Hz onto a uniform
    250 Hz (4 ms) grid spanning the trial, and velocity and acceleration are
    derived on that grid exactly as in process_data.py.

Output: {output_dir}/{session}_trial{id:04d}.h5 with
    task_time  (N,)    uniform 4 ms grid
    task_data  (N, 6)  pos_x, pos_y, vel_x, vel_y, acc_x, acc_y
    target_pos (N, 2)  zeros (unused downstream)
    sua_trains, mua_trains  spike times per unit; identical, as every unit
                            has its own electrode in these recordings
and attributes trial_id, trial_start_time, trial_stop_time, hand_piece; plus
{session}_conversion_report.json: the pieces, how each trial was assigned
(or why it was skipped, per run), and the glitches removed per trial.

Usage:
    python convert_nwb_trials_to_raw_h5.py --nwb-path sub-Jenkins_ses-20090912_behavior+ecephys.nwb \
        --output-dir raw_trials/sub-Jenkins_ses-20090912_behavior+ecephys --overwrite
"""

import argparse
import glob
import json
import os
from collections import Counter

import h5py
import numpy as np

from hkm_despike import DEFAULT_MAX_SPEED, despike_position
from nwb_pieces import SpikePieces, assign_trials, repair_units_0_1, split_pieces, trial_runs

DELTA_TIME = 0.004  # 250 Hz, the native sampling interval of every other dataset


def build_uniform_grid(trial_start, trial_stop, delta_time=DELTA_TIME):
    """Exactly delta_time-spaced grid over [trial_start, trial_stop], built
    from an integer sample count so float steps cannot drift."""
    n_samples = int(np.floor((trial_stop - trial_start) / delta_time)) + 1
    return trial_start + np.arange(n_samples) * delta_time


def convert_one_trial(trial_id, trial_start, trial_stop, hand_timestamps, hand_xy,
                      unit_spike_times, max_gap_s, min_samples, max_speed=DEFAULT_MAX_SPEED,
                      despike_log=None):
    """Returns ((task_time, task_data, target_pos, sua_trains, mua_trains), None),
    or (None, reason) for a skipped trial. Glitches removed from a trial are
    appended to despike_log (a list) when given."""
    mask = (hand_timestamps >= trial_start) & (hand_timestamps <= trial_stop)
    t_trial = hand_timestamps[mask]
    xy_trial = hand_xy[mask]
    if len(t_trial) < 2:
        return None, f"trial {trial_id}: fewer than 2 hand samples in [{trial_start:.3f}, {trial_stop:.3f}]"

    xy_trial, glitches = despike_position(t_trial, xy_trial, max_speed)
    if despike_log is not None and glitches["n_glitch_intervals"]:
        despike_log.append({"trial_id": int(trial_id), **glitches})

    max_gap = np.max(np.diff(t_trial))
    if max_gap > max_gap_s:
        return None, f"trial {trial_id}: hand-tracking gap of {max_gap * 1000:.1f} ms exceeds --max-gap-ms"

    task_time = build_uniform_grid(t_trial[0], t_trial[-1])
    if len(task_time) < min_samples:
        return None, f"trial {trial_id}: only {len(task_time)} samples after resampling, need >= {min_samples}"

    task_pos = np.stack([np.interp(task_time, t_trial, xy_trial[:, k]) for k in range(2)], axis=1)
    # Same formula and padding as process_data.py.
    task_vel = np.diff(task_pos, axis=0) / DELTA_TIME
    task_acc = np.diff(task_vel, axis=0) / DELTA_TIME
    task_vel = np.concatenate((task_vel, task_vel[-1:, :]), axis=0)
    task_acc = np.concatenate((task_acc, task_acc[-2:, :]), axis=0)
    task_data = np.concatenate((task_pos, task_vel, task_acc), axis=1).astype(np.float32)
    target_pos = np.zeros((len(task_time), 2), dtype=np.float32)

    sua_trains = []
    for spk in unit_spike_times:
        spk = np.asarray(spk)
        sua_trains.append(spk[(spk >= task_time[0]) & (spk <= task_time[-1])].astype(np.float64))
    mua_trains = [s.copy() for s in sua_trains]
    return (task_time.astype(np.float64), task_data, target_pos, sua_trains, mua_trains), None


def main(args):
    from pynwb import NWBHDF5IO   # only needed here, so the functions above work without pynwb

    session_id = os.path.splitext(os.path.basename(args.nwb_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    stale = glob.glob(os.path.join(args.output_dir, f"{session_id}_trial*.h5"))
    if stale:
        if not args.overwrite:
            raise SystemExit(f"{len(stale)} trial file(s) of {session_id} already in {args.output_dir}; "
                             f"they would mix with this run's. Re-run with --overwrite to delete them.")
        for path in stale:
            os.remove(path)
        print(f"Removed {len(stale)} trial file(s) of an earlier run from {args.output_dir}")

    print(f"Reading NWB file: {args.nwb_path}")
    io = NWBHDF5IO(args.nwb_path, "r")
    nwbfile = io.read()
    hand_series = nwbfile.processing["behavior"].data_interfaces["Position"]["Hand"]
    hand_xy = np.asarray(hand_series.data)[:, :2]
    hand_timestamps = np.asarray(hand_series.timestamps)
    units_df = nwbfile.units.to_dataframe()
    unit_spike_times = [np.asarray(units_df.iloc[i]["spike_times"]) for i in range(len(units_df))]
    trials_df = nwbfile.trials.to_dataframe()
    n_trials = len(trials_df)
    print(f"Units: {len(unit_spike_times)}, trials: {n_trials}")

    # Runs: hand pieces, the trial -> piece assignment, spikes per piece (nwb_pieces.py).
    pieces = split_pieces(hand_timestamps)
    hand_ranges = [(hand_timestamps[a], hand_timestamps[b - 1]) for a, b in pieces]
    print(f"Hand pieces: {len(pieces)} " + ", ".join(f"[{lo:.0f}, {hi:.0f}] s" for lo, hi in hand_ranges))
    unit_spike_times, repair_note = repair_units_0_1(unit_spike_times)
    if repair_note:
        print(repair_note)
    trial_piece, reasons, assignment = assign_trials(hand_timestamps, hand_xy, pieces, trials_df)
    spikes = SpikePieces(unit_spike_times, hand_ranges)
    if len(pieces) > 1:
        print(f"Trial assignment: {assignment}")
        print(f"Spike pieces: {spikes.summary()}")
    piece_spikes = {p: spikes.trial_spikes(p) for p in sorted(set(trial_piece.tolist())) if p >= 0}
    runs = trial_runs(trials_df["start_time"].to_numpy(float))

    max_gap_s = args.max_gap_ms / 1000.0
    despike_log, skipped = [], Counter()
    n_written = 0
    for trial_id in range(n_trials):
        trial_start = float(trials_df.iloc[trial_id]["start_time"])
        trial_stop = float(trials_df.iloc[trial_id]["stop_time"])
        p = int(trial_piece[trial_id])
        if p < 0:
            skipped[(int(runs[trial_id]), reasons[trial_id])] += 1
            continue
        if spikes.window_is_ambiguous(p, trial_start, trial_stop):
            skipped[(int(runs[trial_id]), "spike pieces overlap in the trial window")] += 1
            continue
        a, b = pieces[p]
        result, skip_reason = convert_one_trial(
            trial_id, trial_start, trial_stop, hand_timestamps[a:b], hand_xy[a:b], piece_spikes[p],
            max_gap_s, args.min_samples, max_speed=args.max_speed, despike_log=despike_log)
        if result is None:
            print(f"  [skip] {skip_reason}")
            skipped[(int(runs[trial_id]), "hand-tracking gap" if "gap" in skip_reason
                     else "too few hand samples")] += 1
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
            f.attrs["hand_piece"] = p
        n_written += 1
    io.close()
    print(f"\n{n_written} trial file(s) written to {args.output_dir}, {n_trials - n_written} skipped")
    for (run, why), count in sorted(skipped.items()):
        print(f"  run {run}: {count} skipped, {why}")

    n_glitches = sum(g["n_glitch_intervals"] for g in despike_log)
    print(f"Despike (--max-speed {args.max_speed:g}): {n_glitches} glitch interval(s) removed in "
          f"{len(despike_log)} trial(s)")
    kept = trial_piece >= 0
    report_path = os.path.join(args.output_dir, f"{session_id}_conversion_report.json")
    with open(report_path, "w") as f:
        json.dump({"session_id": session_id, "n_trials_total": n_trials, "n_trials_written": n_written,
                   "hand_pieces": [{"rows": [a, b], "clock_s": [float(lo), float(hi)]}
                                   for (a, b), (lo, hi) in zip(pieces, hand_ranges)],
                   "units_repair": repair_note, "assignment": assignment, "spikes": spikes.summary(),
                   "trials_per_run_and_piece": [{"run": int(r), "piece": int(p), "n": int(n)} for (r, p), n in
                                                sorted(Counter(zip(runs[kept], trial_piece[kept])).items())],
                   "skipped": [{"run": r, "reason": why, "n": n} for (r, why), n in sorted(skipped.items())],
                   "max_speed": args.max_speed, "n_trials_with_glitches": len(despike_log),
                   "n_glitch_intervals": n_glitches, "glitches": despike_log}, f, indent=1)
    print(f"Conversion report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nwb-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-gap-ms", type=float, default=20.0,
                        help="Skip trials with a larger hand-tracking gap instead of interpolating "
                             "through it (the reference decoder's 20 ms bin)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Minimum 250 Hz samples to keep a trial. The velocity/acceleration "
                             "derivation needs at least 4; short trials still give the SNN data "
                             "even when they are shorter than one 65-sample ANN window.")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_SPEED,
                        help="Native position intervals faster than this (units/s) are tracking "
                             "glitches and are removed before resampling (hkm_despike.py); 0 disables")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete this session's existing trial files in --output-dir first")
    main(parser.parse_args())

"""
Extract spike trains and kinematics from one raw session file (MATLAB v7.3
.mat, as distributed at https://zenodo.org/record/3854034).

Input  (--input_filepath):  raw .mat with cursor_pos, target_pos, t, spikes
                            (a channels x units cell array of spike times;
                            unit 0 on each channel is unsorted).
Output (--output_filepath): .h5 with
    task_time   (N,)       sample times, 4 ms apart
    task_data   (N, 6)     pos_x, pos_y, vel_x, vel_y, acc_x, acc_y
    target_pos  (N, 2)
    sua_trains  ragged     one spike-time array per sorted unit (unit > 0, non-empty)
    mua_trains  ragged     one spike-time array per channel (all units merged)
All spike times are restricted to [task_time[0], task_time[-1]].
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bmi.utils import flatten_list

NUM_CHANNELS = 96
# make_dataset.py and make_snn_dataset.py assume this sampling interval.
EXPECTED_DELTA_TIME = 0.004


def main(args):
    print(f"Reading raw data from file: {args.input_filepath}")
    with h5py.File(args.input_filepath, 'r') as f:
        task_pos = f['cursor_pos'][()].T      # (N, 2)
        target_pos = f['target_pos'][()].T    # (N, 2)
        task_time = f['t'][()].squeeze()      # (N,)
        spike_refs = f['spikes'][()].T        # (channels, units) of HDF5 references
        num_units = spike_refs.shape[1]
        print(f"Number of channels: {NUM_CHANNELS}, number of units: {num_units}")
        # MATLAB stores an empty cell as a 1-D placeholder; real spike lists are (1, n_spikes).
        all_spikes = [[f[spike_refs[c, u]][()].squeeze(axis=0) if f[spike_refs[c, u]].ndim == 2
                       else np.empty(0)
                       for u in range(num_units)]
                      for c in range(NUM_CHANNELS)]

    def in_task(spikes):
        return spikes[(spikes >= task_time[0]) & (spikes <= task_time[-1])]

    sua_trains = [in_task(all_spikes[c][u])
                  for c in range(NUM_CHANNELS) for u in range(1, num_units)
                  if all_spikes[c][u].shape[0] > 0]
    mua_trains = [np.asarray(sorted(flatten_list([in_task(s) for s in all_spikes[c] if s.shape[0] > 0])))
                  for c in range(NUM_CHANNELS)]
    print(f"Number of SUA: {len(sua_trains)}, Number of MUA: {len(mua_trains)}")

    dt_task = np.diff(task_time).mean()
    if not np.isclose(dt_task, EXPECTED_DELTA_TIME, rtol=0.01):
        raise ValueError(
            f"Sampling interval is {dt_task:.6f}s, but the downstream windowing assumes "
            f"{EXPECTED_DELTA_TIME}s. Check this session for dropped samples or a different rate.")
    print(f"Native sampling interval: {dt_task * 1000:.4f} ms")

    # Forward differences, padded at the end to keep N rows.
    task_vel = np.diff(task_pos, axis=0) / dt_task   # mm/s
    task_acc = np.diff(task_vel, axis=0) / dt_task   # mm/s^2
    task_vel = np.concatenate((task_vel, task_vel[-1:, :]), axis=0)
    task_acc = np.concatenate((task_acc, task_acc[-2:, :]), axis=0)
    task_data = np.concatenate((task_pos, task_vel, task_acc), axis=1)

    with h5py.File(args.output_filepath, 'w') as f:
        f['task_time'] = task_time
        f['task_data'] = task_data
        f['target_pos'] = target_pos
        ragged = h5py.special_dtype(vlen=np.dtype('f8'))
        f.create_dataset('sua_trains', data=np.asarray(sua_trains, dtype=ragged))
        f.create_dataset('mua_trains', data=np.asarray(mua_trains, dtype=ragged))
    print(f"Finished processing and storing spike and kinematic data into file: {args.output_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_filepath', type=str, required=True, help='Raw .mat session file')
    parser.add_argument('--output_filepath', type=str, required=True, help='Output .h5 file')
    main(parser.parse_args())

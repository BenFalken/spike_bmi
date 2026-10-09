"""
TRANSITION ONLY: test_all_decoders.py for HKM datasets built before trial IDs
were stored (the hkm_modified-era nwb_conversion). Delete this file once the
datasets are rebuilt with nwb_conversion/run_nwb_pipeline.sh.

Same arguments and outputs as test_all_decoders.py. The only difference is
how each SNN test trial is matched to its ANN test rows. Without trial IDs,
each trial (in file order) is located by its ground-truth velocity:
timesteps [65, T) of a trial must equal T - 65 consecutive ANN test rows,
searched first right after the previous trial's rows, then anywhere (the
method of the old hkm_modified inference). Every trial then gets its file
index as its trial ID, and decoder_eval.py's usual checks run unchanged.

The old ANN and SNN datasets split train/test separately (by rows and by
samples), so near the boundary one may hold out a trial the other trains
on. ANN test rows no SNN trial matches are dropped, so every decoder is
scored on the same rows; SNN test trials that match no ANN rows are dropped
too (and so are left out of the training-loss check). Both are reported.

Usage: as test_all_decoders.py, or run_inference.sbatch ... LEGACY_HKM=1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_all_decoders as tad  # noqa: E402  (sets up threads and TensorFlow first)
import decoder_eval as de  # noqa: E402

import numpy as np  # noqa: E402

_matched_trials = {}   # snn_dataset_path -> file indices of trials with ANN rows
_load_session_data = tad.load_session_data
_load_test_trials = de.load_test_trials


def match_trials_by_velocity(trials, y_test_vel, rtol=1e-4, atol=1e-3):
    """(row_trial (n_rows,) file index of the trial owning each row, -1 for
    none; file indices of the trials with T > 65 that matched no rows)."""
    n_rows = len(y_test_vel)
    row_trial = np.full(n_rows, -1)
    cursor, unmatched = 0, []
    for i, trial in enumerate(trials):
        n = len(trial['velocity']) - de.BASE_NPERSEG
        if n <= 0:
            continue
        target = trial['velocity'][de.BASE_NPERSEG:]

        def fits(a):
            return (a + n <= n_rows and (row_trial[a:a + n] < 0).all()
                    and np.allclose(y_test_vel[a:a + n], target, rtol=rtol, atol=atol))

        start = cursor if fits(cursor) else None
        if start is None:
            candidates = np.flatnonzero(np.all(np.isclose(y_test_vel[:max(n_rows - n + 1, 0)], target[0],
                                                          rtol=rtol, atol=atol), axis=1))
            start = next((int(a) for a in candidates if fits(a)), None)
        if start is None:
            unmatched.append(i)
            continue
        row_trial[start:start + n] = i
        cursor = start + n
    return row_trial, unmatched


def load_session_data(args):
    data = _load_session_data(args)
    if data['test_trial_id'] is not None or args.experiment != 'hkm' or not args.snn_dataset_path:
        return data                       # a rebuilt dataset: nothing to do
    trials = _load_test_trials(args.snn_dataset_path)
    row_trial, unmatched = match_trials_by_velocity(trials, data['y_test_vel'])
    keep = row_trial >= 0
    if not keep.any():
        raise ValueError("No SNN test trial matches the ANN test rows by velocity: the ANN and SNN "
                         "datasets do not describe the same trials")
    n_long = sum(len(t['velocity']) > de.BASE_NPERSEG for t in trials)
    print(f"[legacy] matched {n_long - len(unmatched)}/{n_long} SNN test trials longer than "
          f"{de.BASE_NPERSEG} samples to ANN test rows by velocity, covering {keep.sum()}/{len(keep)} rows")
    if unmatched:
        print(f"[legacy] {len(unmatched)} SNN test trial(s) match no ANN test rows and are left out: "
              f"files {unmatched[:10]}{' ...' if len(unmatched) > 10 else ''}")
    if not keep.all():
        print(f"[legacy] {(~keep).sum()} ANN test rows belong to no SNN test trial and are left out "
              f"for every decoder")
    _matched_trials[args.snn_dataset_path] = set(np.unique(row_trial[keep]).tolist())
    data = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    data['test_trial_id'] = row_trial[keep]
    return data


def load_test_trials(snn_dataset_path):
    """The test trials with their file index as trial ID; with a legacy
    dataset, long trials that matched no ANN rows are left out."""
    trials = _load_test_trials(snn_dataset_path)
    matched = _matched_trials.get(snn_dataset_path)
    kept = []
    for i, trial in enumerate(trials):
        if trial['trial_id'] is None:
            trial['trial_id'] = i
        if (matched is not None and len(trial['velocity']) > de.BASE_NPERSEG
                and trial['trial_id'] not in matched):
            continue
        kept.append(trial)
    return kept


tad.load_session_data = load_session_data
de.load_test_trials = load_test_trials

if __name__ == '__main__':
    args = tad.build_parser().parse_args()
    if ({'snn', 'speck'} & set(args.decoders.split(','))
            and (args.snn_checkpoint_path or args.speck_checkpoint_path) and not args.snn_dataset_path):
        raise SystemExit("--snn_dataset_path is required to evaluate the SNN")
    tad.main(args)

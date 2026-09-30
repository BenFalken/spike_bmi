"""
Evaluate every trained decoder (KF, WF, LSTM, QRNN, SNN, and the SNN on a
Speck2f devkit) on one session and write the results to one JSON file (--output).

The file has three sections, each computed only where missing (or with
--overwrite), so an interrupted run resumes where it stopped:

    full       accuracy of the full-data models on the chronological test split:
               RMSE, per-axis CC and R^2, chunked 95% CIs, and an op-count energy
               estimate (op_energy_estimate.py) per decoder; 'speck' instead has
               its measured on-chip latency and power under 'chip'. Re-run
               (for every decoder, as they share one scored range) when a
               requested decoder with a model is not in it yet.
    durations  the same, per training duration ("1min", ..., from
               --train_durations, none by default), for every decoder except
               'speck' with a duration-tagged model (null for a duration with
               no trained models yet)
    profiles   per machine (--machine): single-sample latency, measured energy
               and its method, and parameter count per decoder (profiling.py;
               'speck' copied from its chip measurement)

The full-data predictions are also saved next to --output as
<session>_arrays.npz, so --figures_dir can redraw the per-session figures
(session_figures.py) without evaluating again, e.g. without the chip.

A session file made on the cluster can therefore be copied to the
Speck-connected machine and completed there with --decoders ...,speck: its
durations and cluster profile are kept, and the laptop adds its own profile.

make_report.py combines these files across sessions into
combined_metrics.json / combined_metrics_durations.json and draws the report
figures. See decoder_eval.py for how the decoders are loaded and aligned.

Usage (run_inference.sbatch runs this once per session):
    python test_all_decoders.py --experiment bmi --session indy_20160407_02 \
        --input_filepath .../dataset/bmi/indy/mua/indy_20160407_02_binning.h5 \
        --model_dir .../results/model_cache/bmi/indy/indy_20160407_02 \
        --snn_checkpoint_path .../snn_checkpoints/bmi/indy/per_session/indy_20160407_02/best_model_weights.pth \
        --snn_dataset_path .../snn_datasets/bmi/indy/mua_8_group/indy_20160407_02 \
        --output .../results/test_all_decoders/bmi/indy/sessions/indy_20160407_02.json
"""

import os
import sys

# TensorFlow is only used for inference here; keep it off the GPU unless asked
# (must be set before TensorFlow is imported).
if os.environ.get("EVAL_ALL_DECODERS_USE_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# energy_meter pins the math-library thread counts, so it must be imported
# before numpy, torch or TensorFlow.
from energy_meter import EnergyMeter, pin_torch_threads  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

pin_torch_threads()

from decoder_eval import (ALL_DECODERS, DEFAULT_CI_N_SPLITS, available_decoders,  # noqa: E402
                          duration_checkpoint_path, evaluate_decoders, test_split_start)
from profiling import profile_decoders  # noqa: E402


def load_session_data(args):
    with h5py.File(args.input_filepath, 'r') as f:
        X = f[f'X_{args.feature}'][()]
        y_task = f['y_task'][()]
        n_train_attr = f.attrs.get('n_train')
    n_train = test_split_start(len(X), args.test_frac, n_train_attr, args.n_train_override)
    print(f"Chronological split: {n_train} train rows, {len(X) - n_train} test rows")
    return {'X_test': X[n_train:], 'y_test_full': y_task[n_train:],
            'y_test_vel': y_task[n_train:, 2:4], 'y_test_pos': y_task[n_train:, 0:2]}


def _json_ready(result):
    return {k: v for k, v in result.items() if k != 'arrays'}


def save_arrays(arrays, path):
    np.savez_compressed(path, y_true=arrays['y_true'], y_pos=arrays['y_pos'],
                        **{f"pred_{name}": y for name, y in arrays['pred'].items()})


def load_arrays(path):
    with np.load(path) as f:
        return {'y_true': f['y_true'], 'y_pos': f['y_pos'],
                'pred': {k[len('pred_'):]: f[k] for k in f.files if k.startswith('pred_')}}


def _load_results(path):
    with open(path, 'r') as f:
        results = json.load(f)
    if 'profile' in results:     # single cluster profile written before 'profiles'
        old = results.pop('profile') or {}
        for p in old.get('decoders', {}).values():
            p.setdefault('energy_method', old.get('energy_method'))
        results.setdefault('profiles', {})['cluster'] = {'decoders': old.get('decoders', {})}
    return results


def draw_figures(full, session, figures_dir):
    from session_figures import save_session_figures
    print(f"\n--- Figures -> {figures_dir} ---")
    save_session_figures(full, session, figures_dir)


def main(args):
    decoders = [d.strip() for d in args.decoders.split(',') if d.strip()]
    unknown = set(decoders) - set(ALL_DECODERS)
    if unknown:
        raise ValueError(f"Unknown decoder(s) {sorted(unknown)}; choose from {ALL_DECODERS}")
    duration_decoders = [d for d in decoders if d != 'speck']
    session = args.session or os.path.basename(args.input_filepath).replace('_binning.h5', '')

    results = {}
    if os.path.exists(args.output) and not args.overwrite:
        results = _load_results(args.output)
    results.update(session=session, experiment=args.experiment)
    results.setdefault('durations', {})
    results.setdefault('profiles', {})

    def save():
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    def snn_duration_path(minutes):
        return duration_checkpoint_path(args.snn_checkpoint_path, minutes) if args.snn_checkpoint_path else None

    def profiled():
        return results['profiles'].get(args.machine, {}).get('decoders', {})

    evaluated = results['full']['decoders'] if 'full' in results else []
    new_decoders = [d for d in available_decoders(args, decoders, None, args.snn_checkpoint_path)
                    if d not in evaluated]
    # The full-data predictions, kept so figures can be redrawn without re-evaluating.
    arrays_path = os.path.splitext(args.output)[0] + '_arrays.npz'
    todo_full = ('full' not in results or new_decoders
                 or (args.figures_dir and not os.path.exists(arrays_path)))
    # A duration recorded as None had no trained models; retry it only if some have appeared.
    durations = [float(d) for d in args.train_durations.split(',') if d.strip()]
    todo_durations = [d for d in durations if f"{d:g}min" not in results['durations']
                      or (results['durations'][f"{d:g}min"] is None and available_decoders(
                          args, duration_decoders, f"{d:g}min", snn_duration_path(d)))]
    todo_profile = not args.skip_profile and (todo_full or any(
        d not in profiled() for d in evaluated if d in decoders))
    if not (todo_full or todo_durations or todo_profile):
        if args.figures_dir:
            draw_figures(dict(results['full'], arrays=load_arrays(arrays_path)), session, args.figures_dir)
        print(f"[skip] {session}: {args.output} is complete")
        return

    print(f"=== {session} ===")
    data = load_session_data(args)

    if todo_full:
        print("\n--- Full-data models ---")
        full = evaluate_decoders(data, args, decoders, snn_checkpoint_path=args.snn_checkpoint_path)
        if full is None:
            raise FileNotFoundError(f"No trained decoders found for {session}")
        results['full'] = _json_ready(full)
        save()
        save_arrays(full['arrays'], arrays_path)
    if args.figures_dir:
        draw_figures(full if todo_full else dict(results['full'], arrays=load_arrays(arrays_path)),
                     session, args.figures_dir)

    for minutes in todo_durations:
        tag = f"{minutes:g}min"
        print(f"\n--- Training duration {tag} ---")
        result = evaluate_decoders(data, args, duration_decoders, duration_minutes=minutes,
                                   snn_checkpoint_path=snn_duration_path(minutes))
        results['durations'][tag] = None if result is None else _json_ready(result)
        save()

    todo = [d for d in results['full']['decoders'] if d in decoders and d not in profiled()]
    if not args.skip_profile and todo:
        print(f"\n--- Latency / energy profiling ({args.machine}) ---")
        profile = profile_decoders(data, args, todo, snn_checkpoint_path=args.snn_checkpoint_path,
                                   energy_meter_cls=None if args.skip_energy else EnergyMeter)
        chip = results['full']['metrics'].get('speck', {}).get('chip')
        if 'speck' in todo and chip:
            profile['speck'] = {'latency_s': chip['latency_s'], 'energy_j': chip['energy_j'],
                                'energy_method': 'chip_power_monitor', 'param_count': chip['param_count']}
        entry = results['profiles'].setdefault(args.machine, {'decoders': {}})
        entry['host'] = platform.node()
        entry['decoders'].update(profile)
        save()
    print(f"\nSaved {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    io = parser.add_argument_group('inputs and output')
    io.add_argument('--experiment', required=True, choices=['bmi', 'hkm'],
                    help='Selects the SNN model module (models/model_{experiment}.py)')
    io.add_argument('--session', default=None, help='Session ID (default: from --input_filepath)')
    io.add_argument('--input_filepath', required=True, help='ANN dataset (make_dataset.py output)')
    io.add_argument('--model_dir', required=True,
                    help='Model cache for this session; bundles are read from <model_dir>/<feature>/')
    io.add_argument('--snn_checkpoint_path', default=None,
                    help='Full-data SNN checkpoint. Duration-tagged SNNs are looked up under a '
                         'sibling duration_sweep/<session>/<N>min/ directory.')
    io.add_argument('--snn_dataset_path', default=None,
                    help="Session's SNN dataset directory (with test/*.pkl)")
    io.add_argument('--output', required=True, help='Session results JSON')
    io.add_argument('--figures_dir', default=None,
                    help='Also save per-session figures and crosshair GIFs here (session_figures.py)')
    io.add_argument('--overwrite', action='store_true', help='Recompute every section')

    ev = parser.add_argument_group('evaluation')
    ev.add_argument('--decoders', default='kf,wf,lstm,qrnn,snn',
                    help=f'Comma-separated subset of {ALL_DECODERS}')
    ev.add_argument('--feature', default='mua', choices=['sua', 'mua'])
    ev.add_argument('--test_frac', type=float, default=0.1,
                    help='Must match the value the models were trained with')
    ev.add_argument('--n_train_override', type=int, default=None,
                    help='Explicit train/test boundary row (default: the dataset n_train '
                         'attribute, else the SNN-aligned chronological split)')
    ev.add_argument('--train_durations', default='',
                    help='Comma-separated training durations (minutes) to evaluate, e.g. 1,2,...,10')
    ev.add_argument('--ci_n_splits', type=int, default=DEFAULT_CI_N_SPLITS,
                    help='Contiguous chunks used for the within-session CIs')
    ev.add_argument('--continuous_snn_test_stream', action='store_true',
                    help='HKM only: carry SNN state across test trials instead of resetting')
    ev.add_argument('--verbose', type=int, default=0)

    speck = parser.add_argument_group("the 'speck' decoder (needs samna and a connected devkit)")
    speck.add_argument('--speck_devkit', default='speck2fdevkit:0')
    speck.add_argument('--speck_wait_time', type=float, default=0.001,
                       help='Seconds to wait after each timestep\'s input before reading output spikes')
    speck.add_argument('--speck_raster_dt', type=float, default=0.1,
                       help='dt passed to ChipFactory.raster_to_events()')

    prof = parser.add_argument_group('profiling')
    prof.add_argument('--machine', default='cluster' if 'SLURM_JOB_ID' in os.environ else 'local',
                      help='Label for this machine\'s profile; latencies are compared only within one')
    prof.add_argument('--skip_profile', action='store_true', help='Skip latency/energy profiling')
    prof.add_argument('--skip_energy', action='store_true', help='Profile latency only')
    prof.add_argument('--n_timing_samples', type=int, default=50,
                      help='Single-sample predictions (SNN: timesteps) per timing pass')
    prof.add_argument('--n_energy_repeats', type=int, default=10,
                      help='Timing passes inside the energy-measurement window')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    if ({'snn', 'speck'} & set(args.decoders.split(',')) and args.snn_checkpoint_path
            and not args.snn_dataset_path):
        raise SystemExit("--snn_dataset_path is required to evaluate the SNN")
    main(args)
